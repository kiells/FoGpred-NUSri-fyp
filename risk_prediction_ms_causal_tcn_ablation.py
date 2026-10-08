import os
import glob
import copy
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体（可选，根据系统环境）
mpl.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
mpl.rcParams['axes.unicode_minus'] = False

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score
)
from scipy.signal import butter, filtfilt

# =========================
# 0. Config
# =========================
SEED = 42
DATA_DIR = "./prefog_5s"
SAVE_CSV = "risk_prediction_ablation_results.csv"
PREDICTION_CSV = "risk_test_predictions.csv"

WINDOW_SIZE = 128
OVERLAP = 0.5
FS_HZ = 64

BATCH_SIZE = 128
MAX_EPOCHS = 100
PATIENCE = 8
LR = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.2

LOWPASS_CUTOFF = 20.0
FILTER_ORDER = 4

VAL_RATIO = 0.20
TEST_RATIO = 0.10

# ===== 原始标签 =====
# 0 = 非实验数据（剔除）
# 1 = normal gait
# 2 = pre-fog
# 3 = fog
NORMAL_ORIG_LABEL = 1
PREFOG_ORIG_LABEL = 2
FOG_ORIG_LABEL = 3

# ===== 投票标签参数 =====
VOTE_MODE = "tail_priority"
TAIL_RATIO = 0.25
MIN_RATIO = 0.4

# ===== 类别处理 =====
USE_WEIGHTED_SAMPLER = True
RISK_CLASS_WEIGHT_BOOST = 2.0

# ===== 推理后处理 =====
USE_RISK_THRESHOLD = True
RISK_THRESHOLD = 0.65
RISK_MARGIN = 0.03
USE_SMOOTHING = True
SMOOTH_MIN_RUN = 4

# ===== 事件级评估 =====
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
STEP_SEC = STEP_SIZE / FS_HZ

# 有效预警窗口（秒）：只在 FoG onset 前的这个时间窗口内寻找有效预警
VALID_WARNING_SEC = 5.0

# ===== 基线模型参数 =====
MS_BRANCH_KERNELS = [3, 5, 7]
MS_DILATIONS = [1, 2, 4]

# =========================
# 消融实验配置（用于论文展示）
# =========================
ABLATION_CONFIGS = [
    # 1. 完整模型（Baseline）
    {
        "name": "Full_Model",
        "description": "完整模型：多尺度 + SE + 时间注意力",
        "use_multiscale": True,
        "kernels": MS_BRANCH_KERNELS,
        "use_se": True,
        "use_temporal_attention": True,
        "use_dilation": True,
        "dilations": MS_DILATIONS,
        "use_postprocessing": True,
    },
    # 2. 移除SE模块
    {
        "name": "w/o_SE",
        "description": "移除 SE 模块",
        "use_multiscale": True,
        "kernels": MS_BRANCH_KERNELS,
        "use_se": False,
        "use_temporal_attention": True,
        "use_dilation": True,
        "dilations": MS_DILATIONS,
        "use_postprocessing": True,
    },
    # 3. 移除时间注意力
    {
        "name": "w/o_Temporal_Attention",
        "description": "移除时间注意力",
        "use_multiscale": True,
        "kernels": MS_BRANCH_KERNELS,
        "use_se": True,
        "use_temporal_attention": False,
        "use_dilation": True,
        "dilations": MS_DILATIONS,
        "use_postprocessing": True,
    },
    # 4. 移除多尺度（使用单一尺度）
    {
        "name": "w/o_Multi_Scale",
        "description": "移除多尺度（使用单一尺度 kernel=5）",
        "use_multiscale": False,
        "kernels": [5],
        "use_se": True,
        "use_temporal_attention": True,
        "use_dilation": True,
        "dilations": MS_DILATIONS,
        "use_postprocessing": True,
    },
    # 5. 移除膨胀卷积
    {
        "name": "w/o_Dilation",
        "description": "移除膨胀卷积（dilation=1）",
        "use_multiscale": True,
        "kernels": MS_BRANCH_KERNELS,
        "use_se": True,
        "use_temporal_attention": True,
        "use_dilation": False,
        "dilations": [1, 1, 1],
        "use_postprocessing": True,
    },
    # 6. 移除后处理（不使用阈值和平滑）
    {
        "name": "w/o_Post_processing",
        "description": "移除后处理（不使用阈值和平滑）",
        "use_multiscale": True,
        "kernels": MS_BRANCH_KERNELS,
        "use_se": True,
        "use_temporal_attention": True,
        "use_dilation": True,
        "dilations": MS_DILATIONS,
        "use_postprocessing": False,
    },
]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# 1. Preprocessing
# =========================
@dataclass
class RecordingSplitInfo:
    file_name: str
    n_total: int
    n_train: int
    n_test: int
    n_val: int
    has_fog: bool
    test_contains_fog: bool
    test_start: Optional[int]
    test_end: Optional[int]
    val_start: int
    val_end: int
    note: str


def butter_lowpass_filter(data: np.ndarray, cutoff_hz: float, fs_hz: float, order: int = 4) -> np.ndarray:
    if cutoff_hz >= fs_hz / 2:
        raise ValueError(f"cutoff_hz={cutoff_hz} must be < Nyquist={fs_hz / 2:.3f}")

    nyq = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    filtered = np.zeros_like(data, dtype=np.float32)
    min_len = max(len(a), len(b)) * 3

    for j in range(data.shape[1]):
        x = data[:, j].astype(np.float64)
        if len(x) < min_len:
            filtered[:, j] = x.astype(np.float32)
        else:
            filtered[:, j] = filtfilt(b, a, x).astype(np.float32)

    return filtered


def find_test_segment_with_fog(
    labels: np.ndarray,
    test_len: int,
    val_start: int,
    preferred_search_start: int,
    fog_label: int = FOG_ORIG_LABEL
) -> Optional[Tuple[int, int]]:
    latest_start = val_start - test_len
    if latest_start < 0 or test_len <= 0:
        return None

    preferred_search_start = max(0, preferred_search_start)
    preferred_search_start = min(preferred_search_start, latest_start)

    for start in range(latest_start, preferred_search_start - 1, -1):
        end = start + test_len
        seg = labels[start:end]
        if np.any(seg == fog_label):
            return start, end

    return None


def temporal_split_single_recording(
    labels: np.ndarray,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    fog_label: int = FOG_ORIG_LABEL
):
    n = len(labels)
    if n < 20:
        raise ValueError(f"recording too short: n={n}")

    val_len = max(1, int(round(n * val_ratio)))
    test_len = max(1, int(round(n * test_ratio)))

    val_start = n - val_len
    val_end = n

    has_fog = bool(np.any(labels == fog_label))

    if not has_fog:
        train_slice = slice(0, val_start)
        test_slice = None
        val_slice = slice(val_start, val_end)
        meta = {
            "has_fog": False,
            "test_contains_fog": False,
            "test_start": None,
            "test_end": None,
            "val_start": val_start,
            "val_end": val_end,
            "note": "No FoG in this recording; test split skipped."
        }
        return train_slice, test_slice, val_slice, meta

    preferred_search_start = max(0, int(round(n * (1.0 - val_ratio - 2 * test_ratio))))

    found = find_test_segment_with_fog(
        labels=labels,
        test_len=test_len,
        val_start=val_start,
        preferred_search_start=preferred_search_start,
        fog_label=fog_label
    )

    if found is None:
        found = find_test_segment_with_fog(
            labels=labels,
            test_len=test_len,
            val_start=val_start,
            preferred_search_start=0,
            fog_label=fog_label
        )

    if found is None:
        train_slice = slice(0, val_start)
        test_slice = None
        val_slice = slice(val_start, val_end)
        meta = {
            "has_fog": True,
            "test_contains_fog": False,
            "test_start": None,
            "test_end": None,
            "val_start": val_start,
            "val_end": val_end,
            "note": "FoG exists, but no valid contiguous test segment found before val."
        }
        return train_slice, test_slice, val_slice, meta

    test_start, test_end = found

    train_slice = slice(0, test_start)
    test_slice = slice(test_start, test_end)
    val_slice = slice(val_start, val_end)

    meta = {
        "has_fog": True,
        "test_contains_fog": True,
        "test_start": test_start,
        "test_end": test_end,
        "val_start": val_start,
        "val_end": val_end,
        "note": "Temporal split with contiguous train/test/val."
    }
    return train_slice, test_slice, val_slice, meta


def make_windows_from_block(
    features: np.ndarray,
    labels: np.ndarray,
    file_id: str,
    split_name: str,
    window_size: int = WINDOW_SIZE,
    overlap: float = OVERLAP,
    vote_mode: str = VOTE_MODE,
    tail_ratio: float = TAIL_RATIO,
    min_ratio: float = MIN_RATIO
):
    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap too large, step <= 0")

    if len(features) < window_size:
        return (
            np.empty((0, window_size, features.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            []
        )

    X_list, y_list, ids_list = [], [], []
    tail_len = max(1, int(round(window_size * tail_ratio)))

    for i in range(0, len(features) - window_size + 1, step):
        x_win = features[i:i + window_size]
        y_win = labels[i:i + window_size]
        y_tail = y_win[-tail_len:]

        if vote_mode == "tail_priority":
            prefog_ratio = np.mean(y_tail == PREFOG_ORIG_LABEL)
            fog_ratio = np.mean(y_tail == FOG_ORIG_LABEL)
            normal_ratio = np.mean(y_tail == NORMAL_ORIG_LABEL)

            if prefog_ratio >= min_ratio:
                win_label = PREFOG_ORIG_LABEL
            elif fog_ratio >= min_ratio:
                win_label = FOG_ORIG_LABEL
            elif normal_ratio >= min_ratio:
                win_label = NORMAL_ORIG_LABEL
            else:
                vals, counts = np.unique(y_tail, return_counts=True)
                win_label = vals[np.argmax(counts)]
        else:
            vals, counts = np.unique(y_tail, return_counts=True)
            win_label = vals[np.argmax(counts)]

        ids_list.append(f"{file_id}::{split_name}::{i}")
        X_list.append(x_win)
        y_list.append(win_label)

    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.int64),
        ids_list
    )


def standardize_by_train(X_train, X_val, X_test):
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def load_data_temporal_split(folder_path=DATA_DIR):
    files = sorted(glob.glob(os.path.join(folder_path, "S*R*.csv")))
    if not files:
        raise FileNotFoundError(f"在 {folder_path} 没找到 S*R*.csv 文件")

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    X_val_all, y_val_all = [], []

    train_ids_all, test_ids_all, val_ids_all = [], [], []
    split_infos = []
    feature_cols = None

    for file_path in files:
        file_name = os.path.basename(file_path).replace(".csv", "")
        df = pd.read_csv(file_path)

        if "Annot" not in df.columns:
            raise ValueError(f"{file_name} 缺少 Annot 列")

        df = df[df["Annot"] != 0].reset_index(drop=True)
        if len(df) == 0:
            continue

        cur_feature_cols = [col for col in df.columns if col not in ["Time", "Annot"]]
        if feature_cols is None:
            feature_cols = cur_feature_cols
        else:
            if cur_feature_cols != feature_cols:
                raise ValueError(f"{file_name} 的特征列与其他文件不一致")

        features = df[cur_feature_cols].values.astype(np.float32)
        labels = df["Annot"].values.astype(np.int64)

        features_filtered = butter_lowpass_filter(features, LOWPASS_CUTOFF, FS_HZ, FILTER_ORDER)

        train_slice, test_slice, val_slice, meta = temporal_split_single_recording(labels)

        feat_train = features_filtered[train_slice]
        lab_train = labels[train_slice]
        feat_val = features_filtered[val_slice]
        lab_val = labels[val_slice]

        if test_slice is not None:
            feat_test = features_filtered[test_slice]
            lab_test = labels[test_slice]
        else:
            feat_test = np.empty((0, features.shape[1]), dtype=np.float32)
            lab_test = np.empty((0,), dtype=np.int64)

        Xtr, ytr, idtr = make_windows_from_block(feat_train, lab_train, file_name, "train")
        Xva, yva, idva = make_windows_from_block(feat_val, lab_val, file_name, "val")
        Xte, yte, idte = make_windows_from_block(feat_test, lab_test, file_name, "test")

        if len(Xtr) > 0:
            X_train_all.append(Xtr)
            y_train_all.append(ytr)
            train_ids_all.extend(idtr)

        if len(Xva) > 0:
            X_val_all.append(Xva)
            y_val_all.append(yva)
            val_ids_all.extend(idva)

        if len(Xte) > 0:
            X_test_all.append(Xte)
            y_test_all.append(yte)
            test_ids_all.extend(idte)

        split_infos.append(RecordingSplitInfo(
            file_name=file_name,
            n_total=len(labels),
            n_train=len(lab_train),
            n_test=len(lab_test),
            n_val=len(lab_val),
            has_fog=meta["has_fog"],
            test_contains_fog=meta["test_contains_fog"],
            test_start=meta["test_start"],
            test_end=meta["test_end"],
            val_start=meta["val_start"],
            val_end=meta["val_end"],
            note=meta["note"]
        ))

    if feature_cols is None:
        raise RuntimeError("没有成功读取任何有效 recording")

    X_train = np.concatenate(X_train_all, axis=0)
    y_train_orig = np.concatenate(y_train_all, axis=0)
    X_test = np.concatenate(X_test_all, axis=0)
    y_test_orig = np.concatenate(y_test_all, axis=0)
    X_val = np.concatenate(X_val_all, axis=0)
    y_val_orig = np.concatenate(y_val_all, axis=0)

    X_train, X_val, X_test = standardize_by_train(X_train, X_val, X_test)

    return (
        X_train, y_train_orig, train_ids_all,
        X_val, y_val_orig, val_ids_all,
        X_test, y_test_orig, test_ids_all,
        feature_cols, split_infos
    )


# =========================
# 2. Risk reformulation
# =========================
def make_risk_labels(y_orig: np.ndarray) -> np.ndarray:
    return np.where(y_orig == NORMAL_ORIG_LABEL, 0, 1).astype(np.int64)


def compute_class_weights(y_train_risk: np.ndarray, boost=RISK_CLASS_WEIGHT_BOOST):
    weights = np.ones(2, dtype=np.float32)
    weights[1] *= boost
    return torch.tensor(weights, dtype=torch.float32)


# =========================
# 3. Sampler / loaders
# =========================
def build_weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    class_counts = np.bincount(y_train)
    class_weights = np.zeros_like(class_counts, dtype=np.float64)

    for cls_idx, cnt in enumerate(class_counts):
        if cnt > 0:
            class_weights[cls_idx] = 1.0 / cnt

    sample_weights = class_weights[y_train]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def make_loaders(X_train, y_train, X_val, y_val, X_test, y_test):
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    sampler = build_weighted_sampler(y_train) if USE_WEIGHTED_SAMPLER else None

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader


# =========================
# 4. Models
# =========================
class CausalConv1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            c_in, c_out,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias
        )

    def forward(self, x):
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)


class CausalConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            CausalConv1d(c_in, c_out, kernel_size=k, dilation=dilation, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale


class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        score = self.attn(x)
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(x * weight, dim=-1)
        return pooled, weight.squeeze(1)


class MultiScaleCausalBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernels: List[int],
        dilation: int,
        dropout: float = 0.2,
        use_se: bool = True
    ):
        super().__init__()
        assert c_out % len(kernels) == 0, "c_out must be divisible by number of kernels."
        branch_out = c_out // len(kernels)

        self.branches = nn.ModuleList([
            nn.Sequential(
                CausalConv1d(c_in, branch_out, kernel_size=k, dilation=dilation, bias=False),
                nn.BatchNorm1d(branch_out),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for k in kernels
        ])

        self.fuse = nn.Sequential(
            nn.Conv1d(c_out, c_out, kernel_size=1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(c_out, c_out, kernel_size=3, dilation=dilation, bias=False),
            nn.BatchNorm1d(c_out),
        )

        self.se = SEBlock(c_out) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False) if c_in != c_out else nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        outs = [branch(x) for branch in self.branches]
        out = torch.cat(outs, dim=1)
        out = self.fuse(out)
        out = self.se(out)
        out = self.relu(out + residual)
        return out


class MSCausalTCNBackbone(nn.Module):
    def __init__(
        self,
        num_features: int,
        use_multiscale: bool = True,
        kernels: List[int] = [3, 5, 7],
        dilations: List[int] = [1, 2, 4],
        use_se: bool = True,
        use_temporal_attention: bool = True,
        hidden_channels: int = 64
    ):
        super().__init__()
        self.input_proj = CausalConvBNAct(num_features, 64, k=3, dilation=1, dropout=DROPOUT)

        # 计算通道数，确保能被kernel数量整除
        n_kernels = len(kernels)
        # 使用126而不是128，因为126能被3整除
        base_channels = 126 if n_kernels == 3 else 128 if n_kernels == 2 else 128

        self.block1 = MultiScaleCausalBlock(
            64, base_channels, kernels=kernels, dilation=dilations[0], dropout=DROPOUT, use_se=use_se
        )
        self.block2 = MultiScaleCausalBlock(
            base_channels, base_channels, kernels=kernels, dilation=dilations[1], dropout=DROPOUT, use_se=use_se
        )
        self.block3 = MultiScaleCausalBlock(
            base_channels, base_channels, kernels=kernels, dilation=dilations[2], dropout=DROPOUT, use_se=use_se
        )

        self.temporal_pool = TemporalAttentionPooling(base_channels, hidden_channels=hidden_channels) if use_temporal_attention else None
        self.output_channels = base_channels

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.input_proj(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        if self.temporal_pool is not None:
            pooled, attn = self.temporal_pool(x)
        else:
            pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
            attn = None
        return pooled, attn


class MSCausalTCNRiskModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_classes=2,
        use_multiscale: bool = True,
        kernels: List[int] = [3, 5, 7],
        dilations: List[int] = [1, 2, 4],
        use_se: bool = True,
        use_temporal_attention: bool = True
    ):
        super().__init__()
        self.backbone = MSCausalTCNBackbone(
            num_features=num_features,
            use_multiscale=use_multiscale,
            kernels=kernels,
            dilations=dilations,
            use_se=use_se,
            use_temporal_attention=use_temporal_attention
        )
        hidden_dim = self.backbone.output_channels
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        pooled, attn = self.backbone(x)
        logits = self.fc(pooled)
        return logits, attn


# =========================
# 5. Metrics / postprocess
# =========================
@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    all_preds, all_true, all_probs = [], [], []

    for bx, by in loader:
        bx = bx.to(device)
        logits, _ = model(bx)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.append(preds)
        all_true.append(by.numpy())

    return np.concatenate(all_true), np.concatenate(all_preds), np.concatenate(all_probs)


def compute_risk_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    acc = accuracy_score(y_true, y_pred)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    risk_recall = recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
    risk_precision = precision_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
    risk_f1 = report["1"]["f1-score"] if "1" in report else 0.0

    # Sensitivity = Risk_Recall (对于正类的召回率)
    sensitivity = risk_recall

    # Specificity = TN / (TN + FP) = 对于负类(类别0)的召回率
    specificity = recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]

    return {
        "Accuracy": acc,
        "Macro_Recall": macro_recall,
        "Macro_F1": macro_f1,
        "Risk_Recall": risk_recall,
        "Risk_Precision": risk_precision,
        "Risk_F1": risk_f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
    }


def parse_record_id(full_id: str) -> str:
    return full_id.split("::")[0]


def smooth_short_runs(seq: np.ndarray, target_class: int, min_run: int, fill_class: int) -> np.ndarray:
    seq = seq.copy()
    n = len(seq)
    i = 0
    while i < n:
        if seq[i] != target_class:
            i += 1
            continue
        j = i
        while j < n and seq[j] == target_class:
            j += 1
        if (j - i) < min_run:
            seq[i:j] = fill_class
        i = j
    return seq


def apply_risk_threshold_and_smoothing(
    y_prob: np.ndarray,
    test_ids: List[str],
    use_threshold: bool = USE_RISK_THRESHOLD,
    risk_threshold: float = RISK_THRESHOLD,
    risk_margin: float = RISK_MARGIN,
    use_smoothing: bool = USE_SMOOTHING,
    smooth_min_run: int = SMOOTH_MIN_RUN
) -> np.ndarray:
    y_pred = np.argmax(y_prob, axis=1)

    if use_threshold:
        risk_prob = y_prob[:, 1]
        normal_prob = y_prob[:, 0]

        risk_mask = (y_pred == 1)
        weak_risk = risk_mask & (
            (risk_prob < risk_threshold) |
            ((risk_prob - normal_prob) < risk_margin)
        )
        y_pred[weak_risk] = 0

    if use_smoothing and smooth_min_run > 1:
        groups: Dict[str, List[int]] = {}
        for i, fid in enumerate(test_ids):
            rec = parse_record_id(fid)
            groups.setdefault(rec, []).append(i)

        for rec, inds in groups.items():
            inds = sorted(inds)
            seq = y_pred[inds]
            seq = smooth_short_runs(seq, target_class=1, min_run=smooth_min_run, fill_class=0)
            y_pred[inds] = seq

    return y_pred


def compute_event_level_metrics(y_true_orig: np.ndarray, y_pred_risk: np.ndarray, test_ids: List[str]) -> Dict[str, float]:
    groups: Dict[str, List[int]] = {}
    for i, fid in enumerate(test_ids):
        rec = parse_record_id(fid)
        groups.setdefault(rec, []).append(i)

    total_fog_events = 0
    detected_fog_events = 0
    prediction_horizons_sec = []

    # 有效预警窗口（步数）
    VALID_WARNING_STEPS = int(VALID_WARNING_SEC / STEP_SEC)

    false_alarm_segments = 0
    total_test_minutes = (len(y_true_orig) * STEP_SEC) / 60.0

    for rec, inds in groups.items():
        inds = sorted(inds)
        y_true_rec = y_true_orig[inds]
        y_pred_rec = y_pred_risk[inds]

        true_risk_rec = (y_true_rec != NORMAL_ORIG_LABEL).astype(np.int64)
        true_fog_rec = (y_true_rec == FOG_ORIG_LABEL).astype(np.int64)

        fog_onsets = []
        for t in range(len(true_fog_rec)):
            if true_fog_rec[t] == 1 and (t == 0 or true_fog_rec[t - 1] == 0):
                fog_onsets.append(t)

        total_fog_events += len(fog_onsets)

        for onset in fog_onsets:
            if onset == 0:
                continue

            # 有效预警窗口: [onset - VALID_WARNING_STEPS, onset)
            valid_window_start = max(0, onset - VALID_WARNING_STEPS)
            pred_in_window = y_pred_rec[valid_window_start:onset]

            # 在有效预警窗口内找所有 risk segment
            segments = []
            t = 0
            while t < len(pred_in_window):
                if pred_in_window[t] != 1:
                    t += 1
                    continue
                j = t
                while j < len(pred_in_window) and pred_in_window[j] == 1:
                    j += 1
                # segment 的实际开始位置（相对于整个序列）
                seg_start_actual = valid_window_start + t
                segments.append((seg_start_actual, valid_window_start + j))
                t = j

            if len(segments) > 0:
                # 取最早的有效预警 segment
                seg_start, _ = segments[0]
                detected_fog_events += 1
                horizon_steps = onset - seg_start
                prediction_horizons_sec.append(horizon_steps * STEP_SEC)

        t = 0
        while t < len(y_pred_rec):
            if y_pred_rec[t] != 1:
                t += 1
                continue
            j = t
            while j < len(y_pred_rec) and y_pred_rec[j] == 1:
                j += 1

            if np.sum(true_risk_rec[t:j]) == 0:
                false_alarm_segments += 1
            t = j

    event_recall = detected_fog_events / total_fog_events if total_fog_events > 0 else np.nan
    false_alarms_per_min = false_alarm_segments / total_test_minutes if total_test_minutes > 0 else np.nan
    prediction_horizon = float(np.mean(prediction_horizons_sec)) if len(prediction_horizons_sec) > 0 else np.nan

    return {
        "FoG_Event_Count": total_fog_events,
        "Detected_FoG_Event_Count": detected_fog_events,
        "Event_Recall": event_recall,
        "False_Alarms_per_Min": false_alarms_per_min,
        "Prediction_Horizon_sec": prediction_horizon,
    }


# =========================
# 6. Training
# =========================
def train_one_model(model, train_loader, val_loader, class_weights, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_state = None
    best_val_score = -1
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits, _ = model(bx)
                loss = criterion(logits, by)
                val_running_loss += loss.item() * bx.size(0)

        val_loss = val_running_loss / len(val_loader.dataset)

        y_val_true, y_val_pred, _ = infer(model, val_loader, device)
        val_metrics = compute_risk_metrics(y_val_true, y_val_pred)

        val_score = (
            0.5 * val_metrics["Risk_Recall"] +
            0.3 * val_metrics["Risk_F1"] +
            0.2 * val_metrics["Sensitivity"]
        )
        scheduler.step(val_score)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_Risk_F1={val_metrics['Risk_F1']:.4f} | val_score={val_score:.4f}"
            )

        if val_score > best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model


# =========================
# 7. Ablation Experiment Run
# =========================
def run_ablation_experiments(
    X_train, y_train_orig,
    X_val, y_val_orig,
    X_test, y_test_orig,
    test_ids,
    configs: List[Dict],
    save_name: str = SAVE_CSV
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train_risk = make_risk_labels(y_train_orig)
    y_val_risk = make_risk_labels(y_val_orig)
    y_test_risk = make_risk_labels(y_test_orig)

    num_features = X_train.shape[2]
    class_weights = compute_class_weights(y_train_risk, boost=RISK_CLASS_WEIGHT_BOOST)

    final_rows = []

    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENTS")
    print("=" * 80)

    for idx, config in enumerate(configs):
        name = config["name"]
        description = config["description"]

        print(f"\n[{idx+1}/{len(configs)}] Training: {name}")
        print(f"    Description: {description}")

        # 为每个配置创建新的数据加载器
        train_loader, val_loader, test_loader = make_loaders(
            X_train, y_train_risk,
            X_val, y_val_risk,
            X_test, y_test_risk
        )

        # 创建模型
        model = MSCausalTCNRiskModel(
            num_features=num_features,
            num_classes=2,
            use_multiscale=config["use_multiscale"],
            kernels=config["kernels"],
            dilations=config["dilations"],
            use_se=config["use_se"],
            use_temporal_attention=config["use_temporal_attention"]
        )

        # 训练模型
        model = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device
        )

        # 推理
        y_true_risk, _, y_prob_risk = infer(model, test_loader, device)

        # 根据配置决定是否使用后处理
        if config["use_postprocessing"]:
            y_pred_risk = apply_risk_threshold_and_smoothing(y_prob_risk, test_ids)
        else:
            # 不使用后处理：直接取 argmax
            y_pred_risk = np.argmax(y_prob_risk, axis=1)

        # 计算指标
        win_metrics = compute_risk_metrics(y_true_risk, y_pred_risk)
        event_metrics = compute_event_level_metrics(
            y_true_orig=y_test_orig,
            y_pred_risk=y_pred_risk,
            test_ids=test_ids
        )

        # 按要求只输出6个指标
        print(
            f"    Results -> "
            f"Accuracy={win_metrics['Accuracy']:.4f}, "
            f"Macro F1={win_metrics['Macro_F1']:.4f}, "
            f"Risk F1={win_metrics['Risk_F1']:.4f}, "
            f"Event Recall={event_metrics['Event_Recall']:.4f}, "
            f"FA/min={event_metrics['False_Alarms_per_Min']:.4f}, "
            f"Prediction Horizon={event_metrics['Prediction_Horizon_sec']:.4f}"
        )

        # 保存结果（只保存要求的6个指标）
        row = {
            "Experiment": name,
            "Description": description,
            "Accuracy": win_metrics['Accuracy'],
            "Macro_F1": win_metrics['Macro_F1'],
            "Risk_F1": win_metrics['Risk_F1'],
            "Event_Recall": event_metrics['Event_Recall'],
            "FA_per_min": event_metrics['False_Alarms_per_Min'],
            "Prediction_Horizon_sec": event_metrics['Prediction_Horizon_sec'],
        }
        final_rows.append(row)

        # 保存预测结果（将文件名中的 / 替换为 _）
        safe_name = name.replace("/", "_")
        pred_df = pd.DataFrame({
            "id": test_ids,
            "y_true_orig": y_test_orig,
            "y_true_risk": y_true_risk,
            "risk_prob": y_prob_risk[:, 1],
            "y_pred_risk_post": y_pred_risk,
        })
        pred_df.to_csv(f"{safe_name}_{PREDICTION_CSV}", index=False)

        # 保存混淆矩阵
        pd.DataFrame(confusion_matrix(y_true_risk, y_pred_risk)).to_csv(
            f"confusion_matrix_{safe_name}.csv", index=False
        )

    # 保存最终结果
    result_df = pd.DataFrame(final_rows)
    result_df.to_csv(save_name, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(result_df.to_string(index=False))
    print(f"\nResults saved to: {save_name}")

    return result_df


# =========================
# 8. Visualization
# =========================
def plot_ablation_comparison(csv_path: str, save_path: str = "ablation_comparison.png"):
    """
    绘制消融实验对比图
    """
    df = pd.read_csv(csv_path)

    experiments = df['Experiment'].tolist()
    metrics = ['Accuracy', 'Macro_F1', 'Risk_F1', 'Event_Recall', 'FA_per_min', 'Prediction_Horizon_sec']

    # 创建子图（增加高度以适应更高纵轴）
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = df[metric].values

        # 为每个实验创建条形图
        colors = plt.cm.Set3(np.linspace(0, 1, len(experiments)))
        bars = ax.bar(experiments, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

        # 标记基线模型
        for i, (exp, bar) in enumerate(zip(experiments, bars)):
            if exp == 'Full_Model':
                bar.set_edgecolor('red')
                bar.set_linewidth(2.5)

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=9, rotation=45)

        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # 根据不同指标设置纵轴范围
        if metric == 'Prediction_Horizon_sec':
            ax.set_ylim(4, 5)  # 有效预警窗口为 5 秒
        elif metric == 'FA_per_min':
            ax.set_ylim(0, 2.0)  # FA/min 量程
        elif metric == 'Risk_F1':
            ax.set_ylim(0.5, 0.8)  # Risk_F1 量程
        elif metric in ['Accuracy', 'Event_Recall', 'Macro_F1']:
            ax.set_ylim(0.6, 0.9)  # Accuracy/Event_Recall/Macro_F1 量程
        else:
            ax.set_ylim(0, 1.0)

    plt.suptitle('Ablation Experiment Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


# =========================
# 9. Main
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    print("=" * 80)
    print("ABLATION STUDY FOR RISK PREDICTION")
    print("=" * 80)

    print("\nLoading data with temporal split + voting labels ...")
    (
        X_train, y_train_orig, train_ids,
        X_val, y_val_orig, val_ids,
        X_test, y_test_orig, test_ids,
        feature_cols, split_infos
    ) = load_data_temporal_split(DATA_DIR)

    print("Data ready:")
    print(f"  X_train={X_train.shape}, y_train_orig={y_train_orig.shape}")
    print(f"  X_val  ={X_val.shape}, y_val_orig={y_val_orig.shape}")
    print(f"  X_test ={X_test.shape}, y_test_orig={y_test_orig.shape}")
    print(f"  Labels(orig)={sorted(np.unique(np.concatenate([y_train_orig, y_val_orig, y_test_orig])).tolist())}")
    print(f"  WeightedRandomSampler = {USE_WEIGHTED_SAMPLER}")
    print(f"  Risk weight boost = {RISK_CLASS_WEIGHT_BOOST}")
    print(f"  Baseline kernels = {MS_BRANCH_KERNELS}, dilations = {MS_DILATIONS}")

    print("\nAblation configurations:")
    for idx, config in enumerate(ABLATION_CONFIGS):
        print(f"  [{idx+1}] {config['name']}: {config['description']}")

    # 运行消融实验
    result_df = run_ablation_experiments(
        X_train=X_train, y_train_orig=y_train_orig,
        X_val=X_val, y_val_orig=y_val_orig,
        X_test=X_test, y_test_orig=y_test_orig,
        test_ids=test_ids,
        configs=ABLATION_CONFIGS,
        save_name=SAVE_CSV
    )

    # 生成对比图
    print("\n" + "=" * 80)
    print("Generating comparison plot...")
    print("=" * 80)

    plot_ablation_comparison(csv_path=SAVE_CSV, save_path="ablation_comparison.png")

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETED")
    print("=" * 80)

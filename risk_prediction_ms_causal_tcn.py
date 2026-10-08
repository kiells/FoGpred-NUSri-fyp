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
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.gridspec import GridSpec

# 设置美观的样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
mpl.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['figure.facecolor'] = '#f8f9fa'
mpl.rcParams['axes.facecolor'] = '#ffffff'
mpl.rcParams['grid.alpha'] = 0.3

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from scipy.signal import butter, filtfilt
from scipy.stats import gaussian_kde

# =========================
# 0. Config
# =========================
SEED = 42
DATA_DIR = "./prefog_5s"
SAVE_CSV = "risk_prediction_ms_causal_tcn_results.csv"
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

# ===== 参数扫描 =====
RUN_PARAM_SWEEP = True  # 是否运行参数扫描
SWEEP_THRESHOLDS = np.arange(0.40, 0.85, 0.05)  # 阈值范围
SWEEP_MARGINS = np.arange(0.0, 0.11, 0.01)      # 边缘范围
SWEEP_SMOOTH_MIN_RUN = 4
SWEEP_USE_SMOOTHING = True
SWEEP_SAVE_CSV = "threshold_margin_sweep_results.csv"

# ===== 事件级评估 =====
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
STEP_SEC = STEP_SIZE / FS_HZ

# 有效预警窗口（秒）：只在 FoG onset 前的这个时间窗口内寻找有效预警
VALID_WARNING_SEC = 5.0

# ===== 模型结构 =====
MS_BRANCH_KERNELS = [3, 5, 7]
MS_DILATIONS = [1, 2, 4]
USE_SE = True
USE_TEMPORAL_ATTENTION = True
ATTN_HIDDEN = 64


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
    def __init__(self, in_channels: int, hidden_channels: int = ATTN_HIDDEN):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.Conv1d(hidden_channels, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        score = self.attn(x)                    # [B, 1, T]
        weight = torch.softmax(score, dim=-1)  # [B, 1, T]
        pooled = torch.sum(x * weight, dim=-1) # [B, C]
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
    def __init__(self, num_features: int):
        super().__init__()
        self.input_proj = CausalConvBNAct(num_features, 64, k=3, dilation=1, dropout=DROPOUT)
        # 使用126而不是128，因为126能被len(MS_BRANCH_KERNELS)=3整除
        self.block1 = MultiScaleCausalBlock(
            64, 126, kernels=MS_BRANCH_KERNELS, dilation=MS_DILATIONS[0], dropout=DROPOUT, use_se=USE_SE
        )
        self.block2 = MultiScaleCausalBlock(
            126, 126, kernels=MS_BRANCH_KERNELS, dilation=MS_DILATIONS[1], dropout=DROPOUT, use_se=USE_SE
        )
        self.block3 = MultiScaleCausalBlock(
            126, 126, kernels=MS_BRANCH_KERNELS, dilation=MS_DILATIONS[2], dropout=DROPOUT, use_se=USE_SE
        )

        self.temporal_pool = TemporalAttentionPooling(126, hidden_channels=ATTN_HIDDEN) if USE_TEMPORAL_ATTENTION else None

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
    def __init__(self, num_features, num_classes=2):
        super().__init__()
        self.backbone = MSCausalTCNBackbone(num_features)
        self.fc = nn.Sequential(
            nn.Linear(126, 64),
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
    median_prediction_horizon = float(np.median(prediction_horizons_sec)) if len(prediction_horizons_sec) > 0 else np.nan

    return {
        "FoG_Event_Count": total_fog_events,
        "Detected_FoG_Event_Count": detected_fog_events,
        "Event_Recall": event_recall,
        "False_Alarms_per_Min": false_alarms_per_min,
        "Prediction_Horizon_sec": prediction_horizon,
        "Median_Prediction_Horizon_sec": median_prediction_horizon,
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
            0.5 * val_metrics["Risk_Recall"] +    # 首要：尽量不漏报风险
            0.3 * val_metrics["Risk_F1"] +        # 平衡：兼顾 Precision
            0.2 * val_metrics["Sensitivity"]      # = Risk_Recall，强化召回
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
# 7. Run
# =========================
def run_risk_prediction(
    X_train, y_train_orig,
    X_val, y_val_orig,
    X_test, y_test_orig,
    test_ids,
    save_name=SAVE_CSV
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train_risk = make_risk_labels(y_train_orig)
    y_val_risk = make_risk_labels(y_val_orig)
    y_test_risk = make_risk_labels(y_test_orig)

    train_loader, val_loader, test_loader = make_loaders(
        X_train, y_train_risk,
        X_val, y_val_risk,
        X_test, y_test_risk
    )

    num_features = X_train.shape[2]
    class_weights = compute_class_weights(y_train_risk, boost=RISK_CLASS_WEIGHT_BOOST)

    model_configs = {
        "MS_Causal_TCN_Risk": lambda: MSCausalTCNRiskModel(num_features, num_classes=2),
    }

    final_rows = []

    for name, factory in model_configs.items():
        print("\n" + "=" * 80)
        print(f"Training model: {name}")

        model = factory()
        model = train_one_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device
        )

        y_true_risk, _, y_prob_risk = infer(model, test_loader, device)
        y_pred_risk = apply_risk_threshold_and_smoothing(y_prob_risk, test_ids)

        win_metrics = compute_risk_metrics(y_true_risk, y_pred_risk)

        try:
            risk_auc = roc_auc_score(y_true_risk, y_prob_risk[:, 1])
        except Exception:
            risk_auc = np.nan

        event_metrics = compute_event_level_metrics(
            y_true_orig=y_test_orig,
            y_pred_risk=y_pred_risk,
            test_ids=test_ids
        )

        print(
            f"[TEST] {name} -> "
            f"Acc={win_metrics['Accuracy']:.4f}, "
            f"Macro_F1={win_metrics['Macro_F1']:.4f}, "
            f"Sensitivity={win_metrics['Sensitivity']:.4f}, "
            f"Specificity={win_metrics['Specificity']:.4f}, "
            f"Risk_Recall={win_metrics['Risk_Recall']:.4f}, "
            f"Risk_Precision={win_metrics['Risk_Precision']:.4f}, "
            f"Risk_F1={win_metrics['Risk_F1']:.4f}, "
            f"Event_Recall={event_metrics['Event_Recall']:.4f}, "
            f"FA/min={event_metrics['False_Alarms_per_Min']:.4f}, "
            f"PredictionHorizon={event_metrics['Prediction_Horizon_sec']:.4f}"
        )

        row = {"Experiment": name, "Risk_AUC": risk_auc}
        row.update(win_metrics)
        row.update(event_metrics)
        final_rows.append(row)

        pd.DataFrame(confusion_matrix(y_true_risk, y_pred_risk)).to_csv(
            f"confusion_matrix_{name}.csv", index=False
        )

        pred_df = pd.DataFrame({
            "id": test_ids,
            "y_true_orig": y_test_orig,
            "y_true_risk": y_true_risk,
            "risk_prob": y_prob_risk[:, 1],
            "y_pred_risk_post": y_pred_risk,
        })
        pred_df.to_csv(f"{name}_{PREDICTION_CSV}", index=False)

    result_df = pd.DataFrame(final_rows).sort_values(
        by=["Risk_F1", "Event_Recall", "Macro_F1", "Accuracy"],
        ascending=False
    )
    result_df.to_csv(save_name, index=False)

    # 保存训练好的模型用于跨数据集验证
    model_save_path = "trained_model.pkl"
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: {model_save_path}")

    print("\nFinal risk-prediction results:")
    print(result_df)
    print(f"\nSaved results to: {save_name}")
    return result_df, model, y_test_orig, y_pred_risk, y_prob_risk[:, 1], event_metrics


# =========================
# 7.5 Parameter Sweep
# =========================
def run_threshold_margin_sweep(
    model, test_loader, device,
    y_test_orig, test_ids,
    thresholds: np.ndarray = np.arange(0.40, 0.85, 0.05),
    margins: np.ndarray = np.arange(0.0, 0.11, 0.01),
    smooth_min_run: int = SMOOTH_MIN_RUN,
    use_smoothing: bool = True,
    save_csv: str = "threshold_margin_sweep_results.csv"
):
    """
    Sweep over threshold and margin combinations to find optimal trade-off.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch device
        y_test_orig: Original labels for event-level metrics
        test_ids: Test sample IDs
        thresholds: Array of threshold values to test
        margins: Array of margin values to test
        smooth_min_run: Minimum run length for smoothing
        use_smoothing: Whether to apply smoothing
        save_csv: Filename to save results
    """
    print("\n" + "=" * 80)
    print(f"Running parameter sweep: {len(thresholds)} thresholds × {len(margins)} margins")
    print(f"Total combinations: {len(thresholds) * len(margins)}")

    # Get model predictions once
    model.eval()
    with torch.no_grad():
        all_probs = []
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits, _ = model(bx)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
        y_prob_risk = np.concatenate(all_probs)

    y_true_risk = make_risk_labels(y_test_orig)

    results = []

    for thresh in thresholds:
        for margin in margins:
            y_pred_risk = apply_risk_threshold_and_smoothing(
                y_prob_risk,
                test_ids,
                use_threshold=True,
                risk_threshold=thresh,
                risk_margin=margin,
                use_smoothing=use_smoothing,
                smooth_min_run=smooth_min_run
            )

            # Window-level metrics
            win_metrics = compute_risk_metrics(y_true_risk, y_pred_risk)

            # Event-level metrics
            event_metrics = compute_event_level_metrics(
                y_true_orig=y_test_orig,
                y_pred_risk=y_pred_risk,
                test_ids=test_ids
            )

            # Compute ROC-AUC
            try:
                risk_auc = roc_auc_score(y_true_risk, y_prob_risk[:, 1])
            except Exception:
                risk_auc = np.nan

            row = {
                "Threshold": thresh,
                "Margin": margin,
                "Risk_AUC": risk_auc,
            }
            row.update(win_metrics)
            row.update(event_metrics)
            results.append(row)

            # Print progress every 10 combinations
            if len(results) % (len(margins) * 2) == 0:
                print(f"  Progress: {len(results)}/{len(thresholds) * len(margins)} combinations")

    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)

    print(f"\nSweep complete! Results saved to: {save_csv}")

    # Find best configs for different metrics
    print("\n" + "-" * 80)
    print("Best configurations:")

    metrics_to_check = [
        "Risk_F1",
        "Risk_Recall",
        "Risk_Precision",
        "Sensitivity",
        "Specificity",
        "Macro_F1",
        "Event_Recall"
    ]

    for metric in metrics_to_check:
        best_idx = df[metric].idxmax()
        best_row = df.iloc[best_idx]
        print(f"  {metric:20s}: Thresh={best_row['Threshold']:.2f}, "
              f"Margin={best_row['Margin']:.2f}, Value={best_row[metric]:.4f}")

    print("-" * 80)

    return df


# =========================
# 7.6 Visualization Functions
# =========================
def set_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True, spine_style=True):
    """设置图表样式"""
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, fontweight='semibold')
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, fontweight='semibold')
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    if spine_style:
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
            spine.set_linewidth(0.8)
    ax.tick_params(axis='both', labelsize=10, colors='#333333')


def plot_confusion_matrix_heatmap(y_true, y_pred, save_path="confusion_matrix_heatmap.png"):
    """
    绘制美观的混淆矩阵热图
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold', y=0.98)

    # 左侧：原始计数
    im1 = ax1.imshow(cm, interpolation='nearest', cmap='Blues', alpha=0.8)
    ax1.set_title('Count Matrix', fontsize=12, fontweight='bold')

    # 添加数值标签
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax1.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                    color=text_color, fontsize=14, fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Normal', 'Risk'], fontsize=11, fontweight='semibold')
    ax1.set_yticklabels(['Normal', 'Risk'], fontsize=11, fontweight='semibold')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='semibold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='semibold')

    # 颜色条
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Count', fontsize=10, fontweight='semibold')

    # 右侧：归一化百分比
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='RdYlGn', vmin=0, vmax=1)
    ax2.set_title('Normalized Matrix (%)', fontsize=12, fontweight='bold')

    # 添加百分比标签
    for i in range(2):
        for j in range(2):
            text_color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax2.text(j, i, f'{cm_norm[i, j]:.1%}', ha='center', va='center',
                    color=text_color, fontsize=14, fontweight='bold')

    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Normal', 'Risk'], fontsize=11, fontweight='semibold')
    ax2.set_yticklabels(['Normal', 'Risk'], fontsize=11, fontweight='semibold')
    ax2.set_ylabel('True Label', fontsize=11, fontweight='semibold')
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='semibold')

    # 颜色条
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Percentage', fontsize=10, fontweight='semibold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Confusion matrix heatmap saved to: {save_path}")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path="roc_curve.png"):
    """
    绘制ROC曲线
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制ROC曲线
    ax.plot(fpr, tpr, color='#e74c3c', linewidth=3, label=f'ROC Curve (AUC = {auc:.4f})')

    # 填充曲线下方区域
    ax.fill_between(fpr, tpr, alpha=0.3, color='#e74c3c')

    # 绘制对角线
    ax.plot([0, 1], [0, 1], color='#95a5a6', linestyle='--', linewidth=2,
           label='Random Classifier (AUC = 0.5)')

    # 添加最佳工作点
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = 0.5  # 近似阈值
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx], color='#27ae60', s=200, zorder=5,
              edgecolors='white', linewidth=2, label=f'Optimal Point\n(TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f})')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='semibold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='semibold')
    ax.set_title('ROC Curve Analysis', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    set_plot_style(ax, grid=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  ROC curve saved to: {save_path}")
    plt.close()


def plot_prediction_distribution(y_true, y_prob, save_path="prediction_distribution.png"):
    """
    绘制预测概率分布直方图
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Prediction Probability Distribution Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 分离正负样本的概率
    neg_probs = y_prob[y_true == 0]
    pos_probs = y_prob[y_true == 1]

    # 左侧：叠加直方图
    ax1 = axes[0]
    bins = np.linspace(0, 1, 50)
    ax1.hist(neg_probs, bins=bins, alpha=0.6, color='#2ecc71', label='Normal', edgecolor='white', linewidth=1.5)
    ax1.hist(pos_probs, bins=bins, alpha=0.6, color='#e74c3c', label='Risk', edgecolor='white', linewidth=1.5)
    ax1.axvline(x=0.5, color='#3498db', linestyle='--', linewidth=2, alpha=0.8, label='Threshold (0.5)')
    ax1.set_xlabel('Predicted Probability', fontsize=11, fontweight='semibold')
    ax1.set_ylabel('Count', fontsize=11, fontweight='semibold')
    ax1.set_title('Class Probability Distribution', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper center', fontsize=10, framealpha=0.9)
    set_plot_style(ax1, grid=True)

    # 右侧：核密度估计
    ax2 = axes[1]
    from scipy.stats import gaussian_kde

    if len(neg_probs) > 1 and len(pos_probs) > 1:
        kde_neg = gaussian_kde(neg_probs)
        kde_pos = gaussian_kde(pos_probs)
        x_range = np.linspace(0, 1, 200)

        ax2.plot(x_range, kde_neg(x_range), color='#2ecc71', linewidth=3, label='Normal')
        ax2.fill_between(x_range, kde_neg(x_range), alpha=0.3, color='#2ecc71')
        ax2.plot(x_range, kde_pos(x_range), color='#e74c3c', linewidth=3, label='Risk')
        ax2.fill_between(x_range, kde_pos(x_range), alpha=0.3, color='#e74c3c')

    ax2.axvline(x=0.5, color='#3498db', linestyle='--', linewidth=2, alpha=0.8, label='Threshold (0.5)')
    ax2.set_xlabel('Predicted Probability', fontsize=11, fontweight='semibold')
    ax2.set_ylabel('Density', fontsize=11, fontweight='semibold')
    ax2.set_title('Class Probability Density (KDE)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)
    set_plot_style(ax2, grid=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Prediction distribution saved to: {save_path}")
    plt.close()


def plot_prediction_timeline(y_true_orig, y_pred_risk, y_prob, save_path="prediction_timeline.png", n_samples=1000):
    """
    绘制预测时间序列可视化
    """
    n_samples = min(n_samples, len(y_true_orig))
    y_true_sub = y_true_orig[:n_samples]
    y_pred_sub = y_pred_risk[:n_samples]
    y_prob_sub = y_prob[:n_samples]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    fig.suptitle('Prediction Timeline Analysis (First {} Samples)'.format(n_samples),
                 fontsize=16, fontweight='bold', y=0.98)

    x = np.arange(len(y_true_sub))

    # 子图1：真实标签
    ax1 = axes[0]
    y_true_risk = (y_true_sub != NORMAL_ORIG_LABEL).astype(int)
    colors_true = ['#2ecc71' if y == 0 else '#e74c3c' for y in y_true_risk]
    ax1.scatter(x, y_true_risk, c=colors_true, alpha=0.6, s=15, edgecolors='white', linewidth=0.5)
    ax1.fill_between(x, y_true_risk, alpha=0.3, where=y_true_risk == 1, color='#e74c3c', label='Risk')
    ax1.fill_between(x, y_true_risk, alpha=0.3, where=y_true_risk == 0, color='#2ecc71', label='Normal')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='semibold')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Normal', 'Risk'], fontsize=10)
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
    set_plot_style(ax1, grid=True)

    # 子图2：预测概率
    ax2 = axes[1]
    ax2.plot(x, y_prob_sub, color='#3498db', linewidth=1.5, alpha=0.8, label='Risk Probability')
    ax2.axhline(y=0.5, color='#95a5a6', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')
    ax2.fill_between(x, y_prob_sub, 0.5, where=y_prob_sub >= 0.5, alpha=0.3, color='#3498db')
    ax2.set_ylabel('Risk Probability', fontsize=11, fontweight='semibold')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    set_plot_style(ax2, grid=True)

    # 子图3：预测标签
    ax3 = axes[2]
    correct = (y_true_risk == y_pred_sub)
    colors_pred = []
    for i, (true, pred) in enumerate(zip(y_true_risk, y_pred_sub)):
        if pred == 1:
            colors_pred.append('#e74c3c' if correct[i] else '#c0392b')
        else:
            colors_pred.append('#2ecc71' if correct[i] else '#27ae60')

    ax3.scatter(x, y_pred_sub, c=colors_pred, alpha=0.6, s=15, edgecolors='white', linewidth=0.5)
    ax3.fill_between(x, y_pred_sub, alpha=0.3, where=y_pred_sub == 1, color='#e74c3c')
    ax3.fill_between(x, y_pred_sub, alpha=0.3, where=y_pred_sub == 0, color='#2ecc71')
    ax3.set_ylabel('Predicted Label', fontsize=11, fontweight='semibold')
    ax3.set_xlabel('Sample Index', fontsize=11, fontweight='semibold')
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(['Normal', 'Risk'], fontsize=10)
    set_plot_style(ax3, grid=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Prediction timeline saved to: {save_path}")
    plt.close()


def plot_threshold_heatmap(csv_path, save_path="threshold_margin_heatmap.png"):
    """
    绘制阈值-边缘参数扫描热图
    """
    df = pd.read_csv(csv_path)

    # 选择 Risk_F1 作为主要指标
    pivot_table = df.pivot(index='Threshold', columns='Margin', values='Risk_F1')

    fig, ax = plt.subplots(figsize=(14, 10))

    # 绘制热图
    im = ax.imshow(pivot_table.values, cmap='RdYlGn', aspect='auto', origin='lower')

    # 设置坐标轴
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_xticklabels([f'{m:.2f}' for m in pivot_table.columns], fontsize=10)
    ax.set_yticklabels([f'{t:.2f}' for t in pivot_table.index], fontsize=10)

    # 添加数值标签
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            value = pivot_table.iloc[i, j]
            text_color = 'white' if value > 0.5 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                   color=text_color, fontsize=8, fontweight='bold')

    # 颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Risk F1 Score', fontsize=11, fontweight='semibold')

    ax.set_xlabel('Margin', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Threshold', fontsize=12, fontweight='semibold')
    ax.set_title('Parameter Sweep Heatmap: Risk F1 Score', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Threshold heatmap saved to: {save_path}")
    plt.close()


def plot_event_detection_analysis(event_metrics, save_path="event_detection_analysis.png"):
    """
    绘制事件检测分析图
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle('Event Detection Performance Analysis',
                 fontsize=16, fontweight='bold', y=0.98)

    # 子图1：事件检测对比
    ax1 = fig.add_subplot(gs[0, 0])
    events = ['Total FoG\nEvents', 'Detected\nEvents']
    counts = [event_metrics['FoG_Event_Count'], event_metrics['Detected_FoG_Event_Count']]
    colors = ['#3498db', '#27ae60']

    bars = ax1.bar(events, counts, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(count)}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.set_ylabel('Count', fontsize=11, fontweight='semibold')
    ax1.set_title('FoG Event Detection', fontsize=12, fontweight='bold')
    set_plot_style(ax1, grid=True)

    # 子图2：事件召回率
    ax2 = fig.add_subplot(gs[0, 1])
    event_recall = event_metrics['Event_Recall']

    # 绘制圆环图
    colors_ring = ['#27ae60', '#e74c3c']
    sizes = [event_recall, 1 - event_recall]

    wedges, texts, autotexts = ax2.pie(sizes, colors=colors_ring, autopct='%1.1f%%',
                                       startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title(f'Event Recall\n({event_recall:.2%})', fontsize=12, fontweight='bold')

    # 中心圆
    centre_circle = Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    ax2.text(0, 0, f'{event_recall:.2%}', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#27ae60')

    # 子图3：预测时间分布
    ax3 = fig.add_subplot(gs[0, 2])
    pred_horizon = event_metrics['Prediction_Horizon_sec']
    median_horizon = event_metrics['Median_Prediction_Horizon_sec']

    metrics_horizon = ['Mean\nPrediction\nHorizon', 'Median\nPrediction\nHorizon']
    values_horizon = [pred_horizon, median_horizon]
    colors_horizon = ['#3498db', '#e74c3c']

    bars3 = ax3.bar(metrics_horizon, values_horizon, color=colors_horizon,
                    alpha=0.7, edgecolor='white', linewidth=2)
    for bar, val in zip(bars3, values_horizon):
        if not pd.isna(val):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.axhline(y=3.0, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7, label='Target (3s)')
    ax3.set_ylabel('Seconds', fontsize=11, fontweight='semibold')
    ax3.set_title('Prediction Horizon', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.9)
    set_plot_style(ax3, grid=True)

    # 子图4：误报率分析
    ax4 = fig.add_subplot(gs[1, 0])
    fa_per_min = event_metrics['False_Alarms_per_Min']

    # 绘制仪表盘样式（使用笛卡尔坐标）
    fa_color = '#27ae60' if fa_per_min <= 0.5 else '#f39c12' if fa_per_min <= 1.0 else '#e74c3c'

    # 绘制背景半圆
    theta = np.linspace(0, np.pi, 100)
    x_bg = np.cos(theta)
    y_bg = np.sin(theta)
    ax4.plot(x_bg, y_bg, color='#ecf0f1', linewidth=30, alpha=0.5)

    # 根据误报率绘制弧线
    max_fa = 2.0
    fa_ratio = min(fa_per_min / max_fa, 1.0)
    theta_fa = np.linspace(0, fa_ratio * np.pi, 100)
    x_fa = np.cos(theta_fa)
    y_fa = np.sin(theta_fa)
    ax4.plot(x_fa, y_fa, color=fa_color, linewidth=30, alpha=0.8)

    # 设置样式
    ax4.set_xlim(-1.3, 1.3)
    ax4.set_ylim(-0.2, 1.3)
    ax4.set_aspect('equal')
    ax4.axis('off')
    ax4.set_title(f'False Alarm Rate\n{fa_per_min:.2f} / min', fontsize=12, fontweight='bold', pad=20)

    # 添加中心数值
    ax4.text(0, 0, f'{fa_per_min:.2f}', ha='center', va='center',
            fontsize=20, fontweight='bold', color=fa_color)

    # 添加目标标记
    ax4.plot([np.cos(np.pi * 0.5 / max_fa)], [np.sin(np.pi * 0.5 / max_fa)],
            'o', color='#27ae60', markersize=8, alpha=0.7, label='Target (0.5)')

    # 子图5：综合性能卡片
    ax5 = fig.add_subplot(gs[1, 1])

    # 创建表格
    card_data = [
        ['Event Recall', f'{event_metrics["Event_Recall"]:.2%}'],
        ['Pred. Horizon', f'{event_metrics["Prediction_Horizon_sec"]:.2f}s'],
        ['FA/min', f'{event_metrics["False_Alarms_per_Min"]:.2f}'],
    ]

    table = ax5.table(cellText=card_data, cellLoc='left', loc='center',
                     colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # 设置单元格样式
    for i, cell in enumerate(table.get_celld().values()):
        if i % 2 == 0:  # 标签列
            cell.set_facecolor('#ecf0f1')
            cell.set_text_props(weight='semibold')
        else:  # 数值列
            cell.set_facecolor('#ffffff')
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('#bdc3c7')

    ax5.axis('off')
    ax5.set_title('Event Detection Summary', fontsize=12, fontweight='bold', pad=20)

    # 子图6：性能雷达图
    ax6 = fig.add_subplot(gs[1, 2], projection='polar')

    # 归一化指标
    metrics_radar = [
        min(event_metrics['Event_Recall'] / 0.8, 1.0),
        min((5.0 - event_metrics['Prediction_Horizon_sec']) / 2.0, 1.0) if not pd.isna(event_metrics['Prediction_Horizon_sec']) else 0.5,
        max(1.0 - event_metrics['False_Alarms_per_Min'] / 2.0, 0.0)
    ]
    labels_radar = ['Event\nRecall', 'Prediction\nHorizon', 'Low\nFA Rate']

    angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
    metrics_radar += [metrics_radar[0]]
    angles += angles[:1]

    ax6.plot(angles, metrics_radar, 'o-', linewidth=3, color='#e74c3c')
    ax6.fill(angles, metrics_radar, alpha=0.25, color='#e74c3c')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels_radar, fontsize=10, fontweight='semibold')
    ax6.set_ylim(0, 1)
    ax6.set_yticks([0.5, 1.0])
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Event Detection Profile', fontsize=12, fontweight='bold', pad=10)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Event detection analysis saved to: {save_path}")
    plt.close()


def plot_comprehensive_dashboard(result_df, sweep_df, save_path="comprehensive_dashboard.png"):
    """
    绘制综合仪表板
    """
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('MS-Causal-TCN Risk Prediction: Comprehensive Dashboard',
                 fontsize=18, fontweight='bold', y=0.995)

    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    # 第1行：关键指标卡片
    metrics_cards = [
        ('Accuracy', 'Accuracy', '#3498db'),
        ('Macro F1', 'Macro_F1', '#2ecc71'),
        ('Risk F1', 'Risk_F1', '#e74c3c'),
        ('Risk AUC', 'Risk_AUC', '#9b59b6'),
    ]

    for idx, (title, col, color) in enumerate(metrics_cards):
        ax = fig.add_subplot(gs[0, idx])
        value = result_df[col].iloc[0]

        # 绘制圆角矩形背景
        rect = FancyBboxPatch((-0.4, 0.2), 0.8, 0.6, boxstyle="round,pad=0.1",
                             facecolor=color, alpha=0.15, edgecolor=color, linewidth=2)
        ax.add_patch(rect)

        # 添加数值
        ax.text(0, 0.5, f'{value:.3f}', ha='center', va='center',
               fontsize=24, fontweight='bold', color=color)
        ax.text(0, 0.85, title, ha='center', va='center',
               fontsize=12, fontweight='semibold', color='#2c3e50')

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.axis('off')

    # 第2行：指标对比
    ax1 = fig.add_subplot(gs[1, :2])
    metrics = ['Sensitivity', 'Specificity', 'Risk_Recall', 'Risk_Precision', 'Event_Recall']
    values = [result_df[m].iloc[0] for m in metrics]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.axhline(y=0.8, color='#27ae60', linestyle='--', linewidth=2, alpha=0.7, label='Target (0.8)')
    ax1.set_ylabel('Score', fontsize=11, fontweight='semibold')
    ax1.set_title('Performance Metrics', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.9)
    set_plot_style(ax1, grid=True)

    # 事件分析
    ax2 = fig.add_subplot(gs[1, 2:])
    event_metrics = {
        'FoG_Event_Count': result_df['FoG_Event_Count'].iloc[0],
        'Detected_FoG_Event_Count': result_df['Detected_FoG_Event_Count'].iloc[0],
        'Event_Recall': result_df['Event_Recall'].iloc[0],
        'Prediction_Horizon_sec': result_df['Prediction_Horizon_sec'].iloc[0],
        'Median_Prediction_Horizon_sec': result_df['Median_Prediction_Horizon_sec'].iloc[0],
        'False_Alarms_per_Min': result_df['False_Alarms_per_Min'].iloc[0],
    }

    categories = ['Total\nEvents', 'Detected\nEvents', 'Event\nRecall',
                  'Pred.\nHorizon', 'Median\nHorizon', 'FA/min']
    values_events = [
        event_metrics['FoG_Event_Count'],
        event_metrics['Detected_FoG_Event_Count'],
        event_metrics['Event_Recall'],
        event_metrics['Prediction_Horizon_sec'] if not pd.isna(event_metrics['Prediction_Horizon_sec']) else 0,
        event_metrics['Median_Prediction_Horizon_sec'] if not pd.isna(event_metrics['Median_Prediction_Horizon_sec']) else 0,
        event_metrics['False_Alarms_per_Min'],
    ]
    colors_events = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#1abc9c']

    bars2 = ax2.bar(categories, values_events, color=colors_events, alpha=0.7, edgecolor='white', linewidth=2)
    for bar, val, cat in zip(bars2, values_events, categories):
        height = bar.get_height()
        if 'Recall' in cat:
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        elif 'FA' in cat:
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        elif 'Horizon' in cat and val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{val:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(val)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_ylabel('Value', fontsize=11, fontweight='semibold')
    ax2.set_title('Event Detection Analysis', fontsize=12, fontweight='bold')
    set_plot_style(ax2, grid=True)

    # 第3行：参数扫描最佳配置
    ax3 = fig.add_subplot(gs[2, 0])
    best_configs = sweep_df.groupby('Threshold').agg({'Risk_F1': 'mean'}).sort_values('Risk_F1', ascending=False).head(5)

    y_pos = np.arange(len(best_configs))
    bars3 = ax3.barh(y_pos, best_configs['Risk_F1'], color='#3498db', alpha=0.7, edgecolor='white', linewidth=2)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([f'{t:.2f}' for t in best_configs.index], fontsize=10)
    ax3.set_xlabel('Risk F1 Score', fontsize=11, fontweight='semibold')
    ax3.set_title('Top 5 Threshold Configurations', fontsize=12, fontweight='bold')
    ax3.set_xlim([0, 1])
    set_plot_style(ax3, grid=True)

    # 添加数值标签
    for bar, val in zip(bars3, best_configs['Risk_F1']):
        ax3.text(val, bar.get_y() + bar.get_height() / 2.,
                f' {val:.3f}', va='center', fontsize=9, fontweight='bold')

    # 雷达图
    ax4 = fig.add_subplot(gs[2, 1], projection='polar')
    radar_metrics = [
        result_df['Accuracy'].iloc[0],
        result_df['Macro_F1'].iloc[0],
        result_df['Risk_F1'].iloc[0],
        result_df['Sensitivity'].iloc[0],
        result_df['Specificity'].iloc[0],
        result_df['Event_Recall'].iloc[0],
    ]
    radar_labels = ['Acc', 'MF1', 'RF1', 'Sens', 'Spec', 'ERec']

    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    radar_metrics += [radar_metrics[0]]
    angles += angles[:1]

    ax4.plot(angles, radar_metrics, 'o-', linewidth=2, color='#e74c3c')
    ax4.fill(angles, radar_metrics, alpha=0.25, color='#e74c3c')
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_labels, fontsize=9, fontweight='semibold')
    ax4.set_ylim(0, 1)
    ax4.set_yticks([0.5, 1.0])
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Performance Profile', fontsize=12, fontweight='bold', pad=10)

    # 阈值分析
    ax5 = fig.add_subplot(gs[2, 2:])

    # 绘制 Risk_F1 vs Threshold 曲线
    threshold_data = sweep_df[sweep_df['Margin'] == 0.0].sort_values('Threshold')
    ax5.plot(threshold_data['Threshold'], threshold_data['Risk_F1'],
            color='#3498db', linewidth=2.5, marker='o', markersize=6, label='Risk F1')
    ax5.plot(threshold_data['Threshold'], threshold_data['Event_Recall'],
            color='#e74c3c', linewidth=2.5, marker='s', markersize=6, label='Event Recall')
    ax5.plot(threshold_data['Threshold'], threshold_data['Risk_Precision'],
            color='#2ecc71', linewidth=2.5, marker='^', markersize=6, label='Risk Precision')

    # 标记最佳点
    best_idx = threshold_data['Risk_F1'].idxmax()
    best_row = threshold_data.loc[best_idx]
    ax5.scatter(best_row['Threshold'], best_row['Risk_F1'],
               color='#9b59b6', s=200, zorder=5, edgecolors='white', linewidth=2,
               label=f'Best: T={best_row["Threshold"]:.2f}')

    ax5.set_xlabel('Threshold', fontsize=11, fontweight='semibold')
    ax5.set_ylabel('Score', fontsize=11, fontweight='semibold')
    ax5.set_title('Threshold Performance Analysis (Margin=0.0)', fontsize=12, fontweight='bold')
    ax5.set_ylim([0, 1.05])
    ax5.legend(loc='lower right', fontsize=9, framealpha=0.9)
    set_plot_style(ax5, grid=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Comprehensive dashboard saved to: {save_path}")
    plt.close()


def plot_threshold_tradeoff(
    csv_path: str = "threshold_margin_sweep_results.csv",
    margin_value: float = 0.0,
    save_path: str = "threshold_tradeoff.png",
    figsize: tuple = (14, 7)
):
    """
    绘制 Recall-Precision-Event vs Threshold 三条线图（改进版）。

    Args:
        csv_path: 参数扫描结果 CSV 文件路径
        margin_value: 指定使用的 margin 值（通常 0.0）
        save_path: 图片保存路径
        figsize: 图片尺寸
    """
    # 读取数据
    df = pd.read_csv(csv_path)

    # 过滤指定 margin 的数据
    df_filtered = df[df['Margin'] == margin_value].copy()
    df_filtered = df_filtered.sort_values('Threshold')

    # 提取数据
    thresholds = df_filtered['Threshold'].values
    risk_recall = df_filtered['Risk_Recall'].values
    risk_precision = df_filtered['Risk_Precision'].values
    event_recall = df_filtered['Event_Recall'].values

    # 创建图表
    fig, ax1 = plt.subplots(figsize=figsize)

    # 绘制三条线（使用更美观的配色）
    ax1.plot(thresholds, risk_recall, 'o-', color='#3498db', label='Risk Recall (Sensitivity)',
             linewidth=2.5, markersize=7, alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)
    ax1.plot(thresholds, risk_precision, 's-', color='#e74c3c', label='Risk Precision',
             linewidth=2.5, markersize=7, alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)

    # 创建第二个 y 轴用于 Event_Recall
    ax2 = ax1.twinx()
    ax2.plot(thresholds, event_recall, '^-', color='#2ecc71', label='Event Recall',
             linewidth=2.5, markersize=7, alpha=0.9, markeredgecolor='white', markeredgewidth=1.5)

    # 设置坐标轴标签和标题
    ax1.set_xlabel('Risk Threshold', fontsize=13, fontweight='semibold')
    ax1.set_ylabel('Risk Metrics', fontsize=13, fontweight='semibold', color='#2c3e50')
    ax2.set_ylabel('Event Recall', fontsize=13, fontweight='semibold', color='#27ae60')
    ax1.tick_params(axis='both', labelsize=11, colors='#2c3e50')
    ax2.tick_params(axis='y', labelsize=11, labelcolor='#27ae60')

    # 设置标题
    ax1.set_title(f'Threshold Trade-off Analysis (Margin={margin_value:.2f})',
                  fontsize=15, fontweight='bold', pad=20)

    # 设置 y 轴范围
    ax1.set_ylim(0.4, 1.02)
    ax2.set_ylim(0.7, 1.02)

    # 添加网格
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='center left', bbox_to_anchor=(1.12, 0.5),
               fontsize=10, framealpha=0.95, edgecolor='#bdc3c7')

    # 添加水平参考线
    ax1.axhline(y=0.8, color='#95a5a6', linestyle=':', linewidth=2, alpha=0.6)
    ax1.text(thresholds[0], 0.805, 'Target: 0.8', fontsize=9, color='#7f8c8d', fontweight='semibold')

    # 设置边框样式
    for spine in ax1.spines.values():
        spine.set_edgecolor('#bdc3c7')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  Threshold tradeoff plot saved to: {save_path}")
    plt.close()


def plot_pr_curve(
    csv_path: str = "threshold_margin_sweep_results.csv",
    margin_value: float = 0.0,
    save_path: str = "pr_curve.png",
    figsize: tuple = (11, 11)
):
    """
    绘制 Precision-Recall 曲线（每个点是一个 threshold 配置）（改进版）。

    Args:
        csv_path: 参数扫描结果 CSV 文件路径
        margin_value: 指定使用的 margin 值
        save_path: 图片保存路径
        figsize: 图片尺寸
    """
    df = pd.read_csv(csv_path)
    df_filtered = df[df['Margin'] == margin_value].copy()

    risk_recall = df_filtered['Risk_Recall'].values
    risk_precision = df_filtered['Risk_Precision'].values
    thresholds = df_filtered['Threshold'].values

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制 PR 曲线
    points = ax.scatter(risk_recall, risk_precision, c=thresholds,
                       cmap='RdYlGn_r', s=120, alpha=0.8, edgecolors='white', linewidth=2,
                       label='Threshold Configurations')

    # 连接点
    ax.plot(risk_recall, risk_precision, color='#34495e', linestyle='--', alpha=0.4, linewidth=1.5)

    # 添加颜色条
    cbar = plt.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label('Threshold Value', fontsize=11, fontweight='semibold')
    cbar.ax.tick_params(labelsize=10)

    # 设置坐标轴
    ax.set_xlabel('Risk Recall (Sensitivity)', fontsize=13, fontweight='semibold')
    ax.set_ylabel('Risk Precision', fontsize=13, fontweight='semibold')
    ax.set_title('Precision-Recall Trade-off Analysis', fontsize=15, fontweight='bold', pad=20)

    # 设置坐标轴范围
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)

    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='lower left', fontsize=10, framealpha=0.95, edgecolor='#bdc3c7')
    ax.tick_params(axis='both', labelsize=11, colors='#2c3e50')

    # 添加对角线参考
    ax.plot([0.4, 1.0], [0.4, 1.0], color='#95a5a6', linestyle=':', alpha=0.3, linewidth=2)

    # 添加理想区域标注
    ax.fill_between([0.8, 1.0], [0.8, 1.0], alpha=0.15, color='#27ae60')
    ax.text(0.9, 0.9, 'Ideal Zone', ha='center', fontsize=11, color='#27ae60',
           fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#27ae60'))

    # 添加最佳点标记
    f1_scores = 2 * risk_recall * risk_precision / (risk_recall + risk_precision + 1e-8)
    best_idx = np.argmax(f1_scores)
    ax.scatter(risk_recall[best_idx], risk_precision[best_idx],
              color='#e74c3c', s=300, zorder=5, edgecolors='white', linewidth=3,
              marker='*', label=f'Best F1: {f1_scores[best_idx]:.3f}')

    # 设置边框样式
    for spine in ax.spines.values():
        spine.set_edgecolor('#bdc3c7')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    print(f"  PR curve saved to: {save_path}")
    plt.close()


# =========================
# 8. Main
# =========================
if __name__ == "__main__":
    set_seed(SEED)

    print("Loading data with temporal split + voting labels ...")
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
    print(f"  labels(orig)={sorted(np.unique(np.concatenate([y_train_orig, y_val_orig, y_test_orig])).tolist())}")
    print(f"  WeightedRandomSampler = {USE_WEIGHTED_SAMPLER}")
    print(f"  Risk weight boost = {RISK_CLASS_WEIGHT_BOOST}")
    print(f"  Multi-scale kernels = {MS_BRANCH_KERNELS}, dilations = {MS_DILATIONS}")
    print(f"  SE = {USE_SE}, TemporalAttention = {USE_TEMPORAL_ATTENTION}")
    print(
        f"  Postprocess: threshold={USE_RISK_THRESHOLD}, "
        f"risk_thresh={RISK_THRESHOLD}, margin={RISK_MARGIN}, "
        f"smoothing={USE_SMOOTHING}, min_run={SMOOTH_MIN_RUN}"
    )

    result_df, trained_model, y_test_orig, y_pred_risk, y_prob_risk, event_metrics = run_risk_prediction(
        X_train=X_train, y_train_orig=y_train_orig,
        X_val=X_val, y_val_orig=y_val_orig,
        X_test=X_test, y_test_orig=y_test_orig,
        test_ids=test_ids,
        save_name=SAVE_CSV
    )

    # Run parameter sweep if enabled
    if RUN_PARAM_SWEEP:
        print("\n" + "=" * 80)
        print("Running parameter sweep for threshold and margin...")
        print("=" * 80)

        # Prepare data for sweep
        y_train_risk = make_risk_labels(y_train_orig)
        y_val_risk = make_risk_labels(y_val_orig)
        y_test_risk = make_risk_labels(y_test_orig)

        _, _, test_loader = make_loaders(
            X_train, y_train_risk,
            X_val, y_val_risk,
            X_test, y_test_risk
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Run sweep using the trained model
        sweep_df = run_threshold_margin_sweep(
            model=trained_model,
            test_loader=test_loader,
            device=device,
            y_test_orig=y_test_orig,
            test_ids=test_ids,
            thresholds=SWEEP_THRESHOLDS,
            margins=SWEEP_MARGINS,
            smooth_min_run=SWEEP_SMOOTH_MIN_RUN,
            use_smoothing=SWEEP_USE_SMOOTHING,
            save_csv=SWEEP_SAVE_CSV
        )

        # Generate plots
        print("\n" + "=" * 80)
        print("Generating trade-off analysis plots...")
        print("=" * 80)

        # Plot 1: Recall-Precision-Event vs Threshold
        plot_threshold_tradeoff(
            csv_path=SWEEP_SAVE_CSV,
            margin_value=0.0,
            save_path="threshold_tradeoff.png"
        )

        # Plot 2: Precision-Recall curve
        plot_pr_curve(
            csv_path=SWEEP_SAVE_CSV,
            margin_value=0.0,
            save_path="pr_curve.png"
        )

        # Plot 3: Threshold heatmap
        plot_threshold_heatmap(
            csv_path=SWEEP_SAVE_CSV,
            save_path="threshold_margin_heatmap.png"
        )

        # Plot 4: Comprehensive dashboard with sweep results
        plot_comprehensive_dashboard(
            result_df=result_df,
            sweep_df=sweep_df,
            save_path="comprehensive_dashboard.png"
        )

    # Generate additional visualizations (always run)
    print("\n" + "=" * 80)
    print("Generating additional analysis visualizations...")
    print("=" * 80)

    y_test_risk = make_risk_labels(y_test_orig)

    # Plot: Confusion matrix
    plot_confusion_matrix_heatmap(
        y_true=y_test_risk,
        y_pred=y_pred_risk,
        save_path="confusion_matrix_heatmap.png"
    )

    # Plot: ROC curve
    plot_roc_curve(
        y_true=y_test_risk,
        y_prob=y_prob_risk,
        save_path="roc_curve.png"
    )

    # Plot: Prediction distribution
    plot_prediction_distribution(
        y_true=y_test_risk,
        y_prob=y_prob_risk,
        save_path="prediction_distribution.png"
    )

    # Plot: Prediction timeline
    plot_prediction_timeline(
        y_true_orig=y_test_orig,
        y_pred_risk=y_pred_risk,
        y_prob=y_prob_risk,
        save_path="prediction_timeline.png"
    )

    # Plot: Event detection analysis
    plot_event_detection_analysis(
        event_metrics=event_metrics,
        save_path="event_detection_analysis.png"
    )

    print("\n" + "=" * 80)
    print("All visualizations generated successfully!")
    print("=" * 80)

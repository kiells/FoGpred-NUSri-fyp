import os
import glob
import copy
import random
import gc
import matplotlib.pyplot as plt
import matplotlib
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, precision_score,
    classification_report, confusion_matrix, roc_auc_score
)
from scipy.signal import butter, filtfilt

# =========================
# 0. Config - Base Configuration
# =========================
SEED = 42
DATA_DIR = "./prefog_5s"
SAVE_CSV = "risk_prediction_ms_causal_tcn_results.csv"
PREDICTION_CSV = "risk_test_predictions.csv"

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
STEP_SIZE = None  # Will be computed based on WINDOW_SIZE and OVERLAP
STEP_SEC = None

# 有效预警窗口（秒）：只在 FoG onset 前的这个时间窗口内寻找有效预警
VALID_WARNING_SEC = 5.0

# ===== 模型结构 =====
MS_BRANCH_KERNELS = [3, 5, 7]
MS_DILATIONS = [1, 2, 4]
USE_SE = True
USE_TEMPORAL_ATTENTION = True
ATTN_HIDDEN = 64


# =========================
# Experiment Configuration
# =========================
# Define different WINDOW_SIZE to test
WINDOW_SIZE_LIST = [64, 96, 128, 160, 192, 256]  # 1s, 1.5s, 2s, 2.5s, 3s, 4s at 64Hz

# ===== FIXED OVERLAP MODE =====
# 设置为 True 则只运行固定 overlap 值的实验
USE_FIXED_OVERLAP = True
FIXED_OVERLAP = 0.25  # 固定 overlap 值 (0.5, 0.6, 0.7, 0.75, 0.8 等)

# Original OVERLAP list (only used when USE_FIXED_OVERLAP = False)
OVERLAP_LIST = [0.2, 0.25, 0.5, 0.75, 0.8]  # Different overlap ratios

# Global variables to be set per experiment
WINDOW_SIZE = None
OVERLAP = None


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cleanup_memory(verbose=False):
    """
    清理内存：垃圾回收和 CUDA 缓存清理
    Args:
        verbose: 是否打印内存使用情况
    """
    # 强制垃圾回收
    gc.collect()

    if torch.cuda.is_available():
        # 同步 CUDA 操作
        torch.cuda.synchronize()
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        # 强制释放未使用的显存
        torch.cuda.ipc_collect()

    if verbose and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


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
    global STEP_SIZE, STEP_SEC
    step = int(window_size * (1 - overlap))
    if step <= 0:
        raise ValueError("overlap too large, step <= 0")

    STEP_SIZE = step
    STEP_SEC = step / FS_HZ

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

        Xtr, ytr, idtr = make_windows_from_block(feat_train, lab_train, file_name, "train", WINDOW_SIZE, OVERLAP)
        Xva, yva, idva = make_windows_from_block(feat_val, lab_val, file_name, "val", WINDOW_SIZE, OVERLAP)
        Xte, yte, idte = make_windows_from_block(feat_test, lab_test, file_name, "test", WINDOW_SIZE, OVERLAP)

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
        x = x.transpose(1, 2)
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
    global STEP_SEC
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
            0.5 * val_metrics["Risk_F1"] +
            0.3 * val_metrics["Macro_F1"] +
            0.2 * val_metrics["Risk_Precision"]
        )
        scheduler.step(val_score)

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_Risk_F1={val_metrics['Risk_F1']:.4f} | val_score={val_score:.4f}"
            )

        # 每个 epoch 后清理内存
        if epoch % 1 == 0:
            cleanup_memory(verbose=(epoch % 5 == 0))

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
    cleanup_memory()
    return model


def run_single_experiment(window_size, overlap):
    """
    Run a single experiment with given WINDOW_SIZE and OVERLAP values.
    """
    global WINDOW_SIZE, OVERLAP
    WINDOW_SIZE = window_size
    OVERLAP = overlap

    # 清理上一个实验的内存
    cleanup_memory()

    print("\n" + "=" * 80)
    print(f"Running experiment: WINDOW_SIZE={window_size}, OVERLAP={overlap}")
    print(f"Window duration: {window_size / FS_HZ:.2f} seconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data with current configuration
    X_train, y_train_orig, train_ids, \
    X_val, y_val_orig, val_ids, \
    X_test, y_test_orig, test_ids, \
    feature_cols, split_infos = load_data_temporal_split(DATA_DIR)

    print(f"Data loaded: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

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

    model = MSCausalTCNRiskModel(num_features, num_classes=2)

    print(f"Training model...")
    model = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        device=device
    )

    # Inference
    y_true_risk, _, y_prob_risk = infer(model, test_loader, device)
    y_pred_risk = apply_risk_threshold_and_smoothing(y_prob_risk, test_ids)

    # Compute metrics
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
        f"[TEST] WINDOW_SIZE={window_size}, OVERLAP={overlap} -> "
        f"Macro_F1={win_metrics['Macro_F1']:.4f}, "
        f"Sensitivity={win_metrics['Sensitivity']:.4f}, "
        f"Specificity={win_metrics['Specificity']:.4f}, "
        f"Risk_F1={win_metrics['Risk_F1']:.4f}, "
        f"Event_Recall={event_metrics['Event_Recall']:.4f}, "
        f"FA/min={event_metrics['False_Alarms_per_Min']:.4f}, "
        f"PredictionHorizon={event_metrics['Prediction_Horizon_sec']:.4f}"
    )

    result = {
        "WINDOW_SIZE": window_size,
        "OVERLAP": overlap,
        "Window_sec": window_size / FS_HZ,
        "Risk_AUC": risk_auc,
    }
    result.update(win_metrics)
    result.update(event_metrics)

    # 清理当前实验的内存
    cleanup_memory()
    del model, train_loader, val_loader, test_loader
    del X_train, y_train_orig, X_val, y_val_orig, X_test, y_test_orig
    del y_train_risk, y_val_risk, y_test_risk
    del y_true_risk, y_prob_risk, y_pred_risk
    cleanup_memory()

    return result


# =========================
# 7. Visualization
# =========================
def plot_experiment_results(all_results, output_dir="experiment_plots"):
    """
    Generate visualization plots comparing different WINDOW_SIZE and OVERLAP combinations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert results to DataFrame
    df = pd.DataFrame(all_results)

    # Set style
    matplotlib.use('Agg')
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create a figure with 4 subplots (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Window Size and Overlap Experiment Results\n(Sampling Rate: {FS_HZ} Hz)', fontsize=16, fontweight='bold')

    # Plot 1: Sensitivity
    ax1 = axes[0, 0]
    for overlap in sorted(df['OVERLAP'].unique()):
        df_overlap = df[df['OVERLAP'] == overlap].sort_values('WINDOW_SIZE')
        ax1.plot(df_overlap['Window_sec'], df_overlap['Sensitivity'],
                marker='o', linewidth=2, markersize=8, label=f'Overlap={overlap:.2f}')
    ax1.set_xlabel('Window Size (seconds)', fontsize=12)
    ax1.set_ylabel('Sensitivity', fontsize=12)
    ax1.set_title('Sensitivity vs Window Size', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Specificity
    ax2 = axes[0, 1]
    for overlap in sorted(df['OVERLAP'].unique()):
        df_overlap = df[df['OVERLAP'] == overlap].sort_values('WINDOW_SIZE')
        ax2.plot(df_overlap['Window_sec'], df_overlap['Specificity'],
                marker='s', linewidth=2, markersize=8, label=f'Overlap={overlap:.2f}')
    ax2.set_xlabel('Window Size (seconds)', fontsize=12)
    ax2.set_ylabel('Specificity', fontsize=12)
    ax2.set_title('Specificity vs Window Size', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Macro F1
    ax3 = axes[1, 0]
    for overlap in sorted(df['OVERLAP'].unique()):
        df_overlap = df[df['OVERLAP'] == overlap].sort_values('WINDOW_SIZE')
        ax3.plot(df_overlap['Window_sec'], df_overlap['Macro_F1'],
                marker='^', linewidth=2, markersize=8, label=f'Overlap={overlap:.2f}')
    ax3.set_xlabel('Window Size (seconds)', fontsize=12)
    ax3.set_ylabel('Macro F1 Score', fontsize=12)
    ax3.set_title('Macro F1 vs Window Size', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Prediction Horizon
    ax4 = axes[1, 1]
    for overlap in sorted(df['OVERLAP'].unique()):
        df_overlap = df[df['OVERLAP'] == overlap].sort_values('WINDOW_SIZE')
        ax4.plot(df_overlap['Window_sec'], df_overlap['Prediction_Horizon_sec'],
                marker='d', linewidth=2, markersize=8, label=f'Overlap={overlap:.2f}')
    ax4.set_xlabel('Window Size (seconds)', fontsize=12)
    ax4.set_ylabel('Prediction Horizon (seconds)', fontsize=12)
    ax4.set_title('Prediction Horizon vs Window Size', fontsize=14, fontweight='bold')
    ax4.legend(loc='best', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_comparison.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'experiment_comparison.pdf'), bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir}/experiment_comparison.png")

    # Also create individual plots for each overlap
    fig2, axes2 = plt.subplots(1, 4, figsize=(20, 5))
    fig2.suptitle('Metrics vs Window Size (All Overlaps)', fontsize=16, fontweight='bold')

    metrics = ['Sensitivity', 'Specificity', 'Macro_F1', 'Prediction_Horizon_sec']
    titles = ['Sensitivity', 'Specificity', 'Macro F1', 'Prediction Horizon (s)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        ax = axes2[idx]
        for overlap in sorted(df['OVERLAP'].unique()):
            df_overlap = df[df['OVERLAP'] == overlap].sort_values('WINDOW_SIZE')
            ax.plot(df_overlap['Window_sec'], df_overlap[metric],
                    marker='o', linewidth=2, markersize=6, label=f'Ov={overlap:.2f}')
        ax.set_xlabel('Window Size (seconds)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_all_in_one.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'metrics_all_in_one.pdf'), bbox_inches='tight')
    print(f"Combined plot saved to: {output_dir}/metrics_all_in_one.png")

    # 关闭图形以释放内存
    plt.close('all')
    cleanup_memory()


# =========================
# 8. Main Experiment Loop
# =========================
def run_all_experiments():
    """
    Run experiments for all WINDOW_SIZE and OVERLAP combinations,
    or only with FIXED_OVERLAP if USE_FIXED_OVERLAP is True.
    """
    set_seed(SEED)

    # 确定要使用的 overlap 值列表
    if USE_FIXED_OVERLAP:
        overlap_to_test = [FIXED_OVERLAP]
    else:
        overlap_to_test = OVERLAP_LIST

    print("=" * 80)
    print("WINDOW SIZE AND OVERLAP EXPERIMENT FRAMEWORK")
    print("=" * 80)
    print(f"WINDOW_SIZE values to test: {WINDOW_SIZE_LIST}")
    if USE_FIXED_OVERLAP:
        print(f"FIXED OVERLAP mode: {FIXED_OVERLAP}")
    else:
        print(f"OVERLAP values to test: {OVERLAP_LIST}")
    print(f"Total experiments: {len(WINDOW_SIZE_LIST) * len(overlap_to_test)}")
    print("=" * 80)

    # 初始内存清理
    cleanup_memory(verbose=True)

    all_results = []

    for window_size in WINDOW_SIZE_LIST:
        for overlap in overlap_to_test:
            try:
                result = run_single_experiment(window_size, overlap)
                all_results.append(result)
            except Exception as e:
                print(f"ERROR in experiment WINDOW_SIZE={window_size}, OVERLAP={overlap}: {e}")
                import traceback
                traceback.print_exc()
                # 出错后清理内存继续下一个实验
                cleanup_memory()

    # Save all results to CSV
    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values(by=["WINDOW_SIZE", "OVERLAP"])

    # 根据模式设置 CSV 文件名
    if USE_FIXED_OVERLAP:
        csv_path = f"window_size_results_overlap_{FIXED_OVERLAP:.2f}.csv"
    else:
        csv_path = "window_overlap_experiment_results.csv"

    result_df.to_csv(csv_path, index=False)
    print(f"\nAll results saved to: {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("EXPERIMENT SUMMARY TABLE")
    print("=" * 100)
    summary_cols = ['WINDOW_SIZE', 'OVERLAP', 'Window_sec', 'Sensitivity', 'Specificity', 'Macro_F1', 'Risk_F1', 'Event_Recall', 'Prediction_Horizon_sec']
    print(result_df[summary_cols].to_string(index=False))

    # Generate visualizations
    plot_experiment_results(all_results)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED!")
    print("=" * 80)

    # 最终内存清理
    cleanup_memory()

    return result_df


if __name__ == "__main__":
    results = run_all_experiments()

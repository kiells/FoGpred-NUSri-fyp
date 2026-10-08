import pandas as pd
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split

def sliding_window_samples(folder_path, window_size=128, overlap=0.5):
    """
    从文件夹读取所有CSV，进行滑动窗口采样。
    """
    all_windows = []
    all_labels = []
    
    # 获取所有处理过的CSV文件
    files = sorted(glob.glob(os.path.join(folder_path, "S*R*.csv")))
    
    if not files:
        print("未找到数据文件，请检查路径。")
        return None, None

    # 步长计算
    step = int(window_size * (1 - overlap))
    
    for file_path in files:
        df = pd.read_csv(file_path)
        
        # 1. 过滤掉非实验数据 (Annot == 0)
        df = df[df['Annot'] != 0].reset_index(drop=True)
        
        # 2. 确定特征列 (排除 Time 和 Annot)
        # 假设剩余列为：Ankle_X, Ankle_Y, Ankle_Z, Thigh_X, ... 等加速度数据
        feature_cols = [col for col in df.columns if col not in ['Time', 'Annot']]
        features = df[feature_cols].values
        labels = df['Annot'].values
        
        # 3. 滑动采样
        for start in range(0, len(df) - window_size, step):
            end = start + window_size
            
            # 获取当前窗口的数据块 (Shape: window_size, num_features)
            window_data = features[start:end]
            
            # 确定窗口标签：取窗口中出现次数最多的标签
            counts = np.bincount(labels[start:end])
            window_label = np.argmax(counts)
            
            all_windows.append(window_data)
            all_labels.append(window_label)
            
    return np.array(all_windows), np.array(all_labels)

def save_datasets(X_train, y_train, X_val, y_val, X_test, y_test, save_dir="./processed_data"):
    """
    将划分好的数据集分别存储为 .npy 文件
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 定义存储映射
    data_map = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    print(f"开始保存数据至 {save_dir}...")
    for name, data in data_map.items():
        file_path = os.path.join(save_dir, f"{name}.npy")
        np.save(file_path, data)
        print(f"  - 已保存: {file_path} (Shape: {data.shape})")
    print("所有数据集已分开储存完成。")

# --- 1. 数据采集 ---
IN_DIR = "./prefog_5s" # 之前生成的带 Pre-FoG 标签的数据目录
X, y = sliding_window_samples(IN_DIR, window_size=128, overlap=0.5)

print(f"原始数据采样完成:")
print(f"窗口总数: {X.shape[0]} | 每个窗口大小: {X.shape[1]} | 特征数: {X.shape[2]}")
print(f"各类别分布: {np.unique(y, return_counts=True)}")

# --- 2. 划分数据集 (8:1:1) ---

# 第一次划分：分出 80% 训练集，剩下 20% 作为临时集 (用于再分验证和测试)
# 使用 stratify=y 确保各集合中行走、Pre-FoG、FoG 的比例一致
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.20, 
    random_state=42, 
    stratify=y
)

# 第二次划分：将 20% 的临时集对半分，得到 10% 验证集和 10% 测试集
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.50, 
    random_state=42, 
    stratify=y_temp
)
save_datasets(X_train, y_train, X_val, y_val, X_test, y_test)

# --- 3. 结果验证 ---
print("\n--- Patient-dependent 数据集划分完成 ---")
print(f"训练集 (80%): {X_train.shape}, 类别分布: {np.bincount(y_train)}")
print(f"验证集 (10%): {X_val.shape}, 类别分布: {np.bincount(y_val)}")
print(f"测试集 (10%): {X_test.shape}, 类别分布: {np.bincount(y_test)}")



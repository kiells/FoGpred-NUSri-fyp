import pandas as pd
import numpy as np
import os
import glob
import re

def sliding_window_with_subject(folder_path, window_size=128, overlap=0.5):
    """
    采样并保留受试者 ID 信息
    """
    all_windows = []
    all_labels = []
    all_subjects = []
    
    files = sorted(glob.glob(os.path.join(folder_path, "S*R*.csv")))
    step = int(window_size * (1 - overlap))
    
    for file_path in files:
        # 从文件名解析受试者 ID (例如 S01R01 -> S01)
        filename = os.path.basename(file_path)
        sub_match = re.search(r'(S\d+)', filename)
        sub_id = sub_match.group(1) if sub_match else "Unknown"
        
        df = pd.read_csv(file_path)
        df = df[df['Annot'] != 0].reset_index(drop=True) # 排除非实验数据
        
        feature_cols = [col for col in df.columns if col not in ['Time', 'Annot']]
        features = df[feature_cols].values
        labels = df['Annot'].values
        
        for start in range(0, len(df) - window_size, step):
            end = start + window_size
            window_data = features[start:end]
            
            # 标签判定：多数原则
            window_label = np.argmax(np.bincount(labels[start:end]))
            
            all_windows.append(window_data)
            all_labels.append(window_label)
            all_subjects.append(sub_id)
            
    return np.array(all_windows), np.array(all_labels), np.array(all_subjects)

def build_and_save_loso(X, y, subjects, base_save_dir="./loso_dataset"):
    """
    执行 LOSO 划分并存储
    """
    unique_subs = np.unique(subjects)
    print(f"检测到受试者: {unique_subs}，共 {len(unique_subs)} 轮实验。")
    
    for i, test_sub in enumerate(unique_subs):
        round_num = i + 1
        round_dir = os.path.join(base_save_dir, f"round_{round_num}")
        os.makedirs(round_dir, exist_ok=True)
        
        # 1. 测试集: 当前受试者
        test_mask = (subjects == test_sub)
        X_test, y_test = X[test_mask], y[test_mask]
        
        # 2. 剩余受试者
        remaining_subs = unique_subs[unique_subs != test_sub]
        
        # 3. 验证集: 取剩余受试者中的一位 (例如最后一位)
        val_sub = remaining_subs[-1]
        val_mask = (subjects == val_sub)
        X_val, y_val = X[val_mask], y[val_mask]
        
        # 4. 训练集: 剩下的所有受试者
        train_subs = remaining_subs[:-1]
        train_mask = np.isin(subjects, train_subs)
        X_train, y_train = X[train_mask], y[train_mask]
        
        # 存储信息
        print(f"[Round {round_num}] 测试集:{test_sub} | 验证集:{val_sub} | 训练集:{train_subs}")
        
        # 执行存储
        data_to_save = {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val,     "y_val": y_val,
            "X_test": X_test,   "y_test": y_test
        }
        
        for name, data in data_to_save.items():
            np.save(os.path.join(round_dir, f"{name}.npy"), data)
            
    # 保存一个说明文件，记录每一轮对应的受试者关系
    print(f"\n所有 LOSO 数据已保存至 {base_save_dir}")

if __name__ == "__main__":
    # 配置
    INPUT_DIR = "./prefog_5s" # 处理过的 CSV 目录
    OUTPUT_DIR = "./processed_loso"
    
    # 采样
    X, y, subs = sliding_window_with_subject(INPUT_DIR, window_size=128, overlap=0.5)
    
    # 划分并保存
    build_and_save_loso(X, y, subs, OUTPUT_DIR)
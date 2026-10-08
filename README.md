# 帕金森病冻结步态风险预测系统

这是一个基于MS-Causal-TCN模型的帕金森病冻结步态风险预测系统，集成了模型训练、消融实验、窗口大小实验和实时验证功能。

## 系统概述

本系统包含以下核心功能：
1. **模型训练** - 使用MS-Causal-TCN模型训练风险预测模型
2. **消融实验** - 分析模型各组件的贡献度
3. **窗口大小实验** - 优化滑动窗口大小和重叠率
4. **实时验证** - 对Subject1数据进行实时预测验证

## 文件结构

```
E:\fyp\daphnet\
├── train_and_validate.py          # 主控制脚本
├── run_training_and_validation.bat # Windows批处理启动脚本
├── risk_prediction_ms_causal_tcn.py          # 主模型训练脚本
├── risk_prediction_ms_causal_tcn_ablation.py # 消融实验脚本
├── experiment_windows.py          # 窗口大小实验脚本
├── validate_subject1.py           # Subject1验证脚本
├── config.json                    # 配置文件
├── README.md                      # 说明文档
├── prefog_5s\                     # 训练数据目录
│   ├── S01R01.csv
│   ├── S01R02.csv
│   └── ...
└── subject1_data\                 # 验证数据目录
    ├── S1R01.csv
    ├── S1R02.csv
    └── ...
```

## 快速开始

### 1. 运行系统

双击 `run_training_and_validation.bat` 文件启动系统。

### 2. 准备数据

1. **训练数据**：将帕金森病数据解压到 `prefog_5s` 目录，确保包含 `S*R*.csv` 格式的文件
2. **验证数据**：将需要验证的Subject1数据（`S1R*.csv`）放入 `subject1_data` 目录

### 3. 选择操作

运行bat文件后，系统会显示菜单：
- **选项1**：运行完整流程（推荐）
- **选项2**：仅训练模型
- **选项3**：运行消融实验
- **选项4**：运行窗口大小实验
- **选项5**：仅验证模型
- **选项6**：查看训练报告
- **选项7**：清理临时文件
- **选项8**：退出

## 配置说明

系统使用 `config.json` 文件进行配置：

```json
{
  "training": {
    "data_dir": "./prefog_5s",      // 训练数据目录
    "window_size": 128,             // 窗口大小（样本点）
    "overlap": 0.5                   // 窗口重叠率
  },
  "experiments": {
    "run_ablation": false,           // 是否运行消融实验
    "run_window_experiment": false   // 是否运行窗口实验
  },
  "validation": {
    "subject1_dir": "./subject1_data", // 验证数据目录
    "realtime_validation": true     // 是否运行实时验证
  }
}
```

## 模型说明

### MS-Causal-TCN 模型

Multi-Scale Causal Temporal Convolutional Network，包含以下特点：
1. **多尺度卷积** - 使用不同大小的卷积核捕获不同时间尺度的特征
2. **因果卷积** - 确保预测不使用未来信息
3. **SE模块** - 空间注意力机制，增强重要特征
4. **时间注意力池化** - 动态加权时序特征

### 标签定义
- 1 = 正常步态 (Normal gait)
- 2 = 冻结前兆 (Pre-fog)
- 3 = 冻结步态 (FoG)

### 风险标签转换
- 正常步态 → 0（低风险）
- 冻结前兆/冻结步态 → 1（高风险）

## 输出说明

### 训练输出
- `output_YYYYMMDD_HHMMSS/` - 时间戳命名的输出目录
  - `trained_model.pkl` - 训练好的模型
  - `training_results.csv` - 训练结果
  - 各种可视化图片
  - `training_report.json` - 训练报告

### 验证输出
- `predictions_S1RXX.csv` - 每个验证文件的预测结果
- `validation_summary.json` - 验证汇总统计
- `prediction_timeline.png` - 预测时间线图

## 使用建议

### 1. 首次使用
- 选择选项1（完整流程）
- 系统会自动检查数据和依赖
- 生成的所有文件都会保存在带时间戳的目录中

### 2. 快速验证
- 如果已有训练好的模型，选择选项5（仅验证）
- 确保模型文件在当前目录

### 3. 参数调优
- 运行窗口大小实验（选项4）找到最佳窗口参数
- 运行消融实验（选项3）理解模型各组件的重要性

### 4. 性能监控
- 训练过程会实时显示日志
- 所有输出都会保存到文件中
- 可以通过选项6查看历史报告

## 故障排除

### 常见问题

1. **Python环境问题**
   - 确保已安装Python 3.7+
   - 首次运行会自动创建虚拟环境

2. **数据格式问题**
   - 确保CSV文件包含Time和Annot列
   - 特征列应以A开头（如A1, A2...）

3. **内存不足**
   - 减小batch_size参数
   - 使用较小的窗口大小

4. **模型加载失败**
   - 确保模型文件存在
   - 检查模型和代码版本匹配

### 获取帮助

如果遇到问题，请检查：
1. 日志文件内容
2. 输出目录中的错误信息
3. 配置文件是否正确

## 扩展功能

### 自定义实验
可以通过修改实验脚本的参数来：
- 调整网络结构
- 更改训练超参数
- 添加新的评估指标

### 实时预测
系统支持实时预测，只需：
1. 准备新的数据文件
2. 放入subject1_data目录
3. 运行验证选项

## 版本历史

- v1.0 - 初始版本，支持基础训练和验证
- v1.1 - 添加消融实验和窗口大小实验
- v1.2 - 集成统一控制流程和批处理启动

## 许可证

本项目仅供研究和学习使用。
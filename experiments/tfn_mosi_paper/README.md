# TFN (Tensor Fusion Network) 论文复现实验

**论文**: Tensor Fusion Network for Multimodal Sentiment Analysis (EMNLP 2017)
**数据集**: CMU-MOSI
**日期**: 2026-03-12

---

## 🎯 实验目标

复现论文中的三个实验 (E1, E2, E3)，特别是多模态对比实验 (E1)。

---

## 📊 论文结果 (目标)

| 实验 | 任务 | 准确率/MAE |
|------|------|-----------|
| E1 | Binary Acc | **77.1%** |
| E1 | 5-class Acc | **42.0%** |
| E1 | MAE | **0.87** |
| E2 | Language only | 74.8% |
| E2 | Visual only | 69.4% |
| E2 | Acoustic only | 65.1% |

---

## 🔧 实验设置 (论文原始)

### 超参数
- **优化器**: Adam
- **学习率**: 5e-4
- **Dropout**: 0.15
- **L2 正则化**: 0.01
- **验证方式**: 5-fold 交叉验证
- **独立说话人**: 是

### 损失函数
- Binary: BCEWithLogitsLoss
- 5-class: CrossEntropyLoss
- Regression: MSELoss

### 评估指标
- Binary Accuracy (threshold=0)
- 5-class Accuracy
- MAE (Mean Absolute Error)

---

## 📁 文件结构

```
tfn_mosi_paper/
├── src/
│   ├── tfn_paper.py          # TFN 模型实现
│   ├── data_loader.py        # MOSI 数据加载
│   └── train.py              # 训练脚本
├── configs/
│   └── tfn_paper.yaml        # 配置文件
├── scripts/
│   ├── run_tfn_binary.sh     # Binary 分类脚本
│   ├── run_tfn_5class.sh     # 5-class 分类脚本
│   └── run_tfn_regression.sh # 回归脚本
├── results/                   # 实验结果
├── logs/                      # 训练日志
└── README.md                  # 本文件
```

---

## 🚀 运行命令

```bash
# Binary 分类 (E1)
bash scripts/run_tfn_binary.sh

# 5-class 分类 (E1)
bash scripts/run_tfn_5class.sh

# 回归 (E1)
bash scripts/run_tfn_regression.sh

# 消融实验 (E2)
bash scripts/run_tfn_ablation.sh
```

---

## 📈 预期 vs 实际结果

| 实验 | 指标 | 论文 | 复现 | 差距 |
|------|------|------|------|------|
| E1 | Binary Acc | 77.1% | TBD | - |
| E1 | 5-class Acc | 42.0% | TBD | - |
| E1 | MAE | 0.87 | TBD | - |

---

## 🔗 相关文件

- **服务器数据**: `/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/`
- **原始代码**: `src/baselines/tfn_complete.py`
- **对比实验**: EAS 将在相同设置下测试

---

**创建时间**: 2026-03-12
**状态**: 🟡 准备中

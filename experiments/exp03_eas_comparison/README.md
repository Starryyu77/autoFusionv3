# Experiment 3: EAS vs TFN Comparison

## 目标
在完全相同的设置下，对比手工EAS与TFN。

## 配置
- embed_dim: 64 (EAS) vs 32 (TFN)
- hidden_dim: 128
- lr: 5e-4, weight_decay: 0.01
- epochs: 100, patience: 20

## 结果

| 任务 | TFN | **EAS** | 提升 |
|------|-----|---------|------|
| **Binary Acc** | 71.03% | **78.18%** | **+10.1%** |
| **5-class Acc** | 42.04% | **49.99%** | **+19.0%** |
| **MAE** | 0.824 | **0.687** | **-16.6%** |

## 架构对比

| 特性 | TFN | EAS |
|------|-----|-----|
| **融合机制** | 张量外积 | 动态门控+交叉注意力 |
| **参数量** | ~0.3M | **0.14M** (53%↓) |
| **动态性** | 静态 | 自适应 |

## 结论
- EAS在所有任务上超越TFN
- 参数量更少但性能更好
- 动态融合优于静态张量融合

## 文件
- `src/eas_model.py` - EAS动态融合模型
- `RESULTS_COMPARISON.md` - 详细对比报告

## 运行
```bash
bash scripts/run_eas_binary.sh
bash scripts/run_eas_5class.sh
bash scripts/run_eas_regression.sh
```

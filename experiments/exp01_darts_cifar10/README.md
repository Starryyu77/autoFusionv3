# DARTS CIFAR-10 实验完整记录

## 实验概述
- **方法**: DARTS (Differentiable Architecture Search) - ICLR 2019
- **数据集**: CIFAR-10
- **日期**: 2026-03-10 至 2026-03-12
- **服务器**: NTU GPU43 (4x RTX A5000)
- **对比方法**: EAS-LLM

## 最终结果对比

| 方法 | 验证准确率 | 测试误差 | 参数量 | 总耗时 |
|------|-----------|---------|--------|-------|
| **DARTS (1st)** | **97.38%** | 待测 | 3.35MB | ~18.5h |
| **EAS-LLM** | 94.58% | **5.42%** | 5.45MB | ~5.8h |

**结论**: DARTS 在 CIFAR-10 上显著优于 EAS-LLM (+2.8%)，但 EAS 训练速度快 3 倍

## DARTS 详细结果

### 搜索阶段 (50 epochs)
- **配置**: 1st order approximation, batch_size=32
- **搜索验证准确率**: 100%
- **搜索耗时**: ~10 小时
- **模型大小**: 3.35MB

### 最佳架构 (Genotype)
```python
Normal: [
  ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
  ('sep_conv_3x3', 0), ('sep_conv_3x3', 1),
  ('skip_connect', 0), ('sep_conv_5x5', 1),
  ('skip_connect', 0), ('sep_conv_5x5', 1)
]
Reduce: [
  ('max_pool_3x3', 0), ('max_pool_3x3', 1),
  ('max_pool_3x3', 0), ('skip_connect', 2),
  ('skip_connect', 2), ('skip_connect', 3),
  ('skip_connect', 2), ('skip_connect', 3)
]
```

### 完整训练阶段 (600 epochs)
- **最终训练准确率**: 99.22%
- **最终验证准确率**: 97.28%
- **最佳验证准确率**: **97.38%**
- **训练时间**: ~8-9 小时

## 文件结构
```
darts_cifar10/
├── results/              # 实验结果
│   ├── CIFAR10_EXPERIMENT_RESULTS.md  # 详细报告
│   └── cifar10_final_results.json     # 结构化数据
├── logs/                 # 训练日志 (需从服务器下载)
├── configs/              # 配置文件
├── scripts/              # 运行脚本
└── models/               # 保存的模型 (需从服务器下载)
```

## 服务器文件位置

### 模型文件
```
/usr1/home/s125mdg43_10/paper_reproduction_2026/
├── baselines/darts/cnn/
│   ├── eval-exp_darts_eval_gpu1-20260311-102202/
│   │   └── weights.pt          # 评估阶段模型 (16MB)
│   └── search-exp_darts_1st_b32_gpu2-20260310-225711/
│       └── weights.pt          # 搜索阶段模型
```

### 日志文件
```
/usr1/home/s125mdg43_10/paper_reproduction_2026/logs/
├── darts/
│   ├── 06_darts_1st_b32_gpu2.log      # 搜索日志
│   └── 08_darts_eval_gpu1.log         # 评估日志
└── eas/
    └── 03_eas_full_train_600epochs.log # EAS对比日志
```

## 复现命令

```bash
# DARTS 搜索
cd baselines/darts/cnn
python train_search.py --batch_size 32 --epochs 50 --unrolled

# DARTS 评估
python train.py --auxiliary --cutout --save exp_darts_eval
```

## GitHub Commit
- `107e7fa` - docs: Add CIFAR-10 baseline experiment final results
- `2b8c020` - chore: Update work context with CIFAR-10 completion status

---
**记录时间**: 2026-03-12

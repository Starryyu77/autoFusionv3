# CIFAR-10 基线实验完整结果

**实验日期**: 2026-03-10 至 2026-03-12
**执行环境**: NTU GPU43 (4x RTX A5000)
**实验目标**: 对比 DARTS (ICLR 2019) 与 EAS-LLM 在 CIFAR-10 上的性能

---

## 📊 最终结果对比

| 方法 | 搜索准确率 | 完整训练准确率 | 测试误差 | 参数量 | 训练时间 |
|------|-----------|---------------|---------|--------|---------|
| **DARTS (1st order)** | 100% (验证) | **97.38%** | 待测 | 3.35MB | ~10h |
| **EAS-LLM** | 92.80% | 94.58% | **5.42%** | 5.45MB | ~6h |

**结论**: DARTS 在 CIFAR-10 上表现优于 EAS-LLM，验证准确率高 **2.8%**。

---

## 🔬 DARTS (ICLR 2019) 详细结果

### 搜索阶段 (50 epochs)
- **配置**: 1st order approximation, batch_size=32
- **搜索准确率**: 100% (验证集)
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
- **保存位置**: `baselines/darts/cnn/eval-exp_darts_eval_gpu1-20260311-102202/`

### 训练过程关键节点
| Epoch | 训练准确率 | 验证准确率 | 学习率 |
|-------|-----------|-----------|--------|
| 149 | 92.50% | 93.79% | 2.13e-2 |
| 203 | 94.00% | 94.44% | 1.85e-2 |
| 219 | 94.79% | 94.73% | 1.74e-2 |
| 326 | 95.02% | 96.03% | 1.07e-2 |
| 375 | 94.99% | 96.03% | 7.60e-3 |
| 480 | 98.21% | 96.97% | 2.31e-3 |
| 598 | 99.26% | 97.20% | 4.28e-8 |
| 599 | 99.26% | 97.18% | 0.00e+0 |

---

## 🤖 EAS-LLM 详细结果

### 搜索阶段 (50 iterations)
- **LLM 模型**: kimi-k2.5 (Aliyun Bailian)
- **搜索成功率**: 100% (50/50)
- **最佳搜索准确率**: 92.80% (Iterations 0, 5, 12)
- **搜索耗时**: ~4小时40分钟
- **API 调用**: 50次

### Top 5 搜索架构
| 排名 | Iteration | 准确率 |
|:---:|---:|:---:|
| 1 | 0, 5, 12 | **92.80%** |
| 2 | 1, 2 | 91.93% |
| 3 | 8 | 91.57% |
| 4 | 4 | 91.37% |
| 5 | 3 | 91.35% |

### 完整训练阶段 (600 epochs)
- **架构来源**: Iteration 12 (搜索准确率 92.80%)
- **最佳验证准确率**: **94.58%** (Epoch 505)
- **测试误差**: **5.42%**
- **模型参数量**: 5.45M
- **训练时间**: 1.15小时
- **提升幅度**: 92.80% → 94.58% (+1.78%)

### 架构代码
EAS 生成的最佳架构为 ResNet-style CNN，包含4个阶段和残差连接。

---

## 📈 实验公平性说明

| 阶段 | DARTS | EAS-LLM |
|------|-------|---------|
| 搜索 epochs | 50 | 50 iter × 50 epochs |
| 评估 epochs | 600 | 600 |
| 数据集 | CIFAR-10 | CIFAR-10 |
| 优化器 | SGD + Cosine | SGD + Cosine |
| 增强 | Auxiliary + Cutout | 无 |

**差异说明**:
- DARTS 使用了 auxiliary towers 和 cutout 增强
- EAS 使用了更简单的 ResNet-style 架构

---

## 🔍 关键发现

1. **DARTS 优势**: 在 CIFAR-10 这种经典 CV 任务上，DARTS 的可微分搜索能找到更优的 cell 结构
2. **EAS 特点**: LLM 生成的架构具有较好的通用性，但针对特定任务可能不如专门搜索的架构
3. **训练效率**: EAS 搜索+训练总时间 (~6h) 短于 DARTS (~18h)

---

## 📁 文件位置

### 服务器路径 (NTU GPU43)
```
/usr1/home/s125mdg43_10/paper_reproduction_2026/
├── baselines/darts/cnn/
│   ├── eval-exp_darts_eval_gpu1-20260311-102202/
│   │   ├── weights.pt          # 最佳模型 (16MB)
│   │   └── log.txt             # 训练日志
│   └── search-exp_darts_1st_b32_gpu2-20260310-225711/
│       └── weights.pt          # 搜索阶段模型
├── baselines/eas/
│   ├── eas_llm_cifar10.py      # EAS 搜索代码
│   └── eas_full_train.py       # EAS 完整训练代码
├── results/
│   ├── darts/
│   │   └── darts_search_results.json
│   └── eas/
│       ├── eas_llm_cifar10_results.json
│       ├── eas_full_train_600epochs.json
│       └── best_model_600epochs.pth
└── logs/
    ├── darts/08_darts_eval_gpu1.log
    └── eas/03_eas_full_train_600epochs.log
```

---

**生成时间**: 2026-03-12
**实验状态**: ✅ 完成

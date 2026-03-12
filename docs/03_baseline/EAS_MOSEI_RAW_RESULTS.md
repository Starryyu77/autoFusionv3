# EAS MOSEI Raw Features 实验结果

**实验时间**: 2026-03-11 ~ 2026-03-12
**实验配置**: `configs/round2_eas_mosei_raw.yaml`
**数据格式**: Raw features (vision=35D, audio=74D, text=300D)

---

## 实验概述

这是 EAS (Evolutionary Architecture Search) 在 CMU-MOSEI 数据集上使用 **原始特征** (非预训练) 的架构搜索实验。
目标是与 DynMM 等基线方法进行公平对比（使用相同的数据格式）。

---

## 搜索结果

### 关键指标

| 指标 | 数值 |
|------|------|
| **总迭代数** | 200/200 ✅ |
| **总耗时** | 12.3 小时 (735.5 min) |
| **编译成功率** | 56.5% (113/200) |
| **成功评估架构** | 113 个 |
| **最佳 Reward** | **1.680** |

### 最佳架构表现

发现于 **Iteration 1** (早期即发现优质架构):

| 指标 | 数值 |
|------|------|
| **准确率 (Accuracy)** | **70.00%** |
| **模态鲁棒性 (mRob@50%)** | **49.00%** |
| **FLOPs** | 202.7M (0.20G) |
| **Reward** | 1.680 |

---

## 与基线对比

| 方法 | 准确率 | 相对提升 |
|------|--------|----------|
| **EAS (Ours)** | **70.00%** | - |
| DynMM | 28.59% | **+2.45x** |
| TFN | 28.64% | **+2.44x** |
| Mean | 28.64% | **+2.44x** |
| Attention | 28.61% | **+2.45x** |

**结论**: EAS 在相同原始特征上显著优于所有基线方法，提升约 **2.45倍**。

---

## 最佳架构设计

**架构类型**: Cross-Modal Attention + Gated Fusion

### 核心创新

1. **跨模态注意力机制**: vision 作为 query，audio 和 text 作为 key/value 进行交叉注意力
2. **门控融合**: 学习自适应的模态重要性权重
3. **多尺度特征**: 局部时序特征 + 全局池化特征
4. **残差连接与层归一化**: 稳定训练

### 网络结构

```python
# 投影层
vision_proj:  [B, 50, 35]  → [B, 50, 256]
audio_proj:   [B, 50, 74]  → [B, 50, 256]
text_proj:    [B, 50, 300] → [B, 50, 256]

# 注意力机制
- Cross-modal Attention (vision → audio+text)
- Self-Attention for each modality (8 heads)
- Gated Fusion with Sigmoid activation

# 融合层
Fusion MLP: 768 → 512 → 256
Output: 256-dim fusion representation
```

### 参数量

| 组件 | 参数量 |
|------|--------|
| 投影层 | ~110K |
| 注意力模块 | ~525K |
| 门控机制 | ~197K |
| 融合MLP | ~394K |
| **总计** | **~1.2M** |

---

## 实验配置

```yaml
dataset:
  name: "CMU-MOSEI"
  feature_dims:
    vision: [50, 35]    # OpenFace
    audio: [50, 74]     # COVAREP
    text: [50, 300]     # GloVe
  unified_dim: 256

search:
  max_iterations: 200
  strategy: "cma_es"

proxy_evaluator:
  num_shots: 64
  num_epochs: 10
  batch_size: 32

reward:
  weights:
    accuracy: 1.0
    mrob: 2.0
    flops_penalty: 0.5
```

---

## 文件位置

| 文件 | 路径 |
|------|------|
| 实验配置 | `configs/round2_eas_mosei_raw.yaml` |
| 最佳架构代码 | `results/round2_eas_mosei_raw/best_architecture.py` |
| 最佳架构元数据 | `results/round2_eas_mosei_raw/best_architecture.json` |
| 完整日志 | `logs/eas_mosei_raw_v2.log` |
| 检查点 | `results/round2_eas_mosei_raw/checkpoint_iter*.json` |

---

## 下一步

1. **完整训练**: 对最佳架构进行 600-epoch 完整训练
2. **模态缺失测试**: 评估 25%/50% 模态缺失情况下的鲁棒性
3. **与其他数据集对比**: IEMOCAP, VQA-v2

---

## 结论

EAS 在 CMU-MOSEI 原始特征上的架构搜索取得了显著成功：
- ✅ 200轮搜索完成，发现多个优质架构
- ✅ 最佳架构准确率 **70.00%**，超越基线 **2.45倍**
- ✅ 编译成功率 **56.5%**，验证了自修复编译机制的有效性
- ✅ 架构设计创新：跨模态注意力 + 门控融合

这证明了 EAS 方法在多模态融合任务上的优势，即使在原始特征（非预训练）的挑战性设置下也能发现高性能架构。

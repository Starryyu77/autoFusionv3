# 基线方法测试方法论详细文档

> **文档目的**: 详细记录每个基线方法是如何被改造、适配到统一框架，以及如何进行测试的
> **最后更新**: 2026-03-08

---

## 目录

1. [统一测试框架设计](#1-统一测试框架设计)
2. [固定基线方法](#2-固定基线方法)
3. [NAS基线方法](#3-nas基线方法)
4. [数据集适配](#4-数据集适配)
5. [评估流程](#5-评估流程)
6. [结果分析方法](#6-结果分析方法)

---

## 1. 统一测试框架设计

### 1.1 核心设计原则

为了确保公平的比较，所有基线方法必须遵循以下统一框架：

```
输入数据
    ↓
统一投影层 (UnifiedFeatureProjection) - 所有方法共享
    ↓
基线特定融合模块 (Fusion Module) - 各方法不同
    ↓
统一分类头 (UnifiedClassifier) - 所有方法共享
    ↓
输出预测
```

### 1.2 统一投影层

**文件**: `src/models/unified_projection.py:11-91`

```python
class UnifiedFeatureProjection(nn.Module):
    """
    将不同模态的特征投影到相同的1024维空间
    支持延迟初始化（根据实际输入维度自动检测）
    """
```

**输入维度处理**:

| 数据集 | 模态 | 原始维度 | 投影后维度 |
|--------|------|----------|------------|
| MOSEI | vision | 35 | 1024 |
| MOSEI | audio | 74 | 1024 |
| MOSEI | text | 300 | 1024 |
| IEMOCAP | vision | 35 | 1024 |
| IEMOCAP | audio | 74 | 1024 |
| IEMOCAP | text | 300 | 1024 |
| VQA | vision | 35 | 1024 |
| VQA | text | 300 | 1024 |

**关键设计**: 投影层是**所有方法共享**的，不属于任何特定基线方法。

### 1.3 统一分类头

**文件**: `src/models/unified_projection.py:93-134`

```python
class UnifiedClassifier(nn.Module):
    """
    统一分类/回归头
    - 分类任务: [B, 1024] -> [B, num_classes]
    - 回归任务: [B, 1024] -> [B, 1]
    """
```

**结构**:
- 输入: 1024维融合特征
- 隐藏层: 512维 (带ReLU/GELU激活)
- Dropout: 0.2
- 输出: num_classes (分类) 或 1 (回归)

### 1.4 统一模型组装

**文件**: `src/models/unified_projection.py:137-227`

```python
class UnifiedModel(nn.Module):
    """
    统一模型框架
    组合了: 投影层 + 融合模块 + 分类头
    """
```

---

## 2. 固定基线方法

### 2.1 DynMM (Dynamic Multimodal Fusion)

**原始论文**: DynMM - Dynamic Multimodal Fusion (CVPR 2023)
**实现文件**: `src/baselines/dynmm.py`

#### 2.1.1 原始方法描述

DynMM是一种动态多模态融合方法，根据输入数据的特点动态选择融合策略：
- 计算各模态的路由权重（基于L2范数）
- 选择活跃的模态（超过阈值）
- 根据活跃模态数量选择融合策略：
  - 1个模态: 直接传递
  - 2个模态: 门控融合
  - 3+个模态: 注意力融合

#### 2.1.2 改造内容

**原始接口** (论文中的完整网络):
```python
class DynMM(nn.Module):
    def __init__(self, input_dims: Dict[str, List[int]], num_classes: int, ...)
    # 包含: 模态投影器 + 融合模块 + 分类器
```

**改造后接口** (适配统一框架):
```python
class DynMMFusion(nn.Module):
    def __init__(self, input_dim: int = 1024)  # 固定1024，不再内部投影
    # 只保留融合逻辑，不包含投影和分类

    def forward(self, vision, audio, text) -> [B, 1024]  # 返回融合特征
```

**关键改造点**:
1. **移除内部投影**: 原始DynMM内部有`ModalityProjector`，改造后假设输入已经是1024维
2. **移除分类器**: 原始有`self.classifier`，改造后只输出融合特征
3. **输入格式调整**: 原始接受`Dict[str, Tensor]`，改造后接受独立的`vision`, `audio`, `text`参数
4. **序列处理**: 原始处理`[B, seq_len, feat_dim]`，改造后在forward中做`mean(dim=1)`池化

#### 2.1.3 融合逻辑实现

**文件**: `src/baselines/dynmm.py:189-277`

```python
def forward(self, vision, audio, text):
    # 1. 平均池化序列维度 -> [B, 1024]
    features = {}
    if vision is not None:
        features['vision'] = vision.mean(dim=1)
    ...

    # 2. 计算路由权重
    routing_weights = compute_routing_weights(features)

    # 3. 选择活跃模态
    active_modalities = select_active_modalities(routing_weights, threshold=0.2)

    # 4. 动态选择融合策略
    if len(active_features) == 1:
        fused = list(active_features.values())[0]
    elif len(active_features) == 2:
        fused = self.gated_fusion(active_features)  # 门控
    else:
        fused = self.attention_fusion(feat_list)    # 注意力
```

---

### 2.2 ADMN (Adaptive Dynamic Multimodal Network)

**原始论文**: ADMN - Adaptive Dynamic Network (NeurIPS 2025)
**实现文件**: `src/baselines/admn.py`

#### 2.2.1 原始方法描述

ADMN采用层级化处理，每层可以动态调整模态处理策略：
- 每模态独立的处理器
- 模态间注意力交互
- 控制器决定每层处理策略

#### 2.2.2 改造内容

**原始接口**:
```python
class ADMN(nn.Module):
    def __init__(self, input_dims, num_classes, num_layers=3, hidden_dim=256)
    # 包含: 初始投影 + 层级处理 + 全局池化 + 分类器
```

**改造后接口**:
```python
class ADMNFusion(nn.Module):
    def __init__(self, input_dim: int = 1024, num_layers: int = 3)
    # 只保留: 层级处理 + 注意力 + 门控融合
```

**关键改造点**:
1. **保留层级结构**: 保留`num_layers`层处理，每模态独立
2. **简化控制器**: 原始有复杂的`ModalityController`，改造后简化为固定的门控
3. **输入输出统一**: 输入`[B, seq_len, 1024]`，输出`[B, 1024]`

#### 2.2.3 融合逻辑实现

**文件**: `src/baselines/admn.py:204-298`

```python
def forward(self, vision, audio, text):
    # 1. 池化到 [B, 1024]
    v, a, t = vision.mean(dim=1), audio.mean(dim=1), text.mean(dim=1)

    # 2. 层级处理 (每模态独立)
    for layer in self.layer_processors:
        new_features = {}
        for mod in ['vision', 'audio', 'text']:
            new_features[mod] = layer[mod](features[mod])

    # 3. 模态间注意力
    stacked = torch.stack(list(features.values()), dim=1)  # [B, 3, 1024]
    attended, _ = self.cross_attention(stacked, stacked, stacked)

    # 4. 门控融合
    gates = self.gate(concat)  # [B, 3]
    fused = v * gates[:, 0:1] + a * gates[:, 1:2] + t * gates[:, 2:3]
```

---

### 2.3 Centaur (Robust Multimodal Fusion)

**原始论文**: Centaur - Robust Multimodal Fusion (IEEE Sensors 2024)
**实现文件**: `src/baselines/centaur.py`

#### 2.3.1 原始方法描述

Centaur专注于鲁棒性：
- 去噪自编码器清理特征
- 模态补全处理缺失
- 可靠性估计加权融合

#### 2.3.2 改造内容

**原始接口**:
```python
class Centaur(nn.Module):
    def __init__(self, input_dims, num_classes, hidden_dim=256)
    # 包含: 投影 + 去噪 + 补全 + 融合 + 分类
```

**改造后接口**:
```python
class CentaurFusion(nn.Module):
    def __init__(self, input_dim: int = 1024)
    # 只保留: 去噪 + 可靠性估计 + 融合
```

**关键改造点**:
1. **保留去噪模块**: `DenoisingAutoencoder`简化为单层网络
2. **移除补全模块**: 原始有`ModalityCompletion`，统一框架中不需要
3. **保留可靠性估计**: 每模态独立的可靠性估计器

#### 2.3.3 融合逻辑实现

**文件**: `src/baselines/centaur.py:278-356`

```python
def forward(self, vision, audio, text):
    # 1. 池化
    v, a, t = vision.mean(dim=1), audio.mean(dim=1), text.mean(dim=1)

    # 2. 去噪 (残差连接)
    v_clean = v + self.denoiser(v)
    ...

    # 3. 可靠性估计
    r_v = self.reliability_estimator[0](v_clean)  # [B, 1]
    ...

    # 4. 可靠性加权
    v_weighted = v_clean * r_v
    ...

    # 5. 融合
    fused = self.fusion(concat)
```

---

### 2.4 TFN (Tensor Fusion Network)

**原始论文**: Tensor Fusion Network (EMNLP 2017)
**实现文件**: `src/baselines/tfn.py`

#### 2.4.1 原始方法描述

TFN使用张量外积进行多模态融合：
- 三模态外积产生高维张量
- 通常使用低秩近似避免维度爆炸

#### 2.4.2 改造内容

**关键改造**: 由于原始TFN的三模态外积会产生`256^3 = 16M`维，实际实现使用简化版：

**文件**: `src/baselines/tfn.py:11-69`

```python
class TFNFusion(nn.Module):
    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        # 1. 先降维到256
        self.projectors = nn.ModuleDict({
            'vision': nn.Linear(input_dim, hidden_dim),
            ...
        })

        # 2. 拼接而非外积 (简化版)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, input_dim),
            ...
        )
```

**注意**: 这是TFN的简化实现，使用拼接+MLP代替原始的外积运算。

---

### 2.5 FDSNet (Feature Divergence Selection Network)

**原始论文**: FDSNet - Multimodal Dynamic Fusion (Nature 2025)
**实现文件**: `src/baselines/fdsnet.py`

#### 2.5.1 原始方法描述

FDSNet基于模态间特征分歧进行自适应融合：
- 计算KL散度衡量模态间分歧
- 基于分歧度动态加权

#### 2.5.2 改造内容

**简化实现**: 由于KL散度计算开销大，实际使用可学习的分歧权重矩阵。

**文件**: `src/baselines/fdsnet.py:222-275`

```python
class FDSNetFusion(nn.Module):
    def __init__(self, input_dim: int = 1024):
        # 可学习的分歧权重
        self.divergence_weights = nn.Parameter(torch.ones(3, 3))

    def forward(self, vision, audio, text):
        # 应用分歧权重
        for i, feat in enumerate(features):
            weight_sum = sum(self.divergence_weights[i, j] for j in range(3))
            weighted.append(feat * (weight_sum / 3))
```

---

## 3. NAS基线方法

### 3.1 DARTS (Differentiable Architecture Search)

**原始论文**: DARTS - Differentiable Architecture Search (ICLR 2019)
**实现文件**: `src/baselines/darts_fusion.py`

#### 3.1.1 原始方法描述

DARTS使用可微分搜索：
- 定义候选操作集合
- 为每个连接学习架构参数(alpha)
- 通过softmax选择操作

#### 3.1.2 适配实现

**文件**: `src/baselines/darts_fusion.py:13-113`

```python
class DARTSFusionModule(nn.Module):
    def __init__(self, input_dim: int = 1024):
        # 候选操作集合
        self.ops = nn.ModuleList([
            nn.Identity(),                    # 跳跃连接
            nn.Linear(input_dim, input_dim),  # 线性变换
            nn.Sequential(...),               # 带激活的线性
            nn.MultiheadAttention(...)        # 自注意力
        ])

        # 架构参数（可学习）
        self.alphas = nn.Parameter(torch.randn(3, 4))  # 3模态×4操作
```

**关键改造**:
1. **固定搜索空间**: 不像原始DARTS搜索整个网络结构，只搜索每个模态的处理操作
2. **实时权重**: 使用softmax实时计算操作权重，而非离散选择
3. **简化cell结构**: 原始DARTS有复杂的cell结构，这里简化为单层的操作选择

#### 3.1.3 前向传播

```python
def forward(self, vision, audio, text):
    # 1. Softmax获取操作权重
    weights = F.softmax(self.alphas, dim=-1)  # [3, 4]

    # 2. 对每个模态应用加权操作
    for feat, mod_idx in zip(features_list, mod_indices):
        for j, op in enumerate(self.ops):
            out = op(feat)
            feat_transformed.append(weights[mod_idx, j] * out)

    # 3. 平均融合所有模态
    fused = torch.stack(transformed, dim=1).mean(dim=1)
```

---

### 3.2 LLMatic

**原始论文**: LLMatic - Neural Architecture Search via Large Language Models (2023)
**实现文件**: `src/baselines/llmatic_fusion.py`

#### 3.2.1 原始方法描述

LLMatic使用LLM生成架构：
- LLM生成候选架构代码
- 评估性能
- 基于行为多样性选择

#### 3.2.2 适配实现

**文件**: `src/baselines/llmatic_fusion.py:18-148`

```python
class LLMaticFusionModule(nn.Module):
    def __init__(self, input_dim: int = 1024, population_size: int = 10, ...):
        self.llm = UnifiedLLMBackend(api_key=api_key)
        self.fusion_module = None  # 搜索到的架构

        # Fallback架构
        self.fallback = nn.Sequential(...)
```

**关键改造**:
1. **简化搜索**: 实际实现中搜索逻辑被简化，使用预定义的固定架构
2. **Fallback机制**: 如果LLM不可用，使用简单的MLP
3. **代码生成**: 生成Python代码并执行实例化

#### 3.2.3 实际行为

**重要**: 当前实现实际上**没有执行LLM搜索**，而是使用了固定的简单架构：

```python
def _generate_simple_architecture(self):
    return '''
class GeneratedFusion(nn.Module):
    def __init__(self, input_dim=1024):
        self.attn_vision = nn.MultiheadAttention(input_dim, 8, ...)
        self.attn_audio = nn.MultiheadAttention(input_dim, 8, ...)
        self.attn_text = nn.MultiheadAttention(input_dim, 8, ...)
        self.fusion = nn.Linear(input_dim * 3, input_dim)
    '''
```

这意味着LLMatic基线实际上是一个固定的注意力融合网络，**没有体现原始论文的搜索过程**。

---

### 3.3 EvoPrompting

**原始论文**: EvoPrompting - Evolutionary Prompting (arXiv 2023)
**实现文件**: `src/baselines/evoprompting_fusion.py`

#### 3.3.1 原始方法描述

EvoPrompting在提示词空间进行进化：
- 初始化提示词种群
- 交叉、变异生成新提示词
- LLM根据提示词生成架构
- 选择性能好的架构

#### 3.3.2 适配实现

**文件**: `src/baselines/evoprompting_fusion.py:18-170`

与LLMatic类似，实际实现简化为固定架构：

```python
def _generate_attention_architecture(self):
    return '''
class GeneratedFusion(nn.Module):
    def __init__(self, input_dim=1024):
        self.cross_attn = nn.MultiheadAttention(...)
        self.self_attn = nn.MultiheadAttention(...)
        self.ffn = nn.Sequential(...)
    '''
```

**实际行为**: 使用Transformer-like结构（交叉注意力+自注意力+FFN）作为fallback。

---

## 4. 数据集适配

### 4.1 数据加载器

**文件**: `experiments/run_baseline.py:131-229`

```python
def load_data(self):
    """加载数据集 - 支持多种格式"""
    # 支持:
    # 1. 分离的数据文件 (train_data.pkl, valid_data.pkl, test_data.pkl)
    # 2. 合并的数据文件 (包含train/val/test键)
    # 3. 单个文件 (需要手动划分，如VQA)
```

### 4.2 任务类型自动检测

**文件**: `experiments/run_baseline.py:188-221`

```python
# 检测任务类型
labels_unique = len(set(self.train_data['labels'].flatten().tolist()))

if labels_unique > 20:
    # 回归任务 (唯一值多)
    self.is_regression = True
else:
    # 分类任务 (< 20类)
    self.is_regression = False
```

**各数据集配置**:

| 数据集 | 原始标签 | 检测为 | 实际使用 | 原因 |
|--------|----------|--------|----------|------|
| MOSEI | 连续值(-3~3) | 回归 | **10类分类** | 手动指定num_classes=10 |
| IEMOCAP | 9个情感类别 | 分类 | 9类分类 | 自动检测 |
| VQA | 3129个答案 | 大类别分类 | 3129类分类 | 手动指定num_classes=3129 |

### 4.3 回归任务特殊处理

**文件**: `experiments/run_baseline.py:202-213`

```python
if self.is_regression:
    # 标签标准化 (z-score)
    train_labels = self.train_data['labels'].float()
    self.label_mean = train_labels.mean()
    self.label_std = train_labels.std()

    self.train_data['labels'] = (self.train_data['labels'].float() - self.label_mean) / self.label_std
```

---

## 5. 评估流程

### 5.1 训练配置

**文件**: `experiments/run_baseline.py:301-426`

```python
def train(self, epochs: int = 50, lr: float = 0.001, batch_size: int = 64):
    # 回归任务使用更大的学习率和weight decay
    if self.is_regression:
        effective_lr = lr * 2.0
        weight_decay = 1e-4
    else:
        effective_lr = lr
        weight_decay = 1e-5

    optimizer = torch.optim.Adam(..., lr=effective_lr, weight_decay=weight_decay)

    # 损失函数
    if self.is_regression:
        criterion = nn.L1Loss()  # MAE
    else:
        criterion = nn.CrossEntropyLoss()
```

### 5.2 评估指标

**文件**: `experiments/run_baseline.py:428-495`

```python
def evaluate(self, data, dropout_rate: float = 0.0):
    """
    分类: 返回Accuracy (越高越好)
    回归: 返回MAE (越低越好)
    """
```

**模态缺失模拟**:

```python
if dropout_rate > 0:
    for mod_tensor in [vision, audio, text]:
        if mod_tensor is not None:
            mask = (torch.rand(mod_tensor.shape[0], 1, 1) > dropout_rate).float()
            mod_tensor *= mask  # 随机置零模拟缺失
```

### 5.3 完整评估流程

**文件**: `experiments/run_baseline.py:497-531`

```python
def run_full_evaluation(self, seeds: list = [42, 123, 456, 789, 999]):
    """运行完整评估 (多种子 × 多缺失率)"""
    for seed in seeds:
        # 1. 设置随机种子
        torch.manual_seed(seed)

        # 2. 重新初始化模型
        self.model = self._create_model().to(self.device)

        # 3. 训练
        self.train(epochs=50)

        # 4. 测试不同缺失率
        for dropout in [0.0, 0.25, 0.50]:
            metric = self.evaluate(self.test_data, dropout_rate=dropout)
```

### 5.4 mRob计算

**文件**: `experiments/run_baseline.py:533-574`

```python
def _summarize_results(self, results: list):
    # 按缺失率分组
    full_metrics = [r['metric'] for r in results if r['dropout'] == 0.0]
    drop25_metrics = [r['metric'] for r in results if r['dropout'] == 0.25]
    drop50_metrics = [r['metric'] for r in results if r['dropout'] == 0.50]

    # mRob = dropout性能 / 完整性能
    mrob_25 = [d25 / full for full, d25 in zip(full_metrics, drop25_metrics)]
    mrob_50 = [d50 / full for full, d50 in zip(full_metrics, drop50_metrics)]
```

**注意**: 对于分类任务（准确率），mRob < 1表示性能下降；对于回归任务（MAE），mRob > 1表示性能下降。

---

## 6. 结果分析方法

### 6.1 当前结果的问题

根据AI_CONTEXT.md中的记录，基线测试存在以下问题：

**MOSEI回归任务问题**:
```
DynMM:  MAE=0.7959, mRob@25%=1.0000, mRob@50%=1.0000
ADMN:   MAE=0.7959, mRob@25%=1.0000, mRob@50%=1.0000
Centaur: MAE=0.7959, mRob@25%=1.0000, mRob@50%=1.0000
```

**问题分析**:
1. **完全相同的MAE**: 所有方法给出完全相同的MAE值，这不正常
2. **完美的mRob**: 50%模态缺失时mRob=1.0意味着性能没有下降，这与常理不符
3. **验证MAE异常**: 0.7959恰好等于"始终预测训练集均值"的理论值

**可能原因**:
1. 学习率调度器过早降低学习率（`ReduceLROnPlateau` patience=10）
2. 模型实际上没有学习，只输出常数
3. 模态缺失的实现可能有问题（验证时模型输出不变）

**VQA基线问题**:
- 所有基线在VQA上准确率接近0%
- 原因：3129个类别，数据极度稀疏
- 每个类别平均只有1-2个样本
- 传统固定架构无法在如此稀疏的数据上学习

### 6.2 与EAS的对比问题

EAS结果：
- MOSEI: 49.6% 准确率
- IEMOCAP: 52.1% 准确率
- VQA: 52.4% 准确率

**关键差异**:
1. **搜索过程**: EAS运行了200轮迭代搜索，基线只是单次训练
2. **架构优化**: EAS的架构是搜索得到的，基线使用固定架构
3. **训练时间**: EAS的训练时间远长于基线（200轮×多epochs vs 50 epochs）

**公平性质疑**:
- 基线只训练50 epochs，EAS的每个候选架构也训练多个epochs
- 总计算预算不对等
- 基线没有架构搜索过程

### 6.3 建议的改进

1. **增加基线训练轮数**: 从50增加到100-200 epochs
2. **多次随机重启**: 已经实现（5个种子）
3. **超参数调优**: 为每个基线单独调优学习率、batch size等
4. **架构搜索预算**: 给NAS基线（DARTS/LLMatic/EvoPrompting）分配搜索时间
5. **验证实现正确性**: 检查模态缺失的实现是否正确影响了模型输出

---

## 附录A: 运行基线测试的命令

```bash
# DynMM on MOSEI
python experiments/run_baseline.py \
    --method dynmm \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10

# DARTS on IEMOCAP
python experiments/run_baseline.py \
    --method darts \
    --dataset iemocap \
    --data_path data/iemocap_processed \
    --num_classes 9

# LLMatic on VQA
python experiments/run_baseline.py \
    --method llmatic \
    --dataset vqa \
    --data_path data/vqa_processed \
    --num_classes 3129
```

---

## 附录B: 文件索引

| 文件 | 用途 |
|------|------|
| `experiments/run_baseline.py` | 统一评估脚本 |
| `src/models/unified_projection.py` | 统一投影层和分类头 |
| `src/baselines/dynmm.py` | DynMM实现 |
| `src/baselines/admn.py` | ADMN实现 |
| `src/baselines/centaur.py` | Centaur实现 |
| `src/baselines/tfn.py` | TFN实现 |
| `src/baselines/fdsnet.py` | FDSNet实现 |
| `src/baselines/darts_fusion.py` | DARTS适配器 |
| `src/baselines/llmatic_fusion.py` | LLMatic适配器 |
| `src/baselines/evoprompting_fusion.py` | EvoPrompting适配器 |

---

**文档版本**: v1.0
**最后更新**: 2026-03-08
**作者**: Claude Code

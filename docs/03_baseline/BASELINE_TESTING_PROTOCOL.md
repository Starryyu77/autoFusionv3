# 基线测试实验方案 (Baseline Testing Protocol)

**版本**: v1.0
**创建日期**: 2026-03-08
**状态**: 待审核

---

## 目录

1. [实验目标与原则](#1-实验目标与原则)
2. [基线方法详细配置](#2-基线方法详细配置)
3. [数据集处理规范](#3-数据集处理规范)
4. [训练与评估协议](#4-训练与评估协议)
5. [实验执行流程](#5-实验执行流程)
6. [结果分析与报告](#6-结果分析与报告)

---

## 1. 实验目标与原则

### 1.1 实验目标

本实验方案旨在建立一套严格、可复现、公平的基线测试流程，用于比较以下8个基线方法与EAS（我们的方法）的性能：

| 类型 | 方法 | 论文来源 | 核心创新 |
|------|------|----------|----------|
| **NAS基线** | DARTS | ICLR 2019 | 可微分架构搜索 |
| | LLMatic | arXiv 2023 | LLM+质量多样性搜索 |
| | EvoPrompting | arXiv 2023 | 进化提示工程 |
| **固定基线** | DynMM | CVPR 2023 | 动态多模态融合 |
| | TFN | EMNLP 2017 | 张量融合网络 |
| | ADMN | NeurIPS 2025 | 自适应动态网络 |
| | Centaur | IEEE Sensors 2024 | 鲁棒多模态融合 |
| | FDSNet | Nature 2025 | 特征分歧选择网络 |

### 1.2 核心原则

#### 原则1: 统一基座模型
所有方法必须使用相同的预训练特征提取器：
- **视觉**: CLIP-ViT-L/14 (768-dim → 1024-dim)
- **音频**: wav2vec 2.0 Large (1024-dim)
- **文本**: BERT-Base (768-dim → 1024-dim)

**原因**: 如果不统一基座，性能差异可能来自基座选择而非融合架构设计。

#### 原则2: 公平计算预算
- **NAS方法** (DARTS, LLMatic, EvoPrompting): 200轮架构搜索 + 50 epoch训练
- **固定方法** (DynMM, TFN, ADMN, Centaur, FDSNet): 直接50 epoch训练

**原因**: NAS方法需要搜索时间，固定方法不需要，这是方法本身的特性差异。

#### 原则3: 一致的训练配置
所有方法使用相同的优化器、学习率、batch size等超参数（详见第4节）。

#### 原则4: 多次随机种子验证
每个实验运行5个不同的随机种子 [42, 123, 456, 789, 999]，报告均值和标准差。

---

## 2. 基线方法详细配置

### 2.1 统一框架设计

```
输入数据
    ↓
统一投影层 (UnifiedFeatureProjection) - 可训练
    - 视觉: Linear(768, 1024) + LayerNorm + GELU
    - 音频: Identity (已是1024)
    - 文本: Linear(768, 1024) + LayerNorm + GELU
    ↓
基线特定融合模块 - 各方法核心差异
    ↓
统一分类头 (UnifiedClassifier)
    - Linear(1024, 512) + ReLU + Dropout(0.2)
    - Linear(512, num_classes)
    ↓
输出预测
```

### 2.2 各基线方法配置

#### 2.2.1 DARTS (Differentiable Architecture Search)

**原始论文**: [DARTS: Differentiable Architecture Search](https://arxiv.org/abs/1806.09055) (ICLR 2019)

**原始设置参考**:
- 搜索空间: 7种操作 (3×3 conv, 5×5 conv, 3×3 pool, skip, zero)
- 优化: 双层优化 (架构参数 + 网络权重)
- 训练: 50 epochs搜索 + 600 epochs重训练

**我们的适配**:

```python
class DARTSConfig:
    """DARTS多模态适配配置"""

    # 搜索空间 (适配多模态融合)
    search_space = {
        'operations': [
            'identity',           # 跳跃连接
            'linear',             # 线性变换
            'mlp',                # 两层MLP
            'self_attention',     # 自注意力
            'cross_attention',    # 交叉注意力
            'gate',               # 门控机制
            'zero'                # 零操作
        ]
    }

    # 搜索配置
    search_epochs = 200           # 搜索迭代次数
    proxy_samples = 256           # 代理评估样本数
    proxy_epochs = 5              # 代理评估训练epoch

    # 架构参数优化
    arch_lr = 0.0003              # 架构参数学习率
    arch_weight_decay = 0.001

    # 网络权重优化 (在代理评估中)
    network_lr = 0.001
    network_weight_decay = 0.0001

    # 最终训练 (与所有基线一致)
    final_epochs = 50
    final_lr = 0.001
    final_batch_size = 64
```

**融合模块设计**:
```python
class DARTSMultimodalFusion(nn.Module):
    """
    DARTS多模态融合适配

    将DARTS的cell结构应用于多模态融合：
    - 每个模态是一个节点
    - 边上是可学习的操作混合
    """

    def __init__(self, input_dim=1024, num_modalities=3):
        super().__init__()

        # 为每对模态创建混合操作
        self.edges = nn.ModuleDict({
            'v_a': MixedOp(input_dim),  # vision -> audio
            'v_t': MixedOp(input_dim),  # vision -> text
            'a_v': MixedOp(input_dim),  # audio -> vision
            'a_t': MixedOp(input_dim),  # audio -> text
            't_v': MixedOp(input_dim),  # text -> vision
            't_a': MixedOp(input_dim),  # text -> audio
        })

        # 架构参数 (可学习)
        self.alphas = nn.Parameter(torch.randn(6, 7))  # 6条边，7种操作

    def forward(self, vision, audio, text):
        # 应用混合操作
        # ... (具体实现)
        pass
```

#### 2.2.2 LLMatic

**原始论文**: [LLMatic: Neural Architecture Search via Large Language Models and Quality-Diversity Optimization](https://arxiv.org/abs/2306.01102)

**原始设置参考**:
- 种群大小: 50
- 迭代次数: 100
- LLM: GPT-4
- 变异策略: 代码级prompt工程

**我们的适配**:

```python
class LLMaticConfig:
    """LLMatic适配配置"""

    # 搜索配置
    population_size = 20          # 种群大小 (减小以降低API成本)
    num_iterations = 200          # 迭代次数
    proxy_samples = 256
    proxy_epochs = 5

    # LLM配置
    llm_model = "kimi-k2.5"       # 使用阿里云百炼
    temperature = 0.7
    max_tokens = 4096

    # 质量-多样性权衡
    qd_params = {
        'novalty_threshold': 0.1,  # 新颖性阈值
        'quality_weight': 0.7,     # 质量权重
        'diversity_weight': 0.3    # 多样性权重
    }

    # 最终训练
    final_epochs = 50
    final_lr = 0.001
    final_batch_size = 64
```

**关键实现**:
- LLM生成PyTorch代码作为架构
- 行为特征提取用于多样性计算
- 基于准确率-多样性帕累托前沿选择

#### 2.2.3 EvoPrompting

**原始论文**: [EvoPrompting: Evolutionary Prompting for Neural Architecture Search](https://arxiv.org/abs/2302.14838)

**原始设置参考**:
- 提示词种群进化
- 交叉和变异操作在prompt空间
- LLM根据prompt生成代码

**我们的适配**:

```python
class EvoPromptingConfig:
    """EvoPrompting适配配置"""

    # 进化配置
    population_size = 20
    num_iterations = 200
    proxy_samples = 256
    proxy_epochs = 5

    # 进化操作
    crossover_rate = 0.8
    mutation_rate = 0.2
    elite_ratio = 0.1             # 保留top 10%

    # Prompt模板
    base_prompt = """
    Design a multimodal fusion architecture for {task}.

    Input dimensions:
    - vision: [B, 576, 1024]
    - audio: [B, 400, 1024]
    - text: [B, 77, 1024]

    Requirements:
    1. Output shape: [B, 1024]
    2. Handle missing modalities (inputs may be zero tensors)
    3. Use standard PyTorch operations only

    Generate a complete nn.Module class:
    """

    # 最终训练
    final_epochs = 50
    final_lr = 0.001
    final_batch_size = 64
```

#### 2.2.4 DynMM (Dynamic Multimodal Fusion)

**原始论文**: [Dynamic Multimodal Fusion](https://arxiv.org/abs/2204.00102) (CVPR 2023)

**原始设置参考**:
- 动态路由机制
- 基于特征重要性选择活跃模态
- 门控融合策略

**我们的适配**:

```python
class DynMMConfig:
    """DynMM固定架构配置 (无需搜索)"""

    # 架构参数
    input_dim = 1024
    routing_threshold = 0.2       # 活跃模态阈值
    num_fusion_strategies = 3     # 单模态/门控/注意力

    # 训练配置
    epochs = 50
    lr = 0.001
    batch_size = 64
    weight_decay = 1e-4

    # 动态路由 (训练时学习，推理时固定)
    routing_learning_rate = 0.01
```

#### 2.2.5 TFN (Tensor Fusion Network)

**原始论文**: [Tensor Fusion Network for Multimodal Sentiment Analysis](https://arxiv.org/abs/1707.07250) (EMNLP 2017)

**原始设置参考**:
- 三模态外积融合
- 低秩近似避免维度爆炸

**我们的适配**:

```python
class TFNConfig:
    """TFN固定架构配置"""

    # 架构参数
    input_dim = 1024
    hidden_dim = 256              # 降维后维度 (避免256^3=16M爆炸)

    # 简化版TFN (使用低秩近似)
    use_low_rank = True
    rank = 64                     # 低秩近似秩

    # 训练配置
    epochs = 50
    lr = 0.001
    batch_size = 64
    weight_decay = 1e-4
```

**注意**: 使用简化版TFN，避免完整外积的高维度。

#### 2.2.6 ADMN (Adaptive Dynamic Multimodal Network)

**原始论文**: [Adaptive Dynamic Multimodal Network](https://arxiv.org/abs/2502.07862) (NeurIPS 2025)

**原始设置参考**:
- 层级自适应处理
- 每层可动态调整模态处理策略

**我们的适配**:

```python
class ADMNConfig:
    """ADMN固定架构配置"""

    # 架构参数
    input_dim = 1024
    num_layers = 3                # 层级数
    hidden_dim = 1024

    # 自适应控制
    controller_hidden = 512
    skip_threshold = 0.6

    # 训练配置
    epochs = 50
    lr = 0.001
    batch_size = 64
    weight_decay = 1e-4
```

#### 2.2.7 Centaur (Robust Multimodal Fusion)

**原始论文**: [Centaur: Robust Multimodal Fusion for Sensor-based Activity Recognition](https://arxiv.org/abs/2303.04636) (IEEE Sensors 2024)

**原始设置参考**:
- 去噪自编码器
- 模态补全机制
- 可靠性加权融合

**我们的适配**:

```python
class CentaurConfig:
    """Centaur固定架构配置"""

    # 架构参数
    input_dim = 1024
    denoising_hidden = 512

    # 可靠性估计
    reliability_layers = [512, 256, 1]

    # 训练配置
    epochs = 50
    lr = 0.001
    batch_size = 64
    weight_decay = 1e-4

    # 去噪损失权重
    denoise_weight = 0.1
```

#### 2.2.8 FDSNet (Feature Divergence Selection Network)

**原始论文**: [FDSNet: Multimodal Dynamic Fusion](https://www.nature.com/articles/s41598-025-25693-y) (Nature 2025)

**原始设置参考**:
- 基于KL散度衡量模态分歧
- 分歧驱动的自适应融合

**我们的适配**:

```python
class FDSNetConfig:
    """FDSNet固定架构配置"""

    # 架构参数
    input_dim = 1024
    divergence_hidden = 256

    # 分歧计算
    use_kl_divergence = True
    temperature = 1.0

    # 训练配置
    epochs = 50
    lr = 0.001
    batch_size = 64
    weight_decay = 1e-4
```

---

## 3. 数据集处理规范

### 3.1 数据集概述

| 数据集 | 模态 | 样本数 | 类别数 | 任务类型 |
|--------|------|--------|--------|----------|
| **CMU-MOSEI** | V+A+T | 23,000 | 10 | 情感分类 |
| **IEMOCAP** | V+A+T | 7,500 | 9 | 情感识别 |
| **VQA-v2** | V+T | 200,000 | 3,129 | 视觉问答 |

### 3.2 数据预处理流程

#### 步骤1: 特征提取 (预计算)

```bash
# 提取CLIP视觉特征
python scripts/extract_features.py \
    --model clip-vit-l14 \
    --dataset mosei \
    --split train,val,test \
    --output_dir data/mosei/features/

# 提取wav2vec音频特征
python scripts/extract_features.py \
    --model wav2vec2-large \
    --dataset mosei \
    --split train,val,test \
    --output_dir data/mosei/features/

# 提取BERT文本特征
python scripts/extract_features.py \
    --model bert-base \
    --dataset mosei \
    --split train,val,test \
    --output_dir data/mosei/features/
```

**输出格式**:
```python
{
    'vision': np.array([N, 576, 768], dtype=float32),   # CLIP特征
    'audio': np.array([N, 400, 1024], dtype=float32),   # wav2vec特征
    'text': np.array([N, 77, 768], dtype=float32),      # BERT特征
    'labels': np.array([N], dtype=int64)                 # 标签
}
```

#### 步骤2: 数据加载

```python
class MultimodalDataset(Dataset):
    """统一多模态数据集"""

    def __init__(self, feature_path, split='train'):
        data = load_pickle(f"{feature_path}/{split}_data.pkl")

        self.vision = data['vision']      # [N, 576, 768]
        self.audio = data['audio']        # [N, 400, 1024]
        self.text = data['text']          # [N, 77, 768]
        self.labels = data['labels']      # [N]

    def __getitem__(self, idx):
        return {
            'vision': torch.tensor(self.vision[idx]),
            'audio': torch.tensor(self.audio[idx]),
            'text': torch.tensor(self.text[idx]),
            'labels': torch.tensor(self.labels[idx])
        }
```

#### 步骤3: 模态缺失模拟

```python
def apply_modality_dropout(batch, dropout_rate=0.0):
    """
    模拟模态缺失

    Args:
        batch: dict with 'vision', 'audio', 'text'
        dropout_rate: 每个模态被置零的概率

    Returns:
        处理后的batch
    """
    if dropout_rate == 0:
        return batch

    batch_size = batch['vision'].shape[0]

    for mod in ['vision', 'audio', 'text']:
        # 为每个样本决定是否缺失该模态
        mask = (torch.rand(batch_size) > dropout_rate).float()
        # 扩展到特征维度 [B] -> [B, 1, 1]
        mask = mask.view(-1, 1, 1).to(batch[mod].device)
        batch[mod] = batch[mod] * mask

    return batch
```

---

## 4. 训练与评估协议

### 4.1 统一训练配置

```python
TRAINING_CONFIG = {
    # 优化器
    'optimizer': 'Adam',
    'lr': 0.001,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),

    # 学习率调度
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_mode': 'max',        # 监控准确率
    'scheduler_factor': 0.5,
    'scheduler_patience': 10,
    'min_lr': 1e-6,

    # 训练
    'batch_size': 64,
    'max_epochs': 50,
    'early_stop_patience': 20,

    # 损失函数
    'criterion': 'CrossEntropyLoss',

    # 梯度裁剪
    'grad_clip_norm': 1.0,
}
```

### 4.2 评估指标

#### 指标1: 完整模态准确率 (Full Accuracy)

```python
def evaluate_full_accuracy(model, test_loader):
    """评估完整模态下的准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            outputs = model(
                batch['vision'],
                batch['audio'],
                batch['text']
            )
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    return correct / total
```

#### 指标2: 模态鲁棒性 (mRob)

```python
def evaluate_mrob(model, test_loader, dropout_rates=[0.0, 0.25, 0.50]):
    """
    计算模态鲁棒性

    mRob@X% = Accuracy_with_X%_dropout / Accuracy_full
    """
    results = {}

    for dropout in dropout_rates:
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                # 应用模态缺失
                batch = apply_modality_dropout(batch, dropout)

                outputs = model(
                    batch['vision'],
                    batch['audio'],
                    batch['text']
                )
                predictions = outputs.argmax(dim=-1)
                correct += (predictions == batch['labels']).sum().item()
                total += batch['labels'].size(0)

        results[f'acc_{int(dropout*100)}'] = correct / total

    # 计算mRob
    results['mrob_25'] = results['acc_25'] / results['acc_0'] if results['acc_0'] > 0 else 0
    results['mrob_50'] = results['acc_50'] / results['acc_0'] if results['acc_0'] > 0 else 0

    return results
```

#### 指标3: FLOPs计算

```python
def count_flops(model, input_dims={'vision': [1, 576, 1024],
                                    'audio': [1, 400, 1024],
                                    'text': [1, 77, 1024]}):
    """计算模型FLOPs"""
    try:
        from thop import profile

        dummy_inputs = (
            torch.randn(*input_dims['vision']),
            torch.randn(*input_dims['audio']),
            torch.randn(*input_dims['text'])
        )

        flops, params = profile(model, inputs=dummy_inputs, verbose=False)
        return flops, params
    except ImportError:
        print("Warning: thop not installed, cannot count FLOPs")
        return None, None
```

### 4.3 多次随机种子评估

```python
def run_multi_seed_evaluation(model_class, config, seeds=[42, 123, 456, 789, 999]):
    """
    在多个随机种子上运行评估

    Returns:
        dict with mean and std for each metric
    """
    all_results = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running with seed: {seed}")
        print('='*60)

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # 创建模型
        model = model_class(config)
        model = model.to('cuda')

        # 训练
        train_model(model, config)

        # 评估
        results = {}
        results['seed'] = seed
        results['acc_full'] = evaluate_full_accuracy(model, test_loader)
        mrob_results = evaluate_mrob(model, test_loader)
        results.update(mrob_results)
        results['flops'], results['params'] = count_flops(model)

        all_results.append(results)

    # 汇总统计
    summary = {}
    for metric in ['acc_full', 'acc_25', 'acc_50', 'mrob_25', 'mrob_50']:
        values = [r[metric] for r in all_results]
        summary[f'{metric}_mean'] = np.mean(values)
        summary[f'{metric}_std'] = np.std(values)

    summary['all_results'] = all_results
    return summary
```

---

## 5. 实验执行流程

### 5.1 实验前准备

```bash
# 1. 环境检查
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 2. 数据检查
python scripts/verify_data.py \
    --dataset mosei \
    --data_path data/mosei_processed

# 3. 特征维度验证
python scripts/verify_features.py \
    --feature_path data/mosei/features/
```

### 5.2 基线测试执行

#### NAS方法 (需要搜索)

```bash
# DARTS
python experiments/run_baseline.py \
    --method darts \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode search_and_evaluate \
    --search_iterations 200 \
    --epochs 50 \
    --seeds 42 123 456 789 999

# LLMatic
python experiments/run_baseline.py \
    --method llmatic \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode search_and_evaluate \
    --search_iterations 200 \
    --epochs 50 \
    --seeds 42 123 456 789 999

# EvoPrompting
python experiments/run_baseline.py \
    --method evoprompting \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode search_and_evaluate \
    --search_iterations 200 \
    --epochs 50 \
    --seeds 42 123 456 789 999
```

#### 固定方法 (直接训练)

```bash
# DynMM
python experiments/run_baseline.py \
    --method dynmm \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode evaluate \
    --epochs 50 \
    --seeds 42 123 456 789 999

# ADMN
python experiments/run_baseline.py \
    --method admn \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode evaluate \
    --epochs 50 \
    --seeds 42 123 456 789 999

# Centaur
python experiments/run_baseline.py \
    --method centaur \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode evaluate \
    --epochs 50 \
    --seeds 42 123 456 789 999

# TFN
python experiments/run_baseline.py \
    --method tfn \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode evaluate \
    --epochs 50 \
    --seeds 42 123 456 789 999

# FDSNet
python experiments/run_baseline.py \
    --method fdsnet \
    --dataset mosei \
    --data_path data/mosei_processed \
    --num_classes 10 \
    --mode evaluate \
    --epochs 50 \
    --seeds 42 123 456 789 999
```

### 5.3 批量执行脚本

```bash
#!/bin/bash
# run_all_baselines.sh

DATASETS=("mosei" "iemocap" "vqa")
NUM_CLASSES=(10 9 3129)

NAS_METHODS=("darts" "llmatic" "evoprompting")
FIXED_METHODS=("dynmm" "admn" "centaur" "tfn" "fdsnet")

# Run NAS methods (with search)
for i in ${!DATASETS[@]}; do
    dataset=${DATASETS[$i]}
    num_cls=${NUM_CLASSES[$i]}

    for method in ${NAS_METHODS[@]}; do
        echo "Running $method on $dataset..."
        python experiments/run_baseline.py \
            --method $method \
            --dataset $dataset \
            --data_path data/${dataset}_processed \
            --num_classes $num_cls \
            --mode search_and_evaluate \
            --search_iterations 200 \
            --epochs 50 \
            --output_dir results/baselines_v2
    done
done

# Run fixed methods (direct evaluation)
for i in ${!DATASETS[@]}; do
    dataset=${DATASETS[$i]}
    num_cls=${NUM_CLASSES[$i]}

    for method in ${FIXED_METHODS[@]}; do
        echo "Running $method on $dataset..."
        python experiments/run_baseline.py \
            --method $method \
            --dataset $dataset \
            --data_path data/${dataset}_processed \
            --num_classes $num_cls \
            --mode evaluate \
            --epochs 50 \
            --output_dir results/baselines_v2
    done
done
```

---

## 6. 结果分析与报告

### 6.1 结果收集

```python
# scripts/collect_results.py

import json
import glob
import pandas as pd

def collect_baseline_results(results_dir='results/baselines_v2'):
    """收集所有基线结果"""

    all_results = []

    for result_file in glob.glob(f"{results_dir}/*.json"):
        with open(result_file) as f:
            data = json.load(f)
            all_results.append(data)

    # 转换为DataFrame
    df = pd.DataFrame(all_results)

    # 保存汇总
    df.to_csv(f"{results_dir}/summary.csv", index=False)

    return df
```

### 6.2 生成对比表格

```python
def generate_comparison_table(df):
    """生成Table 2 (论文主表)"""

    # 按数据集分组
    for dataset in ['mosei', 'iemocap', 'vqa']:
        print(f"\n{'='*80}")
        print(f"Results on {dataset.upper()}")
        print('='*80)

        subset = df[df['dataset'] == dataset]

        # 排序: EAS > 其他
        methods_order = ['eas', 'darts', 'llmatic', 'evoprompting',
                         'dynmm', 'admn', 'centaur', 'tfn', 'fdsnet']

        for method in methods_order:
            row = subset[subset['method'] == method]
            if len(row) == 0:
                continue

            acc_mean = row['accuracy_mean'].values[0] * 100
            acc_std = row['accuracy_std'].values[0] * 100
            mrob25_mean = row['mrob_25_mean'].values[0]
            mrob25_std = row['mrob_25_std'].values[0]
            mrob50_mean = row['mrob_50_mean'].values[0]
            mrob50_std = row['mrob_50_std'].values[0]

            print(f"{method:15s} | Acc: {acc_mean:5.2f}±{acc_std:4.2f} | "
                  f"mRob@25%: {mrob25_mean:.3f}±{mrob25_std:.3f} | "
                  f"mRob@50%: {mrob50_mean:.3f}±{mrob50_std:.3f}")
```

### 6.3 统计显著性检验

```python
from scipy import stats

def statistical_significance_test(eas_results, baseline_results):
    """
    检验EAS与基线的差异是否统计显著

    使用配对t检验 (5个随机种子)
    """
    # 提取5个seed的结果
    eas_accuracies = [r['metric'] for r in eas_results['raw_results']
                      if r['dropout'] == 0.0]
    baseline_accuracies = [r['metric'] for r in baseline_results['raw_results']
                           if r['dropout'] == 0.0]

    # 配对t检验
    t_stat, p_value = stats.ttest_rel(eas_accuracies, baseline_accuracies)

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

---

## 附录A: 验证清单

### 实验前检查

- [ ] 所有数据文件存在且格式正确
- [ ] 特征维度验证通过 (768/1024)
- [ ] GPU可用且内存充足 (>8GB)
- [ ] 所有依赖包已安装
- [ ] 随机种子设置正确

### 实验中检查

- [ ] 模型输出不是常数 (std > 0.01)
- [ ] 损失函数正常下降
- [ ] 学习率调度正常工作
- [ ] 模态缺失模拟有效 (不同dropout率结果不同)

### 实验后检查

- [ ] 所有5个seed的结果都已保存
- [ ] 不同基线给出不同结果
- [ ] 结果在合理范围内 (非0%或100%)
- [ ] 标准差不为0

---

## 附录B: 常见问题

### Q1: 为什么基线结果都一样？

**可能原因**:
1. 模型输出常数 (检查输出std)
2. 学习率太小或太大
3. 数据加载有问题
4. 损失函数设置错误

**诊断**:
```python
# 检查模型输出是否多样
outputs = model(batch)
print(f"Output std: {outputs.std()}")  # 应该 > 0.1
```

### Q2: 模态缺失测试无效 (mRob=1.0)？

**可能原因**:
1. 模型没有使用某些模态
2. 缺失模拟代码有问题
3. 模型输出对输入不敏感

**诊断**:
```python
# 手动测试模态缺失
output_full = model(v, a, t)
output_missing = model(v * 0, a, t)
diff = (output_full - output_missing).abs().mean()
print(f"Difference: {diff}")  # 应该 > 0.1
```

### Q3: 如何确保公平比较？

**关键检查点**:
1. 所有方法使用相同的预提取特征
2. 所有方法的投影层相同
3. 所有方法的训练epochs相同
4. 所有方法的评估代码相同
5. 所有方法使用相同的随机种子

---

**文档结束**

# EAS Round 2 实验文档模板

## 实验基本信息

| 项目 | 内容 |
|------|------|
| **实验名称** | {method_name}_{dataset}_{seed}_{modality_dropout} |
| **方法** | {EAS / DARTS / LLMatic / EvoPrompting / DynMM / TFN / ADMN / Centaur} |
| **数据集** | {CMU-MOSEI / VQA-v2 / IEMOCAP} |
| **随机种子** | {42 / 123 / 456 / 789 / 999} |
| **模态缺失率** | {0% / 25% / 50%} |
| **运行时间** | YYYY-MM-DD HH:MM:SS |
| **执行机器** | NTU GPU43 (GPU {0/1/2/3}) |

---

## 1. 实验方法

### 1.1 方法概述

**方法名称**: {方法全名}

**核心思想**:
- 简要描述该方法的核心创新点
- 与传统NAS方法的区别
- 如何处理多模态融合

**算法流程**:
```
输入: 搜索空间 S, 评估函数 E, 迭代次数 T
输出: 最优架构 A*

for t = 1 to T:
    1. 生成/选择候选架构
    2. 评估候选架构性能
    3. 更新搜索策略
    4. 记录最佳结果

return A*
```

### 1.2 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 迭代次数 | 200 | 搜索迭代次数 |
| 种群大小 | 20 | 每代候选架构数量 |
| 学习率 | 0.001 | 训练学习率 |
| Batch Size | 32 | 训练批次大小 |
| 早停耐心 | 20 | 早停等待轮数 |
| 其他方法特定参数 | ... | ... |

### 1.3 奖励函数 (仅EAS)

```python
reward = w_acc * mAcc + w_rob * mRob - w_flops * GFLOPs/1e9

其中:
- w_acc = 1.0    (准确率权重)
- w_rob = 2.0    (鲁棒性权重)
- w_flops = 0.5  (效率权重)
```

---

## 2. 数据来源

### 2.1 数据集信息

| 属性 | 值 |
|------|-----|
| **数据集名称** | {CMU-MOSEI / VQA-v2 / IEMOCAP} |
| **数据类型** | {多模态情感分析 / 视觉问答 / 多模态情感识别} |
| **模态** | {视觉+音频+文本 / 视觉+文本 / 视觉+音频+文本} |
| **样本总数** | {22,856 / 443,757 / 10,039} |
| **训练/验证/测试** | {70% / 15% / 15%} |

### 2.2 特征提取

| 模态 | 特征提取器 | 输出维度 | 预训练权重 |
|------|-----------|---------|-----------|
| **视觉** | CLIP-ViT-L/14 | [576, 1024] | openai/clip-vit-large-patch14 |
| **音频** | wav2vec 2.0 Large | [400, 1024] | facebook/wav2vec2-large-960h |
| **文本** | BERT-Base | [77, 768] | bert-base-uncased |

### 2.3 模态缺失模拟

```python
# 模态缺失策略
missing_strategy = {
    "mode": "random",  # random / burst / progressive
    "rate": 0.5,       # 缺失率
    "implementation": "zero_fill"  # zero_fill / noise_fill
}
```

**缺失模式说明**:
- **Random**: 随机缺失单个模态
- **Burst**: 连续时间窗缺失（模拟传感器故障）
- **Progressive**: 渐进缺失（模拟信号衰减）

---

## 3. 架构设计

### 3.1 搜索空间

**输入维度**:
```python
input_dims = {
    "vision": [batch_size, 576, 1024],
    "audio": [batch_size, 400, 512],
    "text": [batch_size, 77, 768]
}
```

**输出维度**:
```python
output_dim = [batch_size, num_classes]  # num_classes = 10 (MOSEI)
```

**允许的层类型**:
- Linear, Conv1d, Conv2d
- LSTM, GRU, Transformer
- Attention, MultiheadAttention
- BatchNorm, LayerNorm
- Dropout, DropPath
- 激活函数: ReLU, GELU, Sigmoid, Tanh

### 3.2 生成的最优架构

**架构代码**:
```python
# 最优架构的PyTorch代码
class BestArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        # ... 架构定义

    def forward(self, vision, audio, text):
        # ... 前向传播
        return output
```

**架构特点**:
| 特征 | 值 | 分析 |
|------|-----|------|
| 参数数量 | X,XXX,XXX | 模型复杂度 |
| FLOPs | X.XX G | 计算效率 |
| 注意力层数 | X | 特征交互能力 |
| 条件分支 | Y/N | 是否涌现动态结构 |

---

## 4. 实验结果

### 4.1 主要指标

| 指标 | 完整模态 | 25%缺失 | 50%缺失 | 平均 |
|------|---------|---------|---------|------|
| **Accuracy** | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **F1-Score** | XX.XX% | XX.XX% | XX.XX% | XX.XX% |
| **mRob** | 1.000 | X.XXX | X.XXX | X.XXX |

**mRob 计算**:
```
mRob = Performance_missing / Performance_full
```

### 4.2 计算效率

| 指标 | 值 | 单位 |
|------|-----|------|
| **参数量** | X.XX | Million |
| **FLOPs** | X.XX | Giga |
| **推理时间** | X.XX | ms/batch |
| **训练时间** | X.XX | hours |
| **搜索时间** | X.XX | hours |

### 4.3 收敛曲线

```
迭代次数 vs 最佳奖励:
Iter 000: 0.XXX
Iter 050: 0.XXX
Iter 100: 0.XXX
Iter 150: 0.XXX
Iter 200: 0.XXX
```

### 4.4 与基线对比

| 方法 | mAcc | mRob@50% | GFLOPs | 排名 |
|------|------|----------|--------|------|
| **EAS (ours)** | XX.X% | X.XXX | X.XX | 1 |
| DynMM | XX.X% | X.XXX | X.XX | 2 |
| ADMN | XX.X% | X.XXX | X.XX | 3 |
| ... | ... | ... | ... | ... |

---

## 5. 消融实验 (仅EAS)

### 5.1 组件消融

| 配置 | mAcc | mRob@50% | 说明 |
|------|------|----------|------|
| 完整EAS | XX.X% | X.XXX | 基线 |
| - 内循环 | XX.X% | X.XXX | 移除自修复编译 |
| - 外循环 | XX.X% | X.XXX | 移除进化优化 |
| - mRob奖励 | XX.X% | X.XXX | 仅优化准确率 |

### 5.2 超参数敏感性

| 参数 | 取值 | mAcc | mRob@50% |
|------|------|------|----------|
| w_rob | 1.0 | XX.X% | X.XXX |
| w_rob | 2.0 | XX.X% | X.XXX |
| w_rob | 3.0 | XX.X% | X.XXX |

---

## 6. 可解释性分析

### 6.1 涌现模式

**架构代码分析**:
- 条件分支数量: X
- 模态门控机制: Y/N
- 残差连接: Y/N
- 注意力模式: {self/cross/multi-head}

**典型代码片段**:
```python
# 涌现的条件执行示例
if confidence < 0.3:
    # 跳过不可靠模态
    output = self.fuse_remaining_modalities(...)
```

### 6.2 注意力可视化

[可选: 添加注意力权重热力图]

---

## 7. 实验日志

### 7.1 运行日志

```
[YYYY-MM-DD HH:MM:SS] 实验启动
[YYYY-MM-DD HH:MM:SS] 数据加载完成
[YYYY-MM-DD HH:MM:SS] 第10轮: best_reward=0.XXX
[YYYY-MM-DD HH:MM:SS] 第50轮: best_reward=0.XXX
...
[YYYY-MM-DD HH:MM:SS] 实验完成
```

### 7.2 错误与警告

| 时间 | 类型 | 描述 | 解决方案 |
|------|------|------|----------|
| XX:XX | Warning | OOM警告 | 减小batch size |
| XX:XX | Error | 编译失败 | 重试次数+1 |

---

## 8. 复现信息

### 8.1 代码版本

```bash
# Git commit
commit: xxxxxxx
branch: main
date: YYYY-MM-DD
```

### 8.2 运行命令

```bash
# 复现实验的命令
python experiments/run_round2_main.py \
    --config configs/round2_{method}_{dataset}.yaml \
    --seed 42 \
    --gpu 0
```

### 8.3 环境信息

```
Python: 3.10.12
PyTorch: 2.1.0+cu118
CUDA: 11.8
GPU: NVIDIA RTX A5000 24GB
```

---

## 9. 结论与讨论

### 9.1 主要发现

1. **发现1**: ...
2. **发现2**: ...
3. **发现3**: ...

### 9.2 局限性

1. **局限性1**: ...
2. **局限性2**: ...

### 9.3 未来工作

1. **方向1**: ...
2. **方向2**: ...

---

## 附录

### A. 原始数据文件

| 文件 | 路径 | 说明 |
|------|------|------|
| 配置 | `configs/round2_xxx.yaml` | 实验配置 |
| 结果 | `results/round2/xxx.json` | 原始结果 |
| 日志 | `logs/round2_xxx.log` | 运行日志 |
| 架构 | `results/round2/arch_xxx.py` | 最优架构代码 |

### B. 引用

如果发表，请引用:
```bibtex
@inproceedings{eas2026,
  title={Executable Architecture Synthesis: Open-Space Neural Architecture Search with Emergent Multimodal Robustness},
  author={...},
  booktitle={ICCV/CVPR/NeurIPS},
  year={2026}
}
```

---

*文档生成时间: YYYY-MM-DD HH:MM:SS*
*作者: AutoFusion v3 Team*

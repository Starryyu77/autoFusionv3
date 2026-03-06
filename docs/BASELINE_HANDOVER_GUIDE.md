# 基线测试实验交接指南

**交接日期**: 2026-03-07
**交接人**: AutoFusion Team
**接收人**: [合作同学姓名]

---

## 1. 交接文件清单

### 1.1 必须提供的代码文件

```
autofusionv3/
├── src/
│   ├── baselines/                    # 【核心】基线方法实现
│   │   ├── __init__.py
│   │   ├── darts.py                  # DARTS (已有示例)
│   │   ├── llmatic.py                # 【待实现】
│   │   ├── evo_prompting.py          # 【待实现】
│   │   ├── dynmm.py                  # 【待实现】
│   │   ├── fdsnet.py                 # 【待实现】
│   │   ├── admn.py                   # 【待实现】
│   │   └── centaur.py                # 【待实现】
│   │
│   ├── data/                         # 【核心】数据加载 (不可修改)
│   │   ├── __init__.py
│   │   ├── base_loader.py            # 数据集基类
│   │   ├── mosei_loader.py           # CMU-MOSEI加载器
│   │   ├── vqa_loader.py             # VQA-v2加载器
│   │   └── modality_dropout.py       # 模态缺失模拟 (必须统一)
│   │
│   ├── evaluator/                    # 【核心】评估指标 (不可修改)
│   │   ├── __init__.py
│   │   ├── proxy_evaluator.py        # 代理评估器
│   │   └── multimodal_rob.py         # mRob计算
│   │
│   └── utils/                        # 【核心】工具函数 (不可修改)
│       ├── __init__.py
│       ├── random_control.py         # 随机种子控制
│       └── logging_utils.py          # 日志记录
│
├── configs/                          # 【参考】配置文件
│   ├── round2_main_mosei.yaml        # MOSEI主实验配置
│   ├── round2_main_vqa.yaml          # VQA实验配置
│   ├── round2_main_iemocap.yaml      # IEMOCAP实验配置
│   └── round2_ablation.yaml          # 消融实验配置
│
├── experiments/                      # 【参考】实验脚本模板
│   └── run_round2_main.py            # Round 2主实验 (参考用)
│
└── docs/
    ├── EXPERIMENT_CONTROL_PROTOCOL.md  # 【必须阅读】控制变量协议
    └── BASELINE_HANDOVER_GUIDE.md      # 本文件
```

### 1.2 需要复现的基线方法

| 方法 | 论文 | 优先级 | 难度 | 预计代码量 |
|------|------|--------|------|-----------|
| **DARTS** | [Liu et al., ICLR 2019](https://arxiv.org/abs/1806.09055) | P0 | 中 | 已有示例 |
| **LLMatic** | [Nasir et al., GECCO 2024](https://arxiv.org/abs/2306.01102) | P0 | 中 | ~300行 |
| **EvoPrompting** | [Chen et al., NeurIPS 2023](https://arxiv.org/abs/2302.14838) | P0 | 中 | ~300行 |
| **DynMM** | [Xue & Marculescu, CVPR 2023](https://arxiv.org/abs/2204.00102) | P0 | 高 | ~400行 |
| **FDSNet** | [Mohammed et al., Nature 2025](https://www.nature.com/articles/s41598-025-25693-y) | P1 | 中 | ~400行 |
| **ADMN** | [Wu et al., NeurIPS 2025](https://arxiv.org/abs/2502.07862) | P1 | 高 | ~600行 |
| **Centaur** | [Xaviar et al., IEEE Sensors 2024](https://arxiv.org/abs/2303.04636) | P1 | 中 | ~350行 |

---

## 2. 基线实现规范

### 2.1 接口规范 (必须严格遵守)

每个基线方法必须实现以下接口：

```python
class BaselineModel(nn.Module):
    """
    基线模型接口规范

    所有基线方法必须遵循此接口，以确保评估一致性
    """

    def __init__(
        self,
        input_dims: Dict[str, int],      # {'vision': 1024, 'audio': 512, 'text': 768}
        num_classes: int,                 # 输出类别数
        hidden_dim: int = 256,            # 隐藏层维度 (统一)
        **kwargs                          # 方法特定参数
    ):
        super().__init__()
        # 初始化代码

    def forward(self, **inputs) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: {'vision': [B, T, D], 'audio': [B, T, D], 'text': [B, T, D]}

        Returns:
            output: [B, num_classes]
        """
        pass

    def get_flops(self) -> int:
        """返回模型FLOPs"""
        pass
```

### 2.2 实现模板

```python
# src/baselines/[method_name].py

import torch
import torch.nn as nn
from typing import Dict

class [MethodName]Model(nn.Module):
    """
    [方法名称] 基线实现

    论文: [论文标题]
    链接: [论文链接]

    关键创新:
    - [创新点1]
    - [创新点2]
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        num_classes: int,
        hidden_dim: int = 256,
        **kwargs
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # TODO: 实现模型结构

    def forward(self, **inputs) -> torch.Tensor:
        # TODO: 实现前向传播
        pass

    def get_flops(self) -> int:
        # TODO: 计算FLOPs
        pass


def create_[method_name]_model(
    input_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> [MethodName]Model:
    """
    创建[方法名称]模型的工厂函数

    所有基线必须提供此函数，用于统一调用
    """
    return [MethodName]Model(
        input_dims=input_dims,
        num_classes=num_classes,
        **kwargs
    )
```

---

## 3. 评估流程

### 3.1 统一评估脚本

使用 `experiments/run_round2_main.py` 进行统一评估：

```bash
# 评估单个基线
python experiments/run_round2_main.py \
    --config configs/round2_main_mosei.yaml \
    --method DARTS

# 评估所有基线
python experiments/run_round2_main.py \
    --config configs/round2_main_mosei.yaml
```

### 3.2 评估指标

必须报告的指标：

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **mAcc** | 平均准确率 | `correct / total` |
| **mRob** | 模态鲁棒性 | `Acc_missing / Acc_full` |
| **GFLOPs** | 计算量 | 自动计算 |
| **Latency** | 推理延迟 | 平均延迟(ms) |

### 3.3 模态缺失测试

必须测试以下场景：

```python
from data.modality_dropout import UnifiedModalityDropout

# 完整模态 (0%缺失)
dropout = UnifiedModalityDropout(drop_prob=0.0, mode='random', seed=42)

# 25%缺失
dropout = UnifiedModalityDropout(drop_prob=0.25, mode='random', seed=42)

# 50%缺失
dropout = UnifiedModalityDropout(drop_prob=0.50, mode='random', seed=42)
```

---

## 4. 控制变量 (必须严格遵守)

### 4.1 不可修改的变量

| 变量 | 固定值 | 说明 |
|------|--------|------|
| **随机种子** | [42, 123, 456, 789, 1024] | 5个种子，必须全部测试 |
| **训练轮数** | 15 epochs | 所有基线统一 |
| **Batch Size** | 32 | 统一 |
| **学习率** | 0.001 | AdamW优化器 |
| **隐藏维度** | 256 | 模型隐藏层 |
| **模态缺失模拟** | UnifiedModalityDropout | 必须使用此类 |

### 4.2 数据集规格

**CMU-MOSEI**:
- 输入: `{'vision': [B, 576, 1024], 'audio': [B, 400, 512], 'text': [B, 77, 768]}`
- 输出: `[B, 10]` (情感强度回归)
- 样本数: Train 16,265 / Val 1,869 / Test 4,643

**VQA-v2**:
- 输入: `{'vision': [B, 197, 768], 'text': [B, 20, 768]}`
- 输出: `[B, 3129]` (答案分类)
- 样本数: Train 443K / Val 214K

**IEMOCAP**:
- 输入: `{'vision': [B, T, D], 'audio': [B, T, D], 'text': [B, T, D]}`
- 输出: `[B, 9]` (情感分类)
- 样本数: ~12小时视频

---

## 5. 交付物清单

### 5.1 代码交付物

- [ ] `src/baselines/llmatic.py` - LLMatic实现
- [ ] `src/baselines/evo_prompting.py` - EvoPrompting实现
- [ ] `src/baselines/dynmm.py` - DynMM实现
- [ ] `src/baselines/fdsnet.py` - FDSNet实现 (可选)
- [ ] `src/baselines/admn.py` - ADMN实现 (可选)
- [ ] `src/baselines/centaur.py` - Centaur实现 (可选)

### 5.2 实验结果交付物

每个基线方法需要提供：

```
results/round2/
├── [method_name]_mosei_results.csv      # MOSEI实验结果
├── [method_name]_vqa_results.csv        # VQA实验结果
├── [method_name]_iemocap_results.csv    # IEMOCAP实验结果
└── [method_name]_report.md              # 实验报告
```

**CSV格式**:
```csv
method,seed,dropout_prob,accuracy,mrob,gflops,latency
DARTS,42,0.0,0.582,1.0,12.3,15.2
DARTS,42,0.5,0.320,0.55,12.3,15.2
...
```

### 5.3 文档交付物

- [ ] `docs/baselines/[method_name]_notes.md` - 实现笔记
  - 论文核心方法总结
  - 复现难点及解决方案
  - 超参数设置说明
  - 与原文的差异说明

---

## 6. 时间规划

| 周次 | 任务 | 交付物 |
|------|------|--------|
| Week 1 | DARTS + LLMatic | 2个基线 + 报告 |
| Week 2 | EvoPrompting + DynMM | 2个基线 + 报告 |
| Week 3 | FDSNet + ADMN + Centaur | 3个基线 + 报告 |
| Week 4 | 结果整合 | 所有CSV + 汇总表格 |

---

## 7. 常见问题

### Q1: 论文代码开源，可以直接用吗？

**A**: 可以借鉴，但需要：
1. 适配我们的数据接口 (MOSEI/VQA/IEMOCAP)
2. 使用我们的评估指标 (特别是mRob)
3. 遵守我们的控制变量 (种子、训练轮数等)

### Q2: 如果某篇论文方法太复杂怎么办？

**A**: 优先保证核心创新点的复现：
- 简化次要模块
- 保持关键参数与论文一致
- 在报告中说明简化内容

### Q3: 如何测试自己的实现是否正确？

**A**: 使用DARTS作为对照：
```python
# 测试代码
python experiments/run_round2_main.py \
    --config configs/round2_main_mosei.yaml \
    --method DARTS \
    --quick  # 快速测试 (10样本)
```

### Q4: 实验需要多少GPU资源？

**A**:
- 单基线 × 单数据集 × 5种子 × 3缺失率 = 15次运行
- 每次约30分钟 (A5000)
- 建议并行使用多个GPU

---

## 8. 联系方式

- **GitHub**: https://github.com/Starryyu77/autoFusionv3
- **服务器**: gpu43.dynip.ntu.edu.sg
- **项目路径**: `/usr1/home/s125mdg43_10/AutoFusion_v3`

---

## 9. 检查清单 (交接时勾选)

- [ ] 已阅读 `EXPERIMENT_CONTROL_PROTOCOL.md`
- [ ] 已理解基线接口规范
- [ ] 已获取服务器访问权限
- [ ] 已配置Python环境
- [ ] 已运行DARTS示例成功
- [ ] 已明确时间规划

---

**签名**: _________________
**日期**: _________________

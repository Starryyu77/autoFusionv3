# 基线方法基座模型一致性规范

**版本**: v1.0
**关键原则**: 所有基线方法必须使用相同的冻结基座模型，否则对比不公平

---

## 1. 问题分析

### 1.1 原论文基座 vs 我们的基座

| 基线方法 | 原论文基座 | 原论文任务 | 我们的适配 |
|---------|-----------|-----------|-----------|
| **DARTS** | CIFAR-10 CNN / ImageNet | 图像分类 | 强制使用CLIP特征 |
| **LLMatic** | 可能无特定基座 | 代码生成NAS | 强制使用CLIP特征 |
| **EvoPrompting** | 可能无特定基座 | 代码级NAS | 强制使用CLIP特征 |
| **DynMM** | 可能有自定义基座 | 多模态融合 | 强制使用CLIP特征 |
| **FDSNet** | Camera+LiDAR (自动驾驶) | 多模态检测 | 强制使用CLIP特征 |
| **ADMN** | 可能有自定义基座 | 自适应多模态 | 强制使用CLIP特征 |
| **Centaur** | 传感器数据 | 活动识别 | 强制使用CLIP特征 |

### 1.2 核心原则

**所有基线方法必须使用统一的基座模型**：
- 视觉: CLIP-ViT-L/14 (1024-dim)
- 音频: wav2vec 2.0 Large (1024-dim)
- 文本: BERT-Base (768-dim)

**原因**: 如果不统一基座，性能差异可能来自基座模型而非融合架构本身，导致对比不公平。

---

## 2. 基线方法适配策略

### 2.1 适配方式

对于每个基线方法，我们只复现其**融合架构的核心创新**，替换其基座为我们的标准基座。

```python
# 统一的基线方法接口
class BaselineAdapter(nn.Module):
    """
    所有基线方法的统一适配器

    强制使用相同的基座，只替换融合层
    """

    def __init__(self,
                 baseline_fusion_module,  # 基线方法的核心融合模块
                 input_dims: dict,         # 统一输入维度
                 num_classes: int):
        super().__init__()

        # 统一投影层 (将基座特征投影到统一维度)
        self.projections = nn.ModuleDict({
            'vision': nn.Linear(1024, 256),
            'audio': nn.Linear(1024, 256),
            'text': nn.Linear(768, 256)
        })

        # 基线方法的核心融合模块 (保持其创新设计)
        self.fusion = baseline_fusion_module

        # 统一分类器
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, vision, audio, text):
        # 投影
        v = self.projections['vision'](vision.mean(dim=1))
        a = self.projections['audio'](audio.mean(dim=1))
        t = self.projections['text'](text.mean(dim=1))

        # 基线融合方法
        fused = self.fusion(v, a, t)

        return self.classifier(fused)
```

### 2.2 各基线适配说明

#### DARTS
- **原论文**: 搜索CNN细胞结构
- **我们的适配**: 将其可微分搜索应用于多模态融合层
- **保留**: 可微分架构搜索机制
- **替换**: 卷积操作 → 多模态注意力/融合操作

#### DynMM
- **原论文**: 动态多模态融合，可能有特定的模态编码器
- **我们的适配**: 保持其动态路由机制，替换编码器为标准基座
- **保留**: 动态融合路由策略
- **替换**: 自定义编码器 → CLIP/wav2vec/BERT

#### ADMN
- **原论文**: 层级自适应网络
- **我们的适配**: 保持其自适应机制，应用在我们的基座特征上
- **保留**: 层级自适应计算
- **替换**: 基础网络 → 我们的标准基座

---

## 3. 实现规范

### 3.1 每个基线必须实现

```python
# src/baselines/[method_name].py

import torch
import torch.nn as nn
from typing import Dict

class [MethodName]Fusion(nn.Module):
    """
    [方法名]的核心融合模块

    只实现融合逻辑，不管基座！
    输入是统一维度的特征 [B, hidden_dim]
    """

    def __init__(self,
                 hidden_dim: int = 256,
                 num_modalities: int = 3,
                 **kwargs):
        super().__init__()

        # 实现该方法的核心融合创新
        # 例如：DARTS的可微分搜索、DynMM的动态路由等

    def forward(self,
                vision_feat: torch.Tensor,   # [B, hidden_dim]
                audio_feat: torch.Tensor,    # [B, hidden_dim]
                text_feat: torch.Tensor      # [B, hidden_dim]
               ) -> torch.Tensor:           # [B, hidden_dim]
        """
        融合各模态特征

        Args:
            vision_feat: 视觉特征 (已投影)
            audio_feat: 音频特征 (已投影)
            text_feat: 文本特征 (已投影)

        Returns:
            融合后的特征 [B, hidden_dim]
        """
        pass


def create_[method_name]_model(
    input_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    创建完整的模型（包含统一基座投影 + 基线融合）
    """
    from .base_wrapper import BaselineModelWrapper

    # 创建基线融合模块
    fusion_module = [MethodName]Fusion(**kwargs)

    # 包装成完整模型（自动添加投影层和分类器）
    return BaselineModelWrapper(
        fusion_module=fusion_module,
        input_dims=input_dims,
        num_classes=num_classes,
        backbone_dims={
            'vision': 1024,  # CLIP输出
            'audio': 1024,   # wav2vec输出
            'text': 768      # BERT输出
        }
    )
```

### 3.2 基座包装器（统一使用）

```python
# src/baselines/base_wrapper.py

import torch
import torch.nn as nn
from typing import Dict


class BaselineModelWrapper(nn.Module):
    """
    基线方法统一包装器

    确保所有基线方法：
    1. 使用相同的基座投影
    2. 输入输出格式一致
    3. 便于公平对比
    """

    def __init__(self,
                 fusion_module: nn.Module,
                 input_dims: Dict[str, int],
                 num_classes: int,
                 backbone_dims: Dict[str, int],
                 hidden_dim: int = 256):
        super().__init__()

        # 基座特征投影层（将所有基座特征投影到统一维度）
        self.projections = nn.ModuleDict()
        for mod, dim in backbone_dims.items():
            self.projections[mod] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

        # 基线方法的核心融合模块
        self.fusion = fusion_module

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, **inputs) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: {'vision': [B, T, 1024], 'audio': [B, T, 1024], 'text': [B, T, 768]}

        Returns:
            [B, num_classes]
        """
        # 投影各模态特征
        projected = []
        for mod, x in inputs.items():
            if mod in self.projections:
                # 时间维度平均池化 + 投影
                x = x.mean(dim=1)  # [B, backbone_dim]
                x = self.projections[mod](x)  # [B, hidden_dim]
                projected.append(x)

        # 如果只有一个模态，直接返回
        if len(projected) == 1:
            fused = projected[0]
        else:
            # 使用基线融合方法
            # 注意：这里假设fusion_module接受多个特征并返回融合特征
            fused = self.fusion(*projected)

        # 分类
        return self.classifier(fused)
```

---

## 4. 公平对比保证

### 4.1 必须统一的要素

| 要素 | 统一值 | 说明 |
|------|--------|------|
| **基座模型** | CLIP/wav2vec/BERT | 所有方法相同 |
| **基座输出维度** | 1024/1024/768 | 固定 |
| **投影后维度** | 256 | 所有方法相同 |
| **训练轮数** | 15 | few-shot评估 |
| **优化器** | AdamW, lr=0.001 | 相同 |
| **数据增强** | 无 | 基座特征已提取 |
| **评估指标** | mAcc, mRob, GFLOPs | 统一 |

### 4.2 可以不同的要素

| 要素 | 说明 |
|------|------|
| **融合架构** | 各基线方法的核心创新 |
| **参数量** | 允许不同（在GFLOPs中体现）|
| **训练时间** | 允许不同（但都要在合理范围内）|

---

## 5. 文档交付要求

每个基线方法实现时，必须在文档中明确：

```markdown
# [方法名] 实现说明

## 原论文信息
- 标题: [论文标题]
- 会议/期刊: [如CVPR 2023]
- 链接: [arXiv链接]

## 核心创新
- [创新点1]
- [创新点2]

## 我们的适配
### 保留的部分
- [保留的核心机制]

### 替换的部分
- 原论文基座: [描述]
- 我们的基座: CLIP/wav2vec/BERT
- 适配方式: [如何适配到我们的设置]

## 与原论文的差异
| 要素 | 原论文 | 我们的实现 | 原因 |
|------|--------|-----------|------|
| 基座模型 | [描述] | CLIP/wav2vec/BERT | 统一对比 |
| 数据集 | [描述] | CMU-MOSEI/VQA/IEMOCAP | 多模态评估 |
| 任务 | [描述] | 融合+分类 | 聚焦融合架构 |

## 超参数设置
- [各超参数及取值]
```

---

## 6. 验证清单

在提交基线实现前，确认：

- [ ] 使用了 `BaselineModelWrapper` 或等效机制
- [ ] 基座特征维度与CLIP/wav2vec/BERT一致
- [ ] 投影后维度为256（与EAS一致）
- [ ] 训练配置与EAS一致（15 epochs, AdamW, lr=0.001）
- [ ] 使用了统一的 `UnifiedModalityDropout` 进行模态缺失测试
- [ ] 在5个固定种子上运行
- [ ] 实现了 `get_flops()` 方法用于计算量统计

---

## 7. 常见问题

### Q1: 如果基线方法原论文没有开源代码怎么办？

**A**: 根据论文描述自行实现核心机制。重点关注：
- 论文第3节（Method）的算法描述
- 图/表中的架构图
- 补充材料中的实现细节

### Q2: 如果基线方法原论文的基座与我们的差异很大？

**A**: 仍然强制适配到我们的基座。例如：
- 原论文使用ResNet → 我们使用CLIP
- 保持其融合机制（如注意力、门控、动态路由）
- 抛弃其特定于基座的设计

### Q3: 这样适配是否公平？

**A**: 是的，因为：
1. 所有方法（包括我们的EAS）使用相同的基座
2. 对比的是融合架构设计，而非基座选择
3. 这是NAS领域的标准做法

### Q4: 如何确保基线实现正确？

**A**:
1. 先跑通DARTS示例（`src/baselines/darts.py`）
2. 与论文报告的性能趋势对比（不要求数值相同，因为基座不同）
3. 检查消融实验是否符合论文描述（如移除某模块性能下降）

---

## 8. 总结

**核心原则**: 基线方法的对比必须基于**相同的基座模型**，否则不公平。

**适配策略**: 只复现基线的**融合架构创新**，强制使用我们的标准基座（CLIP/wav2vec/BERT）。

**交付标准**: 每个基线必须提供：
1. 核心融合模块代码
2. 使用统一包装器的完整模型
3. 实现说明文档（包含适配细节）
4. 实验结果（5 seeds × 3 datasets × 3 dropout rates）

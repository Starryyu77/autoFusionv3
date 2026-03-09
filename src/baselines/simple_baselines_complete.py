"""
简单基线模型

用于对比的基础方法:
- MeanFusion: 简单平均
- ConcatFusion: 拼接+线性
- AttentionFusion: 自注意力
- MaxFusion: 最大值池化
"""

import torch
import torch.nn as nn
from typing import Dict

from .base_complete_model import (
    CompleteBaselineModel,
    SimpleMeanFusion,
    SimpleConcatFusion,
    SimpleAttentionFusion,
    SimpleMaxFusion
)


class MeanFusionModel(CompleteBaselineModel):
    """简单平均融合基线"""

    def _create_fusion_module(self) -> nn.Module:
        return SimpleMeanFusion()


class ConcatFusionModel(CompleteBaselineModel):
    """拼接+线性融合基线"""

    def _create_fusion_module(self) -> nn.Module:
        return SimpleConcatFusion(self.hidden_dim, len(self.input_dims))


class AttentionFusionModel(CompleteBaselineModel):
    """自注意力融合基线"""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256,
                 num_classes: int = 10, is_regression: bool = False, **kwargs):
        self.num_heads = kwargs.get('num_heads', 4)
        super().__init__(input_dims, hidden_dim, num_classes, is_regression)

    def _create_fusion_module(self) -> nn.Module:
        return SimpleAttentionFusion(self.hidden_dim, self.num_heads)


class MaxFusionModel(CompleteBaselineModel):
    """最大池化融合基线"""

    def _create_fusion_module(self) -> nn.Module:
        return SimpleMaxFusion()


__all__ = [
    'MeanFusionModel',
    'ConcatFusionModel',
    'AttentionFusionModel',
    'MaxFusionModel'
]

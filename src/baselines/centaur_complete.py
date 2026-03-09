"""
Centaur 完整基线模型

鲁棒多模态融合 - 完整端到端实现
"""

import torch
import torch.nn as nn
from typing import Dict

from .base_complete_model import CompleteBaselineModel


class CentaurFusionModule(nn.Module):
    """Centaur鲁棒融合模块"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # 每个模态的可靠性估计
        self.reliability_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])

        # 模态间注意力
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        B, N, D = x.shape

        # 估计每个模态的可靠性
        reliabilities = []
        for i in range(min(N, len(self.reliability_estimators))):
            rel = self.reliability_estimators[i](x[:, i, :])  # [B, 1]
            reliabilities.append(rel)

        rel_stack = torch.stack(reliabilities, dim=1)  # [B, N, 1]

        # 模态间注意力
        attn_out, _ = self.cross_attn(x, x, x)

        # 可靠性加权
        weighted = attn_out * rel_stack

        # 拼接融合
        flat = weighted.reshape(B, N * D)
        fused = self.fusion(flat)

        return self.norm(fused)


class CentaurCompleteModel(CompleteBaselineModel):
    """Centaur完整模型"""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256,
                 num_classes: int = 10, is_regression: bool = False, **kwargs):
        # 忽略特定参数（Centaur实现中已固定）
        super().__init__(input_dims, hidden_dim, num_classes, is_regression)

    def _create_fusion_module(self) -> nn.Module:
        return CentaurFusionModule(self.hidden_dim, len(self.input_dims))


__all__ = ['CentaurCompleteModel', 'CentaurFusionModule']

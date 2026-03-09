"""
TFN 完整基线模型

张量融合网络 - 完整端到端实现
"""

import torch
import torch.nn as nn
from typing import Dict

from .base_complete_model import CompleteBaselineModel


class TFNFusionModule(nn.Module):
    """TFN张量融合模块"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # 降维投影（避免外积维度爆炸）
        self.reduce_dim = hidden_dim // 4
        self.projectors = nn.ModuleList([
            nn.Linear(hidden_dim, self.reduce_dim)
            for _ in range(num_modalities)
        ])

        # 融合层
        fusion_input_dim = self.reduce_dim * num_modalities
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
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

        # 降维投影
        reduced = []
        for i in range(min(N, len(self.projectors))):
            reduced.append(self.projectors[i](x[:, i, :]))

        # 拼接（简化版张量融合）
        concat = torch.cat(reduced, dim=-1)

        # 融合
        fused = self.fusion(concat)
        return self.norm(fused)


class TFNCompleteModel(CompleteBaselineModel):
    """TFN完整模型"""

    def _create_fusion_module(self) -> nn.Module:
        return TFNFusionModule(self.hidden_dim, len(self.input_dims))


__all__ = ['TFNCompleteModel', 'TFNFusionModule']

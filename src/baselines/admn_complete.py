"""
ADMN 完整基线模型

自适应动态多模态网络 - 完整端到端实现
"""

import torch
import torch.nn as nn
from typing import Dict

from .base_complete_model import CompleteBaselineModel


class ADMNLayer(nn.Module):
    """ADMN层级处理模块"""

    def __init__(self, hidden_dim: int, num_modalities: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # 每模态处理器
        self.processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU()
            )
            for _ in range(num_modalities)
        ])

        # 模态间注意力
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )

        # 门控控制器
        self.controller = nn.Sequential(
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        B, N, D = x.shape

        # 每模态处理
        processed = []
        for i in range(min(N, len(self.processors))):
            processed.append(self.processors[i](x[:, i, :]))

        # 堆叠
        stacked = torch.stack(processed, dim=1)  # [B, N, D]

        # 模态间注意力
        attn_out, _ = self.cross_attn(stacked, stacked, stacked)

        # 控制门
        control_state = attn_out.mean(dim=1)  # [B, D]
        gates = self.controller(control_state)  # [B, N]

        # 应用门控
        gated = attn_out * gates.unsqueeze(-1)

        # 平均池化
        return gated.mean(dim=1)


class ADMNFusionModule(nn.Module):
    """ADMN多层融合模块"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            ADMNLayer(hidden_dim, num_modalities)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        for layer in self.layers:
            x = layer(x).unsqueeze(1)  # [B, 1, D] -> [B, N, D] for next layer

        return self.norm(x.squeeze(1))


class ADMNCompleteModel(CompleteBaselineModel):
    """ADMN完整模型"""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256,
                 num_classes: int = 10, is_regression: bool = False, **kwargs):
        self.num_layers = kwargs.get('num_layers', 2)
        super().__init__(input_dims, hidden_dim, num_classes, is_regression)

    def _create_fusion_module(self) -> nn.Module:
        return ADMNFusionModule(self.hidden_dim, len(self.input_dims), self.num_layers)


__all__ = ['ADMNCompleteModel', 'ADMNLayer', 'ADMNFusionModule']

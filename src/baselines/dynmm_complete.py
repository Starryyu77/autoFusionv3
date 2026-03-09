"""
DynMM 完整基线模型

动态多模态融合 - 完整端到端实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base_complete_model import CompleteBaselineModel


class DynMMFusionModule(nn.Module):
    """DynMM动态融合模块"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3, routing_threshold: float = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.routing_threshold = routing_threshold

        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1)
        )

        # 注意力融合
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        B, N, D = x.shape

        # 计算各模态的L2范数作为路由权重
        norms = torch.norm(x, p=2, dim=-1)  # [B, N]

        # 门控权重
        x_flat = x.view(B, N * D)
        gates = self.gate_net(x_flat)  # [B, N]

        # 选择活跃模态
        active_mask = (norms > self.routing_threshold * norms.max(dim=-1, keepdim=True)[0])

        if active_mask.sum() < N * B * 0.5:  # 如果太多模态被抑制
            active_mask = torch.ones_like(active_mask)

        # 应用门控
        weighted = x * gates.unsqueeze(-1) * active_mask.unsqueeze(-1).float()

        # 注意力融合
        if N > 1:
            attn_out, _ = self.attention(weighted, weighted, weighted)
            attn_out = self.norm(attn_out)
            return attn_out.mean(dim=1)
        else:
            return weighted.mean(dim=1)


class DynMMCompleteModel(CompleteBaselineModel):
    """DynMM完整模型"""

    def __init__(self, input_dims: Dict[str, int], hidden_dim: int = 256,
                 num_classes: int = 10, is_regression: bool = False, **kwargs):
        self.routing_threshold = kwargs.get('routing_threshold', 0.2)
        super().__init__(input_dims, hidden_dim, num_classes, is_regression)

    def _create_fusion_module(self) -> nn.Module:
        return DynMMFusionModule(
            self.hidden_dim,
            len(self.input_dims),
            self.routing_threshold
        )


__all__ = ['DynMMCompleteModel', 'DynMMFusionModule']

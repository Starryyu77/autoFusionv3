"""
FDSNet 完整基线模型

特征分歧选择网络 - 完整端到端实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from .base_complete_model import CompleteBaselineModel


class FDSNetFusionModule(nn.Module):
    """FDSNet特征分歧融合模块"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

        # 分歧权重（可学习）
        self.divergence_weights = nn.Parameter(torch.eye(num_modalities))

        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def compute_divergence_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算模态间分歧分数

        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, N] 每个模态的分歧分数
        """
        B, N, D = x.shape

        # 计算每对模态的相似度
        scores = []
        for i in range(N):
            mod_i = x[:, i, :]  # [B, D]

            # 与其他所有模态的差异
            divergences = []
            for j in range(N):
                if i != j:
                    mod_j = x[:, j, :]
                    # 余弦相似度
                    sim = F.cosine_similarity(mod_i, mod_j, dim=-1)  # [B]
                    divergences.append(1 - sim)  # 差异 = 1 - 相似度

            if divergences:
                avg_divergence = torch.stack(divergences, dim=1).mean(dim=1)  # [B]
                scores.append(avg_divergence)
            else:
                scores.append(torch.zeros(B, device=x.device))

        return torch.stack(scores, dim=1)  # [B, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        B, N, D = x.shape

        # 计算分歧分数
        div_scores = self.compute_divergence_scores(x)  # [B, N]

        # 应用可学习的分歧权重
        weights = torch.softmax(self.divergence_weights[:N, :N], dim=1).sum(dim=1)  # [N]

        # 加权特征
        weighted = x * (weights.view(1, N, 1) * div_scores.unsqueeze(-1))

        # 拼接融合
        flat = weighted.view(B, N * D)
        fused = self.fusion(flat)

        return self.norm(fused)


class FDSNetCompleteModel(CompleteBaselineModel):
    """FDSNet完整模型"""

    def _create_fusion_module(self) -> nn.Module:
        return FDSNetFusionModule(self.hidden_dim, len(self.input_dims))


__all__ = ['FDSNetCompleteModel', 'FDSNetFusionModule']

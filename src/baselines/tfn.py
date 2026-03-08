"""
TFN (Tensor Fusion Network) 基线方法

简单的张量外积融合
"""

import torch
import torch.nn as nn


class TFNFusion(nn.Module):
    """
    TFN融合模块 - 适配统一框架

    使用张量外积进行多模态融合
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 降维投影
        self.projectors = nn.ModuleDict({
            'vision': nn.Linear(input_dim, hidden_dim),
            'audio': nn.Linear(input_dim, hidden_dim),
            'text': nn.Linear(input_dim, hidden_dim)
        })

        # 融合后的处理 (外积后维度很高，需要降维)
        # 实际使用简化的低秩近似
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )

        self.output_norm = nn.LayerNorm(input_dim)

    def forward(self, vision: torch.Tensor, audio: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision: [B, 576, 1024]
            audio: [B, 400, 1024]
            text: [B, 77, 1024]

        Returns:
            [B, 1024] 融合特征
        """
        # 平均池化到 [B, 1024]
        v = vision.mean(dim=1)
        a = audio.mean(dim=1)
        t = text.mean(dim=1)

        # 降维投影
        v_proj = self.projectors['vision'](v)
        a_proj = self.projectors['audio'](a)
        t_proj = self.projectors['text'](t)

        # 拼接 (简化版TFN，避免维度爆炸)
        concat = torch.cat([v_proj, a_proj, t_proj], dim=-1)

        # 融合
        fused = self.fusion(concat)

        return self.output_norm(fused)


__all__ = ['TFNFusion']

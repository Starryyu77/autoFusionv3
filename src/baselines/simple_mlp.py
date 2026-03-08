"""简单MLP融合基线"""
import torch
import torch.nn as nn

class SimpleMLPFusion(nn.Module):
    """简单Concat+MLP融合 - 作为基线对比"""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 简单的MLP融合
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, input_dim)
        )

    def forward(self, vision: torch.Tensor, audio: torch.Tensor, text: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            vision: [B, seq_len, 1024]
            audio: [B, seq_len, 1024]
            text: [B, seq_len, 1024]
        Returns:
            [B, 1024]
        """
        # 平均池化
        v = vision.mean(dim=1)  # [B, 1024]
        a = audio.mean(dim=1)
        t = text.mean(dim=1)

        # 拼接
        concat = torch.cat([v, a, t], dim=-1)  # [B, 3072]

        # MLP融合
        fused = self.fusion(concat)  # [B, 1024]

        return fused

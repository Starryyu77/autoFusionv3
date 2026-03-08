"""
DARTS Fusion Module - 统一框架适配器

将DARTS的可微分搜索适配为统一框架下的融合模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DARTSFusionModule(nn.Module):
    """
    DARTS融合模块 - 适配统一框架

    输入: 投影后的模态特征 (已统一为1024维)
    输出: [B, 1024] 融合特征
    """

    def __init__(self, input_dim: int = 1024):
        super().__init__()

        self.input_dim = input_dim

        # 候选操作集合
        self.ops = nn.ModuleList([
            nn.Identity(),  # 跳跃连接
            nn.Linear(input_dim, input_dim),  # 线性变换
            nn.Sequential(  # 带激活的线性
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, input_dim)
            ),
            nn.MultiheadAttention(  # 自注意力
                input_dim,
                num_heads=8,
                batch_first=True
            )
        ])

        # 架构参数（可学习）- 每个模态每个操作的权重
        self.alphas = nn.Parameter(torch.randn(3, 4))  # 3个模态，4个操作

        # LayerNorm
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, vision: torch.Tensor, audio: torch.Tensor = None,
                text: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        融合多个模态特征

        Args:
            vision: [B, seq_len, 1024] or [B, 1024]
            audio: [B, seq_len, 1024] or [B, 1024] or None
            text: [B, seq_len, 1024] or [B, 1024] or None

        Returns:
            [B, 1024] 融合特征
        """
        # 收集存在的模态
        features_list = []
        mod_indices = []

        if vision is not None:
            features_list.append(vision.mean(dim=1) if vision.dim() == 3 else vision)
            mod_indices.append(0)
        if audio is not None:
            features_list.append(audio.mean(dim=1) if audio.dim() == 3 else audio)
            mod_indices.append(1)
        if text is not None:
            features_list.append(text.mean(dim=1) if text.dim() == 3 else text)
            mod_indices.append(2)

        if not features_list:
            raise ValueError("At least one modality must be provided")

        # Softmax获取操作权重
        weights = F.softmax(self.alphas, dim=-1)  # [3, 4]

        # 对每个模态应用加权操作
        transformed = []
        for feat, mod_idx in zip(features_list, mod_indices):
            feat_transformed = []
            for j, op in enumerate(self.ops):
                if isinstance(op, nn.MultiheadAttention):
                    # 注意力需要特殊处理
                    feat_expanded = feat.unsqueeze(0)  # [1, B, hidden_dim]
                    out, _ = op(feat_expanded, feat_expanded, feat_expanded)
                    out = out.squeeze(0)  # [B, hidden_dim]
                else:
                    out = op(feat)
                feat_transformed.append(weights[mod_idx, j] * out)

            # 加权求和
            transformed.append(sum(feat_transformed))

        # 融合所有模态
        if len(transformed) > 1:
            fused = torch.stack(transformed, dim=1).mean(dim=1)  # [B, hidden_dim]
        else:
            fused = transformed[0]

        # LayerNorm
        fused = self.norm(fused)

        return fused


def create_darts_fusion(**kwargs):
    """创建DARTS融合模块的工厂函数"""
    return DARTSFusionModule(input_dim=1024)

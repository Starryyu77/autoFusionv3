"""
FDSNet 基线方法实现

Feature Divergence Selection Network - 特征分歧选择网络
参考论文: FDSNet - Multimodal Dynamic Fusion (Nature 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np


def compute_kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """
    计算 KL 散度

    Args:
        p: 分布 p [N, D]
        q: 分布 q [N, D]

    Returns:
        KL 散度值
    """
    # 转换为概率分布
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)

    # 添加平滑
    eps = 1e-8
    p = p + eps
    q = q + eps

    # 归一化
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)

    # KL(P || Q) = sum(P * log(P/Q))
    kl = (p * (torch.log(p) - torch.log(q))).sum(dim=-1).mean()

    return kl.item()


def compute_divergence_matrix(features: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    计算特征分歧矩阵

    Args:
        features: Dict of {modality: [B, D]}

    Returns:
        [N, N] 分歧矩阵，N 为模态数量
    """
    modalities = list(features.keys())
    n = len(modalities)

    div_matrix = torch.zeros(n, n)

    for i, mod_i in enumerate(modalities):
        for j, mod_j in enumerate(modalities):
            if i == j:
                div_matrix[i, j] = 0.0
            else:
                kl = compute_kl_divergence(features[mod_i], features[mod_j])
                div_matrix[i, j] = kl

    return div_matrix


def select_by_divergence(divergence_scores: Dict[str, float], top_k: int = 2) -> List[str]:
    """
    基于分歧度选择模态

    选择与其他模态分歧最大的模态

    Args:
        divergence_scores: {modality: divergence_score}
        top_k: 选择的数量

    Returns:
        选择的模态列表
    """
    sorted_mods = sorted(divergence_scores.items(), key=lambda x: x[1], reverse=True)
    return [mod for mod, _ in sorted_mods[:top_k]]


class DivergenceEstimator(nn.Module):
    """分歧度估计器"""

    def __init__(self, input_dim: int):
        super().__init__()
        self.estimator = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat1, feat2: [B, D]
        Returns:
            [B, 1] 分歧度
        """
        concat = torch.cat([feat1, feat2], dim=-1)
        return self.estimator(concat)


class DivergenceBasedFusion(nn.Module):
    """基于分歧的融合模块"""

    def __init__(self, input_dim: int, num_modalities: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_modalities = num_modalities

        # 为每对模态学习分歧权重
        self.divergence_weights = nn.Parameter(torch.ones(num_modalities, num_modalities))

        # 融合投影
        self.fusion_proj = nn.Linear(input_dim * num_modalities, input_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict of {modality: [B, D]}
        Returns:
            [B, D] 融合特征
        """
        modalities = list(features.keys())

        # 计算分歧权重
        weighted_features = []
        for i, mod in enumerate(modalities):
            # 该模态与其他模态的分歧加权和
            weight_sum = sum(
                self.divergence_weights[i, j]
                for j in range(len(modalities))
            )
            weighted_feat = features[mod] * (weight_sum / len(modalities))
            weighted_features.append(weighted_feat)

        # 拼接并投影
        concat = torch.cat(weighted_features, dim=-1)
        fused = self.fusion_proj(concat)

        return fused


class FDSNet(nn.Module):
    """
    FDSNet: 特征分歧选择网络

    基于模态间特征分歧进行自适应融合
    """

    def __init__(
        self,
        input_dims: Dict[str, List[int]],
        num_classes: int,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # 模态投影器
        self.projectors = nn.ModuleDict()
        for mod, dims in input_dims.items():
            input_dim = dims[-1]
            self.projectors[mod] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # 分歧估计器
        self.divergence_estimator = DivergenceEstimator(hidden_dim)

        # 分歧融合
        self.fusion = DivergenceBasedFusion(hidden_dim, len(input_dims))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: Dict of {modality: [B, seq_len, feat_dim]}

        Returns:
            [B, num_classes] logits
        """
        # 投影所有模态
        projected = {}
        for mod, x in inputs.items():
            # 平均池化序列维度
            x_pooled = x.mean(dim=1)
            projected[mod] = self.projectors[mod](x_pooled)

        # 融合
        fused = self.fusion(projected)

        # 分类
        output = self.classifier(fused)

        return output


class FDSNetFusion(nn.Module):
    """
    FDSNet融合模块 - 适配统一框架

    基于特征分歧进行自适应融合
    """

    def __init__(self, input_dim: int = 1024):
        super().__init__()

        self.input_dim = input_dim

        # 分歧权重学习
        self.divergence_weights = nn.Parameter(torch.ones(3, 3))

        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
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

        features = [v, a, t]

        # 应用分歧权重
        weighted = []
        for i, feat in enumerate(features):
            weight_sum = sum(self.divergence_weights[i, j] for j in range(3))
            weighted.append(feat * (weight_sum / 3))

        # 拼接融合
        concat = torch.cat(weighted, dim=-1)
        fused = self.fusion(concat)

        return self.output_norm(fused)


__all__ = [
    'FDSNet',
    'FDSNetFusion',
    'compute_kl_divergence',
    'compute_divergence_matrix',
    'select_by_divergence',
    'DivergenceEstimator',
    'DivergenceBasedFusion'
]

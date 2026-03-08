"""
DynMM 基线方法实现

Dynamic Multimodal Fusion - 数据相关的动态多模态融合
参考论文: DynMM - Dynamic Multimodal Fusion (CVPR 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class AttentionFusion(nn.Module):
    """注意力融合模块"""

    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of [B, D] tensors
        Returns:
            [B, D] fused feature
        """
        # 堆叠为 [B, N, D]
        stacked = torch.stack(features, dim=1)

        # 自注意力
        attended, _ = self.attention(stacked, stacked, stacked)
        attended = self.norm(attended)

        # 平均池化
        return attended.mean(dim=1)


class GatedFusion(nn.Module):
    """门控融合模块 - 支持动态模态数"""

    def __init__(self, input_dim: int, max_modalities: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.max_modalities = max_modalities
        # 动态创建门控网络，支持最多max_modalities个模态
        self.gate_net = nn.Sequential(
            nn.Linear(input_dim * max_modalities, max_modalities),
            nn.Softmax(dim=-1)
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: Dict of {modality: [B, D]}
        Returns:
            [B, D] gated fusion
        """
        num_modalities = len(features)
        feature_list = list(features.values())

        # 拼接所有特征
        concat = torch.cat(feature_list, dim=-1)  # [B, num_modalities * D]

        # 如果需要，填充到max_modalities
        if num_modalities < self.max_modalities:
            batch_size = concat.shape[0]
            padding = torch.zeros(batch_size, (self.max_modalities - num_modalities) * self.input_dim,
                                  device=concat.device, dtype=concat.dtype)
            concat = torch.cat([concat, padding], dim=-1)

        # 计算门控权重 (对实际模态有效)
        gates_full = self.gate_net(concat)  # [B, max_modalities]
        gates = gates_full[:, :num_modalities]  # [B, num_modalities]

        # 重新归一化门控权重
        gates = gates / gates.sum(dim=1, keepdim=True)

        # 加权融合
        result = sum(
            feature_list[i] * gates[:, i:i+1]
            for i in range(num_modalities)
        )

        return result


def compute_routing_weights(features: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    计算路由权重

    基于特征的 L2 范数计算各模态的重要性
    """
    weights = {}
    total = 0.0

    for mod, feat in features.items():
        # 计算特征范数
        norm = torch.norm(feat, p=2, dim=-1).mean().item()
        weights[mod] = norm
        total += norm

    # 归一化
    if total > 0:
        for mod in weights:
            weights[mod] /= total

    return weights


def select_active_modalities(
    routing_weights: Dict[str, float],
    threshold: float = 0.2
) -> List[str]:
    """
    选择活跃的模态

    选择权重超过阈值的模态
    """
    return [mod for mod, weight in routing_weights.items() if weight >= threshold]


class ModalityProjector(nn.Module):
    """模态投影器 - 将不同维度投影到共同空间"""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, input_dim]
        Returns:
            [B, output_dim]
        """
        # 平均池化序列维度
        x = x.mean(dim=1)
        return self.projector(x)


class DynMM(nn.Module):
    """
    DynMM: 动态多模态融合网络

    根据输入数据动态选择融合策略
    """

    def __init__(
        self,
        input_dims: Dict[str, List[int]],
        num_classes: int,
        hidden_dim: int = 256,
        routing_threshold: float = 0.2
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.routing_threshold = routing_threshold

        # 模态投影器
        self.projectors = nn.ModuleDict()
        for mod, dims in input_dims.items():
            input_dim = dims[-1]  # 最后一个维度是特征维度
            self.projectors[mod] = ModalityProjector(input_dim, hidden_dim)

        # 融合模块
        self.gated_fusion = GatedFusion(hidden_dim)
        self.attention_fusion = AttentionFusion(hidden_dim)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )


class DynMMFusion(nn.Module):
    """
    DynMM: 动态多模态融合网络 (适配版本 - 用于统一框架)

    输入: 已经投影到1024维的特征
    输出: [B, 1024] 融合特征

    注意: 这个版本假设输入已经是1024维，不再内部投影
    """

    def __init__(
        self,
        input_dim: int = 1024,  # 固定1024
        routing_threshold: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.routing_threshold = routing_threshold

        # 动态路由和门控融合 (在1024维空间，支持2-3个模态)
        self.gated_fusion = GatedFusion(input_dim, max_modalities=3)
        self.attention_fusion = AttentionFusion(input_dim)

        # 输出处理: 确保返回 [B, 1024]
        self.output_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        vision: torch.Tensor = None,
        audio: torch.Tensor = None,
        text: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            vision: [B, 576, 1024] or None
            audio: [B, 400, 1024] or None
            text: [B, 77, 1024] or None

        Returns:
            [B, 1024] 融合特征
        """
        # 动态构建特征字典（支持2或3个模态）
        features = {}

        if vision is not None:
            features['vision'] = vision.mean(dim=1)  # [B, 1024]
        if audio is not None:
            features['audio'] = audio.mean(dim=1)  # [B, 1024]
        if text is not None:
            features['text'] = text.mean(dim=1)  # [B, 1024]

        # 确保至少有一个模态
        if not features:
            raise ValueError("At least one modality must be provided")

        # 计算路由权重
        routing_weights = compute_routing_weights(features)

        # 选择活跃模态
        active_modalities = select_active_modalities(
            routing_weights,
            self.routing_threshold
        )

        # 如果没有活跃模态，使用所有模态
        if not active_modalities:
            active_modalities = list(features.keys())

        # 提取活跃模态特征
        active_features = {mod: features[mod] for mod in active_modalities}

        # 动态选择融合策略
        if len(active_features) == 1:
            # 只有一个模态
            fused = list(active_features.values())[0]
        elif len(active_features) == 2:
            # 两个模态，使用门控融合
            fused = self.gated_fusion(active_features)
        else:
            # 三个或更多，使用注意力融合
            feat_list = list(active_features.values())
            fused = self.attention_fusion(feat_list)

        # 归一化并返回
        return self.output_norm(fused)  # [B, 1024]


__all__ = ['DynMM', 'DynMMFusion', 'AttentionFusion', 'GatedFusion', 'compute_routing_weights', 'select_active_modalities']

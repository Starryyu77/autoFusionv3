"""
ADMN 基线方法实现

Adaptive Dynamic Multimodal Network - 自适应动态多模态网络
参考论文: ADMN - Adaptive Dynamic Network (NeurIPS 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class AdaptiveSkip(nn.Module):
    """自适应跳过模块"""

    def __init__(self, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold

    def should_skip(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        决定是否跳过

        Args:
            confidence: [B] 置信度分数

        Returns:
            [B] bool tensor, True 表示跳过
        """
        return confidence > self.threshold


class ModalityController(nn.Module):
    """模态控制器 - 决定每层的模态处理策略"""

    def __init__(self, input_dim: int, num_modalities: int):
        super().__init__()
        self.controller = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_modalities),
            nn.Sigmoid()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [B, D] 当前层状态

        Returns:
            [B, num_modalities] 每模态的决策概率
        """
        return self.controller(state)


class HierarchicalLayer(nn.Module):
    """层级处理模块"""

    def __init__(self, input_dim: int, output_dim: int, num_modalities: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities

        # 每个模态的处理
        self.modality_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU()
            )
            for _ in range(num_modalities)
        ])

        # 模态间交互
        self.interaction = nn.MultiheadAttention(
            output_dim, num_heads=4, batch_first=True
        )

        # 控制器
        self.controller = ModalityController(output_dim, num_modalities)

        # 自适应跳过
        self.skip_gate = AdaptiveSkip(threshold=0.6)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Dict of {modality: [B, D]}

        Returns:
            Dict of {modality: [B, D]}
        """
        modalities = list(features.keys())
        processed = {}

        # 处理每个模态
        for i, mod in enumerate(modalities):
            if i < len(self.modality_processors):
                processed[mod] = self.modality_processors[i](features[mod])
            else:
                processed[mod] = features[mod]

        # 计算控制器决策
        concat_state = torch.stack(list(processed.values()), dim=1).mean(dim=1)
        decisions = self.controller(concat_state)  # [B, num_modalities]

        # 应用控制决策
        output = {}
        for i, mod in enumerate(modalities):
            if i < decisions.shape[1]:
                # 根据决策调整特征
                gate = decisions[:, i:i+1]
                # 确保维度匹配
                if processed[mod].shape == features[mod].shape:
                    output[mod] = processed[mod] * gate + features[mod] * (1 - gate)
                else:
                    output[mod] = processed[mod] * gate
            else:
                output[mod] = processed[mod]

        return output


class ADMN(nn.Module):
    """
    ADMN: 自适应动态多模态网络

    层级化自适应处理，每层可以动态调整模态处理策略
    """

    def __init__(
        self,
        input_dims: Dict[str, List[int]],
        num_classes: int,
        num_layers: int = 3,
        hidden_dim: int = 256
    ):
        super().__init__()

        self.input_dims = input_dims
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # 初始投影
        self.input_projectors = nn.ModuleDict()
        for mod, dims in input_dims.items():
            input_dim = dims[-1]
            self.input_projectors[mod] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

        # 层级处理
        self.layers = nn.ModuleList([
            HierarchicalLayer(hidden_dim, hidden_dim, len(input_dims))
            for _ in range(num_layers)
        ])

        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: Dict of {modality: [B, seq_len, feat_dim]}

        Returns:
            [B, num_classes] logits
        """
        # 初始投影
        features = {}
        for mod, x in inputs.items():
            # 平均池化序列
            x_pooled = x.mean(dim=1)
            features[mod] = self.input_projectors[mod](x_pooled)

        # 层级处理
        for layer in self.layers:
            features = layer(features)

        # 拼接所有模态
        concat = torch.cat(list(features.values()), dim=-1)

        # 分类
        output = self.classifier(concat)

        return output


class ADMNFusion(nn.Module):
    """
    ADMN融合模块 - 适配统一框架

    输入: 已投影到1024维的特征
    输出: [B, 1024] 融合特征
    """

    def __init__(self, input_dim: int = 1024, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers

        # 平均池化序列维度
        self.pool = nn.AdaptiveAvgPool1d(1)

        # 层级处理 - 每模态一个处理器
        self.layer_processors = nn.ModuleList([
            nn.ModuleDict({
                'vision': nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU()
                ),
                'audio': nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU()
                ),
                'text': nn.Sequential(
                    nn.Linear(input_dim, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.ReLU()
                )
            })
            for _ in range(num_layers)
        ])

        # 模态间注意力
        self.cross_attention = nn.MultiheadAttention(
            input_dim, num_heads=8, batch_first=True
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        # 输出归一化
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

        features = {'vision': v, 'audio': a, 'text': t}

        # 层级处理
        for layer in self.layer_processors:
            new_features = {}
            for mod in ['vision', 'audio', 'text']:
                new_features[mod] = layer[mod](features[mod])
            features = new_features

        # 模态间注意力
        stacked = torch.stack(list(features.values()), dim=1)  # [B, 3, 1024]
        attended, _ = self.cross_attention(stacked, stacked, stacked)

        # 提取各模态特征
        v_feat = attended[:, 0, :]
        a_feat = attended[:, 1, :]
        t_feat = attended[:, 2, :]

        # 门控融合
        concat = torch.cat([v_feat, a_feat, t_feat], dim=-1)
        gates = self.gate(concat)  # [B, 3]

        fused = (v_feat * gates[:, 0:1] +
                 a_feat * gates[:, 1:2] +
                 t_feat * gates[:, 2:3])

        return self.output_norm(fused)


__all__ = ['ADMN', 'ADMNFusion', 'AdaptiveSkip', 'ModalityController', 'HierarchicalLayer']

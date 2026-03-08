"""
Centaur 基线方法实现

Robust Multimodal Fusion - 鲁棒多模态融合
参考论文: Centaur - Robust Multimodal Fusion (IEEE Sensors 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class DenoisingAutoencoder(nn.Module):
    """去噪自编码器"""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] 可能带噪声的输入
        Returns:
            [B, D] 去噪后的输出
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码"""
        return self.encoder(x)


def estimate_noise_level(features: torch.Tensor) -> float:
    """
    估计噪声水平

    使用特征的标准差作为噪声估计
    """
    # 计算每个维度的标准差
    std = features.std(dim=0).mean().item()
    return std


class ModalityCompletion(nn.Module):
    """模态补全模块"""

    def __init__(self, input_dim: int):
        super().__init__()

        # 从其他模态生成缺失模态
        self.completion_net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),  # 从2个模态生成第3个
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )

    def forward(
        self,
        available: Dict[str, torch.Tensor],
        missing: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        补全缺失的模态

        Args:
            available: 可用的模态特征
            missing: 缺失的模态名称

        Returns:
            包含补全特征的字典
        """
        result = dict(available)

        if not missing or len(available) < 2:
            return result

        # 简单策略：平均可用模态来生成缺失模态
        available_feats = list(available.values())

        for miss_mod in missing:
            # 使用completion_net从可用模态生成
            concat = torch.cat(available_feats[:2], dim=-1)  # 取前两个
            completed = self.completion_net(concat)
            result[miss_mod] = completed

        return result


class RobustFusion(nn.Module):
    """鲁棒融合模块"""

    def __init__(self, input_dim: int, num_modalities: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_modalities = num_modalities

        # 每个模态的可靠性估计
        self.reliability_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])

        # 融合
        self.fusion = nn.Linear(input_dim * num_modalities, input_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        鲁棒融合

        Args:
            features: Dict of {modality: [B, D]}

        Returns:
            [B, D] 融合特征
        """
        modalities = list(features.keys())

        # 估计每个模态的可靠性
        reliabilities = []
        weighted_features = []

        for i, mod in enumerate(modalities):
            if i < len(self.reliability_estimator):
                reliability = self.reliability_estimator[i](features[mod])
            else:
                reliability = torch.ones(features[mod].shape[0], 1, device=features[mod].device) * 0.5

            reliabilities.append(reliability)
            weighted_features.append(features[mod] * reliability)

        # 拼接加权特征
        concat = torch.cat(weighted_features, dim=-1)

        # 融合
        fused = self.fusion(concat)

        return fused


class Centaur(nn.Module):
    """
    Centaur: 鲁棒多模态融合网络

    特点:
    1. 去噪自编码器清理特征
    2. 模态补全处理缺失
    3. 可靠性加权的融合
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
        self.modalities = list(input_dims.keys())

        # 模态投影
        self.projectors = nn.ModuleDict()
        for mod, dims in input_dims.items():
            input_dim = dims[-1]
            self.projectors[mod] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )

        # 去噪自编码器
        self.denoisers = nn.ModuleDict({
            mod: DenoisingAutoencoder(hidden_dim)
            for mod in self.modalities
        })

        # 模态补全
        self.completion = ModalityCompletion(hidden_dim)

        # 鲁棒融合
        self.fusion = RobustFusion(hidden_dim, len(self.modalities))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        missing_modalities: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: Dict of {modality: [B, seq_len, feat_dim]}
            missing_modalities: 缺失的模态列表

        Returns:
            [B, num_classes] logits
        """
        missing_modalities = missing_modalities or []

        # 投影
        projected = {}
        for mod, x in inputs.items():
            if mod in self.projectors:
                x_pooled = x.mean(dim=1)
                projected[mod] = self.projectors[mod](x_pooled)

        # 去噪
        denoised = {}
        for mod, feat in projected.items():
            if mod in self.denoisers:
                denoised[mod] = self.denoisers[mod](feat)
            else:
                denoised[mod] = feat

        # 补全缺失模态
        if missing_modalities:
            available_modalities = [m for m in self.modalities if m not in missing_modalities]
            available = {m: denoised[m] for m in available_modalities if m in denoised}

            if len(available) >= 2:  # 需要至少2个模态来补全
                completed = self.completion(available, missing_modalities)
                denoised.update(completed)

        # 确保所有模态都存在
        fusion_input = {}
        for mod in self.modalities:
            if mod in denoised:
                fusion_input[mod] = denoised[mod]
            else:
                # 使用零填充
                fusion_input[mod] = torch.zeros(
                    list(denoised.values())[0].shape,
                    device=list(denoised.values())[0].device
                )

        # 鲁棒融合
        fused = self.fusion(fusion_input)

        # 分类
        output = self.classifier(fused)

        return output


class CentaurFusion(nn.Module):
    """
    Centaur融合模块 - 适配统一框架

    输入: 已投影到1024维的特征
    输出: [B, 1024] 融合特征

    特点: 去噪 + 可靠性估计 + 加权融合
    """

    def __init__(self, input_dim: int = 1024):
        super().__init__()

        self.input_dim = input_dim

        # 去噪自编码器
        self.denoiser = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim)
        )

        # 可靠性估计
        self.reliability_estimator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
            for _ in range(3)  # vision, audio, text
        ])

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
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

        # 去噪
        v_clean = v + self.denoiser(v)
        a_clean = a + self.denoiser(a)
        t_clean = t + self.denoiser(t)

        # 估计可靠性
        r_v = self.reliability_estimator[0](v_clean)  # [B, 1]
        r_a = self.reliability_estimator[1](a_clean)
        r_t = self.reliability_estimator[2](t_clean)

        # 可靠性加权
        v_weighted = v_clean * r_v
        a_weighted = a_clean * r_a
        t_weighted = t_clean * r_t

        # 拼接融合
        concat = torch.cat([v_weighted, a_weighted, t_weighted], dim=-1)
        fused = self.fusion(concat)

        return self.output_norm(fused)


__all__ = [
    'Centaur',
    'CentaurFusion',
    'DenoisingAutoencoder',
    'ModalityCompletion',
    'RobustFusion',
    'estimate_noise_level'
]

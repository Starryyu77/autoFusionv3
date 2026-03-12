"""
EAS (Executable Architecture Synthesis) - MOSI 版本
使用与 TFN 相同的设置进行对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ModalityEncoder(nn.Module):
    """模态编码器"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DynamicFusion(nn.Module):
    """动态融合模块 - EAS 核心"""

    def __init__(self, embed_dim: int = 64, num_modalities: int = 3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_modalities = num_modalities

        # 门控机制 - 学习模态重要性
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, num_modalities),
            nn.Softmax(dim=-1)
        )

        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, embed_dim] - N modalities
        Returns:
            [B, embed_dim]
        """
        B, N, D = features.shape

        # 门控权重
        flat = features.view(B, -1)
        gates = self.gate(flat)  # [B, N]

        # 应用门控
        gated = features * gates.unsqueeze(-1)  # [B, N, D]

        # 交叉注意力
        attn_out, _ = self.cross_attn(gated, gated, gated)

        # 拼接所有模态
        concat = attn_out.reshape(B, -1)

        # 融合
        fused = self.fusion(concat)

        return fused


class EASMOSI(nn.Module):
    """EAS 模型 - 适配 MOSI"""

    def __init__(self,
                 input_dims: Dict[str, int],
                 embed_dim: int = 64,
                 hidden_dim: int = 128,
                 task: str = 'binary'):
        super().__init__()
        self.task = task

        # 模态编码器
        self.text_encoder = ModalityEncoder(input_dims['language'], hidden_dim, embed_dim)
        self.vision_encoder = ModalityEncoder(input_dims['visual'], hidden_dim, embed_dim)
        self.audio_encoder = ModalityEncoder(input_dims['acoustic'], hidden_dim, embed_dim)

        # 动态融合
        self.fusion = DynamicFusion(embed_dim, num_modalities=3)

        # 输出层
        if task == 'binary':
            output_dim = 1
        elif task == '5class':
            output_dim = 5
        else:
            output_dim = 1

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, language: torch.Tensor, visual: torch.Tensor,
                acoustic: torch.Tensor) -> torch.Tensor:
        # 编码各模态
        text_feat = self.text_encoder(language)
        vision_feat = self.vision_encoder(visual)
        audio_feat = self.audio_encoder(acoustic)

        # 堆叠: [B, 3, embed_dim]
        features = torch.stack([text_feat, vision_feat, audio_feat], dim=1)

        # 动态融合
        fused = self.fusion(features)

        # 分类
        output = self.classifier(fused)

        if self.task == 'binary':
            return output  # raw logits
        elif self.task == 'regression':
            return torch.tanh(output) * 3  # [-3, 3]
        else:
            return output  # 5-class logits


if __name__ == '__main__':
    # 测试
    input_dims = {'language': 300, 'visual': 35, 'acoustic': 74}
    model = EASMOSI(input_dims, task='binary')

    B = 4
    language = torch.randn(B, 300)
    visual = torch.randn(B, 35)
    acoustic = torch.randn(B, 74)

    output = model(language, visual, acoustic)
    print(f"EAS Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

"""
TFN (Tensor Fusion Network) - 稳定实现版本
使用简化的融合策略避免数值不稳定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ModalityEmbedding(nn.Module):
    """单模态嵌入网络"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(0.15),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.15),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimplifiedFusion(nn.Module):
    """
    简化的张量融合 - 使用拼接 + 双线性交互
    避免外积的数值不稳定问题
    """

    def __init__(self, embed_dim: int = 32, fusion_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.fusion_dim = fusion_dim

        # 拼接后的特征维度: 3 * embed_dim + 3 (偏置)
        concat_dim = 3 * (embed_dim + 1)

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(concat_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

    def forward(self, z_l: torch.Tensor, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_l, z_v, z_a: [B, embed_dim]
        Returns:
            [B, fusion_dim]
        """
        B = z_l.size(0)

        # 添加偏置项 1
        ones = torch.ones(B, 1, device=z_l.device)
        z_l_aug = torch.cat([ones, z_l], dim=1)  # [B, embed_dim+1]
        z_v_aug = torch.cat([ones, z_v], dim=1)  # [B, embed_dim+1]
        z_a_aug = torch.cat([ones, z_a], dim=1)  # [B, embed_dim+1]

        # 拼接所有特征
        concat = torch.cat([z_l_aug, z_v_aug, z_a_aug], dim=1)  # [B, 3*(embed_dim+1)]

        # MLP 融合
        fused = self.fusion(concat)

        return fused


class SentimentInference(nn.Module):
    """情感推断子网络"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1,
                 task: str = 'binary'):
        super().__init__()
        self.task = task

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.task == 'binary':
            return x  # raw logits
        elif self.task == 'regression':
            return torch.tanh(x) * 3  # [-3, 3]
        else:  # 5-class
            return x  # raw logits


class TFNStable(nn.Module):
    """
    TFN 稳定版本 - 简化融合策略
    """

    def __init__(self,
                 input_dims: Dict[str, int],
                 embed_dim: int = 32,
                 fusion_dim: int = 128,
                 hidden_dim: int = 128,
                 task: str = 'binary'):
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim

        # 模态嵌入子网络
        self.language_embed = ModalityEmbedding(input_dims['language'], hidden_dim, embed_dim)
        self.visual_embed = ModalityEmbedding(input_dims['visual'], hidden_dim, embed_dim)
        self.acoustic_embed = ModalityEmbedding(input_dims['acoustic'], hidden_dim, embed_dim)

        # 简化的融合层
        self.fusion = SimplifiedFusion(embed_dim, fusion_dim)

        # 输出维度
        if task == 'binary':
            output_dim = 1
        elif task == '5class':
            output_dim = 5
        else:
            output_dim = 1

        # 情感推断子网络
        self.inference = SentimentInference(fusion_dim, hidden_dim, output_dim, task)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, language: torch.Tensor, visual: torch.Tensor,
                acoustic: torch.Tensor) -> torch.Tensor:
        # 模态嵌入
        z_l = self.language_embed(language)
        z_v = self.visual_embed(visual)
        z_a = self.acoustic_embed(acoustic)

        # 融合
        fusion = self.fusion(z_l, z_v, z_a)

        # 推断
        output = self.inference(fusion)

        return output


if __name__ == '__main__':
    # 测试
    input_dims = {'language': 300, 'visual': 35, 'acoustic': 74}
    model = TFNStable(input_dims, task='binary')

    B = 4
    language = torch.randn(B, 300)
    visual = torch.randn(B, 35)
    acoustic = torch.randn(B, 74)

    output = model(language, visual, acoustic)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

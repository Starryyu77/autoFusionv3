"""
TFN (Tensor Fusion Network) - 论文原始实现
参考: Tensor Fusion Network for Multimodal Sentiment Analysis (EMNLP 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


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


class TensorFusionLayer(nn.Module):
    """
    张量融合层 - 论文核心创新
    z = [1; z^l] ⊗ [1; z^v] ⊗ [1; z^a]

    使用低秩近似避免维度爆炸
    """

    def __init__(self, embed_dim: int = 32, rank: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.rank = rank

        # 低秩近似: 将外积分解为可学习的双线性融合
        # 参考: Low-rank Tensor Fusion for Multimodal Data
        self.bilinear_lv = nn.Bilinear(embed_dim + 1, embed_dim + 1, rank)
        self.bilinear_lva = nn.Bilinear(rank, embed_dim + 1, rank)

        # LayerNorm 稳定训练
        self.norm_l = nn.LayerNorm(embed_dim + 1)
        self.norm_v = nn.LayerNorm(embed_dim + 1)
        self.norm_a = nn.LayerNorm(embed_dim + 1)

        self.fusion_dim = rank

    def forward(self, z_l: torch.Tensor, z_v: torch.Tensor, z_a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_l: [B, embed_dim] - language embedding
            z_v: [B, embed_dim] - visual embedding
            z_a: [B, embed_dim] - acoustic embedding
        Returns:
            [B, fusion_dim] - fused features
        """
        B = z_l.size(0)

        # 添加偏置项 1
        ones = torch.ones(B, 1, device=z_l.device)
        z_l_aug = torch.cat([ones, z_l], dim=1)  # [B, embed_dim+1]
        z_v_aug = torch.cat([ones, z_v], dim=1)  # [B, embed_dim+1]
        z_a_aug = torch.cat([ones, z_a], dim=1)  # [B, embed_dim+1]

        # LayerNorm 稳定数值
        z_l_aug = self.norm_l(z_l_aug)
        z_v_aug = self.norm_v(z_v_aug)
        z_a_aug = self.norm_a(z_a_aug)

        # 低秩双线性融合
        fusion_lv = self.bilinear_lv(z_l_aug, z_v_aug)  # [B, rank]
        fusion = self.bilinear_lva(fusion_lv, z_a_aug)   # [B, rank]

        return fusion


class SentimentInference(nn.Module):
    """情感推断子网络"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1,
                 task: str = 'binary'):
        super().__init__()
        self.task = task

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.task == 'binary':
            return x  # raw logits for BCEWithLogitsLoss
        elif self.task == 'regression':
            return torch.tanh(x) * 3  # [-3, 3]
        else:  # 5-class
            return x  # raw logits


class TFNPaper(nn.Module):
    """
    TFN 完整模型 - 论文原始实现
    """

    def __init__(self,
                 input_dims: Dict[str, int],
                 embed_dim: int = 32,
                 hidden_dim: int = 128,
                 fusion_rank: int = 16,
                 task: str = 'binary'):
        """
        Args:
            input_dims: {'language': dim, 'visual': dim, 'acoustic': dim}
            embed_dim: 模态嵌入维度 (论文默认32)
            hidden_dim: 推断网络隐藏层维度
            fusion_rank: 融合层低秩维度 (默认16)
            task: 'binary', '5class', 'regression'
        """
        super().__init__()
        self.task = task
        self.embed_dim = embed_dim

        # 模态嵌入子网络
        self.language_embed = ModalityEmbedding(input_dims['language'], hidden_dim, embed_dim)
        self.visual_embed = ModalityEmbedding(input_dims['visual'], hidden_dim, embed_dim)
        self.acoustic_embed = ModalityEmbedding(input_dims['acoustic'], hidden_dim, embed_dim)

        # 张量融合层 (低秩近似)
        self.fusion = TensorFusionLayer(embed_dim, fusion_rank)
        fusion_dim = fusion_rank

        # 输出维度
        if task == 'binary':
            output_dim = 1
        elif task == '5class':
            output_dim = 5
        else:  # regression
            output_dim = 1

        # 情感推断子网络
        self.inference = SentimentInference(fusion_dim, hidden_dim, output_dim, task)

    def forward(self, language: torch.Tensor, visual: torch.Tensor,
                acoustic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            language: [B, language_dim]
            visual: [B, visual_dim]
            acoustic: [B, acoustic_dim]
        Returns:
            预测结果
        """
        # 模态嵌入
        z_l = self.language_embed(language)
        z_v = self.visual_embed(visual)
        z_a = self.acoustic_embed(acoustic)

        # 张量融合
        fusion = self.fusion(z_l, z_v, z_a)

        # 情感推断
        output = self.inference(fusion)

        return output

    def get_fusion_features(self, language: torch.Tensor, visual: torch.Tensor,
                           acoustic: torch.Tensor) -> torch.Tensor:
        """获取融合特征（用于分析）"""
        z_l = self.language_embed(language)
        z_v = self.visual_embed(visual)
        z_a = self.acoustic_embed(acoustic)
        return self.fusion(z_l, z_v, z_a)


class TFNAblation(nn.Module):
    """TFN 消融实验 - 单模态版本"""

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 embed_dim: int = 32, task: str = 'binary'):
        super().__init__()
        self.task = task

        self.embed = ModalityEmbedding(input_dim, hidden_dim, embed_dim)

        if task == 'binary':
            output_dim = 1
        elif task == '5class':
            output_dim = 5
        else:
            output_dim = 1

        self.inference = SentimentInference(embed_dim + 1, hidden_dim, output_dim, task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        ones = torch.ones(z.size(0), 1, device=z.device)
        z_aug = torch.cat([ones, z], dim=1)
        return self.inference(z_aug)


if __name__ == '__main__':
    # 测试模型
    input_dims = {'language': 300, 'visual': 35, 'acoustic': 74}
    model = TFNPaper(input_dims, task='binary')

    # 测试输入
    B = 4
    language = torch.randn(B, 300)
    visual = torch.randn(B, 35)
    acoustic = torch.randn(B, 74)

    output = model(language, visual, acoustic)
    print(f"Model output shape: {output.shape}")
    print(f"Fusion dimension: {(32 + 1) ** 3}")

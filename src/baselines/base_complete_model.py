"""
完整基线模型基类

方案B：让每个基线成为完整的端到端模型
包含：
1. 模态特定的投影层
2. 融合模块
3. 分类/回归头
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class CompleteBaselineModel(nn.Module, ABC):
    """
    完整基线模型基类

    所有基线方法继承此类，实现端到端的多模态融合
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 10,
        is_regression: bool = False,
        dropout: float = 0.2
    ):
        """
        初始化完整基线模型

        Args:
            input_dims: 各模态输入维度，如 {'vision': 768, 'audio': 1024, 'text': 768}
            hidden_dim: 隐藏层维度
            num_classes: 输出类别数（回归时为1）
            is_regression: 是否为回归任务
            dropout: Dropout率
        """
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.is_regression = is_regression
        self.dropout = dropout

        # 1. 模态投影层 - 每个模态独立
        self.modality_projectors = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.modality_projectors[mod] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        # 2. 融合模块 - 子类实现
        self.fusion = self._create_fusion_module()

        # 3. 分类/回归头
        output_dim = 1 if is_regression else num_classes
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    @abstractmethod
    def _create_fusion_module(self) -> nn.Module:
        """
        创建融合模块 - 子类必须实现

        Returns:
            nn.Module: 融合模块，输入 [B, N, hidden_dim]，输出 [B, hidden_dim]
        """
        pass

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
            vision: [B, seq_len, vision_dim] or None
            audio: [B, seq_len, audio_dim] or None
            text: [B, seq_len, text_dim] or None

        Returns:
            [B, num_classes] 或 [B, 1] 输出
        """
        # 收集并投影各模态
        projected = []

        if vision is not None:
            # 平均池化序列维度 [B, seq_len, dim] -> [B, dim]
            v_pooled = vision.mean(dim=1)
            v_proj = self.modality_projectors['vision'](v_pooled)
            projected.append(v_proj)

        if audio is not None:
            a_pooled = audio.mean(dim=1)
            a_proj = self.modality_projectors['audio'](a_pooled)
            projected.append(a_proj)

        if text is not None:
            t_pooled = text.mean(dim=1)
            t_proj = self.modality_projectors['text'](t_pooled)
            projected.append(t_proj)

        if not projected:
            raise ValueError("At least one modality must be provided")

        # 融合
        if len(projected) == 1:
            fused = projected[0]
        else:
            # 堆叠为 [B, N, hidden_dim]
            stacked = torch.stack(projected, dim=1)
            fused = self.fusion(stacked)

        # 分类/回归
        output = self.head(fused)

        return output

    def count_parameters(self) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in self.parameters())


class SimpleConcatFusion(nn.Module):
    """简单拼接融合 - 用于基线对比"""

    def __init__(self, hidden_dim: int, num_modalities: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        B, N, D = x.shape
        x_flat = x.view(B, N * D)
        return self.fc(x_flat)


class SimpleAttentionFusion(nn.Module):
    """简单自注意力融合"""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm(attn_out)
        # 平均池化 [B, N, D] -> [B, D]
        return attn_out.mean(dim=1)


class SimpleMeanFusion(nn.Module):
    """简单平均融合"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        return x.mean(dim=1)


class SimpleMaxFusion(nn.Module):
    """简单最大池化融合"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, hidden_dim]
        Returns:
            [B, hidden_dim]
        """
        return x.max(dim=1)[0]


__all__ = [
    'CompleteBaselineModel',
    'SimpleConcatFusion',
    'SimpleAttentionFusion',
    'SimpleMeanFusion',
    'SimpleMaxFusion'
]

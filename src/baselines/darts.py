"""
DARTS基线实现

Differentiable Architecture Search (Liu et al., ICLR 2019)
适配到多模态融合场景

原论文:
- 任务: CIFAR-10 / ImageNet 图像分类
- 基座: 搜索CNN细胞结构
- 创新: 可微分架构搜索

我们的适配:
- 任务: 多模态融合
- 基座: 固定CLIP/wav2vec/BERT，只搜索融合架构
- 保留: 可微分架构搜索机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from .base_wrapper import BaselineFusionModule, create_baseline_model


class DARTSFusion(BaselineFusionModule):
    """
    DARTS融合模块

    将DARTS的可微分搜索应用于多模态融合
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_modalities: int = 3,
        num_ops: int = 4,
        **kwargs
    ):
        """
        初始化DARTS融合模块

        Args:
            hidden_dim: 特征维度
            num_modalities: 模态数量
            num_ops: 候选操作数量
        """
        super().__init__(hidden_dim, num_modalities)
        self.num_ops = num_ops

        # 候选操作集合
        self.ops = nn.ModuleList([
            nn.Identity(),  # 跳跃连接
            nn.Linear(hidden_dim, hidden_dim),  # 线性变换
            nn.Sequential(  # 带激活的线性
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ),
            nn.MultiheadAttention(  # 自注意力
                hidden_dim,
                num_heads=4,
                batch_first=True
            )
        ])

        # 架构参数（可学习）
        # alphas[i]表示第i个模态的融合操作权重
        self.alphas = nn.Parameter(torch.randn(num_modalities, num_ops))

        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, *features) -> torch.Tensor:
        """
        融合多个模态特征

        Args:
            *features: 各模态特征 [B, hidden_dim]

        Returns:
            融合后的特征 [B, hidden_dim]
        """
        # Softmax获取操作权重
        weights = F.softmax(self.alphas, dim=-1)  # [num_modalities, num_ops]

        # 对每个模态应用加权操作
        transformed = []
        for i, feat in enumerate(features):
            feat_transformed = []
            for j, op in enumerate(self.ops):
                if isinstance(op, nn.MultiheadAttention):
                    # 注意力需要特殊处理
                    feat_expanded = feat.unsqueeze(0)  # [1, B, hidden_dim]
                    out, _ = op(feat_expanded, feat_expanded, feat_expanded)
                    out = out.squeeze(0)  # [B, hidden_dim]
                else:
                    out = op(feat)
                feat_transformed.append(weights[i, j] * out)

            # 加权求和
            transformed.append(sum(feat_transformed))

        # 拼接所有模态变换后的特征
        if len(transformed) > 1:
            fused = torch.stack(transformed, dim=1).mean(dim=1)  # [B, hidden_dim]
        else:
            fused = transformed[0]

        # LayerNorm
        fused = self.norm(fused)

        return fused

    def get_architecture_weights(self) -> torch.Tensor:
        """获取架构权重（用于分析）"""
        return F.softmax(self.alphas, dim=-1)


class DARTSNetwork(nn.Module):
    """
    DARTS完整网络（兼容旧接口）

    注意：实际使用时应通过create_darts_model()创建，
    以确保使用统一的基座投影
    """

    def __init__(
        self,
        input_dims: dict,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 10
    ):
        """
        初始化DARTS网络（兼容版本）

        Args:
            input_dims: 各模态输入维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            num_classes: 输出类别数
        """
        super().__init__()

        # 输入投影层（将各模态投影到统一维度）
        self.input_projections = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.input_projections[mod] = nn.Linear(dim, hidden_dim)

        # DARTS融合模块
        self.fusion = DARTSFusion(
            hidden_dim=hidden_dim,
            num_modalities=len(input_dims)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, **inputs) -> torch.Tensor:
        """前向传播"""
        # 投影各模态
        projected = []
        for mod, x in inputs.items():
            if mod in self.input_projections:
                x = x.mean(dim=1)  # [B, D]
                x = self.input_projections[mod](x)  # [B, hidden_dim]
                projected.append(x)

        # 融合
        if len(projected) > 1:
            fused = self.fusion(*projected)
        else:
            fused = projected[0]

        # 分类
        return self.classifier(fused)


def create_darts_model(
    input_dims: Dict[str, int] = None,
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    创建DARTS模型（标准接口）

    这是创建DARTS基线模型的推荐方式，会自动使用统一基座包装器。

    Args:
        input_dims: 输入维度（已废弃，保留兼容）
        num_classes: 类别数
        **kwargs: 其他参数传递给DARTSFusion

    Returns:
        包装后的完整模型
    """
    # 创建DARTS融合模块
    fusion = DARTSFusion(
        hidden_dim=256,
        num_modalities=3,
        **kwargs
    )

    # 使用统一包装器
    return create_baseline_model(fusion, num_classes=num_classes)


# 兼容旧接口的别名
def create_model(input_dims: dict, num_classes: int = 10, **kwargs):
    """兼容旧接口"""
    return create_darts_model(input_dims, num_classes, **kwargs)


if __name__ == "__main__":
    print("Testing DARTS baseline...")
    print("=" * 60)

    # 方式1: 使用推荐接口（自动使用统一基座包装器）
    print("\n1. Testing with unified wrapper (recommended):")
    model = create_darts_model(num_classes=10)

    batch_size = 2
    x = {
        'vision': torch.randn(batch_size, 50, 1024),  # CLIP特征
        'audio': torch.randn(batch_size, 100, 1024),  # wav2vec特征
        'text': torch.randn(batch_size, 20, 768)      # BERT特征
    }

    output = model(**x)
    print(f"   Input shapes: {[v.shape for v in x.values()]}")
    print(f"   Output shape: {output.shape}")
    print(f"   Total params: {sum(p.numel() for p in model.parameters()):,}")

    # 查看架构权重
    arch_weights = model.fusion.get_architecture_weights()
    print(f"   Architecture weights:\n{arch_weights}")

    # 方式2: 使用兼容接口
    print("\n2. Testing with compatible interface:")
    model2 = DARTSNetwork(
        input_dims={'vision': 1024, 'audio': 512, 'text': 768},
        num_classes=10
    )
    output2 = model2(**x)
    print(f"   Output shape: {output2.shape}")

    print("\n✅ DARTS test passed!")

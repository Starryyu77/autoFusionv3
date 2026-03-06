"""
DARTS基线实现

Differentiable Architecture Search (Liu et al., ICLR 2019)

简化版DARTS实现，用于对比实验
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class DARTSNetwork(nn.Module):
    """
    DARTS网络

    使用可微分架构搜索找到的多模态融合网络
    """

    def __init__(
        self,
        input_dims: dict,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_classes: int = 10
    ):
        """
        初始化DARTS网络

        Args:
            input_dims: 各模态输入维度 {'vision': 1024, 'audio': 512, ...}
            hidden_dim: 隐藏层维度
            num_layers: 层数
            num_classes: 输出类别数
        """
        super().__init__()

        self.input_dims = input_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 输入投影层
        self.input_projections = nn.ModuleDict()
        for mod, dim in input_dims.items():
            self.input_projections[mod] = nn.Linear(dim, hidden_dim)

        # DARTS单元 (简化版: 使用混合操作)
        self.cells = nn.ModuleList([
            DARTSCell(hidden_dim, num_ops=4)
            for _ in range(num_layers)
        ])

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * len(input_dims), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, **inputs) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: {'vision': [B, T, D], 'audio': [B, T, D], ...}

        Returns:
            输出 [B, num_classes]
        """
        # 投影各模态
        projected = []
        for mod, x in inputs.items():
            if mod in self.input_projections:
                # 平均池化到固定长度
                x = x.mean(dim=1)  # [B, D]
                x = self.input_projections[mod](x)  # [B, hidden_dim]
                projected.append(x)

        # 堆叠各模态特征
        features = torch.stack(projected, dim=1)  # [B, num_mod, hidden_dim]

        # 通过DARTS单元
        for cell in self.cells:
            features = cell(features)

        # 全局平均池化
        features = features.mean(dim=1)  # [B, hidden_dim]

        # 分类
        output = self.classifier(features)

        return output


class DARTSCell(nn.Module):
    """
    DARTS单元

    混合多个操作的单元
    """

    def __init__(self, hidden_dim: int, num_ops: int = 4):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_ops = num_ops

        # 可学习架构参数 (alphas)
        self.alphas = nn.Parameter(torch.randn(num_ops))

        # 操作集合
        self.ops = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),  # 线性
            nn.Sequential(  # 带激活的线性
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ),
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True),  # 注意力
            nn.Identity()  # 跳跃连接
        ])

        # LayerNorm
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_mod, hidden_dim]

        Returns:
            [B, num_mod, hidden_dim]
        """
        # Softmax获取操作权重
        weights = F.softmax(self.alphas, dim=0)

        # 应用加权操作
        outputs = []
        for i, op in enumerate(self.ops):
            if isinstance(op, nn.MultiheadAttention):
                # 注意力需要特殊处理
                out, _ = op(x, x, x)
            else:
                out = op(x)
            outputs.append(weights[i] * out)

        # 加权求和
        output = sum(outputs)

        # 残差连接 + LayerNorm
        output = self.norm(output + x)

        return output


def create_darts_model(input_dims: dict, num_classes: int = 10) -> DARTSNetwork:
    """
    创建DARTS模型

    Args:
        input_dims: 输入维度
        num_classes: 类别数

    Returns:
        DARTS网络
    """
    return DARTSNetwork(
        input_dims=input_dims,
        hidden_dim=256,
        num_layers=3,
        num_classes=num_classes
    )


if __name__ == "__main__":
    print("Testing DARTS baseline...")

    # 创建模型
    input_dims = {'vision': 1024, 'audio': 512, 'text': 768}
    model = create_darts_model(input_dims, num_classes=10)

    # 测试前向传播
    batch_size = 2
    x = {
        'vision': torch.randn(batch_size, 50, 1024),
        'audio': torch.randn(batch_size, 100, 512),
        'text': torch.randn(batch_size, 20, 768)
    }

    output = model(**x)
    print(f"Input shapes: {[v.shape for v in x.values()]}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

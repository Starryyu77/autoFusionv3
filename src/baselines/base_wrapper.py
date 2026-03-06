"""
基线方法统一包装器

确保所有基线方法使用相同的基座模型，实现公平对比
"""

import torch
import torch.nn as nn
from typing import Dict


class BaselineModelWrapper(nn.Module):
    """
    基线方法统一包装器

    强制所有基线方法使用相同的基座投影，确保公平对比。

    架构:
    输入特征 (来自冻结基座 CLIP/wav2vec/BERT)
        ↓
    投影层 (backbone_dim → hidden_dim)
        ↓
    基线融合模块 (各方法的核心创新)
        ↓
    分类器
        ↓
    输出预测

    Example:
        >>> # 创建DARTS融合模块
        >>> fusion = DARTSFusion(hidden_dim=256, num_ops=4)
        >>>
        >>> # 包装成完整模型
        >>> model = BaselineModelWrapper(
        ...     fusion_module=fusion,
        ...     input_dims={'vision': 1024, 'audio': 512, 'text': 768},
        ...     num_classes=10,
        ...     backbone_dims={'vision': 1024, 'audio': 1024, 'text': 768}
        ... )
        >>>
        >>> # 前向传播
        >>> output = model(vision=vision_feat, audio=audio_feat, text=text_feat)
    """

    def __init__(
        self,
        fusion_module: nn.Module,
        input_dims: Dict[str, int],
        num_classes: int,
        backbone_dims: Dict[str, int] = None,
        hidden_dim: int = 256
    ):
        """
        初始化包装器

        Args:
            fusion_module: 基线方法的核心融合模块
            input_dims: 各模态输入维度（已投影后的维度）
            num_classes: 输出类别数
            backbone_dims: 基座模型原始输出维度
                          默认: {'vision': 1024, 'audio': 1024, 'text': 768}
            hidden_dim: 投影后的统一维度
        """
        super().__init__()

        # 默认基座维度
        if backbone_dims is None:
            backbone_dims = {
                'vision': 1024,  # CLIP-ViT-L/14
                'audio': 1024,   # wav2vec 2.0 Large
                'text': 768      # BERT-Base
            }

        self.backbone_dims = backbone_dims
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 基座特征投影层：将不同维度的基座特征投影到统一维度
        self.projections = nn.ModuleDict()
        for mod, dim in backbone_dims.items():
            self.projections[mod] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )

        # 基线方法的核心融合模块（各方法的核心创新）
        self.fusion = fusion_module

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        # 统计参数量
        self.total_params = sum(p.numel() for p in self.parameters())
        self.fusion_params = sum(p.numel() for p in fusion_module.parameters())

    def forward(self, **inputs) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: {'vision': [B, T, 1024], 'audio': [B, T, 1024], 'text': [B, T, 768], ...}
                   来自冻结基座模型的特征

        Returns:
            [B, num_classes]
        """
        # 投影各模态特征
        projected = {}
        for mod, x in inputs.items():
            if mod in self.projections:
                # x: [B, T, backbone_dim]
                x = x.mean(dim=1)  # 时间维度平均池化 [B, backbone_dim]
                x = self.projections[mod](x)  # 投影 [B, hidden_dim]
                projected[mod] = x

        if not projected:
            raise ValueError("No valid modalities in input")

        # 提取特征列表
        features = list(projected.values())

        # 融合
        if len(features) == 1:
            # 单模态，无需融合
            fused = features[0]
        else:
            # 多模态融合（使用基线方法的核心融合模块）
            fused = self.fusion(*features)

        # 分类
        return self.classifier(fused)

    def get_flops(self) -> int:
        """
        估算FLOPs

        Returns:
            估算的FLOPs数量
        """
        # 简化估算：参数量 × 前向传播系数
        # 实际FLOPs计算需要使用工具如thop
        return self.total_params * 2  # 粗略估算

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'total_params': self.total_params,
            'fusion_params': self.fusion_params,
            'backbone_dims': self.backbone_dims,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes
        }


def create_baseline_model(
    fusion_module: nn.Module,
    num_classes: int = 10,
    **kwargs
) -> BaselineModelWrapper:
    """
    创建基线模型的便捷函数

    Args:
        fusion_module: 基线方法的核心融合模块
        num_classes: 类别数
        **kwargs: 其他参数

    Returns:
        包装后的完整模型
    """
    # 标准基座维度（所有基线方法必须一致）
    backbone_dims = {
        'vision': 1024,  # CLIP-ViT-L/14
        'audio': 1024,   # wav2vec 2.0 Large
        'text': 768      # BERT-Base
    }

    return BaselineModelWrapper(
        fusion_module=fusion_module,
        input_dims={'vision': 256, 'audio': 256, 'text': 256},
        num_classes=num_classes,
        backbone_dims=backbone_dims,
        hidden_dim=256
    )


# ============ 基线融合模块基类 ============

class BaselineFusionModule(nn.Module):
    """
    基线融合模块基类

    所有基线方法的核心融合模块应继承此类
    """

    def __init__(self, hidden_dim: int = 256, num_modalities: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities

    def forward(self, *features) -> torch.Tensor:
        """
        融合多个模态特征

        Args:
            *features: 各模态特征 [B, hidden_dim]

        Returns:
            融合后的特征 [B, hidden_dim]
        """
        raise NotImplementedError


if __name__ == "__main__":
    # 测试示例
    print("Testing BaselineModelWrapper...")

    # 创建一个简单的融合模块（示例）
    class SimpleConcatFusion(BaselineFusionModule):
        """简单的拼接融合（示例）"""

        def __init__(self, hidden_dim: int = 256, num_modalities: int = 3):
            super().__init__(hidden_dim, num_modalities)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * num_modalities, hidden_dim),
                nn.ReLU()
            )

        def forward(self, *features):
            concat = torch.cat(features, dim=-1)
            return self.fusion(concat)

    # 创建模型
    fusion = SimpleConcatFusion(hidden_dim=256, num_modalities=3)
    model = create_baseline_model(fusion, num_classes=10)

    # 测试输入
    batch_size = 2
    vision = torch.randn(batch_size, 50, 1024)  # [B, T, CLIP_dim]
    audio = torch.randn(batch_size, 100, 1024)  # [B, T, wav2vec_dim]
    text = torch.randn(batch_size, 20, 768)     # [B, T, BERT_dim]

    # 前向传播
    output = model(vision=vision, audio=audio, text=text)

    print(f"Input shapes:")
    print(f"  Vision: {vision.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Text: {text.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    print("✅ Test passed!")

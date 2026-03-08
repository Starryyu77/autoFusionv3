"""
统一特征投影层

所有方法共享的投影层，将不同模态投影到统一维度(1024)
"""

import torch
import torch.nn as nn


class UnifiedFeatureProjection(nn.Module):
    """
    统一特征投影层 - 所有方法共享

    将不同模态的特征投影到相同的1024维空间。
    支持自适应输入维度（根据第一次前向传播自动检测）。

    注意: 这个投影层是所有方法共享的，不属于任何特定方法
    """

    def __init__(self, output_dim: int = 1024, dropout: float = 0.1):
        super().__init__()

        self.output_dim = output_dim
        self.dropout = dropout

        # 延迟初始化投影层（根据实际输入维度）
        self.vision_proj = None
        self.audio_proj = None
        self.text_proj = None

        # 记录输入维度
        self._vision_input_dim = None
        self._audio_input_dim = None
        self._text_input_dim = None

    def _create_proj(self, input_dim: int) -> nn.Module:
        """创建投影层"""
        if input_dim == self.output_dim:
            return nn.Identity()
        return nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )

    def forward(self, vision: torch.Tensor = None, audio: torch.Tensor = None, text: torch.Tensor = None) -> dict:
        """
        前向传播

        Args:
            vision: [B, seq_len, feat_dim] 或 None
            audio: [B, seq_len, feat_dim] 或 None
            text: [B, seq_len, feat_dim] 或 None

        Returns:
            dict: {
                'vision': [B, seq_len, 1024] (如果输入不为None),
                'audio': [B, seq_len, 1024] (如果输入不为None),
                'text': [B, seq_len, 1024] (如果输入不为None)
            }
        """
        result = {}

        # 延迟初始化并处理vision
        if vision is not None:
            if self.vision_proj is None:
                self._vision_input_dim = vision.shape[-1]
                self.vision_proj = self._create_proj(self._vision_input_dim)
                self.vision_proj = self.vision_proj.to(vision.device)
            result['vision'] = self.vision_proj(vision)

        # 延迟初始化并处理audio
        if audio is not None:
            if self.audio_proj is None:
                self._audio_input_dim = audio.shape[-1]
                self.audio_proj = self._create_proj(self._audio_input_dim)
                self.audio_proj = self.audio_proj.to(audio.device)
            result['audio'] = self.audio_proj(audio)

        # 延迟初始化并处理text
        if text is not None:
            if self.text_proj is None:
                self._text_input_dim = text.shape[-1]
                self.text_proj = self._create_proj(self._text_input_dim)
                self.text_proj = self.text_proj.to(text.device)
            result['text'] = self.text_proj(text)

        return result


class UnifiedClassifier(nn.Module):
    """
    统一分类/回归头 - 所有方法共享

    输入: [B, 1024] (融合后的特征)
    输出: [B, num_classes] 或 [B, 1] (回归)

    注意: 这个分类头在框架层面统一，不属于任何特定方法
    """

    def __init__(self, input_dim: int = 1024, num_classes: int = 10, dropout: float = 0.2, is_regression: bool = False):
        super().__init__()

        self.is_regression = is_regression

        if is_regression:
            # 回归任务使用更简单的结构，避免ReLU导致的信息损失
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.LayerNorm(input_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, 1)
            )
        else:
            # 分类任务保持原结构
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim // 2, num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1024] 融合特征

        Returns:
            [B, num_classes] 或 [B, 1] 输出
        """
        return self.classifier(x)


class UnifiedModel(nn.Module):
    """
    统一模型框架

    组合了:
    1. 投影层 (共享)
    2. 融合模块 (各方法不同)
    3. 分类头 (共享)

    使用示例:
        >>> fusion_module = DynMMFusion()  # 或其他方法
        >>> model = UnifiedModel(fusion_module)
        >>> output = model(vision, audio, text)
    """

    def __init__(self, fusion_module: nn.Module, num_classes: int = 10, is_regression: bool = False):
        super().__init__()

        self.is_regression = is_regression

        # 共享投影层
        self.projection = UnifiedFeatureProjection(output_dim=1024)

        # 方法特定的融合模块 (必须输出 [B, 1024])
        self.fusion = fusion_module

        # 共享分类头
        self.classifier = UnifiedClassifier(input_dim=1024, num_classes=num_classes, is_regression=is_regression)

    def forward(self, vision: torch.Tensor = None, audio: torch.Tensor = None, text: torch.Tensor = None,
                modality_mask: dict = None, **kwargs) -> torch.Tensor:
        """
        前向传播

        Args:
            vision: [B, 576, 768] 或 None (VQA数据可能没有vision)
            audio: [B, 400, 1024] 或 None
            text: [B, 77, 768] 或 None
            modality_mask: 可选，标识哪些模态是缺失的
                {'vision': [B], 'audio': [B], 'text': [B]} (1=存在, 0=缺失)

        Returns:
            [B, num_classes] 分类结果
        """
        # 支持通过kwargs传递模态数据
        if vision is None and 'v' in kwargs:
            vision = kwargs['v']
        if audio is None and 'a' in kwargs:
            audio = kwargs['a']
        if text is None and 't' in kwargs:
            text = kwargs['t']

        # 1. 投影到统一维度 (1024) - 仅对存在的模态
        features = {}
        if vision is not None:
            # 延迟初始化投影层
            if self.projection.vision_proj is None:
                self.projection._vision_input_dim = vision.shape[-1]
                self.projection.vision_proj = self.projection._create_proj(vision.shape[-1])
                self.projection.vision_proj = self.projection.vision_proj.to(vision.device)
            features['vision'] = self.projection.vision_proj(vision)

        if audio is not None:
            if self.projection.audio_proj is None:
                self.projection._audio_input_dim = audio.shape[-1]
                self.projection.audio_proj = self.projection._create_proj(audio.shape[-1])
                self.projection.audio_proj = self.projection.audio_proj.to(audio.device)
            features['audio'] = self.projection.audio_proj(audio)

        if text is not None:
            if self.projection.text_proj is None:
                self.projection._text_input_dim = text.shape[-1]
                self.projection.text_proj = self.projection._create_proj(text.shape[-1])
                self.projection.text_proj = self.projection.text_proj.to(text.device)
            features['text'] = self.projection.text_proj(text)

        # 2. 应用模态缺失mask (如果提供)
        if modality_mask is not None:
            for mod, mask in modality_mask.items():
                if mod in features:
                    # mask: [B] -> [B, 1, 1]
                    mask = mask.view(-1, 1, 1)
                    features[mod] = features[mod] * mask

        # 3. 融合 (各方法不同，必须输出 [B, 1024])
        fused = self.fusion(**features)

        # 4. 分类
        output = self.classifier(fused)

        return output


# 辅助函数: 创建测试输入
def create_test_inputs(batch_size: int = 2) -> tuple:
    """创建测试输入"""
    vision = torch.randn(batch_size, 576, 768)  # CLIP特征
    audio = torch.randn(batch_size, 400, 1024)  # wav2vec特征
    text = torch.randn(batch_size, 77, 768)     # BERT特征
    return vision, audio, text


if __name__ == "__main__":
    # 测试
    print("测试统一投影层...")

    # 1. 测试投影层
    projection = UnifiedFeatureProjection()
    v, a, t = create_test_inputs(batch_size=2)

    features = projection(v, a, t)
    print(f"  视觉: {v.shape} -> {features['vision'].shape}")
    print(f"  音频: {a.shape} -> {features['audio'].shape}")
    print(f"  文本: {t.shape} -> {features['text'].shape}")

    # 验证维度
    assert features['vision'].shape == (2, 576, 1024)
    assert features['audio'].shape == (2, 400, 1024)
    assert features['text'].shape == (2, 77, 1024)
    print("  ✅ 投影层测试通过")

    # 2. 测试分类头
    print("\n测试统一分类头...")
    classifier = UnifiedClassifier(input_dim=1024, num_classes=10)
    test_fused = torch.randn(2, 1024)
    output = classifier(test_fused)
    print(f"  输入: {test_fused.shape} -> 输出: {output.shape}")
    assert output.shape == (2, 10)
    print("  ✅ 分类头测试通过")

    print("\n所有测试通过!")

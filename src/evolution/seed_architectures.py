"""
种子架构库 - 提供确定可工作的基础架构

这些架构作为EAS搜索的起点，确保内循环能够成功编译。
进化过程基于这些种子进行变异和改进。
"""

# 种子1: 简单的注意力融合
SEED_ATTENTION_FUSION = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusion(nn.Module):
    """
    基于注意力机制的多模态融合
    输入: vision [B,576,1024], audio [B,400,1024], text [B,77,1024]
    输出: [B, 1024]
    """

    def __init__(self, dim=1024, num_heads=8, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 时序池化后的投影
        self.vision_pool = nn.Linear(576, 1)
        self.audio_pool = nn.Linear(400, 1)
        self.text_pool = nn.Linear(77, 1)

        # 跨模态注意力
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, vision, audio, text):
        # 时序池化: [B, seq, 1024] -> [B, 1, 1024]
        v = self.vision_pool(vision.transpose(1, 2)).transpose(1, 2)
        a = self.audio_pool(audio.transpose(1, 2)).transpose(1, 2)
        t = self.text_pool(text.transpose(1, 2)).transpose(1, 2)

        # 拼接: [B, 3, 1024]
        fused = torch.cat([v, a, t], dim=1)

        # 自注意力
        attended, _ = self.cross_attn(fused, fused, fused)

        # 平均池化: [B, 3, 1024] -> [B, 1024]
        output = attended.mean(dim=1)

        # 前馈
        return self.ffn(torch.cat([v.squeeze(1), a.squeeze(1), t.squeeze(1)], dim=-1))
'''

# 种子2: 门控融合
SEED_GATED_FUSION = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusion(nn.Module):
    """
    基于门控机制的多模态融合
    """

    def __init__(self, dim=1024, dropout=0.1):
        super().__init__()
        self.dim = dim

        # 时序池化
        self.vision_pool = nn.AdaptiveAvgPool1d(1)
        self.audio_pool = nn.AdaptiveAvgPool1d(1)
        self.text_pool = nn.AdaptiveAvgPool1d(1)

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, vision, audio, text):
        # 池化: [B, seq, 1024] -> [B, 1024]
        v = self.vision_pool(vision.transpose(1, 2)).squeeze(-1)
        a = self.audio_pool(audio.transpose(1, 2)).squeeze(-1)
        t = self.text_pool(text.transpose(1, 2)).squeeze(-1)

        # 拼接
        concat = torch.cat([v, a, t], dim=-1)

        # 门控
        gate = self.gate(concat)

        # 融合
        fused = self.fusion(concat)

        # 应用门控
        return gate * fused
'''

# 种子3: 简单但有效的Concat+MLP
SEED_SIMPLE_MLP = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusion(nn.Module):
    """
    简单有效的MLP融合
    """

    def __init__(self, dim=1024, dropout=0.1):
        super().__init__()

        # 时序池化
        self.pool = nn.AdaptiveAvgPool1d(1)

        # MLP融合
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, vision, audio, text):
        # 池化
        v = self.pool(vision.transpose(1, 2)).squeeze(-1)
        a = self.pool(audio.transpose(1, 2)).squeeze(-1)
        t = self.pool(text.transpose(1, 2)).squeeze(-1)

        # 拼接并融合
        fused = torch.cat([v, a, t], dim=-1)
        return self.fusion(fused)
'''

# 所有种子
ALL_SEEDS = {
    'attention': SEED_ATTENTION_FUSION,
    'gated': SEED_GATED_FUSION,
    'simple_mlp': SEED_SIMPLE_MLP,
}


def get_seed_architecture(seed_type='simple_mlp'):
    """
    获取种子架构

    Args:
        seed_type: 'attention', 'gated', 或 'simple_mlp'

    Returns:
        种子代码字符串
    """
    return ALL_SEEDS.get(seed_type, SEED_SIMPLE_MLP)


def get_all_seeds():
    """获取所有种子架构"""
    return ALL_SEEDS


def test_seed_architecture(code, device='cpu'):
    """
    测试种子架构是否能正确编译和运行

    Args:
        code: 种子代码
        device: 运行设备

    Returns:
        (success, error_message)
    """
    try:
        namespace = {'torch': torch, 'nn': nn, 'F': F}
        exec(code, namespace)

        # 找到模型类
        model_class = None
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and name != 'Module':
                model_class = obj
                break

        if model_class is None:
            return False, "No model class found"

        # 实例化
        model = model_class()
        model = model.to(device)
        model.eval()

        # 测试前向传播
        with torch.no_grad():
            vision = torch.randn(2, 576, 1024, device=device)
            audio = torch.randn(2, 400, 1024, device=device)
            text = torch.randn(2, 77, 1024, device=device)

            output = model(vision, audio, text)

            assert output.shape == (2, 1024), f"Output shape mismatch: {output.shape}"

        return True, ""

    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    # 测试所有种子
    print("Testing seed architectures...\n")

    for name, code in ALL_SEEDS.items():
        print(f"Testing {name}...")
        success, error = test_seed_architecture(code)
        if success:
            print(f"  ✅ {name} passed")
        else:
            print(f"  ❌ {name} failed: {error}")

    print("\nAll tests completed!")

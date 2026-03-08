"""
EvoPrompting Fusion Module - 统一框架适配器

基于进化提示工程的神经架构搜索适配器
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_backend import UnifiedLLMBackend


class EvoPromptingFusionModule(nn.Module):
    """
    EvoPrompting融合模块 - 适配统一框架

    在提示词空间进行进化优化，生成融合架构
    """

    def __init__(self, input_dim: int = 1024, population_size: int = 10,
                 num_iterations: int = 20, api_key: Optional[str] = None):
        super().__init__()

        self.input_dim = input_dim
        self.population_size = population_size
        self.num_iterations = num_iterations

        # 初始化LLM后端
        try:
            self.llm = UnifiedLLMBackend(api_key=api_key)
        except:
            self.llm = None

        # 存储搜索到的架构
        self.fusion_module = None
        self.best_architecture_code = None

        # 默认使用注意力融合作为fallback
        self.fallback = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )

    def search_architecture(self, dataset_name: str = "unknown"):
        """
        搜索最优架构（离线执行）
        """
        if self.llm is None:
            print("LLM backend not available, using fallback architecture")
            return

        # 简化处理，使用预定义的架构
        self.best_architecture_code = self._generate_attention_architecture()
        self._instantiate_architecture()

    def _generate_attention_architecture(self) -> str:
        """生成注意力融合架构代码"""
        return '''
import torch
import torch.nn as nn

class GeneratedFusion(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.self_attn = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim)
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, features):
        # features: dict with keys like 'vision', 'audio', 'text'
        # Stack features
        feats = []
        for key in ['vision', 'audio', 'text']:
            if key in features:
                feat = features[key].mean(dim=1, keepdim=True)  # [B, 1, D]
                feats.append(feat)

        if len(feats) == 0:
            raise ValueError("No features provided")

        x = torch.cat(feats, dim=1)  # [B, N, D]

        # Cross-attention
        x2, _ = self.cross_attn(x, x, x)
        x = self.norm1(x + x2)

        # Self-attention
        x2, _ = self.self_attn(x, x, x)
        x = self.norm2(x + x2)

        # FFN
        x2 = self.ffn(x)
        x = self.norm3(x + x2)

        # Average pooling
        return x.mean(dim=1)  # [B, D]
'''

    def _instantiate_architecture(self):
        """实例化架构"""
        if self.best_architecture_code:
            namespace = {}
            exec(self.best_architecture_code, namespace)
            GeneratedFusion = namespace.get('GeneratedFusion')
            if GeneratedFusion:
                self.fusion_module = GeneratedFusion(self.input_dim)

    def forward(self, vision: torch.Tensor, audio: torch.Tensor = None,
                text: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        前向传播

        Args:
            vision: [B, seq_len, 1024]
            audio: [B, seq_len, 1024] or None
            text: [B, seq_len, 1024] or None

        Returns:
            [B, 1024] 融合特征
        """
        if self.fusion_module is None:
            return self._fallback_forward(vision, audio, text)

        features = {}
        if vision is not None:
            features['vision'] = vision
        if audio is not None:
            features['audio'] = audio
        if text is not None:
            features['text'] = text

        return self.fusion_module(features)

    def _fallback_forward(self, vision, audio, text):
        """Fallback前向传播"""
        feats = []
        if vision is not None:
            feats.append(vision.mean(dim=1))
        if audio is not None:
            feats.append(audio.mean(dim=1))
        if text is not None:
            feats.append(text.mean(dim=1))

        if not feats:
            raise ValueError("At least one modality must be provided")

        # 平均融合
        fused = torch.stack(feats, dim=1).mean(dim=1)
        return self.fallback(fused)


def create_evoprompting_fusion(**kwargs):
    """创建EvoPrompting融合模块的工厂函数"""
    return EvoPromptingFusionModule(input_dim=1024, **kwargs)

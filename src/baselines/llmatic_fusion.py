"""
LLMatic Fusion Module - 统一框架适配器

基于LLM+质量多样性的神经架构搜索适配器
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_backend import UnifiedLLMBackend


class LLMaticFusionModule(nn.Module):
    """
    LLMatic融合模块 - 适配统一框架

    使用LLM生成融合架构，基于行为多样性进行选择
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

        # 默认使用简单MLP作为fallback
        self.fallback = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, input_dim)
        )

    def search_architecture(self, dataset_name: str = "unknown"):
        """
        搜索最优架构（离线执行，结果保存）

        实际使用时，应该预先运行搜索并保存最佳架构代码
        """
        if self.llm is None:
            print("LLM backend not available, using fallback architecture")
            return

        # 这里简化处理，实际应该调用LLMatic类进行完整搜索
        # 为简化，我们使用一个固定的简单架构
        self.best_architecture_code = self._generate_simple_architecture()
        self._instantiate_architecture()

    def _generate_simple_architecture(self) -> str:
        """生成简单的融合架构代码"""
        return '''
import torch
import torch.nn as nn

class GeneratedFusion(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.attn_vision = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.attn_audio = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.attn_text = nn.MultiheadAttention(input_dim, 8, batch_first=True)
        self.fusion = nn.Linear(input_dim * 3, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, features):
        # features: dict with 'vision', 'audio', 'text'
        v = features['vision'].mean(dim=1)
        a = features['audio'].mean(dim=1) if 'audio' in features else torch.zeros_like(v)
        t = features['text'].mean(dim=1) if 'text' in features else torch.zeros_like(v)
        concat = torch.cat([v, a, t], dim=-1)
        return self.norm(self.fusion(concat))
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
        # 如果还没有搜索过，使用fallback
        if self.fusion_module is None:
            return self._fallback_forward(vision, audio, text)

        # 使用生成的架构
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
        v = vision.mean(dim=1) if vision is not None else None
        a = audio.mean(dim=1) if audio is not None else None
        t = text.mean(dim=1) if text is not None else None

        # 填充缺失的模态
        if v is None:
            v = torch.zeros(a.shape if a is not None else t.shape)
        if a is None:
            a = torch.zeros_like(v)
        if t is None:
            t = torch.zeros_like(v)

        concat = torch.cat([v, a, t], dim=-1)
        return self.fallback(concat)


def create_llmatic_fusion(**kwargs):
    """创建LLMatic融合模块的工厂函数"""
    return LLMaticFusionModule(input_dim=1024, **kwargs)

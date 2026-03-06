"""
数据模块

包含数据加载器和模态缺失模拟器
"""

from .modality_dropout import UnifiedModalityDropout

__all__ = [
    'UnifiedModalityDropout',
]

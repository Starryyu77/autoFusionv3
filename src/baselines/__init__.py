"""
基线方法模块

包含所有对比方法:
- DARTS: Differentiable Architecture Search
- LLMatic: LLM-based NAS
- EvoPrompting: Evolutionary Prompting
- DynMM: Dynamic Multimodal Fusion
- FDSNet: Feature Disagreement Scoring Network
- ADMN: Adaptive Multimodal Network
- Centaur: Robust Multimodal Fusion
"""

from .darts import DARTSNetwork, create_darts_model

__all__ = [
    'DARTSNetwork',
    'create_darts_model',
]

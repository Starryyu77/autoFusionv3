"""
Outer Loop 模块 - 进化搜索

包含:
- EASEvolver: 基础进化器
- EASEvolverV2: V2 改进版（推荐，基于 Auto-Fusion-v2）
- EvolutionConfig: 进化配置
- Individual: 进化个体
- SearchResult: 搜索结果
- RewardFunction: 奖励函数
"""

from .evolver import EASEvolver, EvolutionConfig, Individual
from .evolver_v2 import EASEvolverV2, SearchResult
from .reward import RewardFunction

__all__ = [
    'EASEvolver',
    'EASEvolverV2',
    'EvolutionConfig',
    'Individual',
    'SearchResult',
    'RewardFunction'
]

"""
Evaluator 模块 - 架构评估

包含:
- ProxyEvaluator: 基础评估器
- ProxyEvaluatorV2: V2 改进版（推荐，基于 Auto-Fusion-v2）
- compute_mrob: mRob 计算
"""

from .multimodal_rob import compute_mrob
from .proxy_evaluator import ProxyEvaluator
from .proxy_evaluator_v2 import ProxyEvaluatorV2

__all__ = [
    'compute_mrob',
    'ProxyEvaluator',
    'ProxyEvaluatorV2'
]

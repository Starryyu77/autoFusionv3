"""
奖励函数模块 V2 - 基于 Auto-Fusion-v2 改进

核心改进:
1. 指数惩罚机制（约束越界时惩罚快速增长）
2. 效率奖励（低于目标时给予奖励）
3. 约束满足度计算
"""

import numpy as np
import torch
from typing import Dict, Any


class RewardFunction:
    """
    EAS奖励函数 V2

    R = w_acc * accuracy + w_rob * mRob + efficiency_reward - constraint_penalty

    其中 constraint_penalty 使用指数惩罚:
    - 轻微越界: 线性惩罚
    - 严重越界: 指数增长惩罚
    """

    def __init__(
        self,
        w_accuracy: float = 1.0,
        w_robustness: float = 2.0,
        w_efficiency: float = 0.5,
        w_constraint: float = 2.0,
        target_flops: float = 10e9,  # 10 GFLOPs
        target_params: float = 50e6,  # 50M parameters
        penalty_type: str = "exponential"  # "linear" or "exponential"
    ):
        """
        初始化奖励函数

        Args:
            w_accuracy: 准确率权重
            w_robustness: 鲁棒性权重（核心指标，权重最高）
            w_efficiency: 效率奖励权重
            w_constraint: 约束惩罚权重
            target_flops: 目标FLOPs
            target_params: 目标参数量
            penalty_type: 惩罚类型（"linear" 或 "exponential"）
        """
        self.w_accuracy = w_accuracy
        self.w_robustness = w_robustness
        self.w_efficiency = w_efficiency
        self.w_constraint = w_constraint
        self.target_flops = target_flops
        self.target_params = target_params
        self.penalty_type = penalty_type

    def compute(
        self,
        accuracy: float,
        mrob: float,
        flops: float,
        params: float = None,
        latency: float = 0.0
    ) -> float:
        """
        计算综合奖励 V2（基于 Auto-Fusion-v2）

        Args:
            accuracy: 完整模态准确率 [0, 1]
            mrob: 模态鲁棒性 [0, 1]
            flops: 计算量 (FLOPs)
            params: 参数量（可选）
            latency: 延迟 (ms，可选)

        Returns:
            综合奖励值
        """
        # 准确率奖励
        acc_reward = self.w_accuracy * accuracy

        # 鲁棒性奖励 (核心指标)
        rob_reward = self.w_robustness * mrob

        # 效率奖励（低于目标给予正奖励）
        efficiency_reward = 0.0
        flops_ratio = flops / self.target_flops if self.target_flops > 0 else 0
        efficiency_reward = self.w_efficiency * max(0, 1 - flops_ratio)

        # 约束惩罚（指数惩罚）
        constraint_penalty = 0.0

        # FLOPs 约束
        if flops > self.target_flops:
            violation = (flops - self.target_flops) / self.target_flops
            constraint_penalty += self._compute_penalty(violation)

        # Params 约束
        if params and self.target_params > 0 and params > self.target_params:
            violation = (params - self.target_params) / self.target_params
            constraint_penalty += self._compute_penalty(violation)

        constraint_penalty *= self.w_constraint

        # 延迟惩罚 (可选)
        latency_penalty = 0.0
        if latency > 50:  # 50ms阈值
            latency_penalty = 0.01 * (latency - 50)

        total_reward = (
            acc_reward +
            rob_reward +
            efficiency_reward -
            constraint_penalty -
            latency_penalty
        )

        return total_reward

    def _compute_penalty(self, violation: float) -> float:
        """
        计算约束违反惩罚（v2: 指数惩罚）

        Args:
            violation: 违反程度 (0 = 刚好越界, 1 = 越界100%)

        Returns:
            惩罚值
        """
        if self.penalty_type == "exponential":
            # 指数增长: penalty = e^violation - 1
            # violation=0.1 -> penalty=0.105
            # violation=0.5 -> penalty=0.649
            # violation=1.0 -> penalty=1.718
            return np.exp(violation) - 1
        else:
            # 线性惩罚
            return violation

    def compute_from_metrics(self, metrics: Dict[str, float]) -> float:
        """
        从metrics字典计算奖励

        Args:
            metrics: {
                'accuracy': float,
                'mrob': float,
                'flops': float,
                'latency': float (optional)
            }
        """
        return self.compute(
            accuracy=metrics.get('accuracy', 0.0),
            mrob=metrics.get('mrob', 0.0),
            flops=metrics.get('flops', 1e9),
            latency=metrics.get('latency', 0.0)
        )

    def get_reward_components(
        self,
        accuracy: float,
        mrob: float,
        flops: float
    ) -> Dict[str, float]:
        """
        获取奖励的各个组成部分（用于分析）
        """
        return {
            'accuracy_reward': self.w_accuracy * accuracy,
            'robustness_reward': self.w_robustness * mrob,
            'efficiency_penalty': self.w_flops * max(0, flops / self.target_flops - 1),
            'total': self.compute(accuracy, mrob, flops)
        }


if __name__ == "__main__":
    # 测试
    reward_fn = RewardFunction()

    # 测试不同场景
    scenarios = [
        {'name': 'High accuracy, low robustness', 'acc': 0.90, 'mrob': 0.60, 'flops': 5e9},
        {'name': 'Balanced', 'acc': 0.85, 'mrob': 0.85, 'flops': 7e9},
        {'name': 'High robustness, low efficiency', 'acc': 0.80, 'mrob': 0.90, 'flops': 15e9},
    ]

    for s in scenarios:
        r = reward_fn.compute(s['acc'], s['mrob'], s['flops'])
        components = reward_fn.get_reward_components(s['acc'], s['mrob'], s['flops'])
        print(f"\n{s['name']}:")
        print(f"  Reward: {r:.3f}")
        print(f"  Components: {components}")

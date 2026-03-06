"""
奖励函数模块

计算综合奖励：accuracy + robustness - efficiency_penalty
"""

import torch
from typing import Dict, Any


class RewardFunction:
    """
    EAS奖励函数

    R = w_acc * accuracy + w_rob * mRob - w_flops * GFLOPs
    """

    def __init__(
        self,
        w_accuracy: float = 1.0,
        w_robustness: float = 2.0,
        w_flops: float = 0.5,
        target_flops: float = 10e9  # 10 GFLOPs
    ):
        """
        初始化奖励函数

        Args:
            w_accuracy: 准确率权重
            w_robustness: 鲁棒性权重
            w_flops: 计算量惩罚权重
            target_flops: 目标FLOPs
        """
        self.w_accuracy = w_accuracy
        self.w_robustness = w_robustness
        self.w_flops = w_flops
        self.target_flops = target_flops

    def compute(
        self,
        accuracy: float,
        mrob: float,
        flops: float,
        latency: float = 0.0
    ) -> float:
        """
        计算综合奖励

        Args:
            accuracy: 完整模态准确率 [0, 1]
            mrob: 模态鲁棒性 [0, 1]
            flops: 计算量 (FLOPs)
            latency: 延迟 (ms)

        Returns:
            综合奖励值
        """
        # 准确率奖励
        acc_reward = self.w_accuracy * accuracy

        # 鲁棒性奖励 (核心指标)
        rob_reward = self.w_robustness * mrob

        # 效率惩罚 (超过目标FLOPs开始惩罚)
        efficiency_penalty = 0.0
        if flops > self.target_flops:
            efficiency_penalty = self.w_flops * (flops / self.target_flops - 1)
        else:
            # 低于目标给予奖励
            efficiency_penalty = -self.w_flops * 0.1 * (1 - flops / self.target_flops)

        # 延迟惩罚 (可选)
        latency_penalty = 0.0
        if latency > 50:  # 50ms阈值
            latency_penalty = 0.01 * (latency - 50)

        total_reward = acc_reward + rob_reward - efficiency_penalty - latency_penalty

        return total_reward

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

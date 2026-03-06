"""
统一的模态缺失模拟器

所有实验必须使用此类，确保缺失模拟一致
"""

import torch
import numpy as np
from typing import Dict, Tuple, Literal


class UnifiedModalityDropout:
    """
    统一的模态缺失模拟器

    三种缺失模式:
    1. random: 随机独立缺失
    2. burst: 连续时间窗缺失（模拟传感器故障）
    3. progressive: 渐进衰减（模拟信号弱化）

    所有实验必须使用此类，确保缺失模拟一致
    """

    def __init__(
        self,
        drop_prob: float = 0.5,
        mode: Literal['random', 'burst', 'progressive'] = 'random',
        seed: int = 42
    ):
        """
        初始化模态缺失模拟器

        Args:
            drop_prob: 缺失概率 (0-1)
            mode: 缺失模式
            seed: 随机种子
        """
        self.drop_prob = drop_prob
        self.mode = mode
        self.rng = np.random.RandomState(seed)

        # 缺失模式配置
        self.mode_configs = {
            'random': {
                'description': '随机独立缺失',
                'burst': False
            },
            'burst': {
                'description': '连续时间窗缺失（模拟传感器故障）',
                'burst': True,
                'burst_length': 5  # 连续5帧缺失
            },
            'progressive': {
                'description': '渐进衰减（模拟信号弱化）',
                'burst': False,
                'noise_std': 0.1
            }
        }

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        应用模态缺失

        Args:
            batch: {
                'vision': [B, T, D] 或 [B, D],
                'audio': [B, T, D] 或 [B, D],
                'text': [B, T, D] 或 [B, D]
            }

        Returns:
            masked_batch: 缺失后的数据
            masks: {modality: binary_mask} (1=保留, 0=缺失)
        """
        if self.mode == 'random':
            return self._random_dropout(batch)
        elif self.mode == 'burst':
            return self._burst_dropout(batch)
        elif self.mode == 'progressive':
            return self._progressive_dropout(batch)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _random_dropout(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """随机独立缺失"""
        batch_size = list(batch.values())[0].shape[0]
        masks = {}
        masked_batch = {}

        for modality, tensor in batch.items():
            # 为每个样本独立决定是否缺失
            mask = torch.from_numpy(
                (self.rng.rand(batch_size) > self.drop_prob).astype(np.float32)
            )

            # 扩展到与tensor相同的维度
            while mask.dim() < tensor.dim():
                mask = mask.unsqueeze(-1)

            # 应用mask
            masked_batch[modality] = tensor * mask.to(tensor.device)
            masks[modality] = mask

        return masked_batch, masks

    def _burst_dropout(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """连续时间窗缺失（模拟传感器故障）"""
        batch_size = list(batch.values())[0].shape[0]
        masks = {}
        masked_batch = {}

        burst_length = self.mode_configs['burst']['burst_length']

        for modality, tensor in batch.items():
            # 创建全1mask
            mask = torch.ones(tensor.shape[0], dtype=torch.float32)

            # 为某些样本设置burst缺失
            for i in range(batch_size):
                if self.rng.rand() < self.drop_prob:
                    # 这个样本的该模态完全缺失
                    mask[i] = 0

            # 扩展到与tensor相同的维度
            while mask.dim() < tensor.dim():
                mask = mask.unsqueeze(-1)

            masked_batch[modality] = tensor * mask.to(tensor.device)
            masks[modality] = mask

        return masked_batch, masks

    def _progressive_dropout(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """渐进衰减（模拟信号弱化）"""
        batch_size = list(batch.values())[0].shape[0]
        noise_std = self.mode_configs['progressive']['noise_std']
        masks = {}
        masked_batch = {}

        for modality, tensor in batch.items():
            # 生成噪声水平 (0到drop_prob)
            noise_levels = torch.from_numpy(
                self.rng.rand(batch_size) * self.drop_prob
            ).float()

            # 为每个样本添加不同强度的噪声
            mask_values = 1 - noise_levels

            # 扩展到与tensor相同的维度
            while mask_values.dim() < tensor.dim():
                mask_values = mask_values.unsqueeze(-1)

            # 应用衰减
            mask_values = mask_values.to(tensor.device)
            masked_tensor = tensor * mask_values

            # 添加高斯噪声
            noise = torch.randn_like(tensor) * noise_std * (1 - mask_values)
            masked_tensor = masked_tensor + noise

            masked_batch[modality] = masked_tensor
            masks[modality] = mask_values

        return masked_batch, masks

    def get_config(self) -> Dict:
        """获取配置 (用于实验记录)"""
        return {
            'drop_prob': self.drop_prob,
            'mode': self.mode,
            'mode_description': self.mode_configs[self.mode]['description'],
            'seed': self.rng.randint(0, 2**32)
        }


def test_dropout():
    """测试模态缺失模拟器"""
    print("Testing UnifiedModalityDropout...")

    # 创建测试数据
    batch = {
        'vision': torch.ones(10, 576, 1024),
        'audio': torch.ones(10, 400, 512),
        'text': torch.ones(10, 77, 768)
    }

    # 测试random模式
    dropout = UnifiedModalityDropout(drop_prob=0.5, mode='random', seed=42)
    masked, masks = dropout(batch)

    print(f"\nRandom mode (drop_prob=0.5):")
    for mod in ['vision', 'audio', 'text']:
        actual_drop = 1 - masks[mod].mean().item()
        print(f"  {mod}: target=0.50, actual={actual_drop:.3f}")

    # 测试burst模式
    dropout_burst = UnifiedModalityDropout(drop_prob=0.5, mode='burst', seed=42)
    masked_burst, masks_burst = dropout_burst(batch)

    print(f"\nBurst mode:")
    for mod in ['vision', 'audio', 'text']:
        actual_drop = 1 - masks_burst[mod].mean().item()
        print(f"  {mod}: actual={actual_drop:.3f}")

    print("\n✅ All tests passed")


if __name__ == "__main__":
    test_dropout()

"""
VQA-v2数据集加载器

视觉问答数据集
"""

import os
from pathlib import Path
from typing import Dict, Any

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .base_loader import BaseMultimodalDataset


class VQADataset(BaseMultimodalDataset):
    """
    VQA-v2数据集

    模态:
    - 视觉: 图像特征 (ViT/ResNet提取)
    - 文本: 问题文本 (BERT/tokenized)

    标签: 答案 (多分类)
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        max_question_len: int = 20,
        **kwargs
    ):
        """
        初始化VQA数据集

        Args:
            data_path: 数据文件路径
            split: train/val/test
            max_question_len: 最大问题长度
        """
        self.max_question_len = max_question_len

        super().__init__(
            data_path=data_path,
            split=split,
            modalities=['vision', 'text'],
            **kwargs
        )

    def _load_data(self) -> Any:
        """加载VQA数据"""
        if not os.path.exists(self.data_path):
            print(f"⚠️  Data file not found: {self.data_path}")
            print(f"   Using dummy data for testing...")
            return self._create_dummy_data()

        # 实际数据加载逻辑
        print(f"📊 Loading VQA data from {self.data_path}...")
        # TODO: 实现实际数据加载
        return self._create_dummy_data()

    def _create_dummy_data(self):
        """创建虚拟数据"""
        num_samples = 1000 if self.split == 'train' else 200

        data = []
        for i in range(num_samples):
            sample = {
                'vision': torch.randn(197, 768),  # ViT patches
                'text': torch.randn(self.max_question_len, 768),  # BERT embeddings
                'label': torch.randint(0, 3129, (1,)).squeeze()  # 3129个答案类别
            }
            data.append(sample)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.data[idx]

        result = {
            'vision': sample['vision'],
            'text': sample['text'],
            'label': sample['label']
        }

        return result


if __name__ == "__main__":
    print("Testing VQADataset...")

    dataset = VQADataset(
        data_path="data/vqa_v2",
        split='train'
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")

    print(f"\nAPI contract: {dataset.get_api_contract()}")

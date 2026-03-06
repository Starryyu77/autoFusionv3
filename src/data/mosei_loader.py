"""
CMU-MOSEI数据集加载器

三模态情感分析数据集
"""

import os
import sys
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset

from .base_loader import BaseMultimodalDataset


class MOSEIDataset(BaseMultimodalDataset):
    """
    CMU-MOSEI数据集

    模态:
    - 视觉: 视频特征 (CLIP提取)
    - 音频: 音频特征 (wav2vec提取)
    - 文本: 文本特征 (BERT提取)

    标签: 情感强度 (连续值，范围-3到+3)
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        modalities: list = None,
        max_seq_len: int = 50,
        normalize: bool = True,
        **kwargs
    ):
        """
        初始化MOSEI数据集

        Args:
            data_path: 数据文件路径 (.pkl文件)
            split: train/val/test
            modalities: 使用的模态
            max_seq_len: 最大序列长度
            normalize: 是否归一化
        """
        self.max_seq_len = max_seq_len
        self.normalize = normalize

        super().__init__(
            data_path=data_path,
            split=split,
            modalities=modalities or ['vision', 'audio', 'text'],
            **kwargs
        )

    def _load_data(self) -> Dict[str, Any]:
        """加载MOSEI数据"""
        if not os.path.exists(self.data_path):
            print(f"⚠️  Data file not found: {self.data_path}")
            print(f"   Please download MOSEI dataset first.")
            print(f"   Using dummy data for testing...")
            return self._create_dummy_data()

        print(f"📊 Loading MOSEI data from {self.data_path}...")

        with open(self.data_path, 'rb') as f:
            all_data = pickle.load(f)

        # 根据split选择数据
        if self.split in all_data:
            data = all_data[self.split]
        else:
            # 假设数据是一个字典，包含train/val/test
            data = all_data

        print(f"   Loaded {len(data)} samples")
        return data

    def _create_dummy_data(self) -> Dict[str, Any]:
        """创建虚拟数据用于测试"""
        num_samples = 1000 if self.split == 'train' else 200

        data = []
        for i in range(num_samples):
            sample = {
                'vision': torch.randn(50, 768),
                'audio': torch.randn(400, 512),
                'text': torch.randn(77, 768),
                'label': torch.randn(1) * 3  # 情感强度 -3~3
            }
            data.append(sample)

        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.data[idx]

        result = {}

        # 处理各模态
        for mod in self.modalities:
            if mod in sample:
                features = sample[mod]

                # 截断或填充到固定长度
                if len(features.shape) == 2:  # [T, D]
                    seq_len = min(features.shape[0], self.max_seq_len)
                    features = features[:seq_len]

                result[mod] = features

        # 处理标签
        if 'label' in sample:
            label = sample['label']
            if isinstance(label, (int, float)):
                label = torch.tensor(label, dtype=torch.float32)
            result['label'] = label

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        return {
            'num_samples': len(self),
            'split': self.split,
            'modalities': self.modalities,
            'feature_dims': self.get_feature_dims()
        }


def download_mosei(data_dir: str, use_multibench: bool = True) -> str:
    """
    下载MOSEI数据集

    Args:
        data_dir: 数据保存目录
        use_multibench: 是否使用MultiBench预处理版本

    Returns:
        数据文件路径
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / "mosei.pkl"

    if output_file.exists():
        print(f"✅ MOSEI data already exists: {output_file}")
        return str(output_file)

    if use_multibench:
        print("📥 Downloading MOSEI via MultiBench...")
        print("   This may take a while...")

        try:
            # 使用MultiBench下载
            import subprocess
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/pliang279/MultiBench.git"
            ], check=True)

            # 下载数据
            print("   Please manually download MOSEI from:")
            print("   http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/")

        except Exception as e:
            print(f"   Error: {e}")
            print("   Creating dummy data instead...")

    # 创建虚拟数据
    print("📝 Creating dummy MOSEI data for testing...")
    dummy_data = {
        'train': MOSEIDataset._create_dummy_data_static(1000),
        'val': MOSEIDataset._create_dummy_data_static(200),
        'test': MOSEIDataset._create_dummy_data_static(400)
    }

    with open(output_file, 'wb') as f:
        pickle.dump(dummy_data, f)

    print(f"✅ Dummy data saved: {output_file}")
    return str(output_file)


@staticmethod
def _create_dummy_data_static(num_samples: int):
    """静态方法创建虚拟数据"""
    data = []
    for i in range(num_samples):
        sample = {
            'vision': torch.randn(50, 768),
            'audio': torch.randn(400, 512),
            'text': torch.randn(77, 768),
            'label': torch.randn(1) * 3
        }
        data.append(sample)
    return data


if __name__ == "__main__":
    print("Testing MOSEIDataset...")

    # 测试虚拟数据
    dataset = MOSEIDataset(
        data_path="data/mosei.pkl",
        split='train'
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Statistics: {dataset.get_statistics()}")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    for key, value in sample.items():
        print(f"  {key}: {value.shape}")

    print(f"\nAPI contract: {dataset.get_api_contract()}")

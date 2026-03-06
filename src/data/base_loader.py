"""
数据集加载器基类

所有数据集加载器继承此类，实现统一接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import torch
from torch.utils.data import Dataset


class BaseMultimodalDataset(Dataset, ABC):
    """
    多模态数据集基类

    统一的数据接口:
    - 视觉特征
    - 音频特征
    - 文本特征
    - 标签
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        modalities: list = None,
        transform=None
    ):
        """
        初始化数据集

        Args:
            data_path: 数据文件路径
            split: 数据分割 (train/val/test)
            modalities: 使用的模态列表
            transform: 数据变换
        """
        self.data_path = data_path
        self.split = split
        self.modalities = modalities or ['vision', 'audio', 'text']
        self.transform = transform

        # 加载数据
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self) -> Any:
        """加载原始数据，子类必须实现"""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """返回数据集大小"""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            {
                'vision': tensor [T, D],
                'audio': tensor [T, D],
                'text': tensor [T, D],
                'label': tensor [...]
            }
        """
        pass

    def get_feature_dims(self) -> Dict[str, int]:
        """获取各模态特征维度"""
        sample = self[0]
        return {
            mod: sample[mod].shape[-1]
            for mod in self.modalities
            if mod in sample
        }

    def get_api_contract(self) -> Dict[str, Any]:
        """
        生成API契约，用于内循环编译

        Returns:
            {
                'inputs': {
                    'vision': {'shape': [B, T, D], 'dtype': 'float32'},
                    ...
                },
                'output_shape': [B, num_classes]
            }
        """
        sample = self[0]
        batch_size = 2  # dummy batch size

        api_contract = {
            'inputs': {},
            'output_shape': [batch_size, -1]  # -1表示动态
        }

        for mod in self.modalities:
            if mod in sample:
                shape = sample[mod].shape
                # [B] + shape
                full_shape = [batch_size] + list(shape)
                api_contract['inputs'][mod] = {
                    'shape': full_shape,
                    'dtype': 'float32'
                }

        # 根据label确定输出shape
        if 'label' in sample:
            label_shape = list(sample['label'].shape)
            api_contract['output_shape'] = [batch_size] + label_shape
            api_contract['num_classes'] = label_shape[-1] if len(label_shape) > 0 else 1

        return api_contract


class SimpleMultimodalDataset(BaseMultimodalDataset):
    """
    简单的多模态数据集实现

    用于快速测试和演示
    """

    def __init__(
        self,
        num_samples: int = 1000,
        vision_dim: int = 1024,
        audio_dim: int = 512,
        text_dim: int = 768,
        num_classes: int = 10,
        **kwargs
    ):
        self.num_samples = num_samples
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.num_classes = num_classes

        # 忽略data_path，生成随机数据
        super().__init__(data_path='', **kwargs)

    def _load_data(self):
        """生成随机数据"""
        return None  # 不需要加载

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """生成随机样本"""
        # 随机序列长度
        vision_len = torch.randint(50, 600, (1,)).item()
        audio_len = torch.randint(100, 500, (1,)).item()
        text_len = torch.randint(20, 80, (1,)).item()

        sample = {
            'vision': torch.randn(vision_len, self.vision_dim),
            'audio': torch.randn(audio_len, self.audio_dim),
            'text': torch.randn(text_len, self.text_dim),
            'label': torch.randint(0, self.num_classes, (1,)).squeeze()
        }

        # 只返回需要的模态
        return {k: v for k, v in sample.items() if k in self.modalities or k == 'label'}


if __name__ == "__main__":
    # 测试简单数据集
    print("Testing SimpleMultimodalDataset...")

    dataset = SimpleMultimodalDataset(num_samples=100)

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")

    for key, value in sample.items():
        print(f"  {key}: {value.shape}")

    print(f"\nFeature dims: {dataset.get_feature_dims()}")
    print(f"API contract: {dataset.get_api_contract()}")

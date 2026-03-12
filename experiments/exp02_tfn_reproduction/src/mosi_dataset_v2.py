"""
CMU-MOSI 数据集加载器 - 适配实际数据格式
支持 Binary, 5-class, Regression 三种任务
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Tuple


class MOSIDatasetV2(Dataset):
    """
    CMU-MOSI 数据集 V2

    数据格式:
    - text: [N, 50, 300] - GloVe embeddings
    - vision: [N, 50, 35] - Facet features
    - audio: [N, 50, 74] - COVAREP features
    - labels: [N, 1, 1] - sentiment scores [-3, 3]
    """

    def __init__(self,
                 data: Dict,
                 split: str = 'train',
                 task: str = 'binary'):
        """
        Args:
            data: 包含 train/test/valid 的字典
            split: 'train', 'valid', 'test'
            task: 'binary', '5class', 'regression'
        """
        super().__init__()
        self.task = task
        self.split_data = data[split]

        # 提取特征和标签
        self.text = self.split_data['text']      # [N, 50, 300]
        self.vision = self.split_data['vision']  # [N, 50, 35]
        self.audio = self.split_data['audio']    # [N, 50, 74]
        self.labels = self.split_data['labels']  # [N, 1, 1]

        # 对时间维度取平均，得到每个样本的固定维度特征
        self.text = self.text.mean(axis=1)       # [N, 300]
        self.vision = self.vision.mean(axis=1)   # [N, 35]
        self.audio = self.audio.mean(axis=1)     # [N, 74]

        # 处理 Inf/NaN 值
        self.text = np.nan_to_num(self.text, nan=0.0, posinf=0.0, neginf=0.0)
        self.vision = np.nan_to_num(self.vision, nan=0.0, posinf=0.0, neginf=0.0)
        self.audio = np.nan_to_num(self.audio, nan=0.0, posinf=0.0, neginf=0.0)

        self.labels = self.labels.squeeze()      # [N]

        print(f"[{split}] Loaded {len(self)} samples")
        print(f"  Text: {self.text.shape}, Vision: {self.vision.shape}, Audio: {self.audio.shape}")
        print(f"  Labels range: [{self.labels.min():.2f}, {self.labels.max():.2f}]")

    def _process_label(self, label: float) -> torch.Tensor:
        """处理标签为不同任务格式"""
        if self.task == 'binary':
            # Binary: positive (>=0) vs negative (<0)
            binary_label = 1.0 if label >= 0 else 0.0
            return torch.tensor([binary_label], dtype=torch.float32)

        elif self.task == '5class':
            # 5-class: 将 [-3, 3] 分为 5 个区间
            if label < -1.8:
                class_id = 0  # strongly negative
            elif label < -0.6:
                class_id = 1  # negative
            elif label < 0.6:
                class_id = 2  # neutral
            elif label < 1.8:
                class_id = 3  # positive
            else:
                class_id = 4  # strongly positive
            return torch.tensor(class_id, dtype=torch.long)

        else:  # regression
            # 原始 [-3, 3] 范围，归一化到 [-1, 1]
            normalized = label / 3.0
            return torch.tensor([normalized], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            (text, vision, audio, label)
        """
        text = torch.tensor(self.text[idx], dtype=torch.float32)
        vision = torch.tensor(self.vision[idx], dtype=torch.float32)
        audio = torch.tensor(self.audio[idx], dtype=torch.float32)
        label = self._process_label(self.labels[idx])

        return text, vision, audio, label


def load_mosi_data(data_path: str) -> Dict:
    """加载 MOSI 数据"""
    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_mosi_loaders_v2(data_path: str,
                        task: str = 'binary',
                        batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取 MOSI 数据加载器

    Args:
        data_path: 数据路径
        task: 'binary', '5class', 'regression'
        batch_size: batch 大小

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    data = load_mosi_data(data_path)

    train_dataset = MOSIDatasetV2(data, 'train', task)
    valid_dataset = MOSIDatasetV2(data, 'valid', task)
    test_dataset = MOSIDatasetV2(data, 'test', task)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_input_dims_v2() -> Dict[str, int]:
    """获取输入维度"""
    return {
        'language': 300,  # text
        'visual': 35,     # vision
        'acoustic': 74    # audio
    }


if __name__ == '__main__':
    # 测试数据加载
    data_path = '/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl'

    train_loader, valid_loader, test_loader = get_mosi_loaders_v2(
        data_path, task='binary', batch_size=4
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 测试一个 batch
    for text, vision, audio, label in train_loader:
        print(f"\nBatch shapes:")
        print(f"  Text: {text.shape}")
        print(f"  Vision: {vision.shape}")
        print(f"  Audio: {audio.shape}")
        print(f"  Label: {label.shape}")
        print(f"  Label values: {label[:5].squeeze()}")
        break

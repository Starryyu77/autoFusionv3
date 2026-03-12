"""
CMU-MOSI 数据集加载器 - 论文原始设置
支持 Binary, 5-class, Regression 三种任务
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import h5py


class MOSIDataset(Dataset):
    """
    CMU-MOSI 数据集

    数据格式 (对齐后):
    - language: GloVe word embeddings (300-dim)
    - visual: Facet features (35-dim)
    - acoustic: COVAREP features (74-dim)
    - labels: sentiment scores in [-3, 3]
    """

    def __init__(self,
                 data_path: str,
                 split: str = 'train',
                 task: str = 'binary',
                 fold: int = 0):
        """
        Args:
            data_path: 数据文件路径 (.pkl 或 .hdf5)
            split: 'train', 'valid', 'test'
            task: 'binary', '5class', 'regression'
            fold: 5-fold 交叉验证的 fold 索引 (0-4)
        """
        super().__init__()
        self.split = split
        self.task = task
        self.fold = fold

        # 加载数据
        self.data = self._load_data(data_path)

        # 5-fold 划分
        self.indices = self._get_fold_indices()

    def _load_data(self, data_path: str) -> Dict:
        """加载 MOSI 数据"""
        data_path = Path(data_path)

        if data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
        elif data_path.suffix == '.hdf5':
            data = self._load_hdf5(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        return data

    def _load_hdf5(self, data_path: Path) -> Dict:
        """从 HDF5 加载数据"""
        data = {}
        with h5py.File(data_path, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        return data

    def _get_fold_indices(self) -> List[int]:
        """5-fold 交叉验证划分"""
        total = len(self.data['labels'])
        fold_size = total // 5

        # 确定测试集
        test_start = self.fold * fold_size
        test_end = test_start + fold_size if self.fold < 4 else total
        test_indices = list(range(test_start, test_end))

        # 剩余数据分为训练集和验证集 (80/20)
        remaining = [i for i in range(total) if i not in test_indices]
        np.random.seed(42)
        np.random.shuffle(remaining)

        valid_size = len(remaining) // 5
        valid_indices = remaining[:valid_size]
        train_indices = remaining[valid_size:]

        if self.split == 'train':
            return train_indices
        elif self.split == 'valid':
            return valid_indices
        else:  # test
            return test_indices

    def _process_label(self, label: float) -> torch.Tensor:
        """
        处理标签为不同任务格式

        Args:
            label: sentiment score in [-3, 3]
        Returns:
            不同任务的标签格式
        """
        if self.task == 'binary':
            # Binary: positive (>=0) vs negative (<0)
            # 映射到 [0, 1]，然后模型输出映射到 [-3, 3]
            binary_label = 1.0 if label >= 0 else 0.0
            return torch.tensor([binary_label], dtype=torch.float32)

        elif self.task == '5class':
            # 5-class: 将 [-3, 3] 分为 5 个区间
            # [-3, -1.8), [-1.8, -0.6), [-0.6, 0.6), [0.6, 1.8), [1.8, 3]
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
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Returns:
            (language, visual, acoustic, label)
        """
        real_idx = self.indices[idx]

        # 获取特征 (取平均池化后的特征)
        language = torch.tensor(self.data['language'][real_idx].mean(axis=0), dtype=torch.float32)
        visual = torch.tensor(self.data['visual'][real_idx].mean(axis=0), dtype=torch.float32)
        acoustic = torch.tensor(self.data['acoustic'][real_idx].mean(axis=0), dtype=torch.float32)

        # 处理标签
        label = self.data['labels'][real_idx]
        if isinstance(label, np.ndarray):
            label = label.item() if label.size == 1 else label.mean()
        label_tensor = self._process_label(label)

        return language, visual, acoustic, label_tensor


def get_mosi_loaders(data_path: str,
                     task: str = 'binary',
                     fold: int = 0,
                     batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    获取 MOSI 数据加载器

    Args:
        data_path: 数据路径
        task: 'binary', '5class', 'regression'
        fold: 5-fold 索引 (0-4)
        batch_size: batch 大小

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    train_dataset = MOSIDataset(data_path, 'train', task, fold)
    valid_dataset = MOSIDataset(data_path, 'valid', task, fold)
    test_dataset = MOSIDataset(data_path, 'test', task, fold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader


def get_input_dims(data_path: str) -> Dict[str, int]:
    """获取输入维度"""
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return {
        'language': data['language'][0].shape[-1],  # 300
        'visual': data['visual'][0].shape[-1],      # 35
        'acoustic': data['acoustic'][0].shape[-1]   # 74
    }


if __name__ == '__main__':
    # 测试数据加载
    data_path = '/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl'

    train_loader, valid_loader, test_loader = get_mosi_loaders(
        data_path, task='binary', fold=0, batch_size=4
    )

    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Valid size: {len(valid_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")

    # 测试一个 batch
    for language, visual, acoustic, label in train_loader:
        print(f"Language shape: {language.shape}")
        print(f"Visual shape: {visual.shape}")
        print(f"Acoustic shape: {acoustic.shape}")
        print(f"Label shape: {label.shape}")
        print(f"Label values: {label[:5]}")
        break

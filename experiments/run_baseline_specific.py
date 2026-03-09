"""
基线方法特定实验运行脚本

根据每个基线的原始论文和架构特点，执行特定的实验配置
"""

import os
import sys
import yaml
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import (
    MeanFusionModel, ConcatFusionModel, AttentionFusionModel, MaxFusionModel,
    DynMMCompleteModel, TFNCompleteModel, ADMNCompleteModel,
    CentaurCompleteModel, FDSNetCompleteModel,
    DARTSCompleteModel, LLMaticCompleteModel, EvoPromptingCompleteModel
)


# 基线模型映射
BASELINE_MODELS = {
    'mean': MeanFusionModel,
    'concat': ConcatFusionModel,
    'attention': AttentionFusionModel,
    'max': MaxFusionModel,
    'dynmm': DynMMCompleteModel,
    'tfn': TFNCompleteModel,
    'admn': ADMNCompleteModel,
    'centaur': CentaurCompleteModel,
    'fdsnet': FDSNetCompleteModel,
    'darts': DARTSCompleteModel,
    'llmatic': LLMaticCompleteModel,
    'evoprompting': EvoPromptingCompleteModel,
}


class BaselineSpecificExperiment:
    """
    基线特定实验执行器

    根据每个基线的特点执行特定的实验配置
    """

    def __init__(
        self,
        method: str,
        dataset: str,
        config_path: str,
        device: str = 'cuda',
        output_dir: str = 'results/baselines_specific'
    ):
        self.method = method.lower()
        self.dataset = dataset.lower()
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 数据集信息
        self.data_path = None
        self.num_classes = None
        self.input_dims = None
        self.is_regression = False

        # 模型
        self.model = None
        self.label_mean = None
        self.label_std = None

        # 数据
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup_dataset(self):
        """设置数据集参数"""
        dataset_config = self.config['datasets'][self.dataset]
        self.data_path = dataset_config['path']
        self.num_classes = dataset_config['num_classes']
        self.input_dims = dataset_config['input_dims']

        print(f"\n📊 Dataset: {self.dataset.upper()}")
        print(f"   Path: {self.data_path}")
        print(f"   Classes: {self.num_classes}")
        print(f"   Input dims: {self.input_dims}")

    def load_data(self):
        """加载数据集"""
        import pickle

        print(f"\n📥 Loading data...")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        if os.path.isdir(self.data_path):
            # 分离的数据文件
            data_dir = Path(self.data_path)
            with open(data_dir / 'train_data.pkl', 'rb') as f:
                train_raw = pickle.load(f)
            with open(data_dir / 'valid_data.pkl', 'rb') as f:
                val_raw = pickle.load(f)
            with open(data_dir / 'test_data.pkl', 'rb') as f:
                test_raw = pickle.load(f)

            self.train_data = self._convert_to_dict(train_raw)
            self.val_data = self._convert_to_dict(val_raw)
            self.test_data = self._convert_to_dict(test_raw)
        else:
            # 单个文件
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)

            if isinstance(data, dict) and 'train' in data:
                self.train_data = self._convert_to_dict(data['train'])
                # 尝试不同的验证集键名
                if 'valid' in data:
                    self.val_data = self._convert_to_dict(data['valid'])
                elif 'val' in data:
                    self.val_data = self._convert_to_dict(data['val'])
                else:
                    # 如果没有验证集，从训练集划分一部分
                    total = len(self.train_data['labels'])
                    val_size = int(0.1 * total)
                    indices = torch.randperm(total)
                    val_idx = indices[:val_size]
                    train_idx = indices[val_size:]
                    self.val_data = {k: v[val_idx] for k, v in self.train_data.items()}
                    self.train_data = {k: v[train_idx] for k, v in self.train_data.items()}

                if 'test' in data:
                    self.test_data = self._convert_to_dict(data['test'])
                else:
                    self.test_data = self.val_data  # 如果没有测试集，使用验证集
            else:
                # 手动划分
                all_data = self._convert_to_dict(data)
                total = len(all_data['labels'])
                train_size = int(0.8 * total)
                val_size = int(0.1 * total)

                indices = torch.randperm(total)
                train_idx = indices[:train_size]
                val_idx = indices[train_size:train_size + val_size]
                test_idx = indices[train_size + val_size:]

                self.train_data = {k: v[train_idx] for k, v in all_data.items()}
                self.val_data = {k: v[val_idx] for k, v in all_data.items()}
                self.test_data = {k: v[test_idx] for k, v in all_data.items()}

        print(f"   Train: {len(self.train_data['labels'])} samples")
        print(f"   Val: {len(self.val_data['labels'])} samples")
        print(f"   Test: {len(self.test_data['labels'])} samples")

        # 检测任务类型 - 修复squeeze问题
        labels_flat = self.train_data['labels'].squeeze().flatten().tolist()
        if isinstance(labels_flat, (int, float)):
            labels_unique = 1
        else:
            labels_unique = len(set(labels_flat))

        if self.num_classes > 20:
            self.is_regression = False
            print(f"   Task: Classification ({self.num_classes} classes)")
        elif labels_unique > 20:
            self.is_regression = True
            print(f"   Task: Regression ({labels_unique} unique values)")

            # 标签标准化
            train_labels = self.train_data['labels'].float()
            self.label_mean = train_labels.mean()
            self.label_std = train_labels.std()
            if self.label_std < 1e-6:
                self.label_std = 1.0

            self.train_data['labels'] = (self.train_data['labels'].float() - self.label_mean) / self.label_std
            self.val_data['labels'] = (self.val_data['labels'].float() - self.label_mean) / self.label_std
            self.test_data['labels'] = (self.test_data['labels'].float() - self.label_mean) / self.label_std
        else:
            self.is_regression = False
            print(f"   Task: Classification ({self.num_classes} classes)")

    def _convert_to_dict(self, raw_data):
        """转换数据格式"""
        is_regression = self.is_regression if hasattr(self, 'is_regression') else False

        if isinstance(raw_data, dict):
            result = {}
            for key, value in raw_data.items():
                if isinstance(value, np.ndarray):
                    # 跳过字符串类型的数组（如'id'字段）
                    if value.dtype.kind in ['U', 'S', 'O']:
                        continue
                    dtype = torch.float32 if key == 'labels' and is_regression else (torch.long if key == 'labels' else torch.float32)
                    result[key] = torch.tensor(value, dtype=dtype)
                elif isinstance(value, torch.Tensor):
                    result[key] = value
                else:
                    result[key] = value
            return result
        elif isinstance(raw_data, list):
            result = {'vision': [], 'audio': [], 'text': [], 'labels': []}
            for sample in raw_data:
                for mod in ['vision', 'audio', 'text']:
                    if mod in sample:
                        result[mod].append(sample[mod])
                if 'label' in sample:
                    result['labels'].append(sample['label'])

            for key in result:
                if result[key]:
                    if isinstance(result[key][0], torch.Tensor):
                        result[key] = torch.stack(result[key])
                    else:
                        dtype = torch.float32 if key == 'labels' and is_regression else torch.long
                        result[key] = torch.tensor(result[key], dtype=dtype)
            return result
        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")

    def create_model(self) -> nn.Module:
        """根据基线类型创建模型"""
        model_config = self.config.get('model', {})
        hidden_dim = model_config.get('hidden_dim', 256)

        # 获取模型类
        model_class = BASELINE_MODELS.get(self.method)
        if model_class is None:
            raise ValueError(f"Unknown baseline method: {self.method}")

        # 创建模型（传递特定参数）
        kwargs = self._get_method_specific_kwargs()
        model = model_class(
            input_dims=self.input_dims,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            is_regression=self.is_regression,
            **kwargs
        )

        return model.to(self.device)

    def _get_method_specific_kwargs(self) -> Dict:
        """获取方法特定的参数"""
        kwargs = {}
        model_config = self.config.get('model', {})

        if self.method == 'dynmm':
            kwargs['routing_threshold'] = model_config.get('routing_threshold', 0.2)
        elif self.method == 'tfn':
            kwargs['reduce_dim'] = model_config.get('reduce_dim', 64)
        elif self.method == 'admn':
            kwargs['num_layers'] = model_config.get('num_layers', 3)
        elif self.method == 'centaur':
            kwargs['use_modality_completion'] = model_config.get('use_modality_completion', True)
        elif self.method == 'darts':
            kwargs['num_ops'] = model_config.get('num_ops', 5)
        elif self.method == 'llmatic':
            search_config = self.config.get('search', {})
            kwargs['population_size'] = search_config.get('population_size', 10)
            kwargs['num_iterations'] = search_config.get('num_iterations', 20)
        elif self.method == 'evoprompting':
            search_config = self.config.get('search', {})
            kwargs['population_size'] = search_config.get('population_size', 10)
            kwargs['num_iterations'] = search_config.get('num_iterations', 20)

        return kwargs

    def train_epoch(self, optimizer, criterion, batch_size: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_samples = len(self.train_data['labels'])
        num_batches = (num_samples + batch_size - 1) // batch_size

        indices = torch.randperm(num_samples)

        for i in range(num_batches):
            batch_idx = indices[i * batch_size: (i + 1) * batch_size]

            # 获取数据
            vision = self.train_data.get('vision')
            audio = self.train_data.get('audio')
            text = self.train_data.get('text')
            labels = self.train_data['labels'][batch_idx].to(self.device).squeeze()

            if vision is not None:
                vision = vision[batch_idx].to(self.device)
            if audio is not None:
                audio = audio[batch_idx].to(self.device)
            if text is not None:
                text = text[batch_idx].to(self.device)

            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(vision, audio, text)

            if self.is_regression:
                outputs = outputs.squeeze()
                labels = labels.float()
            else:
                # 确保labels是1D
                labels = labels.long().view(-1)

            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        return total_loss / num_samples

    def evaluate(self, data: Dict, dropout_rate: float = 0.0, batch_size: int = 128) -> float:
        """评估模型"""
        self.model.eval()
        num_samples = len(data['labels'])
        num_batches = (num_samples + batch_size - 1) // batch_size

        if self.is_regression:
            total_error = 0.0
        else:
            correct = 0
            total = 0

        with torch.no_grad():
            for i in range(num_batches):
                batch_slice = slice(i * batch_size, (i + 1) * batch_size)

                vision = data.get('vision')
                audio = data.get('audio')
                text = data.get('text')
                labels = data['labels'][batch_slice].to(self.device)

                if vision is not None:
                    vision = vision[batch_slice].to(self.device)
                if audio is not None:
                    audio = audio[batch_slice].to(self.device)
                if text is not None:
                    text = text[batch_slice].to(self.device)

                # 应用模态缺失
                if dropout_rate > 0:
                    for mod_tensor in [vision, audio, text]:
                        if mod_tensor is not None:
                            mask = (torch.rand(mod_tensor.shape[0], 1, 1) > dropout_rate).float().to(self.device)
                            mod_tensor *= mask

                outputs = self.model(vision, audio, text)

                if self.is_regression:
                    predictions = outputs.squeeze()
                    total_error += torch.abs(predictions - labels.float()).sum().item()
                else:
                    predictions = outputs.argmax(dim=-1)
                    correct += (predictions == labels.long()).sum().item()
                    total += labels.size(0)

        if self.is_regression:
            mae = total_error / num_samples
            return mae * self.label_std if self.label_std else mae
        else:
            return correct / total if total > 0 else 0.0

    def run_experiment(self, seed: int) -> Dict:
        """运行单次实验"""
        print(f"\n🚀 Running {self.method.upper()} on {self.dataset.upper()} (seed={seed})")

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # 创建模型
        self.model = self.create_model()
        print(f"   Model: {self.method}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # 获取训练配置
        train_config = self.config.get('training', {})
        epochs = train_config.get('epochs', 100)
        lr = train_config.get('learning_rate', 0.001)
        weight_decay = train_config.get('weight_decay', 0.0001)
        batch_size = train_config.get('batch_size', 64)

        # 数据集特定batch_size
        dataset_config = self.config['datasets'][self.dataset]
        if 'batch_size' in dataset_config:
            batch_size = dataset_config['batch_size']

        # NAS基线：先执行架构搜索
        if self.method in ['darts', 'llmatic', 'evoprompting']:
            if hasattr(self.model, 'search_architecture'):
                print(f"\n🔍 Executing architecture search...")
                self.model.search_architecture(verbose=True)

        # 优化器
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max' if not self.is_regression else 'min',
            patience=10, factor=0.5
        )

        # 损失函数
        criterion = nn.MSELoss() if self.is_regression else nn.CrossEntropyLoss()

        # 早停
        best_metric = 0.0 if not self.is_regression else float('inf')
        patience_counter = 0
        patience = train_config.get('early_stop', {}).get('patience', 20)

        # 训练循环
        print(f"\n📚 Training for max {epochs} epochs...")
        start_time = time.time()

        for epoch in range(epochs):
            train_loss = self.train_epoch(optimizer, criterion, batch_size)
            val_metric = self.evaluate(self.val_data, dropout_rate=0.0)

            scheduler.step(val_metric)

            # 早停检查
            improved = (val_metric > best_metric) if not self.is_regression else (val_metric < best_metric)
            if improved:
                best_metric = val_metric
                patience_counter = 0
                # 保存最佳模型
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                metric_name = 'Acc' if not self.is_regression else 'MAE'
                print(f"   Epoch {epoch+1}/{epochs}: loss={train_loss:.4f}, val_{metric_name}={val_metric:.4f}")

            if patience_counter >= patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break

        training_time = time.time() - start_time

        # 加载最佳模型
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        # 测试评估
        print(f"\n🧪 Testing with different dropout rates...")
        results = {'seed': seed, 'training_time': training_time}

        for dropout in [0.0, 0.25, 0.50]:
            metric = self.evaluate(self.test_data, dropout_rate=dropout)
            results[f'dropout_{int(dropout*100)}'] = metric
            metric_name = 'Acc' if not self.is_regression else 'MAE'
            print(f"   Dropout {int(dropout*100)}%: {metric_name}={metric:.4f}")

        return results

    def run_all_seeds(self) -> Dict:
        """运行所有种子并汇总结果"""
        eval_config = self.config.get('evaluation', {})
        seeds = eval_config.get('seeds', [42, 123, 456, 789, 1024])

        all_results = []

        for seed in seeds:
            result = self.run_experiment(seed)
            all_results.append(result)

        # 汇总
        summary = {
            'method': self.method,
            'dataset': self.dataset,
            'is_regression': self.is_regression,
            'num_seeds': len(seeds),
            'seeds': seeds,
        }

        # 计算均值和标准差
        for key in ['dropout_0', 'dropout_25', 'dropout_50', 'training_time']:
            values = [r[key] for r in all_results if key in r]
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)

        # 计算mRob
        full_acc = [r['dropout_0'] for r in all_results]
        drop25_acc = [r['dropout_25'] for r in all_results]
        drop50_acc = [r['dropout_50'] for r in all_results]

        mrob_25 = [d25 / full if full > 0 else 0 for full, d25 in zip(full_acc, drop25_acc)]
        mrob_50 = [d50 / full if full > 0 else 0 for full, d50 in zip(full_acc, drop50_acc)]

        summary['mrob_25_mean'] = np.mean(mrob_25)
        summary['mrob_25_std'] = np.std(mrob_25)
        summary['mrob_50_mean'] = np.mean(mrob_50)
        summary['mrob_50_std'] = np.std(mrob_50)

        summary['raw_results'] = all_results

        return summary

    def save_results(self, summary: Dict):
        """保存结果"""
        output_file = self.output_dir / f"{self.method}_{self.dataset}.json"

        # 转换numpy类型
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        with open(output_file, 'w') as f:
            json.dump(convert(summary), f, indent=2)

        print(f"\n💾 Results saved to {output_file}")

        # 打印摘要
        print("\n" + "=" * 60)
        print(f"Summary for {self.method.upper()} on {self.dataset.upper()}")
        print("=" * 60)

        metric_name = 'Accuracy' if not self.is_regression else 'MAE'
        print(f"{metric_name}: {summary.get('dropout_0_mean', 0):.4f} ± {summary.get('dropout_0_std', 0):.4f}")
        print(f"mRob@25%: {summary.get('mrob_25_mean', 0):.4f} ± {summary.get('mrob_25_std', 0):.4f}")
        print(f"mRob@50%: {summary.get('mrob_50_mean', 0):.4f} ± {summary.get('mrob_50_std', 0):.4f}")
        print(f"Training time: {summary.get('training_time_mean', 0):.1f}s")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Run baseline-specific experiments')
    parser.add_argument('--method', type=str, required=True,
                        help='Baseline method name')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mosei', 'iemocap', 'vqa'],
                        help='Dataset name')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (auto-detected if not provided)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results/baselines_specific',
                        help='Output directory')

    args = parser.parse_args()

    # 自动检测配置文件
    if args.config is None:
        config_path = f"configs/baselines/{args.method}.yaml"
        if not os.path.exists(config_path):
            # 尝试使用simple_baselines配置
            config_path = "configs/baselines/simple_baselines.yaml"
    else:
        config_path = args.config

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Using config: {config_path}")

    # 创建实验
    experiment = BaselineSpecificExperiment(
        method=args.method,
        dataset=args.dataset,
        config_path=config_path,
        device=args.device,
        output_dir=args.output_dir
    )

    # 设置数据集
    experiment.setup_dataset()

    # 加载数据
    experiment.load_data()

    # 运行所有种子的实验
    summary = experiment.run_all_seeds()

    # 保存结果
    experiment.save_results(summary)


if __name__ == '__main__':
    main()

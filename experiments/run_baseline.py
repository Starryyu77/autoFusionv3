"""
基线方法统一评估脚本

所有基线方法使用相同的:
1. 统一投影层 (UnifiedFeatureProjection)
2. 统一分类头 (UnifiedClassifier)
3. 相同的训练配置
4. 相同的评估方式

支持的基线方法:
- DARTS: 可微分架构搜索
- DynMM: 动态多模态融合
- ADMN: 自适应动态网络
- Centaur: 鲁棒多模态融合
- TFN: 张量融合网络
- LLMatic: LLM-based NAS
- EvoPrompting: 进化提示
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.unified_projection import UnifiedModel, UnifiedFeatureProjection, UnifiedClassifier


def create_baseline_fusion(method: str, input_dim: int = 1024):
    """
    创建基线融合模块

    Args:
        method: 基线方法名称
        input_dim: 输入维度 (固定1024)

    Returns:
        nn.Module: 融合模块 (输出 [B, 1024])
    """
    if method == 'dynmm':
        from src.baselines.dynmm import DynMMFusion
        return DynMMFusion(input_dim=input_dim)

    elif method == 'admn':
        from src.baselines.admn import ADMNFusion
        return ADMNFusion(input_dim=input_dim)

    elif method == 'centaur':
        from src.baselines.centaur import CentaurFusion
        return CentaurFusion(input_dim=input_dim)

    elif method == 'darts':
        from src.baselines.darts_fusion import DARTSFusionModule
        return DARTSFusionModule(input_dim=input_dim)

    elif method == 'tfn':
        from src.baselines.tfn import TFNFusion
        return TFNFusion(input_dim=input_dim)

    elif method == 'fdsnet':
        from src.baselines.fdsnet import FDSNetFusion
        return FDSNetFusion(input_dim=input_dim)

    elif method == 'llmatic':
        from src.baselines.llmatic_fusion import LLMaticFusionModule
        return LLMaticFusionModule(input_dim=input_dim)

    elif method == 'evoprompting':
        from src.baselines.evoprompting_fusion import EvoPromptingFusionModule
        return EvoPromptingFusionModule(input_dim=input_dim)

    else:
        raise ValueError(f"Unknown baseline method: {method}")


class BaselineEvaluator:
    """
    基线方法评估器

    统一评估流程:
    1. 创建融合模块
    2. 包装成统一模型
    3. 加载数据
    4. 训练/评估
    5. 多缺失率测试
    """

    def __init__(
        self,
        method: str,
        dataset: str,
        data_path: str,
        num_classes: int = 10,
        device: str = 'cuda',
        output_dir: str = 'results/baselines'
    ):
        self.method = method
        self.dataset = dataset
        self.data_path = data_path
        self.num_classes = num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 检测任务类型 (在加载数据后确定)
        self.is_regression = None  # Will be set in load_data

        # 创建模型 (在知道任务类型后创建)
        self.model = None

        # 标签标准化统计量 (用于回归任务)
        self.label_mean = None
        self.label_std = None

    def _create_model(self) -> nn.Module:
        """创建统一模型"""
        fusion_module = create_baseline_fusion(self.method)
        # 回归任务输出1维，分类任务输出num_classes维
        output_dim = 1 if self.is_regression else self.num_classes
        print(f"   Creating model with output_dim={output_dim} (is_regression={self.is_regression})")
        return UnifiedModel(fusion_module, num_classes=output_dim, is_regression=self.is_regression)

    def load_data(self):
        """加载数据集 - 支持多种格式"""
        import pickle
        import os
        from pathlib import Path

        print(f"📥 Loading data from {self.data_path}...")

        # 检查是否是目录
        if os.path.isdir(self.data_path):
            data_dir = Path(self.data_path)

            # 加载分离的数据文件
            with open(data_dir / 'train_data.pkl', 'rb') as f:
                train_raw = pickle.load(f)
            with open(data_dir / 'valid_data.pkl', 'rb') as f:
                val_raw = pickle.load(f)
            with open(data_dir / 'test_data.pkl', 'rb') as f:
                test_raw = pickle.load(f)

            # 统一转换为dict格式
            self.train_data = self._convert_to_dict(train_raw)
            self.val_data = self._convert_to_dict(val_raw)
            self.test_data = self._convert_to_dict(test_raw)
        else:
            # 合并的数据文件或单个文件
            with open(self.data_path, 'rb') as f:
                data = pickle.load(f)

            # 检查是否已经是合并格式（有train/val/test键）
            if isinstance(data, dict) and 'train' in data:
                self.train_data = self._convert_to_dict(data['train'])
                self.val_data = self._convert_to_dict(data['val'])
                self.test_data = self._convert_to_dict(data.get('test', data['val']))
            else:
                # 单个文件，需要手动划分（如VQA）
                print("   Single file format detected, splitting data...")
                all_data = self._convert_to_dict(data)
                total = len(all_data['labels'])

                # 80/10/10划分
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

        # 检测任务类型和自动推断类别数
        labels_unique = len(set(self.train_data['labels'].flatten().tolist()))

        # 检测任务类型 (根据labels数据类型和范围，以及用户传入的num_classes)
        # 如果用户传入的num_classes > 20，视为大类别分类任务 (如VQA)
        if self.num_classes > 20:
            print(f"   ⚠️  Large num_classes specified: {self.num_classes}")
            print(f"   Task type: CLASSIFICATION ({self.num_classes} classes)")
            self.is_regression = False
        elif labels_unique > 20:
            # 回归任务 (唯一值多但num_classes小)
            self.is_regression = True
            print(f"   Task type: REGRESSION ({labels_unique} unique values)")

            # 对回归任务进行标签标准化 (z-score)
            train_labels = self.train_data['labels'].float()
            self.label_mean = train_labels.mean()
            self.label_std = train_labels.std()

            if self.label_std < 1e-6:
                self.label_std = 1.0

            print(f"   Label normalization: mean={self.label_mean:.4f}, std={self.label_std:.4f}")
            self.train_data['labels'] = (self.train_data['labels'].float() - self.label_mean) / self.label_std
            self.val_data['labels'] = (self.val_data['labels'].float() - self.label_mean) / self.label_std
            self.test_data['labels'] = (self.test_data['labels'].float() - self.label_mean) / self.label_std
            print(f"   Normalized labels range: [{self.train_data['labels'].min():.4f}, {self.train_data['labels'].max():.4f}]")
        else:
            # 标准分类任务 (< 20类)
            if labels_unique != self.num_classes:
                print(f"   ⚠️  Adjusting num_classes: {self.num_classes} -> {labels_unique}")
                self.num_classes = labels_unique
            self.is_regression = False
            print(f"   Task type: CLASSIFICATION ({self.num_classes} classes)")

        # 现在创建模型 (知道了任务类型)
        self.model = self._create_model()
        self.model = self.model.to(self.device)

        print(f"✅ Created {self.method} model")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _convert_to_dict(self, raw_data):
        """
        统一转换为dict格式

        支持:
        1. dict格式: {'vision': tensor, 'audio': tensor, 'text': tensor, 'labels': tensor}
        2. list格式: [{'vision': tensor, ...}, ...] (MOSEI格式)
        """
        # 检测任务类型 - 检查原始数据中labels的唯一值数量
        is_regression = False
        if isinstance(raw_data, dict) and 'labels' in raw_data:
            labels_data = raw_data['labels']
            if isinstance(labels_data, np.ndarray):
                labels_unique = len(set(labels_data.flatten().tolist()))
                is_regression = labels_unique > 20
            elif isinstance(labels_data, torch.Tensor):
                labels_unique = len(set(labels_data.flatten().tolist()))
                is_regression = labels_unique > 20
        elif isinstance(raw_data, list) and len(raw_data) > 0:
            if 'label' in raw_data[0] or 'labels' in raw_data[0]:
                labels_list = [s.get('label', s.get('labels')) for s in raw_data]
                labels_unique = len(set(labels_list))
                is_regression = labels_unique > 20

        if isinstance(raw_data, dict):
            # 已经是dict格式 (VQA等)
            # 确保所有数据都是tensor
            result = {}
            for key, value in raw_data.items():
                if isinstance(value, np.ndarray):
                    # 回归任务labels保持float32，分类任务labels转为long
                    if key == 'labels':
                        dtype = torch.float32 if is_regression else torch.long
                    else:
                        dtype = torch.float32
                    result[key] = torch.tensor(value, dtype=dtype)
                elif isinstance(value, torch.Tensor):
                    result[key] = value
                else:
                    result[key] = value
            return result

        elif isinstance(raw_data, list):
            # list格式 (MOSEI等)
            result = {'vision': [], 'audio': [], 'text': [], 'labels': []}

            for sample in raw_data:
                for mod in ['vision', 'audio', 'text']:
                    if mod in sample:
                        result[mod].append(sample[mod])
                if 'label' in sample:
                    result['labels'].append(sample['label'])

            # 转换为tensor
            for key in result:
                if result[key]:
                    if isinstance(result[key][0], torch.Tensor):
                        result[key] = torch.stack(result[key])
                    else:
                        # 回归任务labels保持float32，分类任务labels转为long
                        if key == 'labels':
                            dtype = torch.float32 if is_regression else torch.long
                        else:
                            dtype = torch.float32
                        result[key] = torch.tensor(result[key], dtype=dtype)

            return result

        else:
            raise ValueError(f"Unsupported data format: {type(raw_data)}")

    def train(self, epochs: int = 50, lr: float = 0.001, batch_size: int = 64):
        """训练模型（使用小批量）"""
        print(f"\n🚀 Training {self.method} for {epochs} epochs (batch_size={batch_size})...")

        # 对于回归任务，使用较大的学习率和weight decay防止平凡解
        if self.is_regression:
            effective_lr = lr * 2.0  # 更大的学习率
            weight_decay = 1e-4  # 更大的weight decay
        else:
            effective_lr = lr
            weight_decay = 1e-5
        optimizer = torch.optim.Adam(self.model.parameters(), lr=effective_lr, weight_decay=weight_decay)
        print(f"   Using learning rate: {effective_lr:.6f}, weight_decay: {weight_decay} (is_regression={self.is_regression})")

        # 根据任务类型选择损失函数和优化目标
        if self.is_regression:
            # 使用MAE损失代替MSE，避免模型偏向预测均值
            criterion = nn.L1Loss()
            # 增大patience，避免过早降低学习率
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=15, factor=0.5, min_lr=1e-6
            )
        else:
            criterion = nn.CrossEntropyLoss()
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=15, factor=0.5, min_lr=1e-6
            )

        # 根据任务类型设置优化目标
        if self.is_regression:
            best_val_metric = float('inf')  # MAE越低越好
            metric_mode = 'min'
            metric_name = 'MAE'
        else:
            best_val_metric = 0.0  # Accuracy越高越好
            metric_mode = 'max'
            metric_name = 'Acc'

        patience_counter = 0
        max_patience = 30  # 增加早停耐心，给模型更多学习时间

        num_samples = len(self.train_data['labels'])
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            # 训练
            self.model.train()
            epoch_loss = 0.0

            # 小批量训练
            indices = torch.randperm(num_samples)
            for i in range(num_batches):
                batch_indices = indices[i * batch_size: (i + 1) * batch_size]

                # 动态获取可用的模态数据
                vision = self.train_data.get('vision', self.train_data.get('v', None))
                audio = self.train_data.get('audio', self.train_data.get('a', None))
                text = self.train_data.get('text', self.train_data.get('t', None))

                # 获取批次数据
                if vision is not None:
                    vision = vision[batch_indices].to(self.device)
                if audio is not None:
                    audio = audio[batch_indices].to(self.device)
                if text is not None:
                    text = text[batch_indices].to(self.device)

                labels = self.train_data['labels'][batch_indices].to(self.device).squeeze().long()

                optimizer.zero_grad()
                # 根据可用模态调用模型
                if audio is None:
                    outputs = self.model(vision, text=text)
                elif text is None:
                    outputs = self.model(vision, audio)
                else:
                    outputs = self.model(vision, audio, text)

                # 调整输出维度匹配labels
                if self.is_regression:
                    outputs = outputs.squeeze()

                loss = criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches

            # Debug: 打印前3个epoch的详细信息
            if epoch <= 2:
                print(f"   [Debug] Epoch {epoch}: loss={avg_loss:.4f}")
                print(f"   [Debug] Output range (normalized): [{outputs.min().item():.4f}, {outputs.max().item():.4f}]")
                print(f"   [Debug] Labels range (normalized): [{labels.min().item():.4f}, {labels.max().item():.4f}]")
                print(f"   [Debug] Current LR: {optimizer.param_groups[0]['lr']:.6f}")

                self.model.train()  # 确保回到train模式

            # 验证
            val_metric = self.evaluate(self.val_data)
            scheduler.step(val_metric)

            # 早停检查
            improved = (val_metric < best_val_metric) if metric_mode == 'min' else (val_metric > best_val_metric)
            if improved:
                best_val_metric = val_metric
                patience_counter = 0
                self._save_checkpoint(f'best_model_{self.method}.pt')
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_{metric_name}={val_metric:.4f}")

            if patience_counter >= max_patience:
                print(f"   ⏹️ Early stopping at epoch {epoch+1}")
                break

        # 加载最佳模型
        self._load_checkpoint(f'best_model_{self.method}.pt')
        print(f"✅ Training complete. Best val {metric_name}: {best_val_metric:.4f}")

    def evaluate(self, data: Dict, dropout_rate: float = 0.0, batch_size: int = 128) -> float:
        """评估模型（使用小批量）
        分类: 返回Accuracy (越高越好)
        回归: 返回MAE (越低越好)
        """
        self.model.eval()

        num_samples = len(data['labels'])
        num_batches = (num_samples + batch_size - 1) // batch_size

        if self.is_regression:
            total_error = 0.0
            total_samples = 0
        else:
            correct = 0
            total = 0

        with torch.no_grad():
            for i in range(num_batches):
                batch_indices = slice(i * batch_size, (i + 1) * batch_size)

                # 动态获取可用的模态数据
                vision = data.get('vision', data.get('v', None))
                audio = data.get('audio', data.get('a', None))
                text = data.get('text', data.get('t', None))

                if vision is not None:
                    vision = vision[batch_indices].to(self.device)
                if audio is not None:
                    audio = audio[batch_indices].to(self.device)
                if text is not None:
                    text = text[batch_indices].to(self.device)

                labels = data['labels'][batch_indices].to(self.device).squeeze().long()

                # 应用模态缺失
                if dropout_rate > 0:
                    for mod_tensor in [vision, audio, text]:
                        if mod_tensor is not None:
                            mask = (torch.rand(mod_tensor.shape[0], 1, 1) > dropout_rate).float().to(self.device)
                            mod_tensor *= mask

                # 根据可用模态调用模型
                if audio is None:
                    outputs = self.model(vision, text=text)
                elif text is None:
                    outputs = self.model(vision, audio)
                else:
                    outputs = self.model(vision, audio, text)

                if self.is_regression:
                    # 回归: 计算MAE
                    predictions = outputs.squeeze()
                    total_error += torch.abs(predictions - labels).sum().item()
                    total_samples += labels.size(0)
                else:
                    # 分类: 计算Accuracy
                    predictions = outputs.argmax(dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)

        if self.is_regression:
            mae_normalized = total_error / total_samples if total_samples > 0 else float('inf')
            # 反标准化到原始尺度的MAE
            mae = mae_normalized * self.label_std
            return mae
        else:
            return correct / total if total > 0 else 0.0

    def run_full_evaluation(self, seeds: list = [42, 123, 456, 789, 999]) -> Dict:
        """运行完整评估 (多种子 × 多缺失率)"""
        print(f"\n📊 Running full evaluation for {self.method}...")

        results = []

        for seed in seeds:
            # 设置随机种子
            torch.manual_seed(seed)
            np.random.seed(seed)

            # 重新初始化模型
            self.model = self._create_model().to(self.device)

            # 训练
            self.train(epochs=50)

            # 测试不同缺失率
            for dropout in [0.0, 0.25, 0.50]:
                metric = self.evaluate(self.test_data, dropout_rate=dropout)
                results.append({
                    'seed': seed,
                    'dropout': dropout,
                    'metric': metric
                })
                metric_name = 'MAE' if self.is_regression else 'Acc'
                print(f"   Seed {seed}, Dropout {dropout}: {metric_name}={metric:.4f}")

        # 汇总结果
        summary = self._summarize_results(results)

        # 保存结果
        self._save_results(summary)

        return summary

    def _summarize_results(self, results: list) -> Dict:
        """汇总结果
        分类: 返回Accuracy和mRob (越高越好)
        回归: 返回MAE和mRob (mRob=dropout/full, 越低越好)
        """
        # 按缺失率分组
        full_metrics = [r['metric'] for r in results if r['dropout'] == 0.0]
        drop25_metrics = [r['metric'] for r in results if r['dropout'] == 0.25]
        drop50_metrics = [r['metric'] for r in results if r['dropout'] == 0.50]

        # 计算mRob (保持统一: dropout性能 / 完整性能)
        # 分类: acc_dropout / acc_full (越低越差，理想接近1)
        # 回归: mae_dropout / mae_full (越低越差，理想接近1)
        mrob_25 = [d25 / full if full > 0 else 0 for full, d25 in zip(full_metrics, drop25_metrics)] if full_metrics else [0]
        mrob_50 = [d50 / full if full > 0 else 0 for full, d50 in zip(full_metrics, drop50_metrics)] if full_metrics else [0]

        if self.is_regression:
            return {
                'method': self.method,
                'dataset': self.dataset,
                'is_regression': True,
                'mae_mean': np.mean(full_metrics),
                'mae_std': np.std(full_metrics),
                'mrob_25_mean': np.mean(mrob_25),
                'mrob_25_std': np.std(mrob_25),
                'mrob_50_mean': np.mean(mrob_50),
                'mrob_50_std': np.std(mrob_50),
                'raw_results': results
            }
        else:
            return {
                'method': self.method,
                'dataset': self.dataset,
                'is_regression': False,
                'accuracy_mean': np.mean(full_metrics),
                'accuracy_std': np.std(full_metrics),
                'mrob_25_mean': np.mean(mrob_25),
                'mrob_25_std': np.std(mrob_25),
                'mrob_50_mean': np.mean(mrob_50),
                'mrob_50_std': np.std(mrob_50),
                'raw_results': results
            }

    def _save_results(self, summary: Dict):
        """保存结果"""
        # 将numpy/torch类型转换为Python原生类型以便JSON序列化
        def convert_to_native(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        output_file = self.output_dir / f"{self.method}_{self.dataset}.json"
        with open(output_file, 'w') as f:
            json.dump(convert_to_native(summary), f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

    def _save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint_path = self.output_dir / filename
        torch.save(self.model.state_dict(), checkpoint_path)

    def _load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint_path = self.output_dir / filename
        if checkpoint_path.exists():
            self.model.load_state_dict(torch.load(checkpoint_path))


def main():
    parser = argparse.ArgumentParser(description='Run baseline evaluation')
    parser.add_argument('--method', type=str, required=True,
                        choices=['dynmm', 'admn', 'centaur', 'darts', 'tfn', 'fdsnet', 'llmatic', 'evoprompting'],
                        help='Baseline method')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mosei', 'iemocap', 'vqa'],
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset pickle file')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='results/baselines',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999],
                        help='Random seeds for evaluation')

    args = parser.parse_args()

    # 创建评估器
    evaluator = BaselineEvaluator(
        method=args.method,
        dataset=args.dataset,
        data_path=args.data_path,
        num_classes=args.num_classes,
        device=args.device,
        output_dir=args.output_dir
    )

    # 加载数据
    evaluator.load_data()

    # 运行完整评估
    results = evaluator.run_full_evaluation(seeds=args.seeds)

    # 打印结果
    print("\n" + "="*60)
    print(f"Final Results for {args.method} on {args.dataset}")
    print("="*60)
    if results.get('is_regression', False):
        print(f"MAE: {results['mae_mean']:.4f} ± {results['mae_std']:.4f}")
    else:
        print(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"mRob@25%: {results['mrob_25_mean']:.4f} ± {results['mrob_25_std']:.4f}")
    print(f"mRob@50%: {results['mrob_50_mean']:.4f} ± {results['mrob_50_std']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()

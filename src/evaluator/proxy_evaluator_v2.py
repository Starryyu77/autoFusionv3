"""
代理评估器 V2 - 基于 Auto-Fusion-v2 改进

核心改进:
1. ModelWrapper 模式（自动推断输出维度）
2. 完整的 few-shot 训练和评估
3. 标签范围验证
4. mRob 计算（模态鲁棒性）
"""

import re
import time
from typing import Dict, Any, Optional, Tuple
from contextlib import redirect_stdout, redirect_stderr
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np


class ProxyEvaluatorV2:
    """
    代理评估器 V2

    使用 few-shot 学习快速评估架构性能
    基于 Auto-Fusion-v2 的 ModelWrapper 模式
    """

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        num_shots: int = 16,
        num_epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_time: int = 300,
        api_contract: Optional[Dict] = None
    ):
        """
        初始化评估器

        Args:
            dataset: 数据集
            num_shots: few-shot 样本数（每类）
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 计算设备
            max_time: 单次评估最大时间（秒）
            api_contract: API 契约（输入输出规格）
        """
        self.full_dataset = dataset
        self.num_shots = num_shots
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.max_time = max_time
        self.api_contract = api_contract or {
            'inputs': {
                'vision': {'shape': [2, 576, 1024]},
                'audio': {'shape': [2, 400, 512]},
                'text': {'shape': [2, 77, 768]}
            },
            'output_shape': [2, 10]
        }

    def evaluate(self, code: str) -> Dict[str, Any]:
        """
        评估生成的架构

        Returns:
            metrics: {
                'accuracy': float,      # 完整模态准确率
                'mrob': float,          # 模态鲁棒性
                'flops': int,           # FLOPs
                'params': int,          # 参数量
                'training_time': float, # 训练时间
                'success': bool         # 是否成功
            }
        """
        start_time = time.time()

        try:
            # 1. 创建模型（使用 ModelWrapper）
            model = self._instantiate_model(code)
            model = model.to(self.device)

            # 2. 分析模型
            flops, params = self._profile_model(model)

            # 3. 创建数据加载器
            train_loader, val_loader = self._create_dataloaders()

            # 4. 完整模态训练和评估
            training_time = self._train_model(model, train_loader)
            acc_full = self._evaluate_model(model, val_loader, dropout=0.0)

            # 5. 缺失模态评估（计算 mRob）
            acc_missing = self._evaluate_model(model, val_loader, dropout=0.5)
            mrob = acc_missing / acc_full if acc_full > 0 else 0.0

            total_time = time.time() - start_time

            return {
                'accuracy': acc_full,
                'mrob': mrob,
                'flops': flops,
                'params': params,
                'training_time': training_time,
                'total_time': total_time,
                'success': True
            }

        except Exception as e:
            return {
                'accuracy': 0.0,
                'mrob': 0.0,
                'flops': 1e12,
                'params': 1e9,
                'training_time': 0.0,
                'total_time': time.time() - start_time,
                'success': False,
                'error': str(e)
            }

    def _instantiate_model(self, code: str) -> nn.Module:
        """
        从代码实例化模型（v2: ModelWrapper 模式）
        """
        namespace = {}

        # 抑制输出
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            exec(code, namespace)

        # 查找模型类
        model_class = None
        for name, obj in namespace.items():
            if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                model_class = obj
                break

        if model_class is None:
            raise ValueError("No valid model class found in code")

        # v2: ModelWrapper 包装器
        class ModelWrapper(nn.Module):
            """自动包装 Fusion Layer + Classifier"""

            def __init__(self, fusion_class, num_classes, api_contract):
                super().__init__()

                # 尝试无参数初始化
                import inspect
                sig = inspect.signature(fusion_class.__init__)
                params = list(sig.parameters.keys())

                if len(params) <= 1:  # 只有 self
                    self.fusion = fusion_class()
                else:
                    # 尝试 input_dims
                    input_dims = {
                        name: {'shape': spec['shape']}
                        for name, spec in api_contract['inputs'].items()
                    }
                    try:
                        self.fusion = fusion_class(input_dims)
                    except:
                        # 最后的尝试：无参数
                        self.fusion = fusion_class()

                # 通过 dummy forward 推断输出维度
                self.contract = api_contract
                dummy_inputs = self._create_dummy_inputs(device='cpu')
                with torch.no_grad():
                    dummy_output = self.fusion(**dummy_inputs)

                fusion_output_dim = dummy_output.shape[-1]
                self.classifier = nn.Linear(fusion_output_dim, num_classes)

            def _create_dummy_inputs(self, device='cpu'):
                inputs = {}
                for name, spec in self.contract['inputs'].items():
                    shape = spec['shape']
                    # 使用 batch_size=1 推断
                    shape = [1 if s == 'B' else s for s in shape]
                    inputs[name] = torch.randn(shape, device=device)
                return inputs

            def forward(self, **kwargs):
                fused = self.fusion(**kwargs)
                logits = self.classifier(fused)
                return logits

        # 获取类别数
        num_classes = self._get_num_classes()

        return ModelWrapper(model_class, num_classes, self.api_contract)

    def _get_num_classes(self) -> int:
        """从数据集推断类别数"""
        try:
            # 尝试获取标签
            labels = set()
            for i in range(min(len(self.full_dataset), 1000)):
                sample = self.full_dataset[i]
                if 'label' in sample:
                    label_val = sample['label']
                    if isinstance(label_val, torch.Tensor):
                        label_val = label_val.item()
                    labels.add(label_val)

            if labels:
                max_label = max(labels)
                return max(len(labels), max_label + 1)
        except:
            pass

        # 默认值
        return self.api_contract.get('output_shape', [2, 10])[-1]

    def _create_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """创建 few-shot 数据加载器"""
        num_samples = len(self.full_dataset)
        indices = list(range(num_samples))
        np.random.shuffle(indices)

        # 分层采样
        num_classes = self._get_num_classes()
        train_size = min(self.num_shots * num_classes, num_samples // 2)
        val_size = min(self.num_shots * num_classes // 2, num_samples // 4)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]

        train_subset = Subset(self.full_dataset, train_indices)
        val_subset = Subset(self.full_dataset, val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        return train_loader, val_loader

    def _train_model(self, model: nn.Module, train_loader: DataLoader) -> float:
        """训练模型 few-shot"""
        start_time = time.time()

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.num_epochs):
            for batch_idx, batch in enumerate(train_loader):
                # 超时检查
                if time.time() - start_time > self.max_time:
                    return time.time() - start_time

                # 准备输入
                inputs = {}
                for key, value in batch.items():
                    if key != 'label':
                        inputs[key] = value.to(self.device)

                labels = batch['label'].to(self.device)

                # 前向
                optimizer.zero_grad()
                logits = model(**inputs)

                # 验证标签范围
                num_classes = logits.shape[-1]
                if labels.min() < 0 or labels.max() >= num_classes:
                    labels = torch.clamp(labels, 0, num_classes - 1)

                loss = criterion(logits, labels)

                # 反向
                loss.backward()
                optimizer.step()

        return time.time() - start_time

    def _evaluate_model(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        dropout: float = 0.0
    ) -> float:
        """评估模型"""
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                # 准备输入
                inputs = {}
                for key, value in batch.items():
                    if key != 'label':
                        inputs[key] = value.to(self.device)

                labels = batch['label'].to(self.device)

                # 应用模态缺失
                if dropout > 0:
                    inputs = self._apply_modality_dropout(inputs, dropout)

                logits = model(**inputs)
                predictions = logits.argmax(dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def _apply_modality_dropout(
        self,
        inputs: Dict[str, torch.Tensor],
        dropout_prob: float
    ) -> Dict[str, torch.Tensor]:
        """应用模态缺失（计算 mRob）"""
        result = inputs.copy()

        for key in list(result.keys()):
            if torch.rand(1).item() < dropout_prob:
                # 将该模态置零
                result[key] = torch.zeros_like(result[key])

        return result

    def _profile_model(self, model: nn.Module) -> Tuple[int, int]:
        """分析模型 FLOPs 和参数量"""
        # 参数量
        params = sum(p.numel() for p in model.parameters())

        # 简单 FLOPs 估计
        flops = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                flops += 2 * module.in_features * module.out_features
            elif isinstance(module, nn.MultiheadAttention):
                # 简化估计
                flops += 2 * 576 * 576

        return flops, params


# 保持向后兼容
ProxyEvaluator = ProxyEvaluatorV2

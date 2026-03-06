#!/usr/bin/env python3
"""
Round 1 (重新设计): 端到端可行性验证

目标: 验证SelfHealingCompiler的端到端成功率
      不只是编译成功，还要能训练、能推理！

验证流程:
1. 编译验证 (Syntax + Shape)
2. 训练验证 (3 epochs, 检查损失是否下降)
3. 推理验证 (Forward pass)

用法:
    python experiments/run_round1_end2end.py --config configs/round1_end2end.yaml
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inner_loop.self_healing import SelfHealingCompiler, CompilationError
from utils.llm_backend import UnifiedLLMBackend
from utils.random_control import set_seed


def create_code_generation_prompt(variant: int = 0) -> str:
    """创建代码生成prompt"""
    base_prompt = """
Generate a PyTorch nn.Module for multimodal fusion.

API Interface:
- Input 'vision': tensor of shape [batch, 576, 1024], dtype float32
- Input 'audio': tensor of shape [batch, 400, 512], dtype float32
- Input 'text': tensor of shape [batch, 77, 768], dtype float32
- Output: tensor of shape [batch, 10], dtype float32

Requirements:
1. Must be subclass of nn.Module
2. Must have forward() accepting three inputs
3. Must handle missing modalities (check for None or zero tensors)
4. Use fusion mechanism (attention, gating, or concatenation)
5. Return correct shape [batch, 10]
6. Must be trainable with standard backpropagation

Generate only the code, no explanation:
"""
    variants = [
        "",
        "\nHint: Use cross-attention between modalities.",
        "\nHint: Add gating mechanism for modality control.",
        "\nHint: Include residual connections.",
        "\nHint: Use early exit for efficiency.",
    ]
    return base_prompt + variants[variant % len(variants)]


def load_mosei_toy_data(data_path: str, max_samples: int = 200) -> Tuple[DataLoader, DataLoader]:
    """
    加载MOSEI toy数据

    Returns:
        train_loader, val_loader
    """
    # 如果真实数据不存在，创建虚拟数据
    if not Path(data_path).exists():
        print(f"⚠️  Data not found at {data_path}, creating dummy data...")
        return create_dummy_data(max_samples)

    # 加载真实数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 限制样本数
    num_samples = min(len(data), max_samples)
    data = data[:num_samples]

    # 划分训练/验证
    split_idx = int(0.8 * num_samples)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_loader = create_dataloader(train_data, batch_size=16, shuffle=True)
    val_loader = create_dataloader(val_data, batch_size=16, shuffle=False)

    return train_loader, val_loader


def create_dummy_data(num_samples: int = 200) -> Tuple[DataLoader, DataLoader]:
    """创建虚拟MOSEI数据用于测试"""
    print(f"📝 Creating dummy data: {num_samples} samples")

    # 创建随机特征数据
    vision = torch.randn(num_samples, 576, 1024)
    audio = torch.randn(num_samples, 400, 512)
    text = torch.randn(num_samples, 77, 768)
    labels = torch.randint(0, 10, (num_samples,))

    # 划分训练/验证
    split_idx = int(0.8 * num_samples)

    train_dataset = TensorDataset(
        vision[:split_idx], audio[:split_idx], text[:split_idx], labels[:split_idx]
    )
    val_dataset = TensorDataset(
        vision[split_idx:], audio[split_idx:], text[split_idx:], labels[split_idx:]
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    return train_loader, val_loader


def create_dataloader(data: List[Dict], batch_size: int, shuffle: bool) -> DataLoader:
    """从数据列表创建DataLoader"""
    # 提取特征
    visions = torch.stack([d['vision'] for d in data])
    audios = torch.stack([d['audio'] for d in data])
    texts = torch.stack([d['text'] for d in data])
    labels = torch.tensor([d['label'] for d in data])

    dataset = TensorDataset(visions, audios, texts, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def stage1_compile_check(compiler, code: str, api_contract: Dict) -> Tuple[bool, str, str]:
    """
    阶段1: 编译验证

    Returns:
        (success, error_msg, compiled_code)
    """
    try:
        result = compiler.compile(code, api_contract, verbose=False)
        return True, "", result.code
    except CompilationError as e:
        return False, str(e), ""


def stage2_shape_check(model_class, api_contract: Dict) -> Tuple[bool, str]:
    """
    阶段2: 形状验证

    Returns:
        (success, error_msg)
    """
    try:
        model = model_class()

        # 创建dummy输入
        dummy_inputs = {
            'vision': torch.randn(*api_contract['inputs']['vision']['shape']),
            'audio': torch.randn(*api_contract['inputs']['audio']['shape']),
            'text': torch.randn(*api_contract['inputs']['text']['shape'])
        }

        # 前向传播
        with torch.no_grad():
            output = model(**dummy_inputs)

        # 验证输出形状
        expected_shape = api_contract['output_shape']
        if list(output.shape) != expected_shape:
            return False, f"Shape mismatch: {list(output.shape)} vs {expected_shape}"

        return True, ""
    except Exception as e:
        return False, f"Shape check failed: {str(e)}"


def stage3_training_check(model_class, train_loader: DataLoader, val_loader: DataLoader,
                          num_epochs: int = 3, min_improvement: float = 0.01) -> Tuple[bool, str, List[float]]:
    """
    阶段3: 训练验证 (关键!)

    验证模型能否训练（损失下降）

    Returns:
        (success, error_msg, loss_history)
    """
    try:
        model = model_class()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001)

        loss_history = []

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0

            for batch in train_loader:
                vision, audio, text, labels = batch
                vision = vision.to(device)
                audio = audio.to(device)
                text = text.to(device)
                labels = labels.to(device)

                # 时间维度平均池化
                vision = vision.mean(dim=1)
                audio = audio.mean(dim=1)
                text = text.mean(dim=1)

                optimizer.zero_grad()
                outputs = model(vision=vision, audio=audio, text=text)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            print(f"    Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # 检查损失是否下降
        if len(loss_history) >= 2:
            improvement = (loss_history[0] - loss_history[-1]) / loss_history[0]
            if improvement < min_improvement:
                return False, f"Insufficient improvement: {improvement:.2%}", loss_history

        return True, "", loss_history
    except Exception as e:
        return False, f"Training failed: {str(e)}", []


def stage4_inference_check(model_class, val_loader: DataLoader) -> Tuple[bool, str, float]:
    """
    阶段4: 推理验证

    Returns:
        (success, error_msg, accuracy)
    """
    try:
        model = model_class()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                vision, audio, text, labels = batch
                vision = vision.to(device)
                audio = audio.to(device)
                text = text.to(device)
                labels = labels.to(device)

                # 时间维度平均池化
                vision = vision.mean(dim=1)
                audio = audio.mean(dim=1)
                text = text.mean(dim=1)

                outputs = model(vision=vision, audio=audio, text=text)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / max(total, 1)
        return True, "", accuracy
    except Exception as e:
        return False, f"Inference failed: {str(e)}", 0.0


def run_end2end_validation(config_path: str):
    """运行端到端验证实验"""
    import yaml

    print("=" * 70)
    print("Round 1: End-to-End Validation (Redesigned)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 加载配置
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    set_seed(config['experiment']['seed'])

    # 初始化组件
    print("🔧 Initializing components...")
    llm = UnifiedLLMBackend()
    compiler = SelfHealingCompiler(llm_backend=llm, max_retries=3)

    # 加载数据
    print("📊 Loading data...")
    train_loader, val_loader = load_mosei_toy_data(
        config['dataset']['path'],
        config['dataset']['max_samples']
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print()

    # 实验配置
    api_contract = config['api_contract']
    num_samples = config['experiment_size']['num_samples']

    # 结果统计
    results = {
        'stage1_compile': [],
        'stage2_shape': [],
        'stage3_training': [],
        'stage4_inference': [],
        'end2end_success': [],
        'loss_histories': [],
        'accuracies': []
    }

    # 运行实验
    for i in range(num_samples):
        print(f"\n{'-' * 70}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'-' * 70}")

        # 生成代码
        prompt = create_code_generation_prompt(variant=i)

        sample_result = {
            'sample_id': i,
            'stages': {}
        }

        # ===== 阶段1: 编译验证 =====
        print("\n📦 Stage 1: Compilation Check")
        try:
            compile_result = compiler.compile(prompt, api_contract, verbose=False)
            code = compile_result.code
            sample_result['stages']['compile'] = {'success': True, 'attempts': compile_result.attempts}
            print("  ✅ Compilation successful")
        except CompilationError as e:
            sample_result['stages']['compile'] = {'success': False, 'error': str(e)}
            results['stage1_compile'].append(False)
            print(f"  ❌ Compilation failed: {str(e)[:50]}")
            continue

        results['stage1_compile'].append(True)

        # ===== 阶段2: 形状验证 =====
        print("\n📐 Stage 2: Shape Check")

        # 加载模型类
        namespace = {}
        try:
            exec(code, namespace)
            model_class = None
            for obj in namespace.values():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                    model_class = obj
                    break

            if model_class is None:
                raise ValueError("No valid model class found")

            success, error = stage2_shape_check(model_class, api_contract)
            sample_result['stages']['shape'] = {'success': success, 'error': error}

            if not success:
                results['stage2_shape'].append(False)
                print(f"  ❌ Shape check failed: {error}")
                continue

            results['stage2_shape'].append(True)
            print("  ✅ Shape check passed")

        except Exception as e:
            sample_result['stages']['shape'] = {'success': False, 'error': str(e)}
            results['stage2_shape'].append(False)
            print(f"  ❌ Shape check failed: {str(e)[:50]}")
            continue

        # ===== 阶段3: 训练验证 (关键!) =====
        print("\n🎓 Stage 3: Training Check (3 epochs)")
        success, error, loss_history = stage3_training_check(
            model_class, train_loader, val_loader,
            num_epochs=3, min_improvement=0.01
        )

        sample_result['stages']['training'] = {
            'success': success,
            'error': error,
            'loss_history': loss_history
        }

        if not success:
            results['stage3_training'].append(False)
            print(f"  ❌ Training failed: {error}")
            continue

        results['stage3_training'].append(True)
        results['loss_histories'].append(loss_history)
        print(f"  ✅ Training passed (loss: {loss_history[0]:.4f} → {loss_history[-1]:.4f})")

        # ===== 阶段4: 推理验证 =====
        print("\n🚀 Stage 4: Inference Check")
        success, error, accuracy = stage4_inference_check(model_class, val_loader)

        sample_result['stages']['inference'] = {
            'success': success,
            'error': error,
            'accuracy': accuracy
        }

        if not success:
            results['stage4_inference'].append(False)
            print(f"  ❌ Inference failed: {error}")
            continue

        results['stage4_inference'].append(True)
        results['accuracies'].append(accuracy)
        print(f"  ✅ Inference passed (accuracy: {accuracy:.2%})")

        # 端到端成功!
        results['end2end_success'].append(True)
        print("\n🎉 End-to-end SUCCESS!")

    # 统计结果
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    total = num_samples

    # 各阶段成功率
    stage1_rate = sum(results['stage1_compile']) / total
    stage2_rate = sum(results['stage2_shape']) / total
    stage3_rate = sum(results['stage3_training']) / total
    stage4_rate = sum(results['stage4_inference']) / total
    end2end_rate = sum(results['end2end_success']) / total

    print(f"\nTotal samples: {total}")
    print(f"\nStage 1 (Compile):    {sum(results['stage1_compile'])}/{total} ({stage1_rate*100:.1f}%)")
    print(f"Stage 2 (Shape):      {sum(results['stage2_shape'])}/{total} ({stage2_rate*100:.1f}%)")
    print(f"Stage 3 (Training):   {sum(results['stage3_training'])}/{total} ({stage3_rate*100:.1f}%)")  # 关键!
    print(f"Stage 4 (Inference):  {sum(results['stage4_inference'])}/{total} ({stage4_rate*100:.1f}%)")
    print(f"\n🎯 End-to-End Success: {sum(results['end2end_success'])}/{total} ({end2end_rate*100:.1f}%)")

    if results['accuracies']:
        print(f"\nAverage Accuracy: {np.mean(results['accuracies']):.2%}")

    # 保存结果
    output_dir = Path(config['experiment']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "end2end_results.json", 'w') as f:
        json.dump({
            'summary': {
                'total': total,
                'stage1_compile_rate': stage1_rate,
                'stage2_shape_rate': stage2_rate,
                'stage3_training_rate': stage3_rate,
                'stage4_inference_rate': stage4_rate,
                'end2end_success_rate': end2end_rate
            },
            'details': results
        }, f, indent=2)

    print(f"\n✅ Results saved to {output_dir / 'end2end_results.json'}")


def main():
    parser = argparse.ArgumentParser(description="Round 1: End-to-End Validation")
    parser.add_argument("--config", type=str, default="configs/round1_end2end.yaml",
                        help="Path to config file")

    args = parser.parse_args()

    if not os.environ.get('ALIYUN_API_KEY'):
        print("❌ Error: ALIYUN_API_KEY not set")
        sys.exit(1)

    try:
        run_end2end_validation(args.config)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

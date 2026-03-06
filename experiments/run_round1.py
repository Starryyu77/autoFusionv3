#!/usr/bin/env python3
"""
Round 1: 内循环验证实验

目标: 验证SelfHealingCompiler能将编译成功率从5%提升到95%

实验内容:
1. 生成100个初始代码
2. 每个代码最多3次修复尝试
3. 记录编译成功率
4. 分析涌现案例

用法:
    python experiments/run_round1.py --config configs/round1_inner_loop.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inner_loop.self_healing import SelfHealingCompiler, CompilationError
from utils.llm_backend import UnifiedLLMBackend
from utils.random_control import set_seed
from utils.logging_utils import ExperimentLogger


def create_code_generation_prompt(variant: int = 0) -> str:
    """
    创建代码生成prompt

    Args:
        variant: prompt变体编号，用于生成多样性代码

    Returns:
        prompt字符串
    """
    base_prompt = """
Generate a PyTorch nn.Module for multimodal fusion.

API Interface:
- Input 'vision': tensor of shape [batch, 576, 1024], dtype float32
- Input 'audio': tensor of shape [batch, 400, 512], dtype float32
- Input 'text': tensor of shape [batch, 77, 768], dtype float32
- Output: tensor of shape [batch, 10], dtype float32

Requirements:
1. The model must be a subclass of nn.Module
2. Must have a forward() method accepting the three inputs
3. Must handle missing modalities gracefully (check for None or zero tensors)
4. Use appropriate fusion mechanism (attention, gating, or concatenation)
5. Return tensor of correct shape [batch, 10]

Generate only the code, no explanation:
"""

    variants = [
        "",  # 基础版
        "\nHint: Consider using cross-attention between modalities.",
        "\nHint: Add a gating mechanism to control modality contribution.",
        "\nHint: Use residual connections for better gradient flow.",
        "\nHint: Include early exit for efficiency when confidence is high.",
    ]

    return base_prompt + variants[variant % len(variants)]


def analyze_emergent_patterns(code: str) -> Dict[str, any]:
    """
    分析代码中的涌现模式

    Returns:
        模式分析结果
    """
    patterns = {
        'has_conditionals': 'if ' in code or 'else:' in code,
        'has_modality_gate': 'confidence' in code or 'modality' in code.lower(),
        'has_early_exit': 'return' in code and 'if' in code,
        'has_attention': 'attention' in code.lower() or 'Attention' in code,
        'has_residual': 'residual' in code.lower() or '+' in code,
        'code_length': len(code),
        'num_lines': len(code.split('\n'))
    }

    # 检查是否有条件分支处理模态缺失
    if patterns['has_conditionals'] and patterns['has_modality_gate']:
        patterns['emergent_type'] = 'conditional_modality_gating'
    elif patterns['has_conditionals']:
        patterns['emergent_type'] = 'conditional_execution'
    else:
        patterns['emergent_type'] = 'static_fusion'

    return patterns


def run_round1_experiment(config_path: str, quick_mode: bool = False):
    """
    运行Round 1实验

    Args:
        config_path: 配置文件路径
        quick_mode: 快速模式（只测试10个样本）
    """
    print("=" * 60)
    print("Round 1: Inner Loop Validation")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Quick mode: {quick_mode}")
    print()

    # 1. 设置随机种子
    set_seed(42)

    # 2. 初始化组件
    print("🔧 Initializing components...")
    llm = UnifiedLLMBackend()

    api_contract = {
        'inputs': {
            'vision': {'shape': [2, 576, 1024], 'dtype': 'float32'},
            'audio': {'shape': [2, 400, 512], 'dtype': 'float32'},
            'text': {'shape': [2, 77, 768], 'dtype': 'float32'}
        },
        'output_shape': [2, 10],
        'model_kwargs': {'hidden_dim': 256}
    }

    compiler = SelfHealingCompiler(
        llm_backend=llm,
        max_retries=3,
        device='cpu'  # Round 1用CPU即可
    )

    # 3. 设置实验规模
    num_samples = 10 if quick_mode else 100
    print(f"📊 Experiment size: {num_samples} code samples")
    print()

    # 4. 运行实验
    results = {
        'successful': [],
        'failed': [],
        'attempts_distribution': [],
        'emergent_cases': []
    }

    for i in range(num_samples):
        print(f"\n{'-' * 60}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'-' * 60}")

        # 生成prompt
        prompt = create_code_generation_prompt(variant=i)

        try:
            # 尝试编译
            result = compiler.compile(prompt, api_contract, verbose=True)

            # 成功
            results['successful'].append({
                'sample_id': i,
                'attempts': result.attempts,
                'code': result.code,
                'metadata': result.metadata
            })
            results['attempts_distribution'].append(result.attempts)

            # 分析涌现模式
            patterns = analyze_emergent_patterns(result.code)
            if patterns['emergent_type'] != 'static_fusion':
                results['emergent_cases'].append({
                    'sample_id': i,
                    'patterns': patterns,
                    'code': result.code
                })
                print(f"🌟 Emergent pattern detected: {patterns['emergent_type']}")

        except CompilationError as e:
            # 失败
            results['failed'].append({
                'sample_id': i,
                'error_history': e.history
            })
            print(f"❌ Failed after max retries")

    # 5. 计算统计
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)

    total = len(results['successful']) + len(results['failed'])
    success_rate = len(results['successful']) / total if total > 0 else 0
    avg_attempts = np.mean(results['attempts_distribution']) if results['attempts_distribution'] else 0

    print(f"\nTotal samples: {total}")
    print(f"Successful: {len(results['successful'])} ({success_rate*100:.1f}%)")
    print(f"Failed: {len(results['failed'])} ({(1-success_rate)*100:.1f}%)")
    print(f"Average attempts per success: {avg_attempts:.2f}")
    print(f"Emergent cases: {len(results['emergent_cases'])}")

    # 6. 打印详细统计
    compiler.print_stats()

    # 7. 保存结果
    output_dir = Path("results/round1")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存JSON结果
    with open(output_dir / "results.json", 'w') as f:
        # 不保存完整代码以减小文件大小
        json.dump({
            'summary': {
                'total': total,
                'successful': len(results['successful']),
                'failed': len(results['failed']),
                'success_rate': success_rate,
                'avg_attempts': avg_attempts,
                'emergent_cases_count': len(results['emergent_cases'])
            },
            'attempts_distribution': results['attempts_distribution'],
            'emergent_cases': [
                {
                    'sample_id': c['sample_id'],
                    'patterns': c['patterns']
                }
                for c in results['emergent_cases']
            ]
        }, f, indent=2)

    # 8. 绘制图表
    plot_results(results, output_dir)

    # 9. 保存涌现案例代码
    if results['emergent_cases']:
        emergent_dir = output_dir / "emergent_cases"
        emergent_dir.mkdir(exist_ok=True)

        for i, case in enumerate(results['emergent_cases'][:5]):  # 保存前5个
            with open(emergent_dir / f"emergent_case_{i+1}.py", 'w') as f:
                f.write(f"# Emergent Case {i+1}\n")
                f.write(f"# Type: {case['patterns']['emergent_type']}\n")
                f.write(f"# Sample ID: {case['sample_id']}\n\n")
                f.write(case['code'])

    print(f"\n✅ Results saved to {output_dir}")

    return results


def plot_results(results: Dict, output_dir: Path):
    """绘制结果图表"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 图1: 编译成功率
    total = len(results['successful']) + len(results['failed'])
    success = len(results['successful'])
    failed = len(results['failed'])

    axes[0].bar(['Success', 'Failed'], [success, failed], color=['green', 'red'], alpha=0.7)
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Compilation Success Rate: {success/total*100:.1f}%')
    axes[0].grid(axis='y', alpha=0.3)

    # 图2: 尝试次数分布
    if results['attempts_distribution']:
        axes[1].hist(results['attempts_distribution'], bins=[0.5, 1.5, 2.5, 3.5],
                     color='blue', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Number of Attempts')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Compilation Attempts')
        axes[1].set_xticks([1, 2, 3])
        axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "compile_rate_analysis.png", dpi=150)
    plt.close()

    print(f"📊 Plots saved to {output_dir / 'compile_rate_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description="Round 1: Inner Loop Validation")
    parser.add_argument("--config", type=str, default="configs/round1_inner_loop.yaml",
                        help="Path to config file")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode (only 10 samples)")

    args = parser.parse_args()

    # 检查API密钥
    if not os.environ.get('ALIYUN_API_KEY'):
        print("❌ Error: ALIYUN_API_KEY environment variable not set")
        print("Please set it with: export ALIYUN_API_KEY='your-key'")
        sys.exit(1)

    # 运行实验
    try:
        results = run_round1_experiment(args.config, quick_mode=args.quick)
        print("\n🎉 Round 1 experiment completed successfully!")

    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

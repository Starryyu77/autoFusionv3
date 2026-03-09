"""
批量运行所有基线实验

在三个数据集上运行所有基线方法的特定实验
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List
import json

# 基线方法列表
BASELINE_METHODS = [
    # 简单基线
    'mean',
    'concat',
    'attention',
    'max',
    # 固定架构基线
    'dynmm',
    'tfn',
    'admn',
    'centaur',
    'fdsnet',
    # NAS基线（成本高，可单独运行）
    'darts',
    'llmatic',
    'evoprompting',
]

DATASETS = ['mosei', 'iemocap', 'vqa']


def run_baseline_experiment(
    method: str,
    dataset: str,
    device: str = 'cuda',
    output_dir: str = 'results/baselines_specific'
) -> bool:
    """
    运行单个基线实验

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*70}")
    print(f"Running {method.upper()} on {dataset.upper()}")
    print('='*70)

    cmd = [
        'python', 'experiments/run_baseline_specific.py',
        '--method', method,
        '--dataset', dataset,
        '--device', device,
        '--output_dir', output_dir
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {method} on {dataset} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {method} on {dataset} failed: {e}")
        return False


def run_all_experiments(
    methods: List[str] = None,
    datasets: List[str] = None,
    device: str = 'cuda',
    output_dir: str = 'results/baselines_specific',
    parallel: bool = False
):
    """运行所有实验"""

    if methods is None:
        methods = BASELINE_METHODS
    if datasets is None:
        datasets = DATASETS

    print(f"\n{'='*70}")
    print(f"Running {len(methods)} methods on {len(datasets)} datasets")
    print(f"Total experiments: {len(methods) * len(datasets)}")
    print('='*70)

    results = {}

    for method in methods:
        results[method] = {}
        for dataset in datasets:
            success = run_baseline_experiment(method, dataset, device, output_dir)
            results[method][dataset] = 'success' if success else 'failed'

    # 汇总结果
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)

    total = len(methods) * len(datasets)
    success_count = sum(1 for m in results.values() for v in m.values() if v == 'success')

    print(f"\nTotal: {total}, Success: {success_count}, Failed: {total - success_count}")

    # 保存汇总
    summary_file = Path(output_dir) / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Run all baseline experiments')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to run (default: all)')
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=['mosei', 'iemocap', 'vqa'],
                        help='Datasets to run (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='results/baselines_specific',
                        help='Output directory')
    parser.add_argument('--simple_only', action='store_true',
                        help='Run only simple baselines (no NAS)')
    parser.add_argument('--nas_only', action='store_true',
                        help='Run only NAS baselines')

    args = parser.parse_args()

    # 确定要运行的方法
    if args.simple_only:
        methods = ['mean', 'concat', 'attention', 'max', 'dynmm', 'tfn', 'admn', 'centaur', 'fdsnet']
    elif args.nas_only:
        methods = ['darts', 'llmatic', 'evoprompting']
    else:
        methods = args.methods or BASELINE_METHODS

    datasets = args.datasets or DATASETS

    # 运行实验
    run_all_experiments(
        methods=methods,
        datasets=datasets,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

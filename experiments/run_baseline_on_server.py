#!/usr/bin/env python3
"""
在NTU-GPU43服务器上运行基线实验

使用方法:
    python run_baseline_on_server.py --method mean --dataset mosei --gpu 0
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(method: str, dataset: str, gpu: int = 0, seeds: list = None):
    """运行单个实验"""

    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # 构建命令
    cmd = [
        'python', 'experiments/run_baseline_specific.py',
        '--method', method,
        '--dataset', dataset,
        '--device', 'cuda',
        '--output_dir', f'results/baselines_{method}'
    ]

    print(f"\n{'='*70}")
    print(f"Running {method.upper()} on {dataset.upper()} (GPU {gpu})")
    print('='*70)
    print(f"Command: {' '.join(cmd)}")
    print(f"Seeds: {seeds}")
    print('')

    # 运行
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✅ {method} on {dataset} completed successfully")
    else:
        print(f"\n❌ {method} on {dataset} failed with return code {result.returncode}")

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run baseline experiments on NTU-GPU43')
    parser.add_argument('--method', type=str, required=True,
                        help='Baseline method (mean, concat, attention, max, dynmm, tfn, admn, centaur, fdsnet, darts, llmatic, evoprompting)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['mosei', 'iemocap', 'vqa'],
                        help='Dataset name')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (0-3)')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='Random seeds to use')

    args = parser.parse_args()

    # 检查GPU可用性
    import torch
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    else:
        print(f"✅ Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    # 运行实验
    success = run_experiment(args.method, args.dataset, args.gpu, args.seeds)

    if success:
        print(f"\n✅ Experiment completed successfully")
    else:
        print(f"\n❌ Experiment failed")
        sys.exit(1)


if __name__ == '__main__':
    main()

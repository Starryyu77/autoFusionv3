"""
随机性控制模块

确保所有实验的可重复性
"""

import random
import numpy as np
import torch
from typing import Optional

# 实验使用的5个固定种子
# 来源: AutoFusion 2.0验证有效的种子集合
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]


def set_seed(seed: int, deterministic: bool = True):
    """
    设置全局随机种子，确保实验可重复性

    这是所有实验开始前必须调用的函数。
    会设置Python、NumPy、PyTorch的所有随机种子。

    Args:
        seed: 随机种子 (使用EXPERIMENT_SEEDS中的值)
        deterministic: 是否使用确定性算法(会影响性能但确保可重复)

    Example:
        >>> from utils.random_control import set_seed, EXPERIMENT_SEEDS
        >>> for seed in EXPERIMENT_SEEDS:
        ...     set_seed(seed)
        ...     run_experiment()
    """
    # Python内置random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性设置
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"✅ Random seed set: {seed} (deterministic={deterministic})")


def verify_seed_reproducibility(seed: int = 42) -> bool:
    """
    验证相同种子是否产生相同结果

    用于测试随机性控制是否正常工作。

    Args:
        seed: 测试用的种子

    Returns:
        bool: 验证是否通过
    """
    # 第一次生成
    set_seed(seed)
    r1_python = random.random()
    r1_numpy = np.random.rand()
    r1_torch = torch.rand(1).item()

    # 第二次生成(相同种子)
    set_seed(seed)
    r2_python = random.random()
    r2_numpy = np.random.rand()
    r2_torch = torch.rand(1).item()

    # 验证
    passed = (
        r1_python == r2_python and
        r1_numpy == r2_numpy and
        abs(r1_torch - r2_torch) < 1e-6
    )

    if passed:
        print("✅ Seed reproducibility verified")
    else:
        print("❌ Seed reproducibility failed!")
        print(f"  Python: {r1_python} vs {r2_python}")
        print(f"  NumPy: {r1_numpy} vs {r2_numpy}")
        print(f"  PyTorch: {r1_torch} vs {r2_torch}")

    return passed


def get_generator(seed: int):
    """
    获取独立的随机数生成器

    用于需要独立随机流的场景(如DataLoader worker)

    Args:
        seed: 随机种子

    Returns:
        torch.Generator: PyTorch随机数生成器
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


if __name__ == "__main__":
    print("Testing random control...")

    # 验证种子可重复性
    verify_seed_reproducibility(42)

    # 测试不同种子产生不同结果
    set_seed(42)
    r1 = torch.rand(3)

    set_seed(123)
    r2 = torch.rand(3)

    print(f"\nSeed 42: {r1}")
    print(f"Seed 123: {r2}")
    print(f"Different results: {not torch.allclose(r1, r2)}")

"""
LLMatic 基线方法实现

基于 LLM + 质量多样性 (Quality Diversity) 的神经架构搜索
参考论文: LLMatic - Neuroevolution through the lens of Large Language Models (GECCO 2024)

核心思想:
1. 使用 LLM 生成多个候选架构
2. 基于行为特征 (behavioral descriptors) 进行多样性选择
3. 在行为空间中进行 MAP-Elites 优化
"""

import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_backend import UnifiedLLMBackend, LLMResponse


@dataclass
class Solution:
    """解的数据结构"""
    code: str
    metrics: Dict[str, float]
    fitness: float = 0.0
    descriptor: Tuple[int, ...] = field(default_factory=tuple)
    generation: int = 0


class MAPElitesArchive:
    """
    MAP-Elites 归档

    将解映射到行为空间网格中，每个 bin 保存最优解
    """

    def __init__(self, bins_per_dim: int = 5, behavior_ranges: Optional[Dict] = None):
        """
        初始化 MAP-Elites 归档

        Args:
            bins_per_dim: 每个维度的分桶数量
            behavior_ranges: 行为描述符的范围
        """
        self.bins_per_dim = bins_per_dim
        self.behavior_ranges = behavior_ranges or {
            "accuracy": (0.0, 1.0),
            "flops": (0, 50000000)
        }
        self.grid: Dict[Tuple[int, ...], Solution] = {}

    def add(self, solution: Dict[str, Any], descriptor: Tuple[int, ...],
            fitness: float) -> bool:
        """
        添加解到归档

        Args:
            solution: 解字典，包含 code 和 metrics
            descriptor: 行为描述符 (bin indices)
            fitness: 适应度值

        Returns:
            bool: 是否成功添加（新解更好或 bin 为空）
        """
        # 创建 Solution 对象
        sol_obj = Solution(
            code=solution["code"],
            metrics=solution.get("metrics", {}),
            fitness=fitness,
            descriptor=descriptor
        )

        # 检查该 bin 是否已有解
        if descriptor not in self.grid:
            self.grid[descriptor] = sol_obj
            return True

        # 比较 fitness，保留更好的
        existing_fitness = self.grid[descriptor].fitness
        if fitness > existing_fitness:
            self.grid[descriptor] = sol_obj
            return True

        return False

    def get_elites(self) -> List[Solution]:
        """获取所有精英解"""
        return list(self.grid.values())

    def sample(self, num_samples: int = 1) -> List[Solution]:
        """随机采样解"""
        elites = self.get_elites()
        if not elites:
            return []
        return random.sample(elites, min(num_samples, len(elites)))

    def get_best(self) -> Optional[Solution]:
        """获取最佳解"""
        elites = self.get_elites()
        if not elites:
            return None
        return max(elites, key=lambda s: s.fitness)

    def coverage(self) -> float:
        """计算归档覆盖率"""
        total_bins = self.bins_per_dim ** 2  # 假设2维行为空间
        return len(self.grid) / total_bins


def compute_behavior_descriptor(
    metrics: Dict[str, float],
    num_bins: int = 5,
    behavior_ranges: Optional[Dict] = None
) -> Tuple[int, int]:
    """
    计算行为描述符

    将性能指标映射到离散的行为空间 bin

    Args:
        metrics: 包含 accuracy, flops 等的字典
        num_bins: 每个维度的分桶数
        behavior_ranges: 行为描述符的范围

    Returns:
        Tuple[int, int]: (accuracy_bin, flops_bin)
    """
    ranges = behavior_ranges or {
        "accuracy": (0.0, 1.0),
        "flops": (0, 50000000)
    }

    # 获取 accuracy 并分桶
    acc = metrics.get("accuracy", 0.5)
    acc_min, acc_max = ranges.get("accuracy", (0.0, 1.0))
    acc_clamped = max(acc_min, min(acc, acc_max))
    acc_bin = int((acc_clamped - acc_min) / (acc_max - acc_min) * (num_bins - 1))
    acc_bin = max(0, min(acc_bin, num_bins - 1))

    # 获取 flops 并分桶（取对数处理大范围）
    flops = metrics.get("flops", 10000000)
    flops_min, flops_max = ranges.get("flops", (0, 50000000))

    # 使用对数刻度处理 flops 的大范围
    if flops <= 0:
        flops_bin = 0
    else:
        flops_log = np.log10(flops)
        flops_min_log = np.log10(max(1, flops_min))
        flops_max_log = np.log10(flops_max)
        flops_bin = int((flops_log - flops_min_log) /
                       (flops_max_log - flops_min_log) * (num_bins - 1))
        flops_bin = max(0, min(flops_bin, num_bins - 1))

    return (acc_bin, flops_bin)


def select_diverse_parents(
    archive: Dict[Tuple[int, ...], Solution],
    num_parents: int = 2
) -> List[Solution]:
    """
    选择多样性的父代

    优先选择在行为空间上距离较远的解

    Args:
        archive: MAP-Elites 归档
        num_parents: 需要选择的父代数量

    Returns:
        List[Solution]: 选择的父代列表
    """
    if not archive:
        return []

    solutions = list(archive.values())

    if len(solutions) <= num_parents:
        return solutions

    # 随机选择第一个父代
    selected = [random.choice(solutions)]

    # 选择距离当前选择最远的解
    while len(selected) < num_parents:
        max_distance = -1
        farthest = None

        for sol in solutions:
            if sol in selected:
                continue

            # 计算到已选解的最小距离
            min_dist = min(
                np.linalg.norm(np.array(sol.descriptor) - np.array(s.descriptor))
                for s in selected
            )

            if min_dist > max_distance:
                max_distance = min_dist
                farthest = sol

        if farthest:
            selected.append(farthest)

    return selected


class LLMatic:
    """
    LLMatic 基线方法

    基于 LLM 和质量多样性的神经架构搜索
    """

    def __init__(
        self,
        population_size: int = 20,
        num_iterations: int = 50,
        api_contract: Optional[Dict] = None,
        behavior_bins: int = 5,
        llm_backend: Optional[UnifiedLLMBackend] = None
    ):
        """
        初始化 LLMatic

        Args:
            population_size: 每代生成的架构数量
            num_iterations: 迭代次数
            api_contract: API契约定义输入输出
            behavior_bins: 行为空间分桶数
            llm_backend: LLM后端实例
        """
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.api_contract = api_contract or {"inputs": {}, "output_shape": [2, 10]}
        self.behavior_bins = behavior_bins

        # 初始化 LLM 后端 (延迟初始化以支持测试)
        self._llm_backend = llm_backend
        self._llm = None

        # 初始化 MAP-Elites 归档
        self.archive = MAPElitesArchive(bins_per_dim=behavior_bins)

        # 历史记录
        self.history: List[Dict] = []
        self.current_iteration = 0

    @property
    def llm(self):
        """Lazy initialization of LLM backend"""
        if self._llm is None:
            if self._llm_backend is not None:
                self._llm = self._llm_backend
            else:
                self._llm = UnifiedLLMBackend()
        return self._llm

    def _build_generation_prompt(
        self,
        behavioral_descriptor: str = ""
    ) -> str:
        """
        构建架构生成提示词

        Args:
            behavioral_descriptor: 目标行为描述符

        Returns:
            str: 提示词
        """
        base_prompt = f"""You are an expert neural architecture designer.

Task: Design a multimodal fusion architecture using PyTorch.

API Contract:
- Vision input: shape {self.api_contract.get('inputs', {}).get('vision', {}).get('shape', 'N/A')}
- Audio input: shape {self.api_contract.get('inputs', {}).get('audio', {}).get('shape', 'N/A')}
- Text input: shape {self.api_contract.get('inputs', {}).get('text', {}).get('shape', 'N/A')}
- Output: shape {self.api_contract.get('output_shape', 'N/A')}

Requirements:
1. Create a class inheriting from nn.Module
2. Implement __init__ and forward methods
3. Handle multimodal fusion appropriately
4. Use standard PyTorch layers only

{f"Design goal: {behavioral_descriptor}" if behavioral_descriptor else ""}

Generate the complete Python code:"""

        return base_prompt

    def _generate_architecture(
        self,
        target_descriptor: Optional[str] = None
    ) -> Optional[str]:
        """
        使用 LLM 生成架构代码

        Args:
            target_descriptor: 目标行为描述符（用于引导生成）

        Returns:
            Optional[str]: 生成的代码或 None
        """
        prompt = self._build_generation_prompt(target_descriptor or "")

        try:
            response = self.llm.generate(prompt)
            return response.code if response else None
        except Exception as e:
            print(f"  ⚠️ LLM generation failed: {e}")
            return None

    def search_one_iteration(
        self,
        evaluator: Callable[[str], Dict[str, float]],
        use_mock_llm: bool = False
    ) -> Dict[str, Any]:
        """
        执行一次搜索迭代

        Args:
            evaluator: 评估函数，输入代码，返回指标字典
            use_mock_llm: 是否使用 mock LLM（用于测试）

        Returns:
            Dict: 迭代结果统计
        """
        self.current_iteration += 1
        generated_count = 0
        added_count = 0

        for i in range(self.population_size):
            # 确定目标行为描述符（从稀疏区域采样）
            target_desc = self._sample_target_descriptor()

            # 生成架构
            if use_mock_llm:
                # Mock 模式：生成随机代码
                code = self._mock_generate_code()
            else:
                code = self._generate_architecture(target_desc)

            if not code:
                continue

            generated_count += 1

            # 评估架构
            try:
                metrics = evaluator(code)
                fitness = metrics.get("fitness", metrics.get("accuracy", 0.0))

                # 计算行为描述符
                descriptor = compute_behavior_descriptor(
                    metrics, num_bins=self.behavior_bins
                )

                # 添加到归档
                solution = {"code": code, "metrics": metrics}
                added = self.archive.add(solution, descriptor, fitness)
                if added:
                    added_count += 1

            except Exception as e:
                print(f"  ⚠️ Evaluation failed: {e}")
                continue

        # 记录统计
        result = {
            "iteration": self.current_iteration,
            "generated": generated_count,
            "added": added_count,
            "archive_size": len(self.archive.grid),
            "coverage": self.archive.coverage(),
            "best_fitness": self.archive.get_best().fitness if self.archive.get_best() else 0.0
        }
        self.history.append(result)

        return result

    def _sample_target_descriptor(self) -> str:
        """
        采样目标行为描述符

        优先选择归档中稀疏的区域
        """
        if not self.archive.grid:
            return "high_accuracy"

        # 简单策略：随机选择一个现有解的邻居
        existing = random.choice(list(self.archive.grid.values()))
        acc_bin, flops_bin = existing.descriptor

        # 添加随机扰动
        new_acc = max(0, min(self.behavior_bins - 1, acc_bin + random.randint(-1, 1)))
        new_flops = max(0, min(self.behavior_bins - 1, flops_bin + random.randint(-1, 1)))

        desc_map = {0: "low", 1: "medium-low", 2: "medium",
                   3: "medium-high", 4: "high"}

        return f"{desc_map.get(new_acc, 'medium')} accuracy, {desc_map.get(new_flops, 'medium')} efficiency"

    def _mock_generate_code(self) -> str:
        """生成 mock 代码（用于测试）"""
        templates = [
            """import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 10)
    def forward(self, vision, audio, text):
        return self.fc(vision.mean(dim=1))
""",
            """import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8)
        self.fc = nn.Linear(256, 10)
    def forward(self, vision, audio, text):
        out, _ = self.attn(vision, vision, vision)
        return self.fc(out.mean(dim=1))
"""
        ]
        return random.choice(templates)

    def search(
        self,
        evaluator: Callable[[str], Dict[str, float]],
        verbose: bool = True
    ) -> Solution:
        """
        执行完整搜索

        Args:
            evaluator: 评估函数
            verbose: 是否打印进度

        Returns:
            Solution: 最佳解
        """
        if verbose:
            print("=" * 70)
            print("LLMatic Search Started")
            print("=" * 70)
            print(f"Population size: {self.population_size}")
            print(f"Iterations: {self.num_iterations}")
            print(f"Behavior bins: {self.behavior_bins}")
            print()

        for iteration in range(1, self.num_iterations + 1):
            result = self.search_one_iteration(evaluator)

            if verbose and iteration % 10 == 0:
                print(f"  Iteration {iteration}/{self.num_iterations}: "
                      f"archive_size={result['archive_size']}, "
                      f"best_fitness={result['best_fitness']:.3f}")

        best = self.archive.get_best()

        if verbose:
            print()
            print("=" * 70)
            print("Search Complete")
            print(f"Archive size: {len(self.archive.grid)}")
            print(f"Coverage: {self.archive.coverage():.2%}")
            print(f"Best fitness: {best.fitness if best else 'N/A'}")
            print("=" * 70)

        return best


# 向后兼容的导出
__all__ = [
    'LLMatic',
    'MAPElitesArchive',
    'Solution',
    'compute_behavior_descriptor',
    'select_diverse_parents'
]

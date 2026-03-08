"""
EvoPrompting 基线方法实现

进化提示工程 - 在提示词层面进行进化优化
参考论文: EvoPrompting - Large Language Models as Hyper-Heuristics for Neuroevolution (NeurIPS 2023)
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_backend import UnifiedLLMBackend


@dataclass
class PromptGene:
    """提示词基因"""
    content: str
    fitness: float = 0.0
    generation: int = 0


def mutate_prompt(prompt: str, mutation_rate: float = 0.3) -> str:
    """
    变异提示词

    策略:
    1. 添加修饰词
    2. 替换关键词
    3. 改变结构
    """
    mutations = [
        " with attention mechanism",
        " using residual connections",
        " with layer normalization",
        " optimized for efficiency",
        " with multi-scale features",
        " using adaptive pooling",
        " with dropout regularization",
        " using batch normalization"
    ]

    words = prompt.split()

    # 随机添加修饰
    if random.random() < mutation_rate:
        prompt += random.choice(mutations)

    # 随机替换关键词
    if random.random() < mutation_rate and len(words) > 3:
        replacements = {
            "CNN": ["ConvNet", "Convolutional Network"],
            "Transformer": ["Attention Network", "Self-Attention Model"],
            "LSTM": ["GRU", "RNN", "Recurrent Network"],
            "large": ["deep", "wide", "hierarchical"],
            "small": ["shallow", "compact", "efficient"]
        }

        for i, word in enumerate(words):
            if word in replacements and random.random() < 0.5:
                words[i] = random.choice(replacements[word])

    return " ".join(words) if words else prompt


def crossover_prompts(parent1: str, parent2: str) -> str:
    """
    交叉两个提示词

    策略: 取前半部分 + 后半部分组合
    """
    words1 = parent1.split()
    words2 = parent2.split()

    if not words1 or not words2:
        return parent1 or parent2

    # 随机选择交叉点
    cross_point1 = len(words1) // 2
    cross_point2 = len(words2) // 2

    # 组合
    child_words = words1[:cross_point1] + words2[cross_point2:]

    return " ".join(child_words)


def tournament_selection(
    population: List[Dict],
    tournament_size: int = 3
) -> Dict:
    """
    锦标赛选择

    从随机子集中选择最优个体
    """
    if not population:
        raise ValueError("Empty population")

    tournament = random.sample(population, min(tournament_size, len(population)))

    # 选择 fitness 最高的
    return max(tournament, key=lambda x: x.get("fitness", 0.0))


class EvoPrompting:
    """
    EvoPrompting 基线方法

    在提示词空间中进行进化搜索
    """

    def __init__(
        self,
        population_size: int = 20,
        num_iterations: int = 50,
        api_contract: Optional[Dict] = None,
        llm_backend: Optional[UnifiedLLMBackend] = None
    ):
        self.population_size = population_size
        self.num_iterations = num_iterations
        self.api_contract = api_contract or {"inputs": {}, "output_shape": [2, 10]}

        self._llm_backend = llm_backend
        self._llm = None

        self.population: List[PromptGene] = []
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

    def _initialize_population(self):
        """初始化种群"""
        base_prompts = [
            "Generate a neural network for multimodal fusion",
            "Create a deep learning model with attention",
            "Design a multimodal architecture using transformers",
            "Build a fusion network with residual connections",
            "Generate an efficient model with layer normalization",
            "Create a robust architecture with dropout",
            "Design a hierarchical multimodal network",
            "Build a cross-modal attention model"
        ]

        self.population = [
            PromptGene(content=prompt)
            for prompt in random.sample(base_prompts, min(len(base_prompts), self.population_size))
        ]

        # 如果不够，随机变异补充
        while len(self.population) < self.population_size:
            parent = random.choice(self.population)
            child_content = mutate_prompt(parent.content, mutation_rate=0.5)
            self.population.append(PromptGene(content=child_content))

    def _generate_from_prompt(self, prompt: str, use_mock: bool = False) -> Optional[str]:
        """从提示词生成架构代码"""
        if use_mock:
            return self._mock_generate()

        full_prompt = f"""{prompt}

Requirements:
- Create a PyTorch nn.Module class
- Handle multimodal inputs: vision, audio, text
- Output shape: {self.api_contract.get('output_shape', [2, 10])}
- Use standard PyTorch layers only

Generate the complete Python code:"""

        try:
            response = self.llm.generate(full_prompt)
            return response.code if response else None
        except Exception as e:
            print(f"  ⚠️ LLM generation failed: {e}")
            return None

    def _mock_generate(self) -> str:
        """Mock 代码生成"""
        templates = [
            """import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 10)
    def forward(self, vision, audio, text):
        v = vision.mean(dim=1)
        return self.fc(v)
""",
            """import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(256, 8)
        self.fc = nn.Linear(256, 10)
    def forward(self, vision, audio, text):
        v = vision.mean(dim=1).unsqueeze(0)
        out, _ = self.attn(v, v, v)
        return self.fc(out.squeeze(0))
"""
        ]
        return random.choice(templates)

    def _evolve_one_generation(
        self,
        evaluator: Callable[[str], Dict],
        use_mock: bool = False
    ):
        """进化一代"""
        self.current_iteration += 1

        # 评估当前种群
        for gene in self.population:
            if gene.fitness == 0.0:  # 未评估
                code = self._generate_from_prompt(gene.content, use_mock)
                if code:
                    try:
                        metrics = evaluator(code)
                        gene.fitness = metrics.get("fitness", metrics.get("accuracy", 0.0))
                    except Exception as e:
                        gene.fitness = 0.0

        # 选择
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:max(2, self.population_size // 4)]

        # 生成下一代
        new_population = list(elites)  # 保留精英

        while len(new_population) < self.population_size:
            # 选择父代
            parent_dict = tournament_selection(
                [{"content": g.content, "fitness": g.fitness} for g in self.population],
                tournament_size=3
            )

            # 变异或交叉
            if random.random() < 0.7:  # 70% 变异
                child_content = mutate_prompt(parent_dict["content"])
            else:  # 30% 交叉
                parent2_dict = tournament_selection(
                    [{"content": g.content, "fitness": g.fitness} for g in self.population],
                    tournament_size=3
                )
                child_content = crossover_prompts(
                    parent_dict["content"],
                    parent2_dict["content"]
                )

            new_population.append(PromptGene(
                content=child_content,
                generation=self.current_iteration
            ))

        self.population = new_population

        # 记录统计
        best = max(self.population, key=lambda x: x.fitness)
        self.history.append({
            "iteration": self.current_iteration,
            "best_fitness": best.fitness,
            "avg_fitness": sum(g.fitness for g in self.population) / len(self.population),
            "best_prompt": best.content[:50]
        })

    def search(
        self,
        evaluator: Callable[[str], Dict],
        use_mock_llm: bool = False,
        verbose: bool = True
    ) -> Optional[PromptGene]:
        """
        执行完整搜索
        """
        if verbose:
            print("=" * 70)
            print("EvoPrompting Search Started")
            print("=" * 70)

        # 初始化
        self._initialize_population()

        if verbose:
            print(f"Initial population: {len(self.population)}")

        # 进化
        for _ in range(self.num_iterations):
            self._evolve_one_generation(evaluator, use_mock_llm)

            if verbose and self.current_iteration % 10 == 0:
                best = max(self.population, key=lambda x: x.fitness)
                print(f"  Iteration {self.current_iteration}: best_fitness={best.fitness:.3f}")

        # 返回最佳
        best = max(self.population, key=lambda x: x.fitness)

        if verbose:
            print()
            print("=" * 70)
            print(f"Search Complete: best_fitness={best.fitness:.3f}")
            print("=" * 70)

        return best


__all__ = ['EvoPrompting', 'PromptGene', 'mutate_prompt', 'crossover_prompts', 'tournament_selection']

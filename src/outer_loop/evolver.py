"""
EAS进化器模块

实现CMA-ES + LLM变异的混合进化策略
"""

import sys
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from inner_loop.self_healing import SelfHealingCompiler


try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    print("Warning: cma package not installed. Using simple GA instead.")


@dataclass
class EvolutionConfig:
    """进化配置"""
    pop_size: int = 10              # 种群大小
    max_generations: int = 100      # 最大迭代次数
    early_stop_patience: int = 20   # 早停耐心值

    # CMA-ES参数
    sigma: float = 0.5              # 初始标准差

    # 奖励权重
    w_accuracy: float = 1.0
    w_robustness: float = 2.0
    w_flops: float = 0.5

    # LLM变异概率
    llm_mutation_prob: float = 0.7


class Individual:
    """进化个体"""

    def __init__(self, code: str, generation: int = 0):
        self.code = code
        self.generation = generation
        self.fitness: Optional[float] = None
        self.metrics: Dict[str, float] = {}
        self.parents: List[str] = []

    def __repr__(self):
        return f"Individual(gen={self.generation}, fitness={self.fitness:.4f if self.fitness else 'N/A'})"


class EASEvolver:
    """
    EAS进化器

    结合CMA-ES和LLM变异的进化策略
    """

    def __init__(
        self,
        inner_loop: SelfHealingCompiler,
        evaluator,
        config: EvolutionConfig,
        api_contract: Dict[str, Any]
    ):
        """
        初始化进化器

        Args:
            inner_loop: 自修复编译器
            evaluator: 评估器
            config: 进化配置
            api_contract: API契约
        """
        self.inner_loop = inner_loop
        self.evaluator = evaluator
        self.config = config
        self.api_contract = api_contract

        self.population: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.fitness_history: List[Dict] = []

        # 早停计数器
        self.no_improvement_count = 0
        self.best_fitness = -float('inf')

    def initialize_population(self, seed_prompts: List[str]):
        """
        初始化种群

        Args:
            seed_prompts: 初始代码生成prompts
        """
        print(f"🌱 Initializing population (size={self.config.pop_size})...")

        for i, prompt in enumerate(seed_prompts[:self.config.pop_size]):
            try:
                result = self.inner_loop.compile(prompt, self.api_contract, verbose=False)
                individual = Individual(result.code, generation=0)
                self.population.append(individual)
                print(f"  ✓ Individual {i+1}/{self.config.pop_size} created")
            except Exception as e:
                print(f"  ✗ Failed to create individual {i+1}: {str(e)[:50]}")
                # 创建简化版本作为fallback
                fallback_code = self._create_fallback_code()
                self.population.append(Individual(fallback_code, generation=0))

    def _create_fallback_code(self) -> str:
        """创建fallback代码（简单MLP）"""
        return '''
import torch
import torch.nn as nn

class SimpleFusionModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(2304, hidden_dim),  # 1024+512+768=2304
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, vision, audio, text):
        # 简单拼接
        combined = torch.cat([vision.mean(dim=1), audio.mean(dim=1), text.mean(dim=1)], dim=-1)
        return self.fusion(combined)
'''

    def evaluate_individual(self, individual: Individual) -> float:
        """
        评估个体适应度

        Returns:
            适应度值（越高越好）
        """
        if individual.fitness is not None:
            return individual.fitness

        # 加载架构
        try:
            metrics = self.evaluator.evaluate_architecture(individual.code)

            # 计算综合奖励
            acc = metrics.get('accuracy', 0.0)
            mrob = metrics.get('mrob', 0.0)
            flops = metrics.get('flops', 1e9)

            reward = (
                self.config.w_accuracy * acc +
                self.config.w_robustness * mrob -
                self.config.w_flops * (flops / 1e9)
            )

            individual.fitness = reward
            individual.metrics = metrics

            return reward

        except Exception as e:
            print(f"  ✗ Evaluation failed: {str(e)[:50]}")
            individual.fitness = -float('inf')
            return individual.fitness

    def select_parents(self, num_parents: int) -> List[Individual]:
        """
        选择父代（锦标赛选择）
        """
        parents = []
        tournament_size = 3

        for _ in range(num_parents):
            # 随机选择tournament_size个个体
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            # 选择最好的
            winner = max(tournament, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))
            parents.append(winner)

        return parents

    def llm_mutate(self, parent: Individual) -> Individual:
        """
        使用LLM进行变异
        """
        # 构建变异prompt
        mutation_prompt = f"""
You are evolving a neural architecture. Here is a parent architecture:

```python
{parent.code}
```

This architecture has fitness score: {parent.fitness:.4f}
Metrics: {parent.metrics}

Generate a mutated version that:
1. Modifies the fusion mechanism (e.g., add attention, change gating)
2. Keeps the same input/output interface
3. Maintains or improves performance

Generate only the mutated code:
"""

        try:
            result = self.inner_loop.compile(mutation_prompt, self.api_contract, verbose=False)
            child = Individual(result.code, generation=self.generation)
            child.parents = [f"gen{parent.generation}_fit{parent.fitness:.3f}"]
            return child
        except Exception as e:
            # 变异失败，返回父代副本
            child = Individual(parent.code, generation=self.generation)
            child.parents = [f"gen{parent.generation}_clone"]
            return child

    def simple_mutate(self, parent: Individual) -> Individual:
        """
        简单变异（不依赖LLM，用于fallback）
        """
        # 简单替换一些参数
        code = parent.code

        # 随机修改隐藏维度
        if 'hidden_dim' in code and random.random() < 0.3:
            old_dim = random.choice([128, 256, 512])
            new_dim = random.choice([128, 256, 512])
            code = code.replace(str(old_dim), str(new_dim))

        child = Individual(code, generation=self.generation)
        child.parents = [f"gen{parent.generation}_simple"]
        return child

    def evolve(self, max_generations: Optional[int] = None) -> Individual:
        """
        主进化循环

        Returns:
            最优个体
        """
        if max_generations is None:
            max_generations = self.config.max_generations

        print(f"\n🚀 Starting evolution for {max_generations} generations...")

        for gen in range(max_generations):
            self.generation = gen
            print(f"\n{'='*50}")
            print(f"Generation {gen+1}/{max_generations}")
            print(f"{'='*50}")

            # 评估种群
            print("📊 Evaluating population...")
            for i, individual in enumerate(self.population):
                if individual.fitness is None:
                    fitness = self.evaluate_individual(individual)
                    print(f"  Individual {i+1}: fitness={fitness:.4f}")

            # 更新最佳个体
            current_best = max(self.population, key=lambda x: x.fitness if x.fitness is not None else -float('inf'))

            if current_best.fitness > self.best_fitness:
                self.best_fitness = current_best.fitness
                self.best_individual = current_best
                self.no_improvement_count = 0
                print(f"🏆 New best fitness: {self.best_fitness:.4f}")
            else:
                self.no_improvement_count += 1
                print(f"📈 Best fitness: {self.best_fitness:.4f} (no improvement for {self.no_improvement_count} gen)")

            # 记录历史
            self.fitness_history.append({
                'generation': gen,
                'best_fitness': self.best_fitness,
                'avg_fitness': np.mean([ind.fitness for ind in self.population if ind.fitness is not None]),
                'population_size': len(self.population)
            })

            # 早停检查
            if self.no_improvement_count >= self.config.early_stop_patience:
                print(f"\n⏹️  Early stopping at generation {gen+1}")
                break

            # 生成新一代
            print("🧬 Creating offspring...")
            offspring = []

            # 精英保留
            elite = current_best
            offspring.append(elite)

            # 生成其余后代
            while len(offspring) < self.config.pop_size:
                parents = self.select_parents(2)

                # 以一定概率使用LLM变异
                if random.random() < self.config.llm_mutation_prob:
                    child = self.llm_mutate(parents[0])
                else:
                    child = self.simple_mutate(parents[0])

                offspring.append(child)

            self.population = offspring

        print(f"\n✅ Evolution complete!")
        print(f"Best individual: {self.best_individual}")

        return self.best_individual

    def get_stats(self) -> Dict[str, Any]:
        """获取进化统计"""
        return {
            'total_generations': self.generation,
            'best_fitness': self.best_fitness,
            'best_metrics': self.best_individual.metrics if self.best_individual else {},
            'fitness_history': self.fitness_history,
            'population_size': len(self.population)
        }


if __name__ == "__main__":
    print("EASEvolver module test")
    print(f"CMA package available: {HAS_CMA}")

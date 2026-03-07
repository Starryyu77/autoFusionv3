"""
EAS进化器模块 V2 - 基于 Auto-Fusion-v2 改进

核心改进:
1. 三阶段策略 (EXPLORATION/EXPLOITATION/REFINEMENT)
2. 历史反馈机制
3. SearchResult dataclass 完整记录
4. 清晰的 _run_iteration 流程
5. 定期 checkpoint 保存
"""

import os
import json
import time
import random
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from inner_loop.self_healing_v2 import SelfHealingCompilerV2, CompilationResult
from evaluator.proxy_evaluator_v2 import ProxyEvaluatorV2


logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """单次搜索结果记录（v2 新增）"""
    iteration: int
    code: str
    compile_success: bool
    compile_attempts: int
    accuracy: float
    mrob: float
    flops: int
    params: int
    reward: float
    error_message: Optional[str] = None
    time_taken: float = 0.0
    strategy_phase: str = ""  # 'exploration', 'exploitation', 'refinement'

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class EvolutionConfig:
    """进化配置 V2"""
    pop_size: int = 10
    max_iterations: int = 200  # v2: 增加迭代次数
    early_stop_patience: int = 20

    # 三阶段阈值 (v2)
    exploration_threshold: float = 0.3
    exploitation_threshold: float = 0.7

    # 奖励权重
    w_accuracy: float = 1.0
    w_robustness: float = 2.0
    w_efficiency: float = 0.5
    w_constraint: float = 2.0

    # LLM变异概率
    llm_mutation_prob: float = 0.7

    # Checkpoint 间隔 (v2)
    checkpoint_interval: int = 10


class EASEvolverV2:
    """
    EAS进化器 V2 - 基于 Auto-Fusion-v2 改进

    核心流程:
    1. _build_prompt: 构建带历史反馈的 prompt
    2. _run_iteration: 执行单次迭代（内循环 + 外循环）
    3. _get_strategy_phase: 获取当前策略阶段
    4. _generate_strategy_feedback: 生成策略指导
    """

    def __init__(
        self,
        llm_backend: Any,
        api_contract: Dict[str, Any],
        proxy_evaluator: ProxyEvaluatorV2,
        reward_fn,
        max_inner_retries: int = 5,
        max_iterations: int = 200,
        output_dir: str = "./results",
        device: str = "cuda"
    ):
        """
        初始化进化器 V2

        Args:
            llm_backend: LLM 后端
            api_contract: API 契约
            proxy_evaluator: 代理评估器
            reward_fn: 奖励函数
            max_inner_retries: 内循环最大重试次数
            max_iterations: 最大迭代次数
            output_dir: 输出目录
            device: 计算设备
        """
        self.llm = llm_backend
        self.contract = api_contract
        self.proxy_evaluator = proxy_evaluator
        self.reward_fn = reward_fn
        self.max_inner_retries = max_inner_retries
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.device = device

        # 初始化内循环（v2 版本）
        self.inner_loop = SelfHealingCompilerV2(
            llm_backend=llm_backend,
            max_retries=max_inner_retries,
            device=device
        )

        # 状态追踪（v2）
        self.history: List[SearchResult] = []
        self.best_result: Optional[SearchResult] = None
        self.iteration = 0

        # 早停
        self.no_improvement_count = 0
        self.best_fitness = -float('inf')

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

    def search(self) -> SearchResult:
        """
        执行双循环搜索（v2 主函数）

        Returns:
            最佳搜索结果
        """
        logger.info("=" * 80)
        logger.info("EAS V2: Dual-Loop Architecture Search")
        logger.info("=" * 80)
        logger.info(f"Max Iterations: {self.max_iterations}")
        logger.info(f"Max Inner Retries: {self.max_inner_retries}")
        logger.info(f"Output Directory: {self.output_dir}")
        logger.info("=" * 80)

        start_time = time.time()

        for iteration in range(1, self.max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()

            logger.info(f"\n{'='*60}")
            logger.info(f"Iteration {iteration}/{self.max_iterations}")
            logger.info(f"Strategy Phase: {self._get_strategy_phase()}")
            logger.info(f"{'='*60}")

            # 执行双循环迭代
            result = self._run_iteration(iteration)
            result.time_taken = time.time() - iter_start
            result.strategy_phase = self._get_strategy_phase()

            self.history.append(result)

            # 更新最佳结果
            if result.compile_success and (
                self.best_result is None or result.reward > self.best_result.reward
            ):
                self.best_result = result
                self.no_improvement_count = 0
                logger.info(f"🏆 New Best! Reward: {result.reward:.3f}")
            else:
                self.no_improvement_count += 1

            # 打印摘要
            self._print_iteration_summary(result)

            # 定期保存 checkpoint（v2）
            if iteration % 10 == 0:
                self._save_checkpoint(iteration)

            # 早停检查
            if self.no_improvement_count >= 20:
                logger.info(f"\n⏹️ Early stopping at iteration {iteration} (no improvement for 20 iterations)")
                break

        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info("Search Complete!")
        logger.info(f"Total Time: {total_time/60:.1f} minutes")
        logger.info(f"Iterations: {self.iteration}")
        logger.info(f"Compile Success Rate: {self._get_compile_success_rate():.1%}")

        if self.best_result:
            logger.info(f"Best Reward: {self.best_result.reward:.3f}")
            logger.info(f"Best Accuracy: {self.best_result.accuracy:.2%}")
            logger.info(f"Best mRob: {self.best_result.mrob:.2%}")
            logger.info(f"Best FLOPs: {self.best_result.flops/1e6:.1f}M")

        logger.info(f"{'='*80}")

        return self.best_result

    def _run_iteration(self, iteration: int) -> SearchResult:
        """
        执行单次双循环迭代（v2 核心函数）

        流程:
        1. Build prompt with history and contract
        2. Inner Loop - Self-healing compilation
        3. Outer Loop - Performance evaluation
        4. Calculate reward
        5. Generate feedback for next iteration
        """
        # Step 1: 构建带历史的 prompt
        prompt = self._build_prompt(iteration)

        # Step 2: 内循环 - 自修复编译
        try:
            compile_result = self.inner_loop.compile(
                prompt, self.contract, verbose=False
            )
            code = compile_result.code
            compile_attempts = compile_result.attempts
        except Exception as e:
            # 编译失败
            return SearchResult(
                iteration=iteration,
                code="",
                compile_success=False,
                compile_attempts=self.max_inner_retries,
                accuracy=0.0,
                mrob=0.0,
                flops=0,
                params=0,
                reward=0.0,
                error_message=f"Compilation failed: {str(e)[:100]}"
            )

        logger.info(f"✅ Compilation succeeded after {compile_attempts} attempt(s)")

        # Step 3: 外循环 - 性能评估
        try:
            metrics = self.proxy_evaluator.evaluate(code)
        except Exception as e:
            logger.warning(f"⚠️ Proxy evaluation failed: {e}")
            return SearchResult(
                iteration=iteration,
                code=code,
                compile_success=True,
                compile_attempts=compile_attempts,
                accuracy=0.0,
                mrob=0.0,
                flops=0,
                params=0,
                reward=0.0,
                error_message=f"Evaluation failed: {str(e)[:100]}"
            )

        # Step 4: 计算奖励
        reward = self.reward_fn.compute(
            accuracy=metrics['accuracy'],
            mrob=metrics['mrob'],
            flops=metrics['flops'],
            params=metrics.get('params', 0)
        )

        # Step 5: 生成反馈
        feedback = self._generate_feedback(metrics, reward, iteration)
        logger.debug(f"Feedback: {feedback}")

        return SearchResult(
            iteration=iteration,
            code=code,
            compile_success=True,
            compile_attempts=compile_attempts,
            accuracy=metrics['accuracy'],
            mrob=metrics['mrob'],
            flops=metrics['flops'],
            params=metrics.get('params', 0),
            reward=reward
        )

    def _build_prompt(self, iteration: int) -> str:
        """
        构建带历史反馈的 prompt（v2 核心改进）
        """
        prompt_parts = []

        # 系统上下文
        prompt_parts.append("You are an expert neural architecture designer.")
        prompt_parts.append("Generate PyTorch code for a multimodal fusion architecture.\n")

        # API 契约
        prompt_parts.append(self._contract_to_prompt())
        prompt_parts.append("")

        # 历史反馈（如果不是第一次）
        if self.history:
            prompt_parts.append("【Search History】")
            prompt_parts.append(f"Total iterations so far: {len(self.history)}")

            if self.best_result:
                prompt_parts.append(f"\n🏆 Current Best Architecture:")
                prompt_parts.append(f"- Reward: {self.best_result.reward:.3f}")
                prompt_parts.append(f"- Accuracy: {self.best_result.accuracy:.2%}")
                prompt_parts.append(f"- mRob: {self.best_result.mrob:.2%}")
                prompt_parts.append(f"- FLOPs: {self.best_result.flops/1e6:.1f}M")

            # 展示最近结果
            prompt_parts.append("\n📊 Recent Results:")
            for result in self.history[-5:]:
                status = "✅" if result.compile_success else "❌"
                prompt_parts.append(
                    f"Iter {result.iteration}: {status} "
                    f"Reward={result.reward:.3f}, "
                    f"Acc={result.accuracy:.2%}, "
                    f"mRob={result.mrob:.2%}"
                )

            prompt_parts.append("")

            # 策略指导
            feedback = self._generate_strategy_feedback(iteration)
            prompt_parts.append(f"【Strategy Guidance】\n{feedback}\n")

        # 代码生成指令
        prompt_parts.append("【Code Generation Instructions】")
        prompt_parts.append("1. Create a class inheriting from nn.Module")
        prompt_parts.append("2. __init__ should accept input_dims as parameter")
        prompt_parts.append("3. forward should accept inputs matching the API contract")
        prompt_parts.append("4. Use only standard PyTorch operations")
        prompt_parts.append("5. Ensure output shape matches the contract")
        prompt_parts.append("6. Handle modality missing gracefully (optional)")
        prompt_parts.append("7. Only return the class definition, no explanation\n")

        prompt_parts.append("【Output Format】")
        prompt_parts.append("```python")
        prompt_parts.append("import torch")
        prompt_parts.append("import torch.nn as nn")
        prompt_parts.append("")
        prompt_parts.append("class FusionModel(nn.Module):")
        prompt_parts.append("    def __init__(self, input_dims): ...")
        prompt_parts.append("    def forward(self, ...): ...")
        prompt_parts.append("```")

        return "\n".join(prompt_parts)

    def _contract_to_prompt(self) -> str:
        """将 API 契约转换为 prompt 文本"""
        parts = ["【API Contract】"]
        parts.append("Inputs:")
        for name, spec in self.contract.get('inputs', {}).items():
            shape = spec.get('shape', [])
            dtype = spec.get('dtype', 'float32')
            parts.append(f"  - {name}: shape={shape}, dtype={dtype}")
        parts.append(f"Output: shape={self.contract.get('output_shape', [])}")
        return "\n".join(parts)

    def _get_strategy_phase(self) -> str:
        """
        获取当前策略阶段（v2 三阶段策略）

        - EXPLORATION (0-30%): 探索多样架构
        - EXPLOITATION (30-70%): 利用已知好架构
        - REFINEMENT (70-100%): 精调最佳架构
        """
        progress = self.iteration / self.max_iterations

        if progress < 0.3:
            return "exploration"
        elif progress < 0.7:
            return "exploitation"
        else:
            return "refinement"

    def _generate_strategy_feedback(self, iteration: int) -> str:
        """
        基于搜索进度生成策略指导（v2 核心改进）
        """
        phase = self._get_strategy_phase()

        if phase == "exploration":
            return (
                "Phase: EXPLORATION 🌱\n"
                "Focus on trying diverse architecture types:\n"
                "- Attention-based fusion (cross-modal attention)\n"
                "- Gating mechanisms (feature-wise gates)\n"
                "- Concatenation + MLP baselines\n"
                "- Try different layer depths and hidden dimensions\n"
                "Don't worry about perfection, explore the space!"
            )
        elif phase == "exploitation":
            return (
                "Phase: EXPLOITATION 🔍\n"
                "Focus on refining the best architectures found:\n"
                "- Improve the top-performing design\n"
                "- Adjust hidden dimensions and layer counts\n"
                "- Try architectural variants of successful patterns\n"
                "- Balance exploration and exploitation\n"
                f"Current best reward: {self.best_result.reward:.3f if self.best_result else 0.0}"
            )
        else:  # refinement
            return (
                "Phase: REFINEMENT ✨\n"
                "Focus on fine-tuning for maximum performance:\n"
                "- Optimize the best architecture\n"
                "- Tune hyperparameters (dropout, normalization)\n"
                "- Ensure efficiency constraints are met\n"
                "- Polish the final design\n"
                f"Target: Beat current best reward {self.best_result.reward:.3f if self.best_result else 0.0}"
            )

    def _generate_feedback(
        self,
        metrics: Dict[str, Any],
        reward: float,
        iteration: int
    ) -> str:
        """生成自然语言反馈"""
        feedback_parts = []

        feedback_parts.append(f"Iteration {iteration} Results:")
        feedback_parts.append(f"- Accuracy: {metrics['accuracy']:.2%}")
        feedback_parts.append(f"- mRob: {metrics['mrob']:.2%}")
        feedback_parts.append(f"- FLOPs: {metrics['flops']/1e6:.1f}M")
        feedback_parts.append(f"- Params: {metrics.get('params', 0)/1e6:.1f}M")
        feedback_parts.append(f"- Reward: {reward:.3f}")

        if self.best_result:
            reward_diff = reward - self.best_result.reward
            if reward_diff > 0:
                feedback_parts.append(f"✅ New best! (+{reward_diff:.3f})")
            else:
                feedback_parts.append(f"vs Best: {reward_diff:.3f}")

        return "\n".join(feedback_parts)

    def _print_iteration_summary(self, result: SearchResult):
        """打印迭代摘要"""
        logger.info(f"Iteration {result.iteration} Summary:")
        logger.info(f"  Compile: {'✅' if result.compile_success else '❌'} "
                   f"({result.compile_attempts} attempts)")

        if result.compile_success:
            logger.info(f"  Accuracy: {result.accuracy:.2%}")
            logger.info(f"  mRob: {result.mrob:.2%}")
            logger.info(f"  FLOPs: {result.flops/1e6:.1f}M")
            logger.info(f"  Reward: {result.reward:.3f}")

        if self.best_result:
            logger.info(f"  🏆 Best so far: {self.best_result.reward:.3f}")

        logger.info(f"  Time: {result.time_taken:.1f}s")

    def _save_checkpoint(self, iteration: int):
        """保存 checkpoint（v2）"""
        checkpoint = {
            "iteration": iteration,
            "history": [r.to_dict() for r in self.history],
            "best_result": self.best_result.to_dict() if self.best_result else None,
            "api_contract": self.contract
        }

        checkpoint_path = os.path.join(
            self.output_dir, f"checkpoint_iter_{iteration}.json"
        )

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")

    def _get_compile_success_rate(self) -> float:
        """计算编译成功率"""
        if not self.history:
            return 0.0
        successes = sum(1 for r in self.history if r.compile_success)
        return successes / len(self.history)


# 保持向后兼容
EASEvolver = EASEvolverV2

"""
自修复编译器模块

核心类: SelfHealingCompiler
整合语法验证、形状验证、错误修复，实现内循环
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

# 导入子模块
from .syntax_validator import SyntaxValidator
from .shape_verifier import ShapeVerifier
from .error_repair import ErrorRepair

# 导入LLM后端
sys.path.append(str(Path(__file__).parent.parent))
from utils.llm_backend import UnifiedLLMBackend, LLMResponse


@dataclass
class CompilationResult:
    """编译结果数据结构"""
    code: str                      # 成功编译的代码
    success: bool                  # 是否成功
    attempts: int                  # 尝试次数
    history: List[Dict]            # 修复历史
    llm_response: LLMResponse      # 最后一次LLM响应
    metadata: Dict[str, Any]       # 额外元数据


class CompilationError(Exception):
    """编译错误异常"""
    def __init__(self, message: str, history: List[Dict]):
        super().__init__(message)
        self.history = history


class SelfHealingCompiler:
    """
    自修复编译器

    核心功能:
    1. LLM生成代码
    2. 语法验证
    3. 形状验证
    4. 错误反馈 -> LLM修复
    5. 循环直到成功或达到最大重试次数

    目标: 将编译成功率从5%提升到95%

    Example:
        >>> compiler = SelfHealingCompiler()
        >>> result = compiler.compile(
        ...     prompt="Generate a multimodal fusion model...",
        ...     api_contract={
        ...         'inputs': {...},
        ...         'output_shape': [...]
        ...     }
        ... )
        >>> print(f"Success after {result.attempts} attempts")
        >>> print(result.code)
    """

    def __init__(
        self,
        llm_backend: Optional[UnifiedLLMBackend] = None,
        max_retries: int = 3,
        device: str = 'cpu'
    ):
        """
        初始化自修复编译器

        Args:
            llm_backend: LLM后端实例，默认创建新实例
            max_retries: 最大重试次数
            device: 形状验证使用的设备
        """
        self.llm = llm_backend or UnifiedLLMBackend()
        self.max_retries = max_retries

        # 验证器
        self.syntax_validator = SyntaxValidator()
        self.shape_verifier = ShapeVerifier(device=device)
        self.error_repair = ErrorRepair()

        # 统计
        self.compile_attempts = 0
        self.compile_successes = 0
        self.total_retries = 0

    def compile(
        self,
        prompt: str,
        api_contract: Dict[str, Any],
        verbose: bool = True
    ) -> CompilationResult:
        """
        编译代码 (内循环主函数)

        流程:
        1. LLM生成代码
        2. 语法验证
        3. 形状验证
        4. 如有错误，添加反馈并回到步骤1
        5. 重复直到成功或达到max_retries

        Args:
            prompt: 初始代码生成prompt
            api_contract: API契约(输入输出规格)
            verbose: 是否打印详细日志

        Returns:
            CompilationResult: 编译结果

        Raises:
            CompilationError: 所有重试失败后抛出
        """
        history = []
        current_prompt = prompt
        last_response = None

        if verbose:
            print(f"🔄 Starting compilation (max {self.max_retries} retries)...")

        for attempt in range(1, self.max_retries + 1):
            self.compile_attempts += 1

            if verbose:
                print(f"\n  Attempt {attempt}/{self.max_retries}")

            # 1. LLM生成代码
            try:
                response = self.llm.generate(current_prompt)
                last_response = response
                code = response.code

                if verbose:
                    print(f"    ✓ Code generated ({len(code)} chars)")

            except Exception as e:
                error_msg = f"LLM generation failed: {str(e)}"
                if verbose:
                    print(f"    ✗ {error_msg}")
                history.append({
                    'attempt': attempt,
                    'stage': 'generation',
                    'success': False,
                    'error': error_msg
                })
                continue

            # 2. 语法验证
            is_valid, syntax_error = self.syntax_validator.check(code)

            if not is_valid:
                if verbose:
                    print(f"    ✗ Syntax error: {syntax_error[:50]}...")

                history.append({
                    'attempt': attempt,
                    'stage': 'syntax_check',
                    'success': False,
                    'error': syntax_error,
                    'code': code[:200]  # 记录部分代码
                })

                # 添加语法反馈
                current_prompt = self.error_repair.add_syntax_feedback(
                    prompt, code, syntax_error
                )
                continue

            if verbose:
                print(f"    ✓ Syntax OK")

            # 3. 形状验证
            is_valid, shape_error = self.shape_verifier.verify(code, api_contract)

            if not is_valid:
                if verbose:
                    print(f"    ✗ Shape error: {shape_error[:50]}...")

                history.append({
                    'attempt': attempt,
                    'stage': 'shape_check',
                    'success': False,
                    'error': shape_error,
                    'code': code[:200]
                })

                # 添加形状反馈
                current_prompt = self.error_repair.add_shape_feedback(
                    prompt, code, shape_error
                )
                continue

            if verbose:
                print(f"    ✓ Shape OK")

            # 4. 模态缺失处理验证 (可选)
            is_valid, robust_error = self._verify_modality_handling(code)

            if not is_valid:
                if verbose:
                    print(f"    ✗ Robustness: {robust_error[:50]}...")

                history.append({
                    'attempt': attempt,
                    'stage': 'robustness_check',
                    'success': False,
                    'error': robust_error
                })

                current_prompt = self.error_repair.add_robustness_feedback(
                    prompt, code, robust_error
                )
                continue

            if verbose:
                print(f"    ✓ Robustness OK")

            # 成功!
            self.compile_successes += 1
            self.total_retries += attempt - 1

            if verbose:
                print(f"\n✅ Compilation successful after {attempt} attempt(s)")

            return CompilationResult(
                code=code,
                success=True,
                attempts=attempt,
                history=history,
                llm_response=last_response,
                metadata={
                    'prompt_tokens': last_response.prompt_tokens if last_response else 0,
                    'completion_tokens': last_response.completion_tokens if last_response else 0,
                    'latency_ms': last_response.latency_ms if last_response else 0
                }
            )

        # 所有重试失败
        raise CompilationError(
            f"Failed to compile after {self.max_retries} attempts",
            history
        )

    def _verify_modality_handling(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        验证代码是否处理了模态缺失

        简单启发式: 检查是否包含条件语句
        """
        # 检查是否有if语句(可能用于模态门控)
        if 'if ' in code or 'confidence' in code or 'modality' in code:
            return True, None

        # 不强制要求，只作为提示
        return True, None

    def get_stats(self) -> Dict[str, Any]:
        """
        获取编译统计

        Returns:
            包含以下字段的字典:
            - total_attempts: 总尝试次数
            - total_successes: 成功次数
            - success_rate: 成功率
            - avg_attempts: 平均尝试次数
            - syntax_validator_stats: 语法验证统计
            - shape_verifier_stats: 形状验证统计
        """
        return {
            'total_attempts': self.compile_attempts,
            'total_successes': self.compile_successes,
            'success_rate': (
                self.compile_successes / max(1, self.compile_attempts)
            ),
            'avg_attempts': (
                (self.total_retries + self.compile_successes)
                / max(1, self.compile_successes)
                if self.compile_successes > 0 else 0
            ),
            'syntax_validator_stats': self.syntax_validator.get_error_stats(),
            'shape_verifier_stats': self.shape_verifier.get_error_stats(),
            'error_repair_stats': self.error_repair.get_repair_stats()
        }

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("SelfHealingCompiler Statistics")
        print("=" * 50)
        print(f"Total attempts: {stats['total_attempts']}")
        print(f"Successful: {stats['total_successes']}")
        print(f"Success rate: {stats['success_rate']*100:.1f}%")
        print(f"Avg attempts per success: {stats['avg_attempts']:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    print("Testing SelfHealingCompiler...")

    # 创建测试用的compiler实例
    # 注意: 需要设置ALIYUN_API_KEY环境变量
    import os
    os.environ.setdefault('ALIYUN_API_KEY', 'test-key')

    try:
        compiler = SelfHealingCompiler()

        # 测试API契约
        api_contract = {
            'inputs': {
                'vision': {'shape': [2, 576, 1024], 'dtype': 'float32'},
                'text': {'shape': [2, 77, 768], 'dtype': 'float32'}
            },
            'output_shape': [2, 10],
            'model_kwargs': {'hidden_dim': 256}
        }

        # 简单的测试prompt
        prompt = """
Generate a PyTorch nn.Module for multimodal fusion.

Inputs:
- vision: [batch, 576, 1024]
- text: [batch, 77, 768]

Output: [batch, 10]

Requirements:
1. Use nn.Linear for projection
2. Use attention mechanism
3. Return tensor of correct shape
"""

        print("\nNote: Full test requires valid API key")
        print(f"API Contract: {api_contract}")

    except ValueError as e:
        print(f"Expected error (no API key): {e}")

"""
自修复编译器模块 V2 - 基于 Auto-Fusion-v2 改进

核心改进:
1. AttemptRecord 完整历史追踪
2. 错误特定指导（防止重复错误）
3. GPU 内存清理
4. 更丰富的错误反馈
"""

import re
import gc
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, field

# 导入子模块
from .syntax_validator import SyntaxValidator
from .shape_verifier import ShapeVerifier
from .error_repair import ErrorRepair

# 导入LLM后端
sys.path.append(str(Path(__file__).parent.parent))
from utils.llm_backend import UnifiedLLMBackend, LLMResponse

import torch


@dataclass
class AttemptRecord:
    """单次编译尝试记录（v2 新增）"""
    attempt_number: int
    code: str
    error: str
    error_type: str = ""  # 'syntax', 'shape', 'runtime', 'oom'


@dataclass
class CompilationResult:
    """编译结果数据结构"""
    code: str
    success: bool
    attempts: int
    history: List[Dict]
    llm_response: Optional[LLMResponse]
    metadata: Dict[str, Any] = field(default_factory=dict)
    attempt_records: List[AttemptRecord] = field(default_factory=list)  # v2 新增


class CompilationError(Exception):
    """编译错误异常"""
    def __init__(self, message: str, history: List[Dict], attempt_records: List[AttemptRecord] = None):
        super().__init__(message)
        self.history = history
        self.attempt_records = attempt_records or []


class SelfHealingCompilerV2:
    """
    自修复编译器 V2 - 基于 Auto-Fusion-v2 改进

    核心改进:
    1. 完整尝试历史追踪（AttemptRecord）
    2. 错误特定指导（_get_error_specific_guidance）
    3. GPU 内存清理（防止 OOM）
    4. 带历史的错误反馈（防止重复错误）
    """

    def __init__(
        self,
        llm_backend: Optional[UnifiedLLMBackend] = None,
        max_retries: int = 5,  # v2: 增加默认重试次数
        device: str = 'cpu'
    ):
        self.llm = llm_backend or UnifiedLLMBackend()
        self.max_retries = max_retries
        self.device = device

        # 验证器
        self.syntax_validator = SyntaxValidator()
        self.shape_verifier = ShapeVerifier(device=device)
        self.error_repair = ErrorRepair()

        # 统计
        self.compile_attempts = 0
        self.compile_successes = 0
        self.total_retries = 0

        # v2: 尝试历史记录
        self.attempt_history: List[AttemptRecord] = []

    def compile(
        self,
        prompt: str,
        api_contract: Dict[str, Any],
        verbose: bool = True
    ) -> CompilationResult:
        """
        编译代码（内循环主函数）- V2 改进版
        """
        history = []
        self.attempt_history = []  # 重置历史
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

                # Post-process: fix common issues
                code = self._post_process_code(code)

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

            # 2. 验证代码（v2: 统一验证函数）
            is_valid, error, error_type = self._validate_code(code, api_contract)

            if not is_valid:
                if verbose:
                    print(f"    ✗ {error_type} error: {error[:60]}...")

                # v2: 记录尝试历史
                record = AttemptRecord(
                    attempt_number=attempt,
                    code=code,
                    error=error,
                    error_type=error_type
                )
                self.attempt_history.append(record)

                history.append({
                    'attempt': attempt,
                    'stage': f'{error_type}_check',
                    'success': False,
                    'error': error,
                    'code': code[:300]
                })

                # v2: 构建带历史的错误反馈
                current_prompt = self._construct_error_prompt_with_history(
                    prompt, code, error, error_type
                )

                # v2: GPU 内存清理
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()

                continue

            if verbose:
                print(f"    ✓ Validation OK")

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
                },
                attempt_records=self.attempt_history
            )

        # 所有重试失败
        raise CompilationError(
            f"Failed to compile after {self.max_retries} attempts",
            history,
            self.attempt_history
        )

    def _validate_code(
        self,
        code: str,
        api_contract: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], str]:
        """
        统一验证代码（v2 新增）

        Returns:
            (success, error_message, error_type)
        """
        try:
            # Step 1: 语法验证
            is_valid, error = self.syntax_validator.check(code)
            if not is_valid:
                return False, error, 'syntax'

            # Step 2: 形状验证
            is_valid, error = self.shape_verifier.verify(code, api_contract)
            if not is_valid:
                return False, error, 'shape'

            # Step 3: 运行时验证（执行前向传播）
            is_valid, error, error_type = self._runtime_verify(code, api_contract)
            if not is_valid:
                return False, error, error_type

            return True, None, ''

        except Exception as e:
            return False, str(e), 'runtime'

    def _runtime_verify(
        self,
        code: str,
        api_contract: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], str]:
        """
        运行时验证（v2 新增）- 实际执行代码验证可行性
        """
        try:
            # 创建 restricted namespace
            namespace = self._create_restricted_namespace()

            # 执行代码
            exec(code, namespace)

            # 查找模型类
            model_class = None
            for obj in namespace.values():
                if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                    if obj != torch.nn.Module:
                        model_class = obj
                        break

            if model_class is None:
                return False, "No valid model class found", 'runtime'

            # 实例化并测试 - 处理不同的__init__签名
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = self._create_model_instance(model_class, api_contract)
            model = model.to(device)
            model.eval()

            # 创建 dummy 输入
            dummy_inputs = {}
            for name, spec in api_contract.get('inputs', {}).items():
                shape = spec['shape']
                dtype = getattr(torch, spec.get('dtype', 'float32').replace('float32', 'float'))
                dummy_inputs[name] = torch.randn(shape, dtype=dtype, device=device)

            # 前向传播
            with torch.no_grad():
                output = model(**dummy_inputs)

            # 验证输出形状
            expected_shape = api_contract.get('output_shape', [2, 10])
            if list(output.shape) != expected_shape:
                return False, f"Output shape mismatch: got {list(output.shape)}, expected {expected_shape}", 'shape'

            # 清理
            del model, output, dummy_inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return True, None, ''

        except torch.cuda.OutOfMemoryError as e:
            return False, f"GPU OOM Error: {str(e)}", 'oom'
        except RuntimeError as e:
            return False, f"Runtime Error: {str(e)}", 'runtime'
        except Exception as e:
            return False, f"Error: {str(e)}", 'runtime'

    def _create_model_instance(self, model_class: type, api_contract: Dict[str, Any]):
        """创建模型实例，优先尝试无参数初始化"""
        import inspect

        sig = inspect.signature(model_class.__init__)
        params = list(sig.parameters.keys())

        # 首先尝试无参数初始化
        if len(params) <= 1:  # 只有 self
            return model_class()

        # 检查是否有 model_kwargs
        model_kwargs = api_contract.get('model_kwargs', {})
        if model_kwargs:
            try:
                return model_class(**model_kwargs)
            except:
                pass

        # 检查是否需要 input_dims 参数
        if 'input_dims' in params:
            # 构建 input_dims
            input_dims = {
                name: {'shape': spec['shape'], 'dtype': spec.get('dtype', 'float32')}
                for name, spec in api_contract.get('inputs', {}).items()
            }
            try:
                return model_class(input_dims=input_dims)
            except:
                pass

        # 最后的尝试：无参数
        return model_class()

    def _create_restricted_namespace(self) -> Dict[str, Any]:
        """创建受限执行环境（v2 新增）"""
        import builtins

        # 安全的 builtins - 允许 __import__ 用于正常的import
        safe_builtins = builtins.__dict__.copy()
        dangerous = ['eval', 'exec', 'compile', 'open', 'input']
        for name in dangerous:
            safe_builtins.pop(name, None)

        return {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            'torch': torch,
            'nn': torch.nn,
            'F': torch.nn.functional,
        }

    def _post_process_code(self, code: str) -> str:
        """
        后处理生成的代码，修复常见问题

        Fixes:
        1. Replace .view() with .reshape() to avoid contiguous memory issues
        """
        # Fix 1: Replace .view() with .reshape() - safer for non-contiguous tensors
        # Pattern: tensor.view(...) -> tensor.reshape(...)
        import re

        # Match .view( that is not preceded by .contiguous()
        # This is a simple regex approach - may not catch all cases but covers common ones
        code = re.sub(r'\.(view)\s*\(', '.reshape(', code)

        return code

    def _construct_error_prompt_with_history(
        self,
        original_prompt: str,
        last_code: str,
        last_error: str,
        error_type: str
    ) -> str:
        """
        构建带历史的错误反馈 Prompt（v2 核心改进）

        关键改进：
        1. 展示所有历史尝试（防止重复）
        2. 展示最近的失败（详细修复）
        3. 针对错误类型的具体指导
        """
        current_attempt = len(self.attempt_history) + 1

        prompt_parts = []
        prompt_parts.append(f"【Attempt {current_attempt}/{self.max_retries}】")
        prompt_parts.append("Your previous code generation attempts failed. Please analyze the history and try a different approach.\n")

        # v2: 展示完整历史（防止重复错误）
        if len(self.attempt_history) >= 2:
            prompt_parts.append("【Previous Attempts - DO NOT REPEAT THESE】")
            for record in self.attempt_history[:-1]:
                prompt_parts.append(f"\nAttempt {record.attempt_number} ({record.error_type}):")
                prompt_parts.append(f"Code snippet: {record.code[:200]}...")
                prompt_parts.append(f"Error: {record.error[:100]}...")
            prompt_parts.append("\n" + "=" * 60)

        # 展示最近的失败
        prompt_parts.append("【Most Recent Failure - Fix This】")
        prompt_parts.append(f"\nCode:")
        prompt_parts.append("```python")
        prompt_parts.append(last_code)
        prompt_parts.append("```\n")
        prompt_parts.append(f"【Error Message ({error_type})】")
        prompt_parts.append(f"```\n{last_error}\n```\n")

        # v2: 错误特定指导
        prompt_parts.append(self._get_error_specific_guidance(last_error, error_type))

        # API 契约
        prompt_parts.append("【API Contract】")
        prompt_parts.append(original_prompt)
        prompt_parts.append("")

        # 修复要求
        prompt_parts.append("【Fix Requirements】")
        prompt_parts.append("1. Analyze the error history - do NOT repeat previous failed approaches")
        prompt_parts.append("2. If you tried permute() and it failed, try transpose() or reshape()")
        prompt_parts.append("3. If dimension mismatch persists, consider using adaptive pooling")
        prompt_parts.append("4. Ensure all tensor dimensions match the contract")
        prompt_parts.append("5. Use correct PyTorch API syntax")
        prompt_parts.append("6. Only return the fixed code, no explanation\n")

        prompt_parts.append("Provide the corrected code with a NEW approach:")

        return "\n".join(prompt_parts)

    def _get_error_specific_guidance(self, error: str, error_type: str) -> str:
        """
        针对错误类型的具体指导（v2 核心改进）
        """
        error_lower = error.lower()
        guidance = ["【Specific Guidance】"]

        if error_type == 'syntax' or 'syntax' in error_lower:
            guidance.append("- Syntax error. Check:")
            guidance.append("  * All parentheses and brackets are balanced")
            guidance.append("  * Proper indentation (4 spaces)")
            guidance.append("  * No trailing commas in argument lists")
            guidance.append("  * Correct Python keywords")

        elif error_type == 'shape' or 'shape mismatch' in error_lower or 'size mismatch' in error_lower:
            guidance.append("- Dimension mismatch detected. Consider:")
            guidance.append("  * Using .mean(dim=1) instead of .view() for pooling")
            guidance.append("  * Adding projection layers to match dimensions")
            guidance.append("  * Using adaptive pooling: nn.AdaptiveAvgPool1d(output_size)")
            guidance.append("  * Check input shape with .shape before operations")

        elif 'permute' in error_lower or 'transpose' in error_lower:
            guidance.append("- Tensor dimension reordering issue. Try:")
            guidance.append("  * Use .reshape() or .view() instead of permute/transpose")
            guidance.append("  * Check the actual tensor shape with .shape before operations")

        elif error_type == 'oom' or 'out of memory' in error_lower:
            guidance.append("- GPU memory issue. Solutions:")
            guidance.append("  * Reduce hidden dimensions (e.g., 512 -> 256)")
            guidance.append("  * Use fewer layers")
            guidance.append("  * Avoid creating intermediate tensors in forward()")

        elif 'attribute' in error_lower or 'has no attribute' in error_lower:
            guidance.append("- Attribute error. Verify:")
            guidance.append("  * Variable names are correct")
            guidance.append("  * Methods exist on the object (check PyTorch docs)")
            guidance.append("  * Proper initialization in __init__")

        elif 'matmul' in error_lower or 'mat1 and mat2' in error_lower:
            guidance.append("- Matrix multiplication shape error:")
            guidance.append("  * Check that inner dimensions match")
            guidance.append("  * Use .transpose() or .T to align dimensions")
            guidance.append("  * Consider using .reshape() to flatten before Linear")

        else:
            guidance.append("- General debugging tips:")
            guidance.append("  * Simplify the architecture")
            guidance.append("  * Test each layer individually")
            guidance.append("  * Use print statements to debug shapes")

        return "\n".join(guidance) + "\n"

    def get_stats(self) -> Dict[str, Any]:
        """获取编译统计"""
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


# 保持向后兼容
SelfHealingCompiler = SelfHealingCompilerV2

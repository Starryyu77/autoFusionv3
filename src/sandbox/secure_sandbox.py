"""
Secure Sandbox - 安全代码执行环境

基于 Auto-Fusion-v2 实现
提供:
- 多进程隔离执行
- 资源限制 (内存、CPU、GPU)
- 超时处理
- GPU 内存保护
"""

import sys
import gc
import signal
import resource
import multiprocessing
import traceback
from typing import Dict, Any, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SandboxResult:
    """沙箱执行结果"""
    success: bool
    output: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_used_mb: float = 0.0


class SandboxTimeoutError(Exception):
    """沙箱执行超时"""
    pass


class SecureSandbox:
    """
    安全沙箱 - 隔离执行生成代码

    特性:
    - 多进程隔离 (spawn context)
    - 资源限制 (CPU时间、内存)
    - GPU内存限制
    - 超时处理
    - 强制清理
    """

    ALLOWED_MODULES = {
        'torch', 'torch.nn', 'torch.nn.functional',
        'math', 'numpy', 'typing'
    }

    def __init__(
        self,
        timeout: int = 60,
        max_memory_mb: int = 2048,
        max_cpu_time: int = 60,
        max_vram_mb: Optional[int] = None
    ):
        """
        初始化沙箱

        Args:
            timeout: 执行超时时间 (秒)
            max_memory_mb: 最大内存限制 (MB)
            max_cpu_time: 最大CPU时间 (秒)
            max_vram_mb: 最大GPU显存限制 (MB)，默认 2GB
        """
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.max_cpu_time = max_cpu_time
        self.max_vram_mb = max_vram_mb or 2048

    def execute(
        self,
        code: str,
        inputs: Dict[str, torch.Tensor],
        api_contract: Optional[Dict] = None
    ) -> SandboxResult:
        """
        在隔离进程中执行代码

        Args:
            code: Python代码
            inputs: 输入张量
            api_contract: API契约（用于验证）

        Returns:
            SandboxResult: 执行结果
        """
        import time
        start_time = time.time()

        # 预执行清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        # 使用多进程隔离
        ctx = multiprocessing.get_context('spawn')
        queue = ctx.Queue()

        process = ctx.Process(
            target=self._execute_in_process,
            args=(code, inputs, queue, api_contract),
            daemon=True
        )

        try:
            process.start()
            process.join(timeout=self.timeout)

            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
                return SandboxResult(
                    success=False,
                    error=f"Execution timeout after {self.timeout}s"
                )

            if process.exitcode != 0:
                return SandboxResult(
                    success=False,
                    error=f"Process crashed with exit code {process.exitcode}"
                )

            try:
                success, result = queue.get_nowait()
                if success:
                    return SandboxResult(
                        success=True,
                        output=result,
                        execution_time=time.time() - start_time
                    )
                else:
                    return SandboxResult(
                        success=False,
                        error=str(result),
                        execution_time=time.time() - start_time
                    )
            except:
                return SandboxResult(
                    success=False,
                    error="Failed to get result from queue"
                )

        finally:
            # 后执行清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

            if process.is_alive():
                process.kill()
                process.join()

    def _execute_in_process(
        self,
        code: str,
        inputs: Dict[str, torch.Tensor],
        queue: multiprocessing.Queue,
        api_contract: Optional[Dict] = None
    ):
        """
        在子进程中执行代码
        """
        try:
            # 设置资源限制 (Unix only)
            if sys.platform != 'win32':
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.max_memory_mb * 1024 * 1024, -1)
                )
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (self.max_cpu_time, -1)
                )

            # GPU内存限制
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(
                    self.max_vram_mb / (torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
                )
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 创建受限环境
            namespace = self._create_restricted_namespace()

            # 执行代码
            exec(code, namespace)

            # 查找模型类
            model_class = None
            for name, obj in namespace.items():
                if isinstance(obj, type) and issubclass(obj, nn.Module) and obj != nn.Module:
                    model_class = obj
                    break

            if model_class is None:
                queue.put((False, "No valid model class found"))
                return

            # 实例化模型
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model_class()
            model = model.to(device)
            model.eval()

            # 准备输入
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 前向传播
            with torch.no_grad():
                output = model(**inputs)

            # 验证输出形状
            if api_contract and 'output_shape' in api_contract:
                expected_shape = api_contract['output_shape']
                if list(output.shape) != expected_shape:
                    queue.put((
                        False,
                        f"Shape mismatch: got {list(output.shape)}, expected {expected_shape}"
                    ))
                    return

            queue.put((True, output.cpu()))

        except Exception as e:
            queue.put((False, f"{type(e).__name__}: {str(e)}"))

        finally:
            # 强制清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

    def _create_restricted_namespace(self) -> Dict[str, Any]:
        """创建受限执行环境"""
        safe_builtins = {
            'len': len, 'range': range, 'enumerate': enumerate,
            'zip': zip, 'isinstance': isinstance, 'hasattr': hasattr,
            'getattr': getattr, 'slice': slice, 'list': list,
            'dict': dict, 'tuple': tuple, 'int': int, 'float': float,
            'str': str, 'print': print, 'Exception': Exception,
            'RuntimeError': RuntimeError, 'ValueError': ValueError,
            'TypeError': TypeError, 'AttributeError': AttributeError,
        }

        namespace = {
            '__builtins__': safe_builtins,
            '__name__': '__sandbox__',
            'torch': torch,
            'nn': nn,
            'F': nn.functional,
        }

        # 添加可选模块
        try:
            import math
            namespace['math'] = math
        except:
            pass

        try:
            import numpy as np
            namespace['np'] = np
        except:
            pass

        return namespace

    def validate_architecture(
        self,
        code: str,
        api_contract: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        验证架构代码（便捷方法）

        Returns:
            (success, error_message)
        """
        # 创建 dummy 输入
        inputs = {}
        for name, spec in api_contract.get('inputs', {}).items():
            shape = spec['shape']
            dtype = getattr(torch, spec.get('dtype', 'float32').replace('float32', 'float'))
            inputs[name] = torch.randn(shape, dtype=dtype)

        result = self.execute(code, inputs, api_contract)

        if result.success:
            return True, None
        else:
            return False, result.error


@contextmanager
def sandbox_context(timeout: int = 60):
    """沙箱上下文管理器"""
    sandbox = SecureSandbox(timeout=timeout)
    try:
        yield sandbox
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

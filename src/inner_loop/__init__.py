"""
Inner Loop 模块 - 自修复编译器

包含:
- SelfHealingCompiler: 基础编译器
- SelfHealingCompilerV2: V2 改进版（推荐）
"""

from .self_healing import (
    SelfHealingCompiler,
    CompilationResult,
    CompilationError
)

from .self_healing_v2 import (
    SelfHealingCompilerV2,
    AttemptRecord
)

__all__ = [
    'SelfHealingCompiler',
    'SelfHealingCompilerV2',
    'CompilationResult',
    'CompilationError',
    'AttemptRecord'
]

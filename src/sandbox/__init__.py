"""
Sandbox 模块 - 安全代码执行环境

提供安全隔离的代码执行环境，防止:
- 资源泄露
- 恶意代码执行
- GPU OOM
"""

from .secure_sandbox import SecureSandbox, SandboxTimeoutError

__all__ = ['SecureSandbox', 'SandboxTimeoutError']

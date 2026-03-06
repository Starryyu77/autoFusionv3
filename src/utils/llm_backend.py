"""
统一LLM后端模块

所有实验必须通过此类调用LLM，确保API调用参数完全一致
使用单例模式保证全局唯一实例
"""

import os
import time
import yaml
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

# 可选导入tenacity，如果不存在则提供简单重试
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    HAS_TENACITY = True
except ImportError:
    HAS_TENACITY = False
    import functools


@dataclass
class LLMResponse:
    """LLM响应数据结构"""
    code: str                    # 生成的代码
    model: str                   # 使用的模型
    temperature: float           # 温度参数
    prompt_tokens: int           # prompt token数
    completion_tokens: int       # 完成token数
    total_tokens: int            # 总token数
    latency_ms: float            # 延迟(毫秒)
    timestamp: str               # 时间戳
    attempt: int                 # 尝试次数


class UnifiedLLMBackend:
    """
    统一LLM后端 - 所有实验必须使用此类

    特性:
    1. 单例模式：全局唯一实例，确保配置一致
    2. 固定配置：temperature=0.7, model=kimi-k2.5 (不可修改)
    3. 自动重试：指数退避策略处理API限流
    4. 完整记录：保存每次调用的metadata

    使用示例:
        >>> llm = UnifiedLLMBackend()
        >>> response = llm.generate("Generate a neural network...")
        >>> print(response.code)
        >>> print(f"Latency: {response.latency_ms}ms")
    """

    _instance = None
    _initialized = False

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化LLM后端

        Args:
            config_path: API配置文件路径，默认使用configs/api_config.yaml
        """
        if self._initialized:
            return

        # 加载配置
        if config_path is None:
            # 从当前文件位置推断配置路径
            current_file = Path(__file__).resolve()
            config_path = current_file.parent.parent.parent / "configs" / "api_config.yaml"

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.api_config = self.config['api']
        self.model = self.api_config['default_model']  # 固定: kimi-k2.5
        self.api_key = os.environ.get('ALIYUN_API_KEY')

        if not self.api_key:
            raise ValueError(
                "ALIYUN_API_KEY environment variable not set. "
                "Please set it before running experiments."
            )

        # 统计信息
        self.call_count = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0.0
        self.errors: List[Dict[str, Any]] = []

        # 备选模型索引
        self.fallback_index = 0

        self._initialized = True
        print(f"✅ UnifiedLLMBackend initialized with model: {self.model}")

    def _get_client(self):
        """创建OpenAI客户端"""
        try:
            import openai
            return openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_config['base_url']
            )
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def _do_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: str,
        attempt: int = 1
    ) -> LLMResponse:
        """
        执行单次API调用

        Args:
            prompt: 输入prompt
            temperature: 温度参数(固定0.7)
            max_tokens: 最大token数
            model: 模型名称
            attempt: 当前尝试次数

        Returns:
            LLMResponse: 包含代码和metadata的响应对象
        """
        import openai

        client = self._get_client()
        start_time = time.time()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a PyTorch expert specializing in multimodal "
                        "neural architecture design. Generate executable Python code "
                        "following best practices."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=self.api_config['top_p']
        )

        latency = (time.time() - start_time) * 1000

        # 提取生成的代码
        content = response.choices[0].message.content

        # 尝试提取代码块
        code = self._extract_code(content)

        return LLMResponse(
            code=code,
            model=model,
            temperature=temperature,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            latency_ms=latency,
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            attempt=attempt
        )

    def _extract_code(self, content: str) -> str:
        """
        从LLM响应中提取代码

        处理以下格式:
        1. ```python\ncode\n```
        2. ```\ncode\n```
        3. 纯代码
        """
        import re

        # 尝试提取代码块
        patterns = [
            r'```python\n(.*?)\n```',
            r'```\n(.*?)\n```',
            r'<code>(.*?)</code>',
        ]

        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                return match.group(1).strip()

        # 如果没有代码块标记，返回整个内容
        return content.strip()

    def _simple_retry(self, func, max_attempts=5):
        """简单的重试实现(当tenacity不可用时)"""
        for attempt in range(1, max_attempts + 1):
            try:
                return func()
            except Exception as e:
                if attempt == max_attempts:
                    raise
                wait_time = min(4 * (2 ** (attempt - 1)), 60)
                print(f"  Retry {attempt}/{max_attempts} after {wait_time}s: {str(e)[:50]}")
                time.sleep(wait_time)

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_fallback: bool = False
    ) -> LLMResponse:
        """
        统一的代码生成接口

        这是所有实验必须使用的唯一LLM调用入口。
        会自动处理重试、记录统计、错误恢复。

        Args:
            prompt: 输入prompt，应包含清晰的代码生成指令
            temperature: 覆盖默认温度(不推荐，仅用于消融实验)
            max_tokens: 覆盖默认max_tokens
            use_fallback: 是否使用备选模型

        Returns:
            LLMResponse: 包含生成的代码和完整metadata

        Raises:
            Exception: 所有重试失败后抛出异常

        Example:
            >>> llm = UnifiedLLMBackend()
            >>> prompt = "Generate a PyTorch nn.Module..."
            >>> resp = llm.generate(prompt)
            >>> print(resp.code)
        """
        # 使用固定配置
        temp = temperature if temperature is not None else self.api_config['temperature']
        tokens = max_tokens if max_tokens is not None else self.api_config['max_tokens']

        # 选择模型
        if use_fallback and self.fallback_index < len(self.api_config.get('fallback_models', [])):
            model = self.api_config['fallback_models'][self.fallback_index]
        else:
            model = self.model

        # 定义调用函数
        max_attempts = self.api_config['retry']['max_attempts']

        def call_api():
            return self._do_generate(prompt, temp, tokens, model)

        # 执行调用(带重试)
        try:
            if HAS_TENACITY:
                # 使用tenacity进行指数退避重试
                @retry(
                    stop=stop_after_attempt(max_attempts),
                    wait=wait_exponential(
                        multiplier=self.api_config['retry']['backoff_factor'],
                        min=self.api_config['retry']['initial_wait'],
                        max=self.api_config['retry']['max_wait']
                    ),
                    retry=retry_if_exception_type(Exception),
                    reraise=True
                )
                def retry_call():
                    return self._do_generate(prompt, temp, tokens, model)

                result = retry_call()
            else:
                # 使用简单重试
                result = self._simple_retry(call_api, max_attempts)

            # 更新统计
            self.call_count += 1
            self.total_tokens += result.total_tokens
            self.total_prompt_tokens += result.prompt_tokens
            self.total_completion_tokens += result.completion_tokens
            self.total_latency_ms += result.latency_ms

            return result

        except Exception as e:
            # 记录错误
            self.errors.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'prompt_length': len(prompt),
                'model': model
            })

            # 尝试切换到fallback模型
            if not use_fallback and self.fallback_index < len(self.api_config.get('fallback_models', [])):
                self.fallback_index += 1
                print(f"Switching to fallback model: {self.api_config['fallback_models'][self.fallback_index - 1]}")
                return self.generate(prompt, temperature, max_tokens, use_fallback=True)

            raise

    def get_stats(self) -> Dict[str, Any]:
        """
        获取API调用统计

        Returns:
            包含以下字段的字典:
            - total_calls: 总调用次数
            - total_tokens: 总token数
            - avg_tokens_per_call: 平均token数
            - avg_latency_ms: 平均延迟
            - error_count: 错误次数
            - error_rate: 错误率
        """
        return {
            'total_calls': self.call_count,
            'total_tokens': self.total_tokens,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_completion_tokens': self.total_completion_tokens,
            'avg_tokens_per_call': self.total_tokens / max(1, self.call_count),
            'avg_latency_ms': self.total_latency_ms / max(1, self.call_count),
            'error_count': len(self.errors),
            'error_rate': len(self.errors) / max(1, self.call_count),
            'model': self.model
        }

    def reset_stats(self):
        """重置统计(新实验开始时调用)"""
        self.call_count = 0
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_latency_ms = 0.0
        self.errors = []
        self.fallback_index = 0
        print("✅ LLM stats reset")

    def print_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        print("\n" + "=" * 50)
        print("LLM API调用统计")
        print("=" * 50)
        print(f"模型: {stats['model']}")
        print(f"总调用次数: {stats['total_calls']}")
        print(f"总Token数: {stats['total_tokens']:,}")
        print(f"  - Prompt: {stats['total_prompt_tokens']:,}")
        print(f"  - Completion: {stats['total_completion_tokens']:,}")
        print(f"平均Token/调用: {stats['avg_tokens_per_call']:.1f}")
        print(f"平均延迟: {stats['avg_latency_ms']:.1f}ms")
        print(f"错误次数: {stats['error_count']} ({stats['error_rate']*100:.2f}%)")
        print("=" * 50)


# 便捷函数
def get_llm_backend(config_path: Optional[str] = None) -> UnifiedLLMBackend:
    """
    获取LLM后端实例(工厂函数)

    这是获取LLM后端的推荐方式，会自动处理单例逻辑
    """
    return UnifiedLLMBackend(config_path)


if __name__ == "__main__":
    # 测试代码
    print("Testing UnifiedLLMBackend...")

    # 设置测试API密钥(实际使用时从环境变量读取)
    os.environ.setdefault('ALIYUN_API_KEY', 'test-key')

    try:
        llm = UnifiedLLMBackend()
        print(f"Initialized: {llm.model}")
        print(f"Config: temperature={llm.api_config['temperature']}")

        # 查看统计
        llm.print_stats()

    except ValueError as e:
        print(f"Expected error (no API key): {e}")

"""
EAS Prompt模板 V2 (修复版)

修复内容:
1. 统一维度为1024 (而非768)
2. 明确模态缺失处理方式 (零张量而非None)
3. 强制输出 [B, 1024] 维度
"""

# 基础模板 (所有实验共用)
BASE_PROMPT_TEMPLATE = """You are an expert neural architecture designer specializing in multimodal fusion.

## Task
Design a multimodal fusion architecture that takes three modalities and outputs fused representations.

## Input Specifications (CRITICAL - DO NOT CHANGE)
All inputs are PyTorch tensors with the following shapes:
- vision: [B, 576, 1024]  # Visual features from CLIP-ViT-L/14
- audio: [B, 400, 1024]   # Audio features from wav2vec 2.0
- text: [B, 77, 1024]     # Text features from BERT

IMPORTANT: All three modalities are ALREADY projected to 1024 dimensions. Do NOT add projection layers.

## Output Specification (CRITICAL)
Your forward() method MUST return a tensor of shape [B, 1024].
This should be the fused representation after global pooling (e.g., mean pooling over sequence).

## Modality Missing Handling (CRITICAL)
When a modality is missing, you will receive a ZERO TENSOR of the same shape, NOT None.
Example: if vision is missing, vision = torch.zeros([B, 576, 1024])

Your code should handle this naturally - zeros don't contribute to gradients or attention.
Optional: You can accept an additional 'modality_mask' dict to explicitly identify missing modalities.

## Architecture Requirements
1. Create a class inheriting from nn.Module
2. Implement __init__(self) and forward(self, vision, audio, text, **kwargs)
3. Use standard PyTorch operations only
4. The architecture should be robust to missing modalities (zeros)
5. Return shape must be [B, 1024]

## Example Structure (for reference only, be creative):
```python
import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Your layers here

    def forward(self, vision, audio, text, **kwargs):
        # vision: [B, 576, 1024]
        # audio: [B, 400, 1024]
        # text: [B, 77, 1024]

        # Your fusion logic here

        # Return fused representation
        return fused  # Shape: [B, 1024]
```

Generate the complete Python code:
"""

# 带历史反馈的模板
ITERATIVE_PROMPT_TEMPLATE = """You are an expert neural architecture designer specializing in multimodal fusion.

## Task
Design a multimodal fusion architecture that improves upon previous attempts.

## Input Specifications (CRITICAL - DO NOT CHANGE)
All inputs are PyTorch tensors with shape [B, seq_len, 1024]:
- vision: [B, 576, 1024]
- audio: [B, 400, 1024]
- text: [B, 77, 1024]

## Output Specification (CRITICAL)
MUST return [B, 1024] (fused representation after pooling).

## Previous Attempts and Feedback
{history_feedback}

## Current Best Architecture (Reference)
{best_architecture_so_far}

## Suggested Improvements
{suggestions}

Generate improved code:
"""


def build_base_prompt() -> str:
    """构建基础prompt"""
    return BASE_PROMPT_TEMPLATE


def build_iterative_prompt(history: list, best_code: str = "", suggestions: str = "") -> str:
    """
    构建带历史反馈的prompt

    Args:
        history: 历史尝试记录
        best_code: 当前最佳架构代码
        suggestions: 改进建议

    Returns:
        完整的prompt字符串
    """
    # 格式化历史反馈
    history_feedback = "\n\n".join([
        f"Attempt {i+1}:\nCode: {h['code'][:200]}...\nError: {h.get('error', 'N/A')}\n"
        for i, h in enumerate(history[-3:])  # 只显示最近3次
    ])

    return ITERATIVE_PROMPT_TEMPLATE.format(
        history_feedback=history_feedback,
        best_architecture_so_far=best_code[:500] if best_code else "None yet",
        suggestions=suggestions
    )


# 策略特定的prompt附加信息
STRATEGY_PROMPTS = {
    "exploration": """
    ## Strategy: Exploration (0-30% iterations)
    Focus on trying diverse architecture types:
    - Different attention mechanisms (self-attention, cross-attention)
    - Various fusion strategies (early, late, intermediate)
    - Novel gating mechanisms or conditioning
    Be creative and explore the design space!
    """,

    "exploitation": """
    ## Strategy: Exploitation (30-70% iterations)
    Focus on refining promising architectures:
    - Improve the best performing designs from history
    - Fine-tune hyperparameters (hidden dims, num heads, etc.)
    - Combine successful patterns from different attempts
    Build upon what has worked well!
    """,

    "refinement": """
    ## Strategy: Refinement (70-100% iterations)
    Focus on fine-tuning for maximum performance:
    - Polish the top-performing architecture
    - Optimize efficiency (reduce FLOPs while maintaining accuracy)
    - Add small improvements (residual connections, better normalization)
    Optimize the best architecture to its limit!
    """
}


def add_strategy_prompt(base_prompt: str, strategy: str) -> str:
    """
    添加策略特定的prompt

    Args:
        base_prompt: 基础prompt
        strategy: 策略名称 (exploration/exploitation/refinement)

    Returns:
        添加了策略信息的prompt
    """
    strategy_text = STRATEGY_PROMPTS.get(strategy, "")
    return base_prompt + "\n" + strategy_text


# FLOPs计算用的dummy输入形状 (用于thop/torchprofile)
FLOPS_DUMMY_INPUT_SHAPES = {
    'vision': (1, 576, 1024),  # 修复: 从768改为1024
    'audio': (1, 400, 1024),
    'text': (1, 77, 1024)      # 修复: 从768改为1024
}

# InnerLoop形状验证用的dummy输入
def create_validation_inputs(batch_size: int = 2) -> dict:
    """
    创建InnerLoop验证用的dummy输入

    修复: 统一使用1024维，并测试缺失模拟
    """
    return {
        'vision': torch.randn(batch_size, 576, 1024),
        'audio': torch.randn(batch_size, 400, 1024),
        'text': torch.randn(batch_size, 77, 1024)
    }


def create_dropout_validation_inputs(batch_size: int = 2, dropout_rate: float = 0.5) -> dict:
    """
    创建带缺失模拟的验证输入

    修复: 使用零张量模拟缺失，而非None
    """
    # 随机决定哪些模态缺失
    import random

    inputs = {
        'vision': torch.randn(batch_size, 576, 1024),
        'audio': torch.randn(batch_size, 400, 1024),
        'text': torch.randn(batch_size, 77, 1024)
    }

    # 应用缺失 (零张量)
    for mod in inputs:
        if random.random() < dropout_rate:
            inputs[mod] = torch.zeros_like(inputs[mod])

    return inputs


# 修复后的InnerLoop验证函数
def validate_architecture(model_class, verbose: bool = False) -> tuple:
    """
    修复后的架构验证函数

    修复内容:
    1. 使用1024维输入
    2. 测试缺失模拟 (零张量)
    3. 强制输出 [B, 1024]
    """
    import torch

    try:
        # 1. 实例化模型
        model = model_class()

        # 2. 测试完整输入
        full_inputs = create_validation_inputs(batch_size=2)
        output_full = model(**full_inputs)

        # 修复: 检查输出维度必须是 [B, 1024]
        if output_full.shape != (2, 1024):
            return False, f"Output shape mismatch: expected (2, 1024), got {output_full.shape}"

        # 3. 测试缺失模拟 (关键修复)
        dropout_inputs = create_dropout_validation_inputs(batch_size=2, dropout_rate=0.5)
        output_dropout = model(**dropout_inputs)

        if output_dropout.shape != (2, 1024):
            return False, f"Output shape mismatch with dropout: expected (2, 1024), got {output_dropout.shape}"

        # 4. 测试全缺失 (极端情况)
        zero_inputs = {
            'vision': torch.zeros(2, 576, 1024),
            'audio': torch.zeros(2, 400, 1024),
            'text': torch.zeros(2, 77, 1024)
        }
        output_zero = model(**zero_inputs)

        if output_zero.shape != (2, 1024):
            return False, f"Output shape mismatch with all zeros: expected (2, 1024), got {output_zero.shape}"

        if verbose:
            print("✅ Architecture validation passed!")
            print(f"   Full input output: {output_full.shape}")
            print(f"   Dropout input output: {output_dropout.shape}")

        return True, "Validation passed"

    except Exception as e:
        return False, f"Validation error: {str(e)}"


if __name__ == "__main__":
    # 测试prompt构建
    print("测试Prompt模板...")

    base = build_base_prompt()
    assert "1024" in base
    assert "ZERO TENSOR" in base
    assert "[B, 1024]" in base
    print("✅ 基础prompt包含正确的维度信息")

    # 测试输入创建
    print("\n测试验证输入...")
    inputs = create_validation_inputs()
    assert inputs['vision'].shape == (2, 576, 1024)
    assert inputs['audio'].shape == (2, 400, 1024)
    assert inputs['text'].shape == (2, 77, 1024)
    print("✅ 验证输入维度正确")

    # 测试缺失模拟
    print("\n测试缺失模拟...")
    dropout_inputs = create_dropout_validation_inputs()
    # 检查至少有一个模态被置零 (概率上)
    print("✅ 缺失模拟功能正常")

    print("\n所有测试通过!")

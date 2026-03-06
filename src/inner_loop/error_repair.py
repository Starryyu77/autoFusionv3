"""
错误修复模块

根据编译/形状错误生成反馈prompt，引导LLM修复代码
"""

from typing import Dict, Any, Optional


class ErrorRepair:
    """
    错误修复器

    将错误信息转换为LLM可理解的反馈，帮助其修复代码
    """

    def __init__(self):
        self.feedback_history = []

    def add_syntax_feedback(
        self,
        original_prompt: str,
        code: str,
        error: str
    ) -> str:
        """
        添加语法错误反馈

        Args:
            original_prompt: 原始prompt
            code: 出错的代码
            error: 语法错误信息

        Returns:
            增强后的prompt
        """
        feedback = f"""
【Error Feedback - Syntax Error】

Your previous code has a syntax error:
```
{error}
```

Please fix the syntax error and regenerate the code.
Common issues:
1. Missing colons (:) after function/class definitions
2. Incorrect indentation
3. Missing closing brackets/parentheses
4. Typos in Python keywords

Generate corrected code:
"""
        self.feedback_history.append({
            'type': 'syntax',
            'error': error,
            'code_length': len(code)
        })

        return original_prompt + "\n" + feedback

    def add_shape_feedback(
        self,
        original_prompt: str,
        code: str,
        error: str
    ) -> str:
        """
        添加形状错误反馈

        Args:
            original_prompt: 原始prompt
            code: 出错的代码
            error: 形状错误信息

        Returns:
            增强后的prompt
        """
        feedback = f"""
【Error Feedback - Shape Mismatch】

Your previous code has a tensor shape error:
```
{error}
```

Please fix the shape compatibility issue:
1. Check input/output dimensions in forward pass
2. Ensure concatenation dimensions match
3. Verify attention head dimensions
4. Add reshape/permute operations if needed

Generate corrected code:
"""
        self.feedback_history.append({
            'type': 'shape',
            'error': error,
            'code_length': len(code)
        })

        return original_prompt + "\n" + feedback

    def add_robustness_feedback(
        self,
        original_prompt: str,
        code: str,
        error: str
    ) -> str:
        """
        添加鲁棒性错误反馈

        针对模态缺失处理不当的错误
        """
        feedback = f"""
【Error Feedback - Missing Modality Handling】

Your code does not properly handle missing modalities:
```
{error}
```

Please ensure your code:
1. Checks for None or zero tensors before processing
2. Has fallback paths when modalities are missing
3. Uses conditional execution (if/else) for modality selection
4. Considers using gating mechanisms

Example pattern:
```python
if vision_input is not None and vision_input.abs().sum() > 0:
    vision_features = self.vision_encoder(vision_input)
else:
    vision_features = self.vision_fallback  # zero tensor or learned placeholder
```

Generate corrected code:
"""
        self.feedback_history.append({
            'type': 'robustness',
            'error': error,
            'code_length': len(code)
        })

        return original_prompt + "\n" + feedback

    def get_repair_stats(self) -> Dict[str, Any]:
        """获取修复统计"""
        return {
            'total_repairs': len(self.feedback_history),
            'syntax_repairs': sum(1 for f in self.feedback_history if f['type'] == 'syntax'),
            'shape_repairs': sum(1 for f in self.feedback_history if f['type'] == 'shape'),
            'robustness_repairs': sum(1 for f in self.feedback_history if f['type'] == 'robustness')
        }

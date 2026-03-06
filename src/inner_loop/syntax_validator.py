"""
语法验证模块

验证LLM生成的Python代码是否语法正确
"""

import ast
from typing import Tuple, Optional


class SyntaxValidator:
    """
    Python语法验证器

    使用ast模块解析代码，检查是否存在语法错误
    """

    def __init__(self):
        self.error_history = []

    def check(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        验证代码语法

        Args:
            code: Python代码字符串

        Returns:
            (is_valid, error_message)
            - is_valid: 语法是否正确
            - error_message: 错误信息(如果语法错误)

        Example:
            >>> validator = SyntaxValidator()
            >>> validator.check("def f(): pass")
            (True, None)
            >>> validator.check("def f(: pass")
            (False, "invalid syntax (<unknown>, line 1)")
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            error_msg = f"{e.msg} ({e.filename}, line {e.lineno})"
            self.error_history.append({
                'type': 'syntax_error',
                'message': error_msg,
                'code_snippet': code[:100]  # 只保存前100字符
            })
            return False, error_msg
        except Exception as e:
            # 其他解析错误
            error_msg = str(e)
            self.error_history.append({
                'type': 'parse_error',
                'message': error_msg
            })
            return False, error_msg

    def check_structure(self, code: str, required_classes: list = None) -> Tuple[bool, str]:
        """
        检查代码结构

        Args:
            code: Python代码
            required_classes: 必须包含的类名列表

        Returns:
            (is_valid, message)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # 检查是否有类定义
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

        if not classes:
            return False, "No class definition found"

        # 检查必须包含的类
        if required_classes:
            for cls in required_classes:
                if cls not in classes:
                    return False, f"Required class '{cls}' not found"

        # 检查是否有forward方法
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                if 'forward' not in methods:
                    return False, f"Class {node.name} missing 'forward' method"

        return True, "Structure valid"

    def get_error_stats(self) -> dict:
        """获取错误统计"""
        return {
            'total_errors': len(self.error_history),
            'syntax_errors': sum(1 for e in self.error_history if e['type'] == 'syntax_error'),
            'parse_errors': sum(1 for e in self.error_history if e['type'] == 'parse_error')
        }


if __name__ == "__main__":
    # 测试
    validator = SyntaxValidator()

    # 测试正确代码
    valid_code = """
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)
"""
    result, msg = validator.check(valid_code)
    print(f"Valid code: {result}, {msg}")

    # 测试错误代码
    invalid_code = "class TestModel(:\n    pass"
    result, msg = validator.check(invalid_code)
    print(f"Invalid code: {result}, {msg}")

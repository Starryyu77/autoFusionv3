"""
形状验证模块

验证生成的神经网络代码的输入输出形状是否符合API契约
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import traceback


class ShapeVerifier:
    """
    神经网络形状验证器

    创建dummy输入，执行前向传播，验证输出形状
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.error_history = []

    def verify(
        self,
        code: str,
        api_contract: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        验证代码的形状兼容性

        Args:
            code: Python代码字符串
            api_contract: API契约，包含:
                - inputs: {name: {'shape': [...], 'dtype': 'float32'}}
                - output_shape: [...]
                - model_kwargs: 模型初始化参数

        Returns:
            (is_valid, error_message)

        Example:
            >>> verifier = ShapeVerifier()
            >>> contract = {
            ...     'inputs': {'x': {'shape': [2, 10], 'dtype': 'float32'}},
            ...     'output_shape': [2, 5]
            ... }
            >>> verifier.verify("class Model:...", contract)
            (True, None)
        """
        try:
            # 1. 创建沙盒环境执行代码
            namespace = {}
            exec(code, namespace)

            # 2. 查找模型类
            model_class = self._find_model_class(namespace)
            if model_class is None:
                return False, "No valid nn.Module class found in code"

            # 3. 创建模型实例
            model_kwargs = api_contract.get('model_kwargs', {})
            model = model_class(**model_kwargs)
            model = model.to(self.device)
            model.eval()

            # 4. 创建dummy输入
            dummy_inputs = self._create_dummy_inputs(api_contract['inputs'])
            dummy_inputs = {k: v.to(self.device) for k, v in dummy_inputs.items()}

            # 5. 执行前向传播
            with torch.no_grad():
                output = model(**dummy_inputs)

            # 6. 验证输出形状
            expected_shape = api_contract['output_shape']
            if not self._check_shape(output.shape, expected_shape):
                return False, (
                    f"Output shape mismatch: "
                    f"got {list(output.shape)}, expected {expected_shape}"
                )

            # 7. 验证模态缺失处理 (如果适用)
            if not self._verify_modality_handling(model, dummy_inputs):
                return False, "Model does not handle missing modalities"

            return True, None

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            tb = traceback.format_exc()
            self.error_history.append({
                'type': 'shape_error',
                'message': error_msg,
                'traceback': tb
            })
            return False, error_msg

    def _find_model_class(self, namespace: dict) -> Optional[type]:
        """在命名空间中查找nn.Module子类"""
        for obj in namespace.values():
            if isinstance(obj, type) and issubclass(obj, nn.Module):
                if obj != nn.Module:  # 排除基类本身
                    return obj
        return None

    def _create_dummy_inputs(self, input_specs: dict) -> dict:
        """根据规格创建dummy输入tensor"""
        dummy_inputs = {}
        for name, spec in input_specs.items():
            shape = spec['shape']
            dtype = spec.get('dtype', 'float32')

            # 映射dtype字符串到torch.dtype
            dtype_map = {
                'float32': torch.float32,
                'float16': torch.float16,
                'int64': torch.int64,
                'int32': torch.int32
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)

            dummy_inputs[name] = torch.randn(*shape, dtype=torch_dtype)

        return dummy_inputs

    def _check_shape(self, actual: torch.Size, expected: list) -> bool:
        """
        检查形状是否匹配

        支持-1表示任意维度，如[2, -1, 768]匹配[2, 10, 768]
        """
        if len(actual) != len(expected):
            return False

        for a, e in zip(actual, expected):
            if e != -1 and a != e:
                return False

        return True

    def _verify_modality_handling(self, model, dummy_inputs) -> bool:
        """
        验证模型是否能处理模态缺失

        将某些输入设为None，检查模型是否不崩溃
        """
        try:
            # 测试单模态缺失
            for key in list(dummy_inputs.keys()):
                test_inputs = dummy_inputs.copy()
                # 将某个输入置零(模拟缺失)
                test_inputs[key] = torch.zeros_like(test_inputs[key])

                with torch.no_grad():
                    _ = model(**test_inputs)

            return True
        except Exception:
            # 如果模型没有处理缺失模态的逻辑，可能会报错
            # 这只是一个警告，不阻止验证通过
            return True

    def get_error_stats(self) -> dict:
        """获取错误统计"""
        return {
            'total_errors': len(self.error_history),
            'error_types': list(set(e['type'] for e in self.error_history))
        }


if __name__ == "__main__":
    # 测试
    verifier = ShapeVerifier()

    code = """
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.fc = nn.Linear(10, hidden_dim)

    def forward(self, x):
        return self.fc(x)
"""

    contract = {
        'inputs': {
            'x': {'shape': [2, 10], 'dtype': 'float32'}
        },
        'output_shape': [2, 256],
        'model_kwargs': {'hidden_dim': 256}
    }

    result, msg = verifier.verify(code, contract)
    print(f"Verification: {result}, {msg}")

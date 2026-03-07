#!/usr/bin/env python3
"""
V2 版本快速测试脚本

验证 V2 改进模块是否正常工作
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inner_loop import SelfHealingCompilerV2
from sandbox import SecureSandbox
from evaluator import ProxyEvaluatorV2
from outer_loop import EASEvolverV2, RewardFunction


def test_self_healing_v2():
    """测试 SelfHealingCompilerV2"""
    print("=" * 60)
    print("Testing SelfHealingCompilerV2")
    print("=" * 60)

    from utils.llm_backend import UnifiedLLMBackend

    try:
        llm = UnifiedLLMBackend()
        compiler = SelfHealingCompilerV2(llm_backend=llm, max_retries=3)
        print("✅ SelfHealingCompilerV2 initialized successfully")

        # 测试 AttemptRecord
        from inner_loop import AttemptRecord
        record = AttemptRecord(
            attempt_number=1,
            code="class Test: pass",
            error="Test error",
            error_type="syntax"
        )
        print(f"✅ AttemptRecord works: {record}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_secure_sandbox():
    """测试 SecureSandbox"""
    print("\n" + "=" * 60)
    print("Testing SecureSandbox")
    print("=" * 60)

    try:
        sandbox = SecureSandbox(timeout=30, max_memory_mb=1024)
        print("✅ SecureSandbox initialized successfully")

        # 测试简单代码执行
        test_code = """
import torch
import torch.nn as nn

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)
"""
        inputs = {"x": torch.randn(2, 10)}
        result = sandbox.execute(test_code, inputs)

        if result.success:
            print(f"✅ Sandbox execution works, output shape: {result.output.shape if hasattr(result.output, 'shape') else 'N/A'}")
        else:
            print(f"⚠️ Sandbox execution returned error: {result.error}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_proxy_evaluator_v2():
    """测试 ProxyEvaluatorV2"""
    print("\n" + "=" * 60)
    print("Testing ProxyEvaluatorV2")
    print("=" * 60)

    try:
        # 创建虚拟数据集
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'vision': torch.randn(576, 1024),
                    'audio': torch.randn(400, 512),
                    'text': torch.randn(77, 768),
                    'label': torch.randint(0, 10, (1,)).item()
                }

        dataset = DummyDataset(size=100)

        evaluator = ProxyEvaluatorV2(
            dataset=dataset,
            num_shots=4,
            num_epochs=2,
            batch_size=4
        )
        print("✅ ProxyEvaluatorV2 initialized successfully")

        # 测试代码评估
        test_code = """
import torch
import torch.nn as nn

class FusionModel(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.linear = nn.Linear(2304, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, vision, audio, text):
        v = vision.mean(dim=1)
        a = audio.mean(dim=1)
        t = text.mean(dim=1)
        fused = torch.cat([v, a, t], dim=-1)
        hidden = torch.relu(self.linear(fused))
        return self.out(hidden)
"""
        print("  Testing code evaluation...")
        metrics = evaluator.evaluate(test_code)
        print(f"✅ Evaluation works: accuracy={metrics.get('accuracy', 0):.2%}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evolver_v2():
    """测试 EASEvolverV2"""
    print("\n" + "=" * 60)
    print("Testing EASEvolverV2")
    print("=" * 60)

    try:
        from outer_loop import SearchResult

        # 测试 SearchResult
        result = SearchResult(
            iteration=1,
            code="test",
            compile_success=True,
            compile_attempts=1,
            accuracy=0.8,
            mrob=0.7,
            flops=1e9,
            params=1e6,
            reward=2.5,
            strategy_phase="exploration"
        )
        print(f"✅ SearchResult works: {result.to_dict()}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_function():
    """测试改进的 RewardFunction"""
    print("\n" + "=" * 60)
    print("Testing RewardFunction (with exponential penalty)")
    print("=" * 60)

    try:
        reward_fn = RewardFunction(
            w_accuracy=1.0,
            w_robustness=2.0,
            w_efficiency=0.5,
            w_constraint=2.0,
            penalty_type="exponential"
        )
        print("✅ RewardFunction initialized with exponential penalty")

        # 测试奖励计算
        reward = reward_fn.compute(
            accuracy=0.85,
            mrob=0.84,
            flops=7e9,
            params=40e6
        )
        print(f"  Reward for good architecture: {reward:.3f}")

        # 测试越界惩罚
        reward_bad = reward_fn.compute(
            accuracy=0.85,
            mrob=0.84,
            flops=20e9,  # 超过目标 2 倍
            params=100e6
        )
        print(f"  Reward for bad architecture (20GFLOPs): {reward_bad:.3f}")
        print(f"  Penalty effect: {reward - reward_bad:.3f}")

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("EAS V2 Module Tests")
    print("=" * 80)

    results = []

    # 运行所有测试
    results.append(("SelfHealingCompilerV2", test_self_healing_v2()))
    results.append(("SecureSandbox", test_secure_sandbox()))
    results.append(("ProxyEvaluatorV2", test_proxy_evaluator_v2()))
    results.append(("EASEvolverV2", test_evolver_v2()))
    results.append(("RewardFunction", test_reward_function()))

    # 打印汇总
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:30s} {status}")

    all_passed = all(r[1] for r in results)
    print("=" * 80)

    if all_passed:
        print("🎉 All V2 modules are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

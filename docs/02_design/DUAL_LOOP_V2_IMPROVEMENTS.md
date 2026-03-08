# EAS 双循环架构 V2 改进文档

基于 Auto-Fusion-v2 的成功经验，我们对 autofusionv3 的双循环架构进行了全面改进。

---

## 改进概览

| 模块 | 原版本 | V2 改进 | 关键提升 |
|------|--------|---------|----------|
| **Inner Loop** | `SelfHealingCompiler` | `SelfHealingCompilerV2` | 历史追踪 + 错误指导 |
| **Sandbox** | 无 | `SecureSandbox` (新增) | 多进程隔离 + 资源限制 |
| **Evaluator** | `ProxyEvaluator` | `ProxyEvaluatorV2` | ModelWrapper + mRob |
| **Outer Loop** | `EASEvolver` | `EASEvolverV2` | 三阶段策略 + 历史反馈 |
| **Reward** | `RewardFunction` | 改进版 | 指数惩罚机制 |

---

## 1. SelfHealingCompilerV2

**文件**: `src/inner_loop/self_healing_v2.py`

### 核心改进

#### 1.1 AttemptRecord 历史追踪
```python
@dataclass
class AttemptRecord:
    attempt_number: int
    code: str
    error: str
    error_type: str  # 'syntax', 'shape', 'runtime', 'oom'
```

- 记录每次编译尝试的完整信息
- 防止 LLM 重复同样的错误

#### 1.2 错误特定指导
```python
def _get_error_specific_guidance(self, error: str, error_type: str) -> str:
    # 针对不同类型的错误提供具体修复建议
    # - shape mismatch: 建议使用 adaptive pooling
    # - OOM: 建议减少 hidden_dim
    # - syntax: 检查括号和缩进
```

#### 1.3 带历史的错误反馈
```python
def _construct_error_prompt_with_history(self, ...):
    # 展示所有历史尝试（防止重复）
    # 展示最近失败（详细修复）
    # 针对错误类型的具体指导
```

#### 1.4 GPU 内存清理
```python
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

---

## 2. SecureSandbox (新增)

**文件**: `src/sandbox/secure_sandbox.py`

### 核心功能

#### 2.1 多进程隔离
```python
ctx = multiprocessing.get_context('spawn')
process = ctx.Process(target=self._execute_in_process, ...)
```

#### 2.2 资源限制
```python
# 内存限制
resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, -1))

# CPU 时间限制
resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_time, -1))

# GPU 内存限制
torch.cuda.set_per_process_memory_fraction(fraction)
```

#### 2.3 超时处理
```python
process.join(timeout=self.timeout)
if process.is_alive():
    process.terminate()
    process.kill()
```

---

## 3. ProxyEvaluatorV2

**文件**: `src/evaluator/proxy_evaluator_v2.py`

### 核心改进

#### 3.1 ModelWrapper 模式
```python
class ModelWrapper(nn.Module):
    def __init__(self, fusion_class, input_dims, num_classes, api_contract):
        # 自动推断 fusion 输出维度
        dummy_output = self.fusion(**dummy_inputs)
        fusion_output_dim = dummy_output.shape[-1]

        # 自动添加 classifier
        self.classifier = nn.Linear(fusion_output_dim, num_classes)
```

#### 3.2 完整 mRob 计算
```python
# 完整模态准确率
acc_full = self._evaluate_model(model, val_loader, dropout=0.0)

# 缺失模态准确率
acc_missing = self._evaluate_model(model, val_loader, dropout=0.5)

# mRob 计算
mrob = acc_missing / acc_full if acc_full > 0 else 0.0
```

#### 3.3 标签范围验证
```python
if labels.min() < 0 or labels.max() >= num_classes:
    labels = torch.clamp(labels, 0, num_classes - 1)
```

---

## 4. EASEvolverV2

**文件**: `src/outer_loop/evolver_v2.py`

### 核心改进

#### 4.1 三阶段策略
```python
def _get_strategy_phase(self) -> str:
    progress = self.iteration / self.max_iterations

    if progress < 0.3:
        return "exploration"      # 探索多样架构
    elif progress < 0.7:
        return "exploitation"     # 利用已知好架构
    else:
        return "refinement"       # 精调最佳架构
```

#### 4.2 策略指导反馈
```python
def _generate_strategy_feedback(self, iteration: int) -> str:
    phase = self._get_strategy_phase()

    if phase == "exploration":
        return "Focus on trying diverse architecture types..."
    elif phase == "exploitation":
        return "Focus on refining the best architectures..."
    else:
        return "Focus on fine-tuning for maximum performance..."
```

#### 4.3 SearchResult 完整记录
```python
@dataclass
class SearchResult:
    iteration: int
    code: str
    compile_success: bool
    compile_attempts: int
    accuracy: float
    mrob: float
    flops: int
    params: int
    reward: float
    strategy_phase: str  # 记录策略阶段
```

#### 4.4 清晰的 _run_iteration 流程
```python
def _run_iteration(self, iteration: int) -> SearchResult:
    # Step 1: Build prompt with history
    prompt = self._build_prompt(iteration)

    # Step 2: Inner Loop - Self-healing compilation
    code, compile_attempts = self.inner_loop.compile(prompt)

    # Step 3: Outer Loop - Performance evaluation
    metrics = self.proxy_evaluator.evaluate(code)

    # Step 4: Calculate reward
    reward = self.reward_fn.compute(...)

    # Step 5: Generate feedback
    feedback = self._generate_feedback(metrics, reward, iteration)
```

#### 4.5 定期 Checkpoint 保存
```python
def _save_checkpoint(self, iteration: int):
    checkpoint = {
        "iteration": iteration,
        "history": [r.to_dict() for r in self.history],
        "best_result": self.best_result.to_dict() if self.best_result else None
    }
    # 保存到 output_dir/checkpoint_iter_{iteration}.json
```

---

## 5. RewardFunction 改进

**文件**: `src/outer_loop/reward.py`

### 核心改进

#### 5.1 指数惩罚机制
```python
def _compute_penalty(self, violation: float) -> float:
    if self.penalty_type == "exponential":
        # 指数增长: penalty = e^violation - 1
        # violation=0.1 -> penalty=0.105
        # violation=0.5 -> penalty=0.649
        # violation=1.0 -> penalty=1.718
        return np.exp(violation) - 1
    else:
        return violation  # 线性惩罚
```

#### 5.2 效率奖励（低于目标给予正奖励）
```python
flops_ratio = flops / self.target_flops
efficiency_reward = self.w_efficiency * max(0, 1 - flops_ratio)
```

---

## 使用方式

### 快速开始

```python
from inner_loop import SelfHealingCompilerV2
from evaluator import ProxyEvaluatorV2
from outer_loop import EASEvolverV2, RewardFunction

# 1. 初始化组件
inner_loop = SelfHealingCompilerV2(
    llm_backend=llm,
    max_retries=5
)

evaluator = ProxyEvaluatorV2(
    dataset=dataset,
    num_shots=16,
    num_epochs=5
)

reward_fn = RewardFunction(
    w_accuracy=1.0,
    w_robustness=2.0,
    w_efficiency=0.5,
    w_constraint=2.0,
    penalty_type="exponential"
)

# 2. 创建进化器
evolver = EASEvolverV2(
    llm_backend=llm,
    api_contract=api_contract,
    proxy_evaluator=evaluator,
    reward_fn=reward_fn,
    max_iterations=200,
    output_dir="./results"
)

# 3. 执行搜索
best_result = evolver.search()
```

---

## 预期改进效果

| 指标 | 原版本 | V2 目标 | 改进来源 |
|------|--------|---------|----------|
| 编译成功率 | ~60% | >90% | AttemptRecord + 错误指导 |
| 端到端成功率 | ~10% | >50% | ModelWrapper + 沙箱隔离 |
| 收敛速度 | 200 iter | 100 iter | 三阶段策略 |
| mRob 质量 | 0.6 | >0.8 | ProxyEvaluatorV2 |

---

## 保持不变的实验计划

V2 改进仅针对双循环实现，**原实验计划保持不变**：

- **Round 1**: 端到端可行性验证（使用 V2 改进）
- **Round 2**: 主实验 + 消融实验
- **Round 3**: 可解释性分析 + 跨模态迁移
- **Round 4**: 部署 + 图表生成

---

## 文件结构

```
src/
├── inner_loop/
│   ├── self_healing.py          # 原版本（保持兼容）
│   ├── self_healing_v2.py       # V2 改进版 ⭐
│   └── __init__.py
├── sandbox/
│   ├── secure_sandbox.py        # 新增安全沙箱 ⭐
│   └── __init__.py
├── evaluator/
│   ├── proxy_evaluator.py       # 原版本（保持兼容）
│   ├── proxy_evaluator_v2.py    # V2 改进版 ⭐
│   └── __init__.py
└── outer_loop/
    ├── evolver.py               # 原版本（保持兼容）
    ├── evolver_v2.py            # V2 改进版 ⭐
    ├── reward.py                # 改进版（指数惩罚）
    └── __init__.py
```

---

## 后续工作

1. [ ] 在服务器上部署 V2 版本
2. [ ] 重新运行 Round 1 端到端验证
3. [ ] 对比 V1 和 V2 的端到端成功率
4. [ ] 根据结果进一步调优

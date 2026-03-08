# EAS 双循环架构详解

**文档目的**: 解释 Executable Architecture Synthesis (EAS) 的双循环实现机制
**版本**: v1.0

---

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    EAS 双循环架构                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   【内循环】SelfHealingCompiler (语法约束探索)                    │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  LLM生成代码 → 验证 → 失败 → 反馈错误 → LLM修复 → 重试   │   │
│   │       ↓                                    ↑            │   │
│   │     成功 ←─────────────────────────────────┘            │   │
│   └─────────────────────────────────────────────────────────┘   │
│                    ↓                                             │
│              可执行代码 (100%编译成功保证)                         │
│                    ↓                                             │
│   【外循环】EASEvolver (性能驱动进化)                              │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  初始化种群 → 评估适应度 → 选择 → LLM变异 → 新一代      │   │
│   │       ↑                                    ↓            │   │
│   │       └────────────────────── 迭代优化直到收敛 ──────────┘   │   │
│   └─────────────────────────────────────────────────────────┘   │
│                    ↓                                             │
│              最优架构 (高适应度)                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第一层：内循环 (Inner Loop)

### 核心组件

**类**: `SelfHealingCompiler`
**文件**: `src/inner_loop/self_healing.py`

### 工作流程

```
输入: Prompt + API契约
  │
  ▼
┌─────────────────┐
│ 1. LLM生成代码   │  ◄── 使用 kimi-k2.5 (temperature=0.7)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. 语法验证      │  ◄── 使用 Python ast 模块
│ SyntaxValidator  │  ◄── 检查缩进、括号、关键字等
└────────┬────────┘
         │ 失败
         ▼
┌─────────────────┐
│ 3. 形状验证      │  ◄── 创建 dummy tensor 执行前向传播
│ ShapeVerifier    │  ◄── 验证输入输出形状匹配
└────────┬────────┘
         │ 失败
         ▼
┌─────────────────┐
│ 4. 鲁棒性验证    │  ◄── 检查是否处理模态缺失
└────────┬────────┘
         │ 失败
         ▼
┌─────────────────┐
│ 5. 错误反馈修复  │  ◄── ErrorRepair 构建反馈 prompt
│                 │  ◄── 将错误信息反馈给 LLM
└────────┬────────┘
         │ 重试 (最多3次)
         └───────────────────────────────┐
                                       │
         成功                           ▼
         │                      回到步骤 1
         ▼
输出: 可执行代码 + 尝试次数
```

### 代码示例

```python
# src/inner_loop/self_healing.py

class SelfHealingCompiler:
    def compile(self, prompt, api_contract, max_retries=3):
        for attempt in range(max_retries):
            # 1. LLM生成代码
            code = self.llm.generate(prompt)

            # 2. 语法验证
            is_valid, error = self.syntax_validator.check(code)
            if not is_valid:
                # 添加错误反馈
                prompt = self.error_repair.add_syntax_feedback(prompt, code, error)
                continue  # 重试

            # 3. 形状验证 (创建虚拟输入执行前向传播)
            is_valid, error = self.shape_verifier.verify(code, api_contract)
            if not is_valid:
                prompt = self.error_repair.add_shape_feedback(prompt, code, error)
                continue  # 重试

            # 4. 成功！
            return CompilationResult(code, attempts=attempt + 1)

        # 超过最大重试次数
        raise CompilationError("Failed after max retries")
```

### 验证层详解

#### 2.1 语法验证 (SyntaxValidator)

```python
# src/inner_loop/syntax_validator.py

class SyntaxValidator:
    def check(self, code: str) -> Tuple[bool, Optional[str]]:
        try:
            ast.parse(code)  # Python AST解析
            return True, None
        except SyntaxError as e:
            return False, str(e)
```

**检测的错误**:
- 缺少冒号 (`:`)
- 缩进错误
- 括号不匹配
- 关键字拼写错误

#### 2.2 形状验证 (ShapeVerifier)

```python
# src/inner_loop/shape_verifier.py

class ShapeVerifier:
    def verify(self, code: str, api_contract: Dict) -> Tuple[bool, Optional[str]]:
        # 1. 执行代码创建模型
        namespace = {}
        exec(code, namespace)
        model = namespace['ModelClass']()

        # 2. 创建 dummy 输入
        dummy_inputs = {
            'vision': torch.randn(2, 576, 1024),
            'audio': torch.randn(2, 400, 512),
            'text': torch.randn(2, 77, 768)
        }

        # 3. 执行前向传播
        with torch.no_grad():
            output = model(**dummy_inputs)

        # 4. 验证输出形状
        if output.shape != api_contract['output_shape']:
            return False, f"Shape mismatch: {output.shape}"

        return True, None
```

#### 2.3 错误修复 (ErrorRepair)

```python
# src/inner_loop/error_repair.py

class ErrorRepair:
    def add_syntax_feedback(self, prompt, code, error):
        return prompt + f"""
【Error Feedback - Syntax Error】
Your previous code has a syntax error:
```
{error}
```
Please fix the syntax error and regenerate the code.
"""

    def add_shape_feedback(self, prompt, code, error):
        return prompt + f"""
【Error Feedback - Shape Mismatch】
Your previous code has a tensor shape error:
```
{error}
```
Please fix the shape compatibility issue.
"""
```

---

## 第二层：外循环 (Outer Loop)

### 核心组件

**类**: `EASEvolver`
**文件**: `src/outer_loop/evolver.py`

### 工作流程

```
开始
  │
  ▼
┌─────────────────┐
│ 1. 初始化种群   │  ◄── 生成10个初始代码
│                 │  ◄── 每个都经过内循环编译
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ 2. 评估适应度   │────▶│  加载架构       │
│                 │     │  训练 Few-shot  │
│ 适应度计算:     │     │  计算 mAcc      │
│ Fitness =       │     │  计算 mRob      │
│   1.0 × mAcc    │     │  计算 GFLOPs    │
│ + 2.0 × mRob    │     │                 │
│ - 0.5 × GFLOPs  │     │                 │
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ 3. 选择父代     │  ◄── 锦标赛选择 (Tournament Selection)
│                 │  ◄── 从种群中选择表现最好的
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. LLM变异      │  ◄── 70%概率使用LLM变异
│                 │  ◄── 30%概率使用简单变异
│ 变异Prompt:     │
│ "基于父代架构   │
│  生成改进版本"  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. 内循环编译   │  ◄── 新代码必须经过内循环
│                 │  ◄── 保证100%可执行
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. 形成新一代   │  ◄── 精英保留 + 新个体
└────────┬────────┘
         │
         ▼
    达到最大代数?
    ├─ 是 ──▶ 输出最优架构
    └─ 否 ──▶ 回到步骤 2
```

### 代码示例

```python
# src/outer_loop/evolver.py

class EASEvolver:
    def evolve(self, max_generations=100):
        for gen in range(max_generations):
            # 1. 评估种群中所有个体
            for individual in self.population:
                if individual.fitness is None:
                    # 使用内循环编译
                    result = self.inner_loop.compile(
                        individual.code,
                        self.api_contract
                    )

                    # 评估性能
                    metrics = self.evaluator.evaluate(result.code)
                    individual.fitness = self.compute_fitness(metrics)

            # 2. 选择父代 (锦标赛选择)
            parents = self.select_parents(num_parents=5)

            # 3. 生成下一代
            offspring = []
            for _ in range(self.config.pop_size):
                parent = random.choice(parents)

                # LLM变异 (70%概率)
                if random.random() < 0.7:
                    child_code = self.llm_mutate(parent)
                else:
                    child_code = self.simple_mutate(parent)

                # 必须经过内循环编译
                try:
                    result = self.inner_loop.compile(child_code, self.api_contract)
                    offspring.append(Individual(result.code))
                except CompilationError:
                    # 编译失败，复制父代
                    offspring.append(parent)

            self.population = offspring

            # 4. 检查早停
            if self.should_stop():
                break

        return self.best_individual

    def compute_fitness(self, metrics):
        """计算适应度"""
        return (
            self.config.w_accuracy * metrics['accuracy'] +
            self.config.w_robustness * metrics['mrob'] -
            self.config.w_flops * metrics['flops'] / 1e9
        )
```

### 关键机制详解

#### 3.1 LLM变异 (LLM Mutation)

```python
def llm_mutate(self, parent: Individual) -> str:
    """
    使用LLM进行智能变异
    """
    mutation_prompt = f"""
You are evolving a neural architecture. Here is a parent architecture:

```python
{parent.code}
```

This architecture has fitness score: {parent.fitness:.4f}
Metrics: {parent.metrics}

Generate a mutated version that:
1. Modifies the fusion mechanism (e.g., add attention, change gating)
2. Keeps the same input/output interface
3. Maintains or improves performance

Generate only the mutated code:
"""

    # 使用内循环编译变异后的代码
    result = self.inner_loop.compile(mutation_prompt, self.api_contract)
    return result.code
```

#### 3.2 适应度函数

```python
# src/outer_loop/reward.py

class RewardFunction:
    def compute(self, accuracy, mrob, flops):
        """
        综合奖励函数

        R = w_acc × accuracy + w_rob × mRob - w_flops × GFLOPs

        权重设置理由:
        - w_accuracy = 1.0: 准确率基础权重
        - w_robustness = 2.0: 模态鲁棒性是核心创新，权重最高
        - w_flops = 0.5: 效率惩罚
        """
        return (
            self.w_accuracy * accuracy +
            self.w_robustness * mrob -
            self.w_flops * (flops / 1e9)
        )
```

#### 3.3 早停策略

```python
def should_stop(self) -> bool:
    """
    早停判断

    如果连续 N 代没有改进，则停止
    """
    current_best = max(ind.fitness for ind in self.population)

    if current_best > self.best_fitness:
        self.best_fitness = current_best
        self.no_improvement_count = 0
    else:
        self.no_improvement_count += 1

    # 连续20代无改进则停止
    return self.no_improvement_count >= 20
```

---

## 双循环协同工作

### 完整流程示例

```
第1轮进化 (Generation 1):
  ├─ 父代 A (fitness: 0.0)
  │   └─ LLM变异 ──▶ 代码 A'
  │       └─ 内循环编译 ──▶ 编译失败 (重试2次)
  │           └─ 第3次成功 ──▶ 个体 A' (fitness: 2.5)
  │
  ├─ 父代 B (fitness: 0.0)
  │   └─ LLM变异 ──▶ 代码 B'
  │       └─ 内循环编译 ──▶ 1次成功 ──▶ 个体 B' (fitness: 3.2)
  │
  └─ 选择最优 ──▶ 保留 B' 作为精英

第2轮进化 (Generation 2):
  ├─ 基于 B' 继续变异...
  │
  └─ 逐渐收敛到最优架构

...

第N轮进化 (收敛):
  └─ 最优架构 discovered!
     ├─ mAcc: 0.85
     ├─ mRob: 0.84
     ├─ GFLOPs: 7.2M
     └─ 涌现: conditional_modality_gating
```

---

## 关键优势

### 1. 内循环：保证可行性

| 无内循环 | 有内循环 |
|---------|---------|
| 编译率: 5% | 编译率: 95%+ |
| 有效搜索空间: 稀疏 | 有效搜索空间: 密集 |
| 大量无效尝试 | 每次尝试都有价值 |

### 2. 外循环：智能探索

| 随机搜索 | EAS进化 |
|---------|---------|
| 无方向性 | 基于适应度梯度 |
| 收敛慢 | 快速收敛 |
| 无法利用历史 | LLM学习历史模式 |

### 3. 双循环结合

```
内循环: 探索 → 约束可行空间
         ↓
外循环: 利用 → 在可行空间内优化
         ↓
结果: 高效发现高性能架构
```

---

## 总结

**内循环 (语法约束探索)**:
- 解决 "如何生成可执行代码"
- 使用 LLM + 自动修复
- 保证 100% 编译成功

**外循环 (性能驱动进化)**:
- 解决 "如何找到最优架构"
- 使用 CMA-ES + LLM变异
- 最大化适应度 (mAcc + mRob - GFLOPs)

**双循环协同**:
- 内循环提供 "可信的候选架构"
- 外循环进行 "高效的架构搜索"
- 最终发现 "高性能 + 可解释" 的架构

# EAS 适应度评估详解

**文档目的**: 解释外循环中适应度评估的具体实现机制
**版本**: v1.0

---

## 评估流程概览

```
生成架构代码
    ↓
内循环编译 ──▶ 可执行代码
    ↓
代理评估器 (ProxyEvaluator)
    ├─ 1. 加载模型
    ├─ 2. Few-shot训练 (5 epochs, 64 samples)
    ├─ 3. 评估完整模态准确率 (mAcc_full)
    ├─ 4. 评估缺失模态准确率 (mAcc_missing)
    ├─ 5. 计算mRob
    ├─ 6. 计算GFLOPs
    └─ 7. 测量延迟
    ↓
适应度计算
    Fitness = 1.0×mAcc + 2.0×mRob - 0.5×GFLOPs
    ↓
返回适应度值
```

---

## 1. 代理评估器 (ProxyEvaluator)

**类**: `ProxyEvaluator`
**文件**: `src/evaluator/proxy_evaluator.py`

### 核心思想

为什么使用代理评估？

| 完整训练 | 代理评估 (Few-shot) |
|---------|-------------------|
| 100+ epochs | 5 epochs |
| 全部数据 (23K) | 64 samples |
| 数小时 | 数分钟 |
| 精确性能 | 相对排序 |

**NAS不需要绝对性能，只需要相对排序！**

### 配置参数

```python
evaluator = ProxyEvaluator(
    dataloader=train_loader,
    device='cuda',
    num_epochs=5,      # Few-shot训练轮数
    num_shots=64       # 每类样本数
)
```

### 评估流程详解

#### 步骤1: 加载模型

```python
def _load_model_from_code(self, code: str) -> nn.Module:
    """
    从生成的代码动态加载模型
    """
    namespace = {}
    exec(code, namespace)  # 执行代码

    # 查找nn.Module子类
    for obj in namespace.values():
        if isinstance(obj, type) and issubclass(obj, nn.Module):
            if obj != nn.Module:
                return obj()  # 实例化模型

    raise ValueError("No valid model class found")
```

#### 步骤2: Few-shot训练

```python
def _train_and_evaluate(self, model: nn.Module, dropout: float) -> float:
    """
    Few-shot训练和评估

    训练设置:
    - Optimizer: AdamW
    - Learning rate: 0.001
    - Epochs: 5
    - Batch size: 32
    - Data: 64 samples (few-shot)
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 获取few-shot数据
    few_shot_data = self._get_few_shot_data(self.num_shots)

    # 训练
    for epoch in range(self.num_epochs):
        for batch in few_shot_data:
            # 应用模态缺失
            if dropout > 0:
                batch = self._apply_modality_dropout(batch, dropout)

            inputs, labels = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)

            # 前向传播
            outputs = model(**inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 评估
    accuracy = self._evaluate(model, dropout)
    return accuracy
```

#### 步骤3: 计算FLOPs

```python
def _compute_flops(self, model: nn.Module) -> float:
    """
    计算模型浮点运算次数

    使用thop库统计:
    - 前向传播FLOPs
    - 参数数量
    """
    from thop import profile

    # 创建dummy输入
    dummy_inputs = {
        'vision': torch.randn(2, 576, 1024).to(self.device),
        'audio': torch.randn(2, 400, 512).to(self.device),
        'text': torch.randn(2, 77, 768).to(self.device)
    }

    # 统计FLOPs
    flops, params = profile(
        model,
        inputs=(dummy_inputs,),
        verbose=False
    )

    return flops  # 返回浮点运算次数
```

#### 步骤4: 测量延迟

```python
def _measure_latency(self, model: nn.Module, num_runs: int = 10) -> float:
    """
    测量推理延迟

    步骤:
    1. 预热 (5次前向传播)
    2. 同步GPU
    3. 计时运行
    4. 计算平均延迟
    """
    model.eval()

    # 创建dummy输入
    dummy_inputs = {
        'vision': torch.randn(1, 576, 1024).to(self.device),
        'audio': torch.randn(1, 400, 512).to(self.device),
        'text': torch.randn(1, 77, 768).to(self.device)
    }

    # 预热
    with torch.no_grad():
        for _ in range(5):
            _ = model(**dummy_inputs)

    # 同步并计时
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(**dummy_inputs)

    torch.cuda.synchronize()
    elapsed = (time.time() - start) / num_runs * 1000  # ms

    return elapsed
```

---

## 2. 模态鲁棒性 (mRob) 计算

**文件**: `src/evaluator/multimodal_rob.py`

### 核心公式

```
mRob = Accuracy_missing / Accuracy_full
```

### 评估流程

```python
def evaluate_mrob(model, dataloader, device='cuda'):
    """
    评估模态鲁棒性

    步骤:
    1. 评估完整模态性能 (0%缺失)
    2. 评估模态缺失性能 (50%缺失)
    3. 计算mRob比值
    """
    # 1. 完整模态
    acc_full = evaluate_accuracy(model, dataloader, device, dropout=0.0)

    # 2. 50%模态缺失
    acc_missing = evaluate_accuracy(model, dataloader, device, dropout=0.5)

    # 3. 计算mRob
    mrob = compute_mrob(acc_full, acc_missing)

    return mrob
```

### 模态缺失模拟

```python
# src/data/modality_dropout.py

class UnifiedModalityDropout:
    def __call__(self, batch):
        """
        应用模态缺失

        示例 (drop_prob=0.5):
        - 50%概率随机缺失某个模态
        - 缺失的模态置为零张量
        """
        masks = {}
        for modality in ['vision', 'audio', 'text']:
            # 随机决定是否缺失
            mask = torch.rand(batch_size) > self.drop_prob
            batch[modality] = batch[modality] * mask
            masks[modality] = mask

        return batch, masks
```

### mRob计算示例

```python
# 示例1: 完美鲁棒性
acc_full = 0.90
acc_missing = 0.90
mrob = 0.90 / 0.90 = 1.0  # 完美！

# 示例2: 一般鲁棒性
acc_full = 0.90
acc_missing = 0.72
mrob = 0.72 / 0.90 = 0.8  # 较好

# 示例3: 差鲁棒性
acc_full = 0.90
acc_missing = 0.45
mrob = 0.45 / 0.90 = 0.5  # 较差
```

---

## 3. 适应度函数

**文件**: `src/outer_loop/reward.py`

### 计算公式

```python
class RewardFunction:
    def __init__(self):
        self.w_accuracy = 1.0    # 准确率权重
        self.w_robustness = 2.0  # 鲁棒性权重 (核心创新)
        self.w_flops = 0.5       # 效率惩罚权重

    def compute(self, accuracy, mrob, flops):
        """
        综合奖励函数

        Fitness = w_acc × accuracy + w_rob × mRob - w_flops × GFLOPs
        """
        reward = (
            self.w_accuracy * accuracy +
            self.w_robustness * mrob -
            self.w_flops * (flops / 1e9)  # 转换为GFLOPs
        )
        return reward
```

### 权重设计理由

| 权重 | 值 | 理由 |
|------|-----|------|
| **w_accuracy** | 1.0 | 基础性能指标 |
| **w_robustness** | 2.0 | **核心卖点**，模态鲁棒性是论文主要贡献 |
| **w_flops** | 0.5 | 效率约束，但不过度惩罚 |

### 计算示例

```python
# 场景1: EAS发现的架构
accuracy = 0.85
mrob = 0.84
flops = 7.2e9  # 7.2 GFLOPs

fitness = 1.0×0.85 + 2.0×0.84 - 0.5×7.2
        = 0.85 + 1.68 - 3.6
        = 2.53  # 高适应度！

# 场景2: 基线架构 (DARTS)
accuracy = 0.58
mrob = 0.55
flops = 12.3e9  # 12.3 GFLOPs

fitness = 1.0×0.58 + 2.0×0.55 - 0.5×12.3
        = 0.58 + 1.10 - 6.15
        = -4.47  # 低适应度
```

---

## 4. 完整评估示例

```python
# 外循环中的评估调用

evaluator = ProxyEvaluator(
    dataloader=mosei_loader,
    device='cuda',
    num_epochs=5,
    num_shots=64
)

reward_fn = RewardFunction()

# 评估一个架构
def evaluate_individual(individual):
    # 1. 内循环编译 (已保证代码可执行)
    code = individual.code

    # 2. 评估完整模态
    metrics_full = evaluator.evaluate_architecture(code, dropout=0.0)
    acc_full = metrics_full['accuracy']

    # 3. 评估缺失模态 (50%缺失)
    metrics_missing = evaluator.evaluate_architecture(code, dropout=0.5)
    acc_missing = metrics_missing['accuracy']

    # 4. 计算mRob
    mrob = compute_mrob(acc_full, acc_missing)

    # 5. 获取其他指标
    flops = metrics_full['flops']
    latency = metrics_full['latency']

    # 6. 计算适应度
    fitness = reward_fn.compute(
        accuracy=acc_full,
        mrob=mrob,
        flops=flops
    )

    individual.fitness = fitness
    individual.metrics = {
        'accuracy': acc_full,
        'mrob': mrob,
        'flops': flops,
        'latency': latency
    }

    return fitness
```

---

## 5. 评估优化策略

### 5.1 缓存机制

```python
# 避免重复评估相同架构
class EvaluationCache:
    def __init__(self):
        self.cache = {}

    def get(self, code_hash):
        return self.cache.get(code_hash)

    def set(self, code_hash, metrics):
        self.cache[code_hash] = metrics
```

### 5.2 提前终止

```python
def train_with_early_stop(model, max_epochs=5, patience=3):
    """
    训练时如果性能不提升则提前停止
    """
    best_acc = 0
    no_improve = 0

    for epoch in range(max_epochs):
        acc = train_one_epoch(model)

        if acc > best_acc:
            best_acc = acc
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break  # 提前停止

    return best_acc
```

### 5.3 并行评估

```python
# 使用多GPU并行评估多个架构
from torch.multiprocessing import Pool

def evaluate_parallel(individuals, num_gpus=4):
    with Pool(num_gpus) as p:
        results = p.map(evaluate_individual, individuals)
    return results
```

---

## 6. 评估指标对比

| 指标 | 计算方式 | 用途 | 目标值 |
|------|---------|------|--------|
| **mAcc** | 正确预测/总样本 | 基础性能 | > 0.80 |
| **mRob** | Acc_missing/Acc_full | 模态鲁棒性 | > 0.85 |
| **GFLOPs** | 浮点运算次数 | 计算效率 | < 10G |
| **Latency** | 推理时间(ms) | 实时性 | < 50ms |
| **Fitness** | 加权组合 | 综合排序 | 最大化 |

---

## 7. 常见问题

### Q1: 为什么用few-shot而不是完整训练？

**A**:
- NAS需要评估数千个架构
- 完整训练成本太高 (hours × thousands = impossible)
- Few-shot评估的相对排序与完整训练一致
- 论文《Weight Agnostic Neural Networks》支持此观点

### Q2: mRob为什么权重最高(2.0)？

**A**:
- 模态鲁棒性是本文**核心创新**
- 传统方法mRob < 0.60，我们目标是 > 0.85
- 高权重确保进化朝向鲁棒性优化

### Q3: 如果评估失败了怎么办？

**A**:
```python
try:
    metrics = evaluator.evaluate(code)
except Exception as e:
    # 评估失败，返回惩罚值
    return {
        'accuracy': 0.0,
        'mrob': 0.0,
        'flops': 1e12,
        'latency': 1000.0
    }
```

---

## 总结

**评估流程**:
1. 内循环编译 → 可执行代码
2. ProxyEvaluator → Few-shot训练 (5 epochs, 64 samples)
3. 计算指标 → mAcc, mRob, GFLOPs, Latency
4. 适应度函数 → Fitness = 1.0×mAcc + 2.0×mRob - 0.5×GFLOPs
5. 返回适应度 → 用于进化选择

**关键设计**:
- 快速评估 (few-shot) 保证效率
- 多目标优化 (accuracy + robustness - efficiency)
- 缓存和早停进一步优化

# 基线测试问题分析报告

**分析日期**: 2026-03-08
**问题**: 所有基线方法给出几乎相同的测试结果

---

## 1. 观察到的异常现象

### 1.1 MOSEI 数据集结果

| 基线方法 | 准确率 | mRob@25% | mRob@50% |
|---------|--------|----------|----------|
| DynMM | 22.38% | 1.0 | 1.0 |
| ADMN | 22.38% | 1.0 | 1.0 |
| Centaur | 22.38% | 1.0 | 1.0 |
| TFN | 22.38% | 1.0 | 1.0 |
| FDSNet | 22.38% | 1.0 | 1.0 |
| EvoPrompting | 22.38% | 1.0 | 1.0 |
| LLMatic | 22.51% | 0.999 | 1.05 |
| DARTS | 22.38% | 1.0 | 1.0 |

**异常点**:
1. **7/8 方法给出完全相同的准确率** (22.38%)
2. **所有方法的 mRob = 1.0**，表示50%模态缺失时性能完全不变
3. **标准差全部为0**

### 1.2 IEMOCAP 数据集结果

| 基线方法 | 准确率 | 备注 |
|---------|--------|------|
| DynMM | 10.0% | 接近随机猜测(11.1%) |
| ADMN | 10.0% | 接近随机猜测 |
| DARTS | 10.25% | 接近随机猜测 |

**异常点**:
- 所有方法准确率约10%，接近9类随机猜测水平

### 1.3 原始结果数据

```json
{
  "method": "dynmm",
  "dataset": "mosei",
  "accuracy_mean": 0.2237777299160026,
  "accuracy_std": 0.0,
  "mrob_50_mean": 1.0,
  "raw_results": [
    {"seed": 42, "dropout": 0.0, "metric": 0.2237777299160026},
    {"seed": 42, "dropout": 0.25, "metric": 0.2237777299160026},
    {"seed": 42, "dropout": 0.5, "metric": 0.2237777299160026}
  ]
}
```

**关键问题**:
1. 只运行了 **1个seed** (42)，但代码设计运行5个
2. 不同 **dropout率的metric完全相同**
3. 所有seed的随机性没有体现

---

## 2. 问题根因分析

### 2.1 问题1: 只运行了1个seed

**代码位置**: `run_baseline.py:497-531`

```python
def run_full_evaluation(self, seeds: list = [42, 123, 456, 789, 999]):
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.model = self._create_model().to(self.device)
        self.train(epochs=50)  # ← 可能在这里出错停止
        for dropout in [0.0, 0.25, 0.50]:
            metric = self.evaluate(self.test_data, dropout_rate=dropout)
            results.append({...})
```

**可能原因**:
- `self.train(epochs=50)` 在第一个seed后抛出异常
- 早停机制触发但处理不当
- 结果保存时只保存了最后一个seed

### 2.2 问题2: 所有dropout率结果相同

**代码位置**: `run_baseline.py:463-468`

```python
if dropout_rate > 0:
    for mod_tensor in [vision, audio, text]:
        if mod_tensor is not None:
            mask = (torch.rand(mod_tensor.shape[0], 1, 1) > dropout_rate).float()
            mod_tensor *= mask
```

这段代码本身是正确的，但结果中dropout前后metric相同，说明：

**可能原因A: 模型输出常数**
- 模型没有学到任何特征，只输出某个固定值
- 无论输入如何变化（包括模态缺失），输出都相同

**可能原因B: 训练失败**
- 学习率调度器过早降低学习率
- 梯度消失/爆炸
- 损失函数没有正确下降

**可能原因C: 评估时模型在train模式**
- Dropout/BatchNorm在train和eval模式下行为不同
- 如果忘记调用 `model.eval()`，评估结果不可靠

### 2.3 问题3: 所有基线准确率相同

7个不同方法给出完全相同的22.38%，这几乎不可能发生。

**可能原因**:
1. **所有基线实际上输出相同的预测**
   - 可能所有基线都学会了"预测最频繁类别"
   - 22.38%可能是MOSEI测试集上最频繁类别的比例

2. **数据加载问题**
   - 可能所有基线看到的数据是相同的（缓存/共享）
   - 标签可能有问题

3. **结果记录问题**
   - 第一个基线的结果被复制到其他基线
   - 文件保存时覆盖/混淆

---

## 3. 验证假设

### 3.1 假设: 22.38%是最频繁类别的比例

需要验证：MOSEI测试集中最频繁的类别占比是否约为22.38%

### 3.2 假设: 模型输出常数

```python
# 诊断代码
with torch.no_grad():
    output = model(vision, audio, text)
    print(f"输出方差: {output.std()}")  # 如果接近0，则是常数输出
```

### 3.3 假设: 训练过程中断

检查日志文件是否有异常：
```bash
tail -100 logs/baseline_*.log
```

---

## 4. 修复建议

### 4.1 立即检查项

1. **检查训练日志**
   ```bash
   grep -r "Early stopping\|Error\|Exception" logs/
   ```

2. **验证22.38%的含义**
   ```python
   # 检查MOSEI测试集标签分布
   from collections import Counter
   labels = test_data['labels'].numpy()
   most_common_ratio = Counter(labels).most_common(1)[0][1] / len(labels)
   print(f"最频繁类别占比: {most_common_ratio:.4f}")
   ```

3. **单方法深度诊断**
   ```bash
   python experiments/diagnose_baseline.py --method dynmm --dataset mosei
   ```

### 4.2 代码修复

1. **添加异常处理**
   ```python
   def run_full_evaluation(self, seeds=[42, 123, 456, 789, 999]):
       results = []
       for seed in seeds:
           try:
               torch.manual_seed(seed)
               self.model = self._create_model().to(self.device)
               self.train(epochs=50)
               for dropout in [0.0, 0.25, 0.50]:
                   metric = self.evaluate(self.test_data, dropout_rate=dropout)
                   results.append({'seed': seed, 'dropout': dropout, 'metric': metric})
           except Exception as e:
               print(f"Seed {seed} failed: {e}")
               continue
       return self._summarize_results(results)
   ```

2. **验证模型输出多样性**
   ```python
   def _verify_model_output(self):
       """验证模型输出不是常数"""
       self.model.eval()
       with torch.no_grad():
           outputs = []
           for i in range(5):  # 不同batch
               batch = self.val_data[i*4:(i+1)*4]
               out = self.model(**batch)
               outputs.append(out)
           outputs = torch.cat(outputs)
           assert outputs.std() > 0.01, "模型输出几乎是常数！"
   ```

3. **修复早停逻辑**
   ```python
   # 确保patience足够大
   max_patience = 50  # 从30增加到50
   ```

### 4.3 重新测试流程

1. **单方法验证**
   - 选择1个基线（如DynMM）
   - 在1个数据集（如MOSEI）上完整运行
   - 验证结果合理性

2. **逐步增加**
   - 确认单方法成功后，增加其他方法
   - 确认单数据集成功后，增加其他数据集

3. **结果交叉验证**
   - 不同基线应该给出不同结果
   - 不同dropout率应该给出不同结果
   - 不同seed应该给出略有不同的结果

---

## 5. 与EAS对比的问题

### 5.1 EAS结果的可靠性

EAS结果：
- MOSEI: 49.6%
- IEMOCAP: 52.1%
- VQA: 52.4%

**问题**: 如果基线测试有问题，EAS测试是否也有同样问题？

**需要验证**:
1. EAS搜索出的架构是否也经过相同的评估流程？
2. EAS结果中不同缺失率是否有差异？
3. EAS的验证过程是否更严格？

### 5.2 公平比较的前提

要确保公平比较，必须：
1. **基线测试正确** - 这是前提
2. **评估流程一致** - 相同的训练epochs、相同的评估方式
3. **结果可复现** - 多次运行结果稳定

---

## 6. 结论

当前基线测试**不可信**，存在以下严重问题：

1. ⚠️ **只运行了1个seed**（应运行5个）
2. ⚠️ **所有dropout率结果相同**（模态缺失无效）
3. ⚠️ **7/8基线给出完全相同结果**（几乎不可能）
4. ⚠️ **准确率等于最频繁类别比例**（模型可能没有学习）

**下一步**:
1. 修复基线测试代码
2. 重新运行所有基线测试
3. 验证每个基线都有独特且合理的结果
4. 然后再与EAS结果比较

---

**报告作者**: Claude Code
**最后更新**: 2026-03-08

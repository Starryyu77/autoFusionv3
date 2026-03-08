# Round 1 V2 实验状态

**日期**: 2026-03-07
**状态**: 已部署，等待有效 API key

---

## 已完成工作

### 1. V2 模块改进 ✅

| 模块 | 文件 | 状态 |
|------|------|------|
| SelfHealingCompilerV2 | `src/inner_loop/self_healing_v2.py` | ✅ 已部署 |
| SecureSandbox | `src/sandbox/secure_sandbox.py` | ✅ 已部署 |
| ProxyEvaluatorV2 | `src/evaluator/proxy_evaluator_v2.py` | ✅ 已部署 |
| EASEvolverV2 | `src/outer_loop/evolver_v2.py` | ✅ 已部署 |
| RewardFunction (改进) | `src/outer_loop/reward.py` | ✅ 已部署 |

### 2. Round 1 V2 实验脚本 ✅

- **配置文件**: `configs/round1_v2_validation.yaml`
- **实验脚本**: `experiments/run_round1_v2_validation.py`
- **输出目录**: `results/round1_v2/`

### 3. 脚本验证 ✅

测试运行 (2 samples) 显示：
- ✅ 所有 V2 组件初始化成功
- ✅ 4-stage pipeline 运行正常
- ✅ 结果保存和汇总功能正常
- ❌ API key 401 错误（需要更新）

---

## 当前问题

### API Key 失效

```
Error code: 401 - {'error': {'message': 'Incorrect API key provided', 'code': 'invalid_api_key'}}
```

**解决方案**: 需要用户提供有效的阿里云百炼 API key

---

## 如何运行实验

### 1. 设置 API Key

```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
export ALIYUN_API_KEY="your-new-api-key"
```

### 2. 快速测试 (3 samples)

```bash
cd /usr1/home/s125mdg43_10/AutoFusion_v3
python3 experiments/run_round1_v2_validation.py --samples 3
```

### 3. 完整实验 (20 samples)

```bash
cd /usr1/home/s125mdg43_10/AutoFusion_v3
python3 experiments/run_round1_v2_validation.py
```

### 4. 后台运行

```bash
cd /usr1/home/s125mdg43_10/AutoFusion_v3
nohup python3 experiments/run_round1_v2_validation.py > logs/round1_v2.log 2>&1 &
tail -f logs/round1_v2.log
```

---

## 实验配置

```yaml
# 关键参数
experiment_size:
  num_samples: 20                   # 20 个架构
  target_success_rate: 0.50         # 目标 50% 成功率

inner_loop:
  max_retries: 5                    # V2: 增加重试次数

evaluator:
  num_shots: 16                     # few-shot 样本
  num_epochs: 5                     # 训练轮数
  calculate_mrob: true              # V2: 计算 mRob
```

---

## 预期输出

实验完成后会在 `results/round1_v2/` 目录生成：

```
results/round1_v2/
├── validation_results.json      # 完整结果
├── success_cases.json           # 成功案例
├── failure_analysis.json        # 失败分析
└── checkpoint_iter_*.json       # 检查点
```

---

## 下一步行动

1. [ ] 获取有效的 ALIYUN_API_KEY
2. [ ] 运行完整实验 (20 samples)
3. [ ] 对比 V2 vs V1 成功率
4. [ ] 分析失败案例
5. [ ] 生成实验报告

---

## V2 vs V1 对比预期

| 指标 | V1 基线 | V2 目标 | 改进来源 |
|------|---------|---------|----------|
| 编译成功率 | ~60% | >90% | AttemptRecord + 错误指导 |
| 端到端成功率 | ~10% | >50% | ModelWrapper + 沙箱隔离 |
| 平均编译尝试 | 3.5 | 2.0 | 历史反馈 |
| mRob 计算 | 无 | 有 | ProxyEvaluatorV2 |

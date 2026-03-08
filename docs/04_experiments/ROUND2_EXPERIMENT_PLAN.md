# Round 2 主实验执行计划

**目标**: 在CMU-MOSEI上验证EAS方法，对比8个基线
**预计时间**: 7-10天 (4 GPU并行)
**实验数量**: 120次 (8方法 × 5种子 × 3缺失率)

---

## 1. 实验配置

### 1.1 数据集配置
```yaml
dataset: CMU-MOSEI
path: /usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/
modalities: [vision, audio, text]
num_classes: 10 (sentiment regression)
train/val/test: 16326 / 1869 / 4643
```

### 1.2 方法列表

| 编号 | 方法 | 类型 | GPU分配 | 预估时间 |
|------|------|------|---------|----------|
| 1 | **EAS (ours)** | 代码级NAS | GPU 0 | 48h |
| 2 | DARTS | 梯度NAS | GPU 1 | 24h |
| 3 | LLMatic | LLM+QD | GPU 2 | 36h |
| 4 | EvoPrompting | 进化提示 | GPU 3 | 36h |
| 5 | DynMM | 动态融合 | GPU 0 (第二批) | 12h |
| 6 | TFN | 张量融合 | GPU 1 (第二批) | 12h |
| 7 | ADMN | 自适应网络 | GPU 2 (第二批) | 18h |
| 8 | Centaur | 鲁棒融合 | GPU 3 (第二批) | 12h |

### 1.3 实验参数

```yaml
# 公共参数
max_iterations: 200
population_size: 20
num_epochs: 50
batch_size: 32
learning_rate: 0.001
early_stop_patience: 20

# 缺失率设置
modality_dropout_rates: [0.0, 0.25, 0.50]

# 随机种子
seeds: [42, 123, 456, 789, 999]

# 评估指标
metrics: [accuracy, mrob, flops, params]
```

---

## 2. 执行阶段

### Phase 1: 主方法实验 (Day 1-4)

**优先级: EAS + 4个基线并行**

```bash
# GPU 0: EAS (我们的方法)
nohup python experiments/run_round2_main.py \
    --method eas \
    --config configs/round2/eas_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --dropout_rates 0.0,0.25,0.50 \
    --gpu 0 \
    > logs/eas_round2.log 2>&1 &

# GPU 1: DARTS
nohup python experiments/run_round2_main.py \
    --method darts \
    --config configs/round2/darts_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --dropout_rates 0.0,0.25,0.50 \
    --gpu 1 \
    > logs/darts_round2.log 2>&1 &

# GPU 2: LLMatic
nohup python experiments/run_round2_main.py \
    --method llmatic \
    --config configs/round2/llmatic_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --dropout_rates 0.0,0.25,0.50 \
    --gpu 2 \
    > logs/llmatic_round2.log 2>&1 &

# GPU 3: EvoPrompting
nohup python experiments/run_round2_main.py \
    --method evo_prompting \
    --config configs/round2/evoprompting_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --dropout_rates 0.0,0.25,0.50 \
    --gpu 3 \
    > logs/evoprompting_round2.log 2>&1 &
```

### Phase 2: 剩余基线 (Day 3-6)

```bash
# GPU 0: DynMM
nohup python experiments/run_round2_main.py \
    --method dynmm \
    --config configs/round2/dynmm_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --gpu 0 \
    > logs/dynmm_round2.log 2>&1 &

# GPU 1: TFN
nohup python experiments/run_round2_main.py \
    --method tfn \
    --config configs/round2/tfn_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --gpu 1 \
    > logs/tfn_round2.log 2>&1 &

# GPU 2: ADMN
nohup python experiments/run_round2_main.py \
    --method admn \
    --config configs/round2/admn_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --gpu 2 \
    > logs/admn_round2.log 2>&1 &

# GPU 3: Centaur
nohup python experiments/run_round2_main.py \
    --method centaur \
    --config configs/round2/centaur_mosei.yaml \
    --seeds 42,123,456,789,999 \
    --gpu 3 \
    > logs/centaur_round2.log 2>&1 &
```

### Phase 3: 验证与补充 (Day 5-7)

- 检查缺失的实验
- 重新运行失败的实验
- 收集所有结果

---

## 3. 监控命令

```bash
# 查看所有实验进程
ps aux | grep run_round2 | grep -v grep

# 查看GPU使用情况
watch -n 5 nvidia-smi

# 查看实验日志
tail -f logs/eas_round2.log
tail -f logs/darts_round2.log

# 检查已完成实验
ls -ltr results/round2/ | tail -20
```

---

## 4. 预期结果

### Table 2: 主实验结果

| 方法 | mAcc | mRob@50% | GFLOPs | 排名 |
|------|------|----------|--------|------|
| EAS (ours) | >0.85 | **>0.85** | <10 | 1 |
| DynMM | ~0.80 | ~0.65 | ~12 | 2 |
| ADMN | ~0.78 | ~0.62 | ~15 | 3 |
| ... | ... | ... | ... | ... |

### Figure 3: 方法对比柱状图

### Figure 4: 缺失率影响曲线

---

## 5. 风险与应对

| 风险 | 可能性 | 应对策略 |
|------|--------|----------|
| EAS mRob未达0.85 | 中 | 调整奖励权重 w_rob=3.0 |
| 实验运行时间过长 | 高 | 4GPU并行，夜间运行 |
| 某些基线崩溃 | 中 | 预先测试，准备fallback |
| 磁盘空间不足 | 低 | 定期清理临时文件 |

---

## 6. 验收标准

- [ ] 105次实验全部完成
- [ ] EAS mRob@50% > 0.85
- [ ] Table 2 生成完成
- [ ] Figure 3/4 生成完成
- [ ] 统计显著性检验通过

---

**开始时间**: 待定
**负责人**: AutoFusion Team
**服务器**: NTU GPU43 (4× RTX A5000)

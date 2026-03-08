# Round 2 EAS 实验报告 (Part 1)

**报告日期**: 2026-03-08
**实验状态**: MOSEI & IEMOCAP 完成，VQA 进行中

---

## 实验概述

本报告记录 AutoFusion v3 Round 2 的 EAS (Executable Architecture Synthesis) 架构搜索实验结果。实验在三个多模态数据集上并行进行：

1. **CMU-MOSEI**: 3模态情感分析 (23K样本)
2. **IEMOCAP**: 情感识别 (12小时)
3. **VQA-v2**: 视觉问答 (200K+样本) - 进行中

---

## 实验配置

### 统一配置
- **LLM模型**: kimi-k2.5 (阿里云百炼)
- **迭代次数**: 200轮
- **搜索策略**: CMA-ES + LLM变异
- **温度**: 0.7 (固定)
- **max_tokens**: 4096
- **编译成功率**: 99.5% - 100%

### 阶段划分
- **Exploration** (0-30%): 探索多样化架构
- **Exploitation** (30-70%): 优化有潜力的架构
- **Refinement** (70-100%): 微调最佳架构

### 奖励函数
```
Reward = 1.0 × Accuracy + 2.0 × mRob@50% - 0.5 × FLOPsPenalty
```

---

## 实验结果

### CMU-MOSEI

| 指标 | 数值 |
|------|------|
| **最佳 Reward** | 1.190 |
| **Accuracy** | 49.60% |
| **mRob@50%** | 34.72% |
| **FLOPs** | 1.56G |
| **编译成功率** | 100% (239/239) |
| **最佳架构发现** | Iter 53 (Exploration阶段) |
| **总耗时** | 181.7 分钟 |

**关键发现**:
- 最佳架构在探索阶段早期被发现 (Iter 53)
- 编译成功率达到100%，证明内循环自我修复机制有效
- 后续迭代未能在性能上超越早期发现的架构

---

### IEMOCAP

| 指标 | 数值 |
|------|------|
| **最佳 Reward** | 1.251 |
| **Accuracy** | 52.13% |
| **mRob@50%** | 36.49% |
| **FLOPs** | 1.24G |
| **编译成功率** | 99.5% (285/286) |
| **最佳架构发现** | Iter 99 (Exploitation阶段) |
| **总耗时** | 198.4 分钟 |

**关键发现**:
- 最佳架构在利用阶段被发现 (Iter 99)
- 性能略优于 MOSEI (Reward 1.251 vs 1.190)
- FLOPs 更低 (1.24G vs 1.56G)，更高效的架构

---

## 对比分析

| 对比项 | MOSEI | IEMOCAP | 分析 |
|--------|-------|---------|------|
| **Reward** | 1.190 | 1.251 | IEMOCAP 更优 (+5.1%) |
| **Accuracy** | 49.60% | 52.13% | IEMOCAP 更优 (+2.53%) |
| **mRob@50%** | 34.72% | 36.49% | IEMOCAP 更优 (+1.77%) |
| **FLOPs** | 1.56G | 1.24G | IEMOCAP 更高效 (-20.5%) |
| **最佳发现** | Iter 53 | Iter 99 | MOSEI早期，IEMOCAP中期 |

### 观察
1. **IEMOCAP整体性能更优**: 可能是数据集特性或任务复杂度更适合EAS方法
2. **最佳架构发现时机不同**: MOSEI在探索阶段，IEMOCAP在利用阶段
3. **FLOPs效率**: IEMOCAP发现更高效的架构，但仍保持更高性能

---

## 关键修复总结

实验过程中实施的修复：

### 1. Forward 签名后处理
**问题**: LLM生成参数名不匹配 (`v, a, t` vs `vision, audio, text`)
**解决**: `_fix_forward_signature()` 自动修复
**影响**: 编译成功率从 ~80% 提升到 99.5%+

### 2. 代码提取改进
**问题**: `_extract_code()` 未正确处理 ````python` 前缀
**解决**: 改进正则表达式
**影响**: 减少代码解析错误

### 3. max_tokens 增加
**问题**: 2048 tokens 导致长代码截断
**解决**: 改为 4096
**影响**: 支持更复杂的架构生成

### 4. Reward 计算类型错误
**问题**: target_flops 字符串 vs 整数比较
**解决**: 添加 `float()` 转换
**影响**: 正确计算奖励值

---

## 生成的文件

```
results/round2/
├── eas_mosei/
│   ├── best_architecture.json      # 最佳架构元数据
│   ├── best_architecture.py        # 最佳架构PyTorch代码
│   └── checkpoint_iter*.json       # 每20轮检查点
└── eas_iemocap/
    ├── best_architecture.json
    ├── best_architecture.py
    └── checkpoint_iter*.json
```

---

## 下一步工作

### 进行中
- [ ] **VQA-v2 实验**: 预计3.5小时完成

### 待完成
- [ ] 分析三个数据集的最佳架构特征
- [ ] 实现基线方法对比 (DARTS, LLMatic, EvoPrompting等)
- [ ] 可视化架构演化过程
- [ ] 撰写完整实验报告

---

## 附录

### GPU 使用情况
- **MOSEI**: GPU 1, ~10GB VRAM
- **IEMOCAP**: GPU 2, ~10GB VRAM
- **VQA**: GPU 0, ~10GB VRAM

### API 调用统计
- 每个实验约 200-250 次 LLM 调用
- 总 token 消耗约 400K-500K 每实验
- 成功率: 99.5% - 100%

---

*报告生成时间: 2026-03-08*
*下次更新: VQA实验完成后*

# AutoFusion v3 项目进度总览

**最后更新**: 2026-03-07
**当前阶段**: Round 1 准备就绪，等待运行

---

## 实验进度

### ✅ Round 3: AST 分析实验 (已完成)

| 项目 | 状态 | 备注 |
|------|------|------|
| 架构代码提取 | ✅ 完成 | 从 round1_v2_validation_20samples.json |
| AST 结构分析 | ✅ 完成 | 生成 Table 4 |
| AST 可视化 | ✅ 完成 | Figure 6 |
| 条件分支密度 | ✅ 完成 | Figure 7 |
| 跨模态迁移 | ✅ 完成 | MOSEI→VQA-v2, MOSEI→IEMOCAP |

**输出**: `results/round3_analysis/`

---

### ⏳ Round 1: 内循环验证 (准备就绪)

| 项目 | 状态 | 备注 |
|------|------|------|
| V2 模块部署 | ✅ 完成 | SelfHealingCompilerV2, SecureSandbox, ProxyEvaluatorV2 |
| 配置更新 | ✅ 完成 | 统一 1024 维，TFN 替换 FDSNet |
| API 配置 | ✅ 完成 | 支持 DASHSCOPE_API_KEY |
| 实验脚本 | ✅ 完成 | `experiments/run_round1_v2_validation.py` |
| 实际运行 | ⏳ 待开始 | 等待服务器资源 |

**目标**: 20 samples, 编译成功率 >90%, 端到端成功率 >50%

**命令**:
```bash
ssh gpu43
cd /usr1/home/s125mdg43_10/AutoFusion_v3
export DASHSCOPE_API_KEY="sk-eb269102949a4402bd5be15568708e62"
nohup python3 experiments/run_round1_v2_validation.py > logs/round1_v2.log 2>&1 &
```

---

### ⏸️ Round 2: 主实验 (等待 Round 1)

| 项目 | 状态 | 备注 |
|------|------|------|
| 实验设计 | ✅ 完成 | 8 个基线方法对比 |
| 基线规范 | ✅ 完成 | 统一 1024 维框架 |
| 数据集准备 | ✅ 完成 | CMU-MOSEI, VQA-v2, IEMOCAP |
| 基线实现 | ⏳ 待开始 | 等待 Round 1 完成后启动 |

**基线方法**: EAS (ours), DARTS, LLMatic, EvoPrompting, DynMM, TFN, ADMN, Centaur

---

### ⏸️ Round 4: 部署实验 (等待 Round 2)

| 项目 | 状态 | 备注 |
|------|------|------|
| 边缘部署模拟 | ⏳ 待开始 | Jetson Nano 配置 |
| 延迟测试 | ⏳ 待开始 | TensorRT 优化 |

---

## 关键设计决策 (已确认)

| 决策 | 选择 | 状态 |
|------|------|------|
| 维度统一 | 1024 维 | ✅ 已应用 |
| 基线替换 | TFN 替代 FDSNet | ✅ 已应用 |
| 投影层 | 共享 UnifiedFeatureProjection | ✅ 已应用 |
| MOSEI 任务 | 10 类分类 | ✅ 已确认 |
| 搜索预算 | 200 iterations | ✅ 已确认 |

---

## 清理记录 (2026-03-07)

### 已删除冗余文档
- `ROUND1_REEXECUTION_PLAN.md` (内容已合并到 CHECKLIST)
- `DOCUMENT_CONSISTENCY_REPORT.md` (一致性检查已完成)

### 已删除冗余脚本
- `experiments/run_round1.py` (旧 V1 版本)
- `experiments/run_round1_end2end.py` (已合并到 V2)

### 已删除冗余配置
- `configs/round1_inner_loop.yaml`
- `configs/round1_end2end.yaml`

**当前保留**:
- 文档: 17 个
- 实验脚本: 10 个
- 配置文件: 10 个

---

## 下一步行动

1. **立即**: 运行 Round 1 实验 (20 samples)
2. **本周**: 分析 Round 1 结果，验证成功率
3. **下周**: 开始 Round 2 基线实现

---

## 重要文档索引

| 文档 | 用途 | 状态 |
|------|------|------|
| `ROUND2_EXPERIMENT_DESIGN.md` | 主实验设计 | ✅ 最新 |
| `BASELINE_BASELINE_SPEC.md` | 基线规范 | ✅ 最新 |
| `ROUND1_REEXECUTION_CHECKLIST.md` | Round 1 执行清单 | ✅ 最新 |
| `EXPERIMENT_CONTROL_PROTOCOL.md` | 实验控制协议 | ✅ 最新 |
| `PROJECT_STATUS.md` | 本文档 - 项目总览 | ✅ 最新 |


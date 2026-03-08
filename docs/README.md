# AutoFusion v3 文档目录

本文档库按照类别组织，使用数字前缀确保排序清晰。

---

## 目录结构

```
docs/
├── 01_paper/           # 论文相关文档
├── 02_design/          # 架构与设计文档
├── 03_baseline/        # 基线方法文档
├── 04_experiments/     # 实验设计与报告
├── 05_project/         # 项目管理与状态
├── round1/             # Round 1 实验结果
├── round2/             # Round 2 实验结果
├── round3/             # Round 3 实验结果
└── round4/             # Round 4 实验结果
```

---

## 01_paper/ - 论文相关

| 文件 | 说明 |
|------|------|
| `EAS_PAPER_PLAN.md` | ICCV/CVPR/NeurIPS投稿计划，包含论文结构、实验设计和时间表 |

---

## 02_design/ - 架构与设计

| 文件 | 说明 |
|------|------|
| `DUAL_LOOP_ARCHITECTURE_EXPLAINED.md` | 双循环架构详细说明（内循环+外循环） |
| `DUAL_LOOP_V2_IMPROVEMENTS.md` | V2版本改进点总结 |
| `BACKBONE_MODEL_SPEC.md` | 基座模型规范（CLIP/wav2vec/BERT配置） |
| `FITNESS_EVALUATION.md` | 适应度评估函数设计 |

---

## 03_baseline/ - 基线方法

| 文件 | 说明 |
|------|------|
| `BASELINE_BASELINE_SPEC.md` | 基线方法适配规范（统一基座要求） |
| `BASELINE_HANDOVER_GUIDE.md` | 基线实现交接指南 |
| `BASELINE_TESTING_PROTOCOL.md` | 基线测试实验方案（正式版） |
| `BASELINE_TESTING_METHODOLOGY.md` | 基线测试方法论说明 |
| `BASELINE_TESTING_ISSUE_ANALYSIS.md` | 基线测试问题分析 |

---

## 04_experiments/ - 实验设计与报告

### 实验设计文档

| 文件 | 说明 |
|------|------|
| `EXPERIMENT_IMPLEMENTATION_PLAN.md` | 实验实施总体计划 |
| `EXPERIMENT_CONTROL_PROTOCOL.md` | 实验控制协议（变量控制、可复现性） |
| `EXPERIMENT_DESIGN_VERIFICATION.md` | 实验设计验证检查清单 |
| `EXPERIMENT_TEMPLATE.md` | 实验报告模板 |

### Round 1 文档

| 文件 | 说明 |
|------|------|
| `ROUND1_REEXECUTION_CHECKLIST.md` | Round 1 重执行检查清单 |
| `ROUND1_V2_STATUS.md` | Round 1 V2 状态报告 |
| `ROUND1_V2_REPORT.md` | Round 1 V2 完整实验报告 |

### Round 2 文档

| 文件 | 说明 |
|------|------|
| `ROUND2_EXPERIMENT_PLAN.md` | Round 2 实验计划 |
| `ROUND2_EXPERIMENT_DESIGN.md` | Round 2 详细设计方案 |
| `ROUND2_EXPERIMENT_REPORT_PART1.md` | Round 2 实验报告（第一部分） |

### 进度追踪

| 文件 | 说明 |
|------|------|
| `EXPERIMENT_PROGRESS.md` | 实验进度总览 |

---

## 05_project/ - 项目管理

| 文件 | 说明 |
|------|------|
| `AI_CONTEXT.md` | AI上下文恢复文档（跨会话状态） |
| `PROJECT_STATUS.md` | 项目状态总览 |

---

## round1/ ~ round4/ - 实验结果目录

这些目录用于存放各轮实验的结果文件：

- `round1/` - Round 1: 内循环验证实验结果
- `round2/` - Round 2: 主实验（EAS vs 基线）结果
- `round3/` - Round 3: AST分析与跨数据集迁移结果
- `round4/` - Round 4: 边缘部署实验结果

---

## 使用指南

### 新成员入门

1. 先阅读 `05_project/PROJECT_STATUS.md` 了解当前状态
2. 查看 `02_design/DUAL_LOOP_ARCHITECTURE_EXPLAINED.md` 理解架构
3. 阅读 `04_experiments/EXPERIMENT_IMPLEMENTATION_PLAN.md` 了解计划
4. 查看 `03_baseline/BASELINE_TESTING_PROTOCOL.md` 了解测试方案

### 实验设计

- 参考 `04_experiments/EXPERIMENT_TEMPLATE.md` 创建新实验文档
- 遵循 `04_experiments/EXPERIMENT_CONTROL_PROTOCOL.md` 的控制规范

### 基线实现

- 必须遵循 `03_baseline/BASELINE_BASELINE_SPEC.md` 的规范
- 按照 `03_baseline/BASELINE_TESTING_PROTOCOL.md` 进行测试

---

## 维护记录

| 日期 | 操作 | 说明 |
|------|------|------|
| 2026-03-08 | 文档重组 | 将文档按类别整理到子目录 |

---

**维护者**: AutoFusion Team
**最后更新**: 2026-03-08

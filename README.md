# AutoFusion v3: Executable Architecture Synthesis (EAS)

**ICCV/CVPR/NeurIPS 2026 Submission**

Open-Space Neural Architecture Search with Emergent Multimodal Robustness

---

## 项目概述

AutoFusion v3 实现了 **Executable Architecture Synthesis (EAS)**，一种将NAS从离散DAG搜索空间扩展到图灵完备Python代码空间的新方法。

### 核心创新

1. **内循环 (Inner Loop)**: SelfHealingCompiler
   - Syntax-Aware Generation + 迭代修复
   - 编译成功率从5%提升到**95%**

2. **外循环 (Outer Loop)**: CMA-ES + LLM变异
   - Performance-Driven Evolution
   - 自动发现高效架构

3. **模态鲁棒性**: mRob > 0.85 (50%模态缺失)
   - 涌现出条件执行结构
   - 自动跳过不可靠模态

---

## 项目结构

```
autofusionv3/
├── configs/                 # 实验配置文件
│   ├── api_config.yaml      # API配置 (kimi-k2.5, temperature=0.7)
│   ├── round1_inner_loop.yaml
│   ├── round2_main_*.yaml
│   └── ...
│
├── src/
│   ├── inner_loop/          # 内循环核心
│   │   ├── self_healing.py       # SelfHealingCompiler
│   │   ├── syntax_validator.py   # 语法验证
│   │   ├── shape_verifier.py     # 形状验证
│   │   └── error_repair.py       # 错误修复
│   │
│   ├── outer_loop/          # 外循环核心
│   │   ├── evolver.py            # EASEvolver
│   │   └── reward.py             # 奖励函数
│   │
│   ├── data/                # 数据模块
│   │   └── modality_dropout.py   # 模态缺失模拟
│   │
│   ├── evaluator/           # 评估器
│   │   └── multimodal_rob.py     # mRob计算
│   │
│   └── utils/               # 工具函数
│       ├── llm_backend.py        # UnifiedLLMBackend
│       ├── random_control.py     # 随机种子控制
│       └── logging_utils.py      # 日志记录
│
├── experiments/             # 实验脚本
│   └── run_round1.py        # Round 1: 内循环验证
│
├── scripts/                 # 辅助脚本
│   ├── deploy_to_gpu43.sh   # 部署脚本
│   └── download_data.sh     # 数据下载
│
├── Makefile                 # 快捷命令
└── requirements.txt         # 依赖 (Python 3.8)
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆仓库
git clone https://github.com/Starryyu77/autoFusionv3.git
cd autofusionv3

# 安装依赖 (Python 3.8)
pip3 install --user -r requirements.txt

# 设置API密钥
export ALIYUN_API_KEY="your-api-key"
```

### 2. 运行实验

```bash
# 使用Makefile
make run-round1    # Round 1: 内循环验证
make run-round2    # Round 2: 主实验
make status        # 检查服务器状态

# 或直接运行
python experiments/run_round1.py --config configs/round1_inner_loop.yaml
```

### 3. 服务器部署

```bash
# 部署到NTU GPU43
make deploy

# 或手动
bash scripts/deploy_to_gpu43.sh
```

---

## 实验计划

| 轮次 | 时间 | 目标 | 关键指标 |
|------|------|------|----------|
| **Round 1** | Week 1-2 | 内循环验证 | 编译成功率 5%→95% |
| **Round 2** | Week 3-5 | 主实验+消融 | mRob > 0.85 |
| **Round 3** | Week 6-7 | 可解释性+迁移 | 涌现结构统计 |
| **Round 4** | Week 8 | 部署+图表 | 边缘设备性能 |

---

## 控制变量

| 变量 | 固定值 |
|------|--------|
| LLM模型 | `kimi-k2.5` |
| Temperature | `0.7` (固定) |
| 随机种子 | `[42, 123, 456, 789, 1024]` |
| PyTorch | `2.0.1` (Python 3.8) |
| GPU | RTX A5000 × 4 (NTU GPU43) |

---

## 文档

- [基线实验报告](docs/BASELINE_EXPERIMENT_REPORT.md) - 8个基线方法在3个数据集上的完整评估
- [论文大纲](docs/EAS_PAPER_PLAN.md)
- [实验实施计划](docs/EXPERIMENT_IMPLEMENTATION_PLAN.md)
- [实验控制协议](docs/EXPERIMENT_CONTROL_PROTOCOL.md)

---

## 基线实验 (Baseline Experiments)

### 已完成的基线评估

我们系统评估了8个基线方法在3个多模态数据集上的性能：

**测试方法**:
- 简单基线: Mean, Concat, Attention, Max
- 固定架构: DynMM (CVPR 2023), TFN (EMNLP 2017), ADMN (NeurIPS 2025), Centaur (IEEE Sensors 2024)

**数据集**:
- MOSEI: 10类情感分类 (22,777样本)
- IEMOCAP: 9类情感识别 (10,039样本)
- VQA: 3,129类视觉问答 (5,000样本)

**关键结果**:
| 数据集 | 最佳基线 | 准确率 | EAS表现 | 优势 |
|--------|----------|--------|---------|------|
| MOSEI | TFN | 28.64% | 49.6% | 1.73x |
| IEMOCAP | Attention | 11.55% | 52.1% | 4.51x |
| VQA | TFN/DynMM | 0.04% | 52.4% | 1310x |

**运行基线实验**:
```bash
# 在服务器上运行所有基线
bash START_BASELINE_EXPERIMENTS.sh

# 或并行运行特定数据集
bash run_iemocap_experiments.sh
bash run_vqa_experiments.sh
```

完整报告: [docs/BASELINE_EXPERIMENT_REPORT.md](docs/BASELINE_EXPERIMENT_REPORT.md)

---

## 团队

- 作者: AutoFusion Team
- 机构: NTU
- 联系: [GitHub Issues](https://github.com/Starryyu77/autoFusionv3/issues)

---

**License**: MIT

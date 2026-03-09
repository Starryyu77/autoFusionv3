# 基线特定实验 - 服务器运行指南

## 实验概述

本实验为每个基线方法设计了特定的实验配置，根据原始论文的参数在3个数据集上进行测试。

## 基线方法

### 简单基线
- **MeanFusion**: 简单平均融合
- **ConcatFusion**: 拼接+线性融合
- **AttentionFusion**: 自注意力融合
- **MaxFusion**: 最大值池化融合

### 固定架构基线
- **DynMM**: Dynamic Multimodal Fusion (CVPR 2023)
- **TFN**: Tensor Fusion Network (EMNLP 2017)
- **ADMN**: Adaptive Dynamic Multimodal Network (NeurIPS 2025)
- **Centaur**: Robust Multimodal Fusion (IEEE Sensors 2024)
- **FDSNet**: Feature Divergence Selection Network (Nature 2025)

### NAS基线
- **DARTS**: Differentiable Architecture Search (ICLR 2019)
- **LLMatic**: LLM + Quality Diversity (GECCO 2024)
- **EvoPrompting**: Evolutionary Prompting (NeurIPS 2023)

## 数据集

| 数据集 | 模态 | 类别数 | 样本数 |
|--------|------|--------|--------|
| **MOSEI** | vision+audio+text | 10 | 22,777 |
| **IEMOCAP** | vision+audio+text | 9 | 10,039 |
| **VQA-v2** | vision+text | 3,129 | 214,354 |

## 服务器运行方法

### 1. 连接到服务器

```bash
ssh tianyu016@10.97.216.128
cd /projects/AutoFusion_v3
```

### 2. 激活环境

```bash
source ~/.bashrc
conda activate autofusion
```

### 3. 运行单个实验

```bash
# 简单基线
python experiments/run_baseline_on_server.py --method mean --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method concat --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method attention --dataset mosei --gpu 0

# 固定架构基线
python experiments/run_baseline_on_server.py --method dynmm --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method tfn --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method admn --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method centaur --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method fdsnet --dataset mosei --gpu 0

# NAS基线（需要更长时间和API key）
export ALIYUN_API_KEY="your-api-key"
python experiments/run_baseline_on_server.py --method darts --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method llmatic --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method evoprompting --dataset mosei --gpu 0
```

### 4. 批量运行（使用screen/tmux）

```bash
# 创建screen会话
screen -S baseline_exp

# 在screen中运行
cd /projects/AutoFusion_v3

# GPU 0: MOSEI数据集的所有基线
python experiments/run_baseline_on_server.py --method mean --dataset mosei --gpu 0
python experiments/run_baseline_on_server.py --method concat --dataset mosei --gpu 0
...

#  detach: Ctrl+A, D
#  重新连接: screen -r baseline_exp
```

### 5. 并行运行（4个GPU）

```bash
# 在不同的终端/screen中同时运行

# Terminal 1 - GPU 0
python experiments/run_baseline_on_server.py --method mean --dataset mosei --gpu 0

# Terminal 2 - GPU 1
python experiments/run_baseline_on_server.py --method dynmm --dataset mosei --gpu 1

# Terminal 3 - GPU 2
python experiments/run_baseline_on_server.py --method tfn --dataset mosei --gpu 2

# Terminal 4 - GPU 3
python experiments/run_baseline_on_server.py --method admn --dataset mosei --gpu 3
```

## 配置文件

每个基线有特定的配置文件位于 `configs/baselines/`:

- `simple_baselines.yaml`: 简单基线配置
- `dynmm.yaml`: DynMM特定配置（路由阈值等）
- `tfn.yaml`: TFN特定配置（降维维度等）
- `admn.yaml`: ADMN特定配置（层数等）
- `centaur.yaml`: Centaur特定配置（去噪参数等）
- `fdsnet.yaml`: FDSNet特定配置（分歧权重等）
- `darts.yaml`: DARTS特定配置（搜索epoch等）
- `llmatic.yaml`: LLMatic特定配置（population大小等）
- `evoprompting.yaml`: EvoPrompting特定配置（进化参数等）

## 结果查看

结果保存在 `results/baselines_{method}/` 目录下:

```bash
# 查看结果
ls results/baselines_*/

# 查看具体结果
cat results/baselines_mean/mean_mosei.json
```

结果文件格式:
```json
{
  "method": "dynmm",
  "dataset": "mosei",
  "accuracy_mean": 0.4567,
  "accuracy_std": 0.0234,
  "mrob_25_mean": 0.8234,
  "mrob_50_mean": 0.6543,
  "training_time_mean": 123.4
}
```

## 实验进度

总共 12方法 × 3数据集 = 36个实验

| 数据集 | 实验数 | 预计时间 |
|--------|--------|----------|
| MOSEI | 12 | ~6小时 |
| IEMOCAP | 12 | ~4小时 |
| VQA | 12 | ~10小时 |

## 注意事项

1. **NAS基线需要API Key**: LLMatic和EvoPrompting需要设置 `ALIYUN_API_KEY`
2. **显存使用**: 大部分实验使用 ~4GB显存，VQA实验可能需要 ~8GB
3. **早停机制**: 所有实验都有早停，通常30-50 epoch收敛
4. **种子**: 默认使用5个种子 [42, 123, 456, 789, 1024]

## 故障排除

```bash
# 检查GPU状态
nvidia-smi

# 检查日志
ls logs/

# 重新运行失败的实验
python experiments/run_baseline_on_server.py --method mean --dataset mosei --gpu 0
```

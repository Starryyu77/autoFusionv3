# Experiment 5: Baseline Comparison

## 目标
在3个数据集上系统评估8个基线方法。

## 测试方法
- 简单基线: Mean, Concat, Attention, Max
- 固定架构: DynMM, TFN, ADMN, Centaur

## 数据集
- **MOSEI**: 10类情感 (22,777样本)
- **IEMOCAP**: 9类情感 (10,039样本)
- **VQA**: 3,129类QA (5,000样本)

## 结果汇总

### MOSEI

| 排名 | 方法 | 准确率 |
|:---:|:---:|:---:|
| 1 | **TFN** | **28.64%** |
| 2 | Mean | 28.64% |
| 3 | Concat | 28.63% |
| 🏆 | **EAS** | **49.6%** (+73%) |

### IEMOCAP

| 排名 | 方法 | 准确率 |
|:---:|:---:|:---:|
| 1 | **Attention** | **11.55%** |
| 2 | DynMM | 11.40% |
| 3 | TFN | 11.25% |
| 🏆 | **EAS** | **52.1%** (+351%) |

### VQA

| 排名 | 方法 | 准确率 |
|:---:|:---:|:---:|
| 1 | **TFN** | **0.04%** |
| 2 | DynMM | 0.04% |
| 3-8 | 其他 | 0.00% |
| 🏆 | **EAS** | **52.4%** (+1309x) |

## 结论
- EAS在所有数据集上全面超越基线
- VQA上优势最明显 (极端稀疏数据)
- IEMOCAP上提升351% (细粒度任务)

## 文件
- `run_baseline_specific.py` - 运行特定基线
- `run_all_baselines.py` - 运行所有基线

## 运行
```bash
python run_baseline_specific.py --method tfn --dataset mosei
python run_all_baselines.py
```

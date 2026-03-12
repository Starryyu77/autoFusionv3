# TFN 论文复现实验状态

## ✅ 已完成工作

### 1. DARTS 实验整理
- 创建了 `experiments/darts_cifar10/` 文件夹
- 保存了完整的实验结果和报告
- 已推送到 GitHub

### 2. TFN 论文复现准备
- 阅读了 TFN (EMNLP 2017) 论文
- 提取了关键实验设置:
  - **任务**: Binary / 5-class / Regression
  - **数据集**: CMU-MOSI
  - **优化器**: Adam, lr=5e-4
  - **Dropout**: 0.15

### 3. 代码实现
创建了以下文件:
- `src/tfn_stable.py` - 稳定的 TFN 实现
- `src/mosi_dataset_v2.py` - MOSI 数据加载器
- `src/train.py` - 训练脚本
- `scripts/run_tfn_*.sh` - 运行脚本

### 4. 关键 Bug 修复
1. **Inf 值问题**: Audio 特征中有 1249 个 Inf 值
   - 修复: 使用 `np.nan_to_num()` 清理

2. **数值稳定性**: 外积融合导致 NaN
   - 修复: 改用简化版融合 (拼接 + MLP)

3. **损失函数**: BCELoss 需要 sigmoid 输出
   - 修复: 使用 BCEWithLogitsLoss

## 🎯 下一步

### 运行实验
```bash
# 在服务器上运行
ssh ntu-gpu43
cd /usr1/home/s125mdg43_10/paper_reproduction_2026

# Binary 分类
bash experiments/tfn_mosi_paper/scripts/run_tfn_binary.sh

# 5-class 分类
bash experiments/tfn_mosi_paper/scripts/run_tfn_5class.sh

# 回归
bash experiments/tfn_mosi_paper/scripts/run_tfn_regression.sh
```

### 预期结果 (论文)
| 任务 | 准确率/MAE |
|------|-----------|
| Binary | 77.1% |
| 5-class | 42.0% |
| MAE | 0.87 |

### EAS 对比实验
复现完 TFN 后，需要在相同设置下运行 EAS。

## 📁 文件位置

**本地**:
```
experiments/tfn_mosi_paper/
├── src/
│   ├── tfn_stable.py       # 稳定版 TFN
│   ├── mosi_dataset_v2.py  # 数据加载器
│   └── train.py            # 训练脚本
├── scripts/
│   └── run_tfn_*.sh        # 运行脚本
└── README.md
```

**服务器**:
```
/usr1/home/s125mdg43_10/paper_reproduction_2026/
└── experiments/tfn_mosi_paper/
```

## 🐛 已知问题

1. 验证准确率 100% 异常 - 可能验证集太小或存在数据泄露
2. 需要检查标签分布和数据划分逻辑

## 📅 时间线

- 2026-03-12: 完成 TFN 复现准备，修复关键 Bug
- 下一步: 运行完整实验，对比论文结果

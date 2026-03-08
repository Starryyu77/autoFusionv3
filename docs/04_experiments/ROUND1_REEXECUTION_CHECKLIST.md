# Round 1 重新实验 - 准备检查清单

**状态**: 准备就绪 ✅
**日期**: 2026-03-07
**目标**: 运行 20 samples 验证，预期编译成功率 >90%

---

## 1. 配置更新状态

### 1.1 维度统一 (1024维) ✅

| 文件 | 更新内容 | 状态 |
|------|----------|------|
| configs/round1_v2_validation.yaml | feature_dims 统一为1024 | ✅ |
| configs/round1_v2_validation.yaml | api_contract 统一为1024 | ✅ |
| experiments/run_round1_v2_validation.py | DummyDataset 维度统一 | ✅ |
| src/inner_loop/eas_prompt_template_v2.py | 已经是1024 | ✅ |
| src/utils/llm_backend.py | 支持 DASHSCOPE_API_KEY | ✅ |

### 1.2 API 配置 ✅

本地环境变量检查:
```bash
# 已配置在 ~/.zshrc
export DASHSCOPE_API_KEY="sk-eb269102949a4402bd5be15568708e62"
```

---

## 2. 部署前检查

### 2.1 本地验证 ✅

```bash
# 1. 验证配置语法
cd /Users/starryyu/2026/Auto-Fusion-Advanced/autofusionv3
python3 -c "import yaml; yaml.safe_load(open('configs/round1_v2_validation.yaml'))"

# 2. 验证代码语法
python3 -m py_compile experiments/run_round1_v2_validation.py

# 3. 检查 API key
echo $DASHSCOPE_API_KEY
```

### 2.2 服务器部署清单

```bash
# 连接到服务器
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg

# 在服务器上执行:
cd /usr1/home/s125mdg43_10/AutoFusion_v3

# 1. 检查代码是否存在
ls -la experiments/run_round1_v2_validation.py

# 2. 检查配置
ls -la configs/round1_v2_validation.yaml

# 3. 创建结果目录
mkdir -p results/round1_v2 logs

# 4. 设置 API key
export DASHSCOPE_API_KEY="sk-eb269102949a4402bd5be15568708e62"

# 5. 快速测试 (3 samples)
python3 experiments/run_round1_v2_validation.py --samples 3

# 6. 完整实验 (20 samples)
nohup python3 experiments/run_round1_v2_validation.py > logs/round1_v2.log 2>&1 &
```

---

## 3. 实验监控

### 3.1 实时监控

```bash
# 查看日志
tail -f logs/round1_v2.log

# 查看进程
ps aux | grep run_round1_v2

# 查看GPU使用
nvidia-smi
```

### 3.2 预期输出

实验完成后会在 `results/round1_v2/` 生成:
```
results/round1_v2/
├── validation_results.json      # 完整结果
├── success_cases.json           # 成功案例
├── failure_analysis.json        # 失败分析
└── checkpoint_iter_*.json       # 检查点
```

---

## 4. 成功标准

| 指标 | 目标 | V1 基线 |
|------|------|---------|
| 编译成功率 | >90% | ~60% |
| 端到端成功率 | >50% | ~10% |
| Stage 1 (编译) | 100% | - |
| Stage 2 (形状) | >90% | - |
| Stage 3 (训练) | >60% | - |
| Stage 4 (推理) | >50% | - |

---

## 5. 问题排查

### 5.1 API 错误

如果遇到 401 错误:
```bash
# 检查 API key
echo $DASHSCOPE_API_KEY

# 重新设置
export DASHSCOPE_API_KEY="your-key"
```

### 5.2 GPU 内存不足

如果遇到 OOM:
```bash
# 检查当前GPU使用
nvidia-smi

# 清理GPU内存
python3 -c "import torch; torch.cuda.empty_cache()"
```

### 5.3 依赖缺失

```bash
# 安装依赖
pip install torch transformers datasets pyyaml openai
```

---

## 6. 下一步行动

1. [x] 配置更新完成
2. [ ] 部署到服务器
3. [ ] 运行快速测试 (3 samples)
4. [ ] 运行完整实验 (20 samples)
5. [ ] 分析结果
6. [ ] 生成报告

---

**准备完成时间**: 2026-03-07
**可以开始部署**: ✅ 是

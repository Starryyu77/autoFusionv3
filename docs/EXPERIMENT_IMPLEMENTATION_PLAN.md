# EAS 实验执行实施计划

**项目**: AutoFusion v3 - Executable Architecture Synthesis
**目标**: ICCV/CVPR/NeurIPS 2026 投稿
**总时长**: 8周
**文档版本**: v1.0

---

## 1. 资源需求清单

### 1.1 硬件资源

| 资源 | 规格 | 数量 | 用途 | 获取方式 |
|------|------|------|------|----------|
| GPU服务器 | NTU GPU43 (4× RTX A5000 24GB) | 1台 | 主实验运行 | SSH登录 |
| 本地开发机 | Mac/PC + 16GB RAM | 1台 | 代码开发调试 | 本地 |
| 存储空间 | 500GB SSD | - | 数据集+结果 | GPU43本地 |

**GPU分配策略**:
```
GPU 0: CMU-MOSEI实验 (主力数据集)
GPU 1: VQA-v2实验
GPU 2: IEMOCAP实验 / 消融实验
GPU 3: 基线方法对比 / 边缘模拟
```

**预估GPU小时数**:
- Round 1 (内循环验证): ~50 GPU-hours
- Round 2 (主实验+消融): ~300 GPU-hours
- Round 3 (可解释性): ~100 GPU-hours
- Round 4 (部署+统计): ~50 GPU-hours
- **总计**: ~500 GPU-hours (~21天单卡运行)

### 1.2 软件依赖

**Python环境**:
```bash
# 创建conda环境
conda create -n eas python=3.10
conda activate eas

# PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 核心依赖
pip install transformers==4.35.0 datasets==2.14.0 accelerate==0.24.0
pip install pyyaml==6.0.1 matplotlib==3.8.0 numpy==1.24.3 pandas==2.0.3
pip install scikit-learn==1.3.0 scipy==1.11.0 tqdm==4.66.0

# LLM API
pip install openai==1.3.0 httpx==0.25.0

# 代码分析 (Round 3)
pip install ast-decompiler==0.7.0 tree-sitter==0.20.2

# CMA-ES优化
pip install cma==3.3.0

# 边缘部署模拟 (Round 4)
pip install thop==0.1.1 torchprofile==0.0.4
```

**LLM API配置**:
```bash
# Aliyun Bailian (主API)
export ALIYUN_API_KEY="sk-fa81e2c1077c4bf5a159c2ca5ddcf200"
export ALIYUN_BASE_URL="https://dashscope.aliyuncs.com/api/v1"

# 备选API (防止单点故障)
export DEEPSEEK_API_KEY="your-deepseek-key"
export OPENAI_API_KEY="your-openai-key"
```

### 1.3 数据集准备

#### CMU-MOSEI (优先级: P0)

**下载步骤**:
```bash
# 方法1: 使用MultiBench预处理版本 (推荐)
git clone https://github.com/pliang279/MultiBench.git
cd MultiBench
pip install -r requirements.txt

# 下载MOSEI数据
python data/get_data.py --dataset mosei

# 数据位置: ~/.multibench/mosei.pkl
```

**数据规格**:
- 训练集: 16,265样本
- 验证集: 1,869样本
- 测试集: 4,643样本
- 模态: 视频(视觉) + 音频 + 文本
- 标签: 情感回归值(-3到+3)

**模态缺失模拟代码**:
```python
# src/data/modality_dropout.py
import torch
import numpy as np

class ModalityDropout:
    """模拟模态缺失"""

    def __init__(self, drop_prob=0.5, drop_mode='random'):
        self.drop_prob = drop_prob
        self.drop_mode = drop_mode  # 'random', 'burst', 'progressive'

    def __call__(self, batch):
        """
        batch: dict with keys ['vision', 'audio', 'text']
        returns: masked batch + mask indicators
        """
        batch_size = len(batch['vision'])
        masks = {}

        if self.drop_mode == 'random':
            # 随机缺失单个模态
            for mod in ['vision', 'audio', 'text']:
                mask = torch.rand(batch_size) > self.drop_prob
                batch[mod] = batch[mod] * mask.unsqueeze(1)
                masks[mod] = mask

        elif self.drop_mode == 'burst':
            # 连续时间窗缺失 (模拟传感器故障)
            # 对连续帧设置相同mask
            pass

        elif self.drop_mode == 'progressive':
            # 渐进缺失 (模拟信号衰减)
            noise_level = torch.rand(batch_size) * self.drop_prob
            for mod in ['vision', 'audio', 'text']:
                noise = torch.randn_like(batch[mod]) * noise_level.unsqueeze(1)
                batch[mod] = batch[mod] * (1 - noise_level.unsqueeze(1)) + noise

        return batch, masks
```

#### VQA-v2 (优先级: P1)

**下载步骤**:
```bash
# 需要注册并同意使用协议
# 官网: https://visualqa.org/download.html

# 使用HuggingFace datasets (简化版)
pip install datasets
python -c "from datasets import load_dataset; load_dataset('HuggingFaceM4/VQAv2')"
```

**数据规格**:
- 训练集: 443,757图像-问题对
- 验证集: 214,354对
- 测试集: 447,793对
- 模态: 图像 + 文本(问题)

**处理方式**:
- 使用CLIP特征提取器预处理图像
- 文本使用BERT编码
- 模态缺失: 图像替换为黑色/噪声

#### IEMOCAP (优先级: P1)

**下载步骤**:
```bash
# 需要向USC申请: https://sail.usc.edu/iemocap/iemocap_release.htm
# 或使用预处理版本

# 使用Multimodal-Transformer预处理数据
wget https://github.com/yaohungt/Multimodal-Transformer/raw/master/IEMOCAP_features.pkl
```

**数据规格**:
- 5个session, 10个speaker
- 约12小时视频
- 9种情感类别
- 模态: 视频 + 音频 + 文本

### 1.4 基线方法实现

需要在代码中实现的基线:

| 基线 | 实现难度 | 预计代码行数 | 依赖 |
|------|---------|-------------|------|
| DARTS | 中 | ~500 | pytorch |
| LLMatic | 低 | ~300 | 复用我们的LLM backend |
| EvoPrompting | 低 | ~200 | 复用进化框架 |
| DynMM | 中 | ~400 | pytorch |
| FDSNet | 中 | ~400 | 论文复现 |
| ADMN | 高 | ~600 | 层级控制逻辑复杂 |
| Centaur | 中 | ~350 | denoising autoencoder |

**基线代码位置**: `src/baselines/`

---

## 2. 代码架构设计

### 2.1 项目结构

```
autofusionv3/
├── src/
│   ├── inner_loop/              # 内循环: 语法约束探索
│   │   ├── __init__.py
│   │   ├── code_generator.py    # LLM代码生成
│   │   ├── syntax_validator.py  # 语法验证
│   │   ├── error_repair.py      # 错误修复
│   │   ├── shape_verifier.py    # 形状验证(dummy forward)
│   │   └── self_healing.py      # 自修复循环整合
│   │
│   ├── outer_loop/              # 外循环: 性能驱动进化
│   │   ├── __init__.py
│   │   ├── evolver.py           # CMA-ES进化器
│   │   ├── llm_mutation.py      # LLM变异算子
│   │   ├── crossover.py         # 交叉操作
│   │   ├── selection.py         # 选择策略
│   │   └── population.py        # 种群管理
│   │
│   ├── evaluator/               # 评估器
│   │   ├── __init__.py
│   │   ├── proxy_evaluator.py   # 代理评估
│   │   ├── multimodal_rob.py    # mRob计算
│   │   ├── efficiency_metrics.py # FLOPs/延迟
│   │   └── early_stop.py        # 早停判断
│   │
│   ├── architectures/           # 架构模板
│   │   ├── __init__.py
│   │   ├── base_arch.py         # 基础架构类
│   │   └── code_executor.py     # 代码执行沙盒
│   │
│   ├── data/                    # 数据处理
│   │   ├── __init__.py
│   │   ├── mosei_loader.py
│   │   ├── vqa_loader.py
│   │   ├── iemocap_loader.py
│   │   └── modality_dropout.py  # 模态缺失模拟
│   │
│   ├── utils/                   # 工具函数
│   │   ├── __init__.py
│   │   ├── llm_backend.py       # LLM API封装
│   │   ├── logging_utils.py     # 日志记录
│   │   └── checkpoint.py        # 检查点管理
│   │
│   ├── baselines/               # 基线方法
│   │   ├── __init__.py
│   │   ├── darts.py
│   │   ├── llmatic.py
│   │   ├── evo_prompting.py
│   │   ├── dynmm.py
│   │   ├── fdsnet.py
│   │   ├── admn.py
│   │   └── centaur.py
│   │
│   └── main.py                  # 主入口
│
├── configs/                     # 配置文件
│   ├── round1_inner_loop.yaml
│   ├── round2_main_mosei.yaml
│   ├── round2_main_vqa.yaml
│   ├── round2_main_iemocap.yaml
│   ├── round2_ablation.yaml
│   ├── round3_analysis.yaml
│   └── round4_deployment.yaml
│
├── experiments/                 # 实验脚本
│   ├── run_round1.py            # Round 1执行
│   ├── run_round2_main.py       # Round 2主实验
│   ├── run_round2_ablation.py   # Round 2消融
│   ├── run_round3_analysis.py   # Round 3分析
│   ├── run_round4_deployment.py # Round 4部署
│   └── utils.py                 # 实验工具
│
├── scripts/                     # 辅助脚本
│   ├── download_data.sh         # 数据下载
│   ├── setup_env.sh             # 环境配置
│   ├── run_on_gpu43.sh          # 集群运行
│   └── generate_figures.py      # 图表生成
│
├── docs/                        # 文档
│   ├── EAS_PAPER_PLAN.md        # 论文大纲
│   └── EXPERIMENT_IMPLEMENTATION_PLAN.md  # 本文件
│
├── results/                     # 实验结果 (gitignore)
│   ├── round1/
│   ├── round2/
│   ├── round3/
│   └── round4/
│
├── logs/                        # 日志 (gitignore)
├── checkpoints/                 # 检查点 (gitignore)
├── tests/                       # 单元测试
├── Makefile                     # 常用命令
├── requirements.txt             # 依赖
└── README.md                    # 项目说明
```

### 2.2 核心类设计

#### InnerLoop (内循环)

```python
# src/inner_loop/self_healing.py

class SelfHealingCompiler:
    """
    自修复编译器: 保证100%生成可执行代码
    """

    def __init__(self, llm_backend, max_retries=3):
        self.llm = llm_backend
        self.max_retries = max_retries
        self.validator = SyntaxValidator()
        self.shape_verifier = ShapeVerifier()
        self.error_repair = ErrorRepair()

        # 统计
        self.compile_attempts = 0
        self.compile_successes = 0
        self.avg_retries = 0

    def compile(self, prompt, api_contract):
        """
        主编译循环

        Args:
            prompt: 初始LLM prompt
            api_contract: API契约 (输入输出形状)

        Returns:
            code: 可执行代码字符串
            attempts: 尝试次数
            history: 修复历史
        """
        history = []
        current_prompt = prompt

        for attempt in range(self.max_retries):
            self.compile_attempts += 1

            # 1. LLM生成代码
            code = self.llm.generate(current_prompt)

            # 2. 语法验证
            is_valid, syntax_error = self.validator.check(code)
            if not is_valid:
                history.append({
                    'attempt': attempt,
                    'code': code,
                    'error': syntax_error,
                    'type': 'syntax'
                })
                current_prompt = self.error_repair.add_syntax_feedback(
                    current_prompt, code, syntax_error
                )
                continue

            # 3. 形状验证 (dummy forward)
            is_valid, shape_error = self.shape_verifier.verify(code, api_contract)
            if not is_valid:
                history.append({
                    'attempt': attempt,
                    'code': code,
                    'error': shape_error,
                    'type': 'shape'
                })
                current_prompt = self.error_repair.add_shape_feedback(
                    current_prompt, code, shape_error
                )
                continue

            # 4. 模态缺失处理验证
            is_valid, robust_error = self.verify_modality_handling(code)
            if not is_valid:
                history.append({
                    'attempt': attempt,
                    'code': code,
                    'error': robust_error,
                    'type': 'robustness'
                })
                current_prompt = self.error_repair.add_robustness_feedback(
                    current_prompt, code, robust_error
                )
                continue

            # 成功
            self.compile_successes += 1
            self.avg_retries = (self.avg_retries * (self.compile_attempts - 1) + attempt + 1) / self.compile_attempts

            return code, attempt + 1, history

        # 超过最大重试次数
        raise CompilationError(f"Failed after {self.max_retries} attempts", history)

    def get_stats(self):
        """获取编译统计"""
        return {
            'attempts': self.compile_attempts,
            'successes': self.compile_successes,
            'success_rate': self.compile_successes / max(1, self.compile_attempts),
            'avg_retries': self.avg_retries
        }
```

#### OuterLoop (外循环)

```python
# src/outer_loop/evolver.py

class EASEvolver:
    """
    EAS进化器: CMA-ES + LLM变异
    """

    def __init__(self, inner_loop, evaluator, config):
        self.inner_loop = inner_loop
        self.evaluator = evaluator
        self.config = config

        # CMA-ES参数
        self.pop_size = config.get('pop_size', 10)
        self.sigma = config.get('sigma', 0.5)

        # 奖励权重
        self.w_acc = config.get('w_accuracy', 1.0)
        self.w_flops = config.get('w_flops', 0.5)
        self.w_rob = config.get('w_robustness', 2.0)

        # 种群
        self.population = []
        self.fitness_history = []

    def initialize_population(self, seed_architectures):
        """初始化种群"""
        for seed in seed_architectures:
            code, attempts, _ = self.inner_loop.compile(seed['prompt'], seed['contract'])
            self.population.append({
                'code': code,
                'fitness': None,
                'metrics': None
            })

    def evaluate_fitness(self, individual):
        """评估适应度"""
        # 加载架构
        arch = self.load_architecture(individual['code'])

        # 评估完整性能
        acc_full = self.evaluator.evaluate(arch, modality_missing=0.0)

        # 评估缺失性能 (50%缺失)
        acc_missing = self.evaluator.evaluate(arch, modality_missing=0.5)

        # 计算mRob
        mrob = acc_missing / max(acc_full, 1e-6)

        # 计算FLOPs
        flops = self.evaluator.compute_flops(arch)

        # 综合奖励
        reward = (self.w_acc * acc_full +
                  self.w_rob * mrob -
                  self.w_flops * flops / 1e9)

        return {
            'fitness': reward,
            'metrics': {
                'accuracy': acc_full,
                'mrob': mrob,
                'flops': flops,
                'acc_missing': acc_missing
            }
        }

    def evolve(self, max_generations=100):
        """主进化循环"""
        for gen in range(max_generations):
            print(f"\n=== Generation {gen+1}/{max_generations} ===")

            # 评估种群
            for ind in self.population:
                if ind['fitness'] is None:
                    result = self.evaluate_fitness(ind)
                    ind['fitness'] = result['fitness']
                    ind['metrics'] = result['metrics']

            # 选择
            parents = self.select_parents(self.population)

            # LLM变异生成新一代
            offspring = []
            for _ in range(self.pop_size):
                parent = random.choice(parents)
                mutation_prompt = self.build_mutation_prompt(parent)

                try:
                    code, attempts, _ = self.inner_loop.compile(
                        mutation_prompt,
                        self.config['api_contract']
                    )
                    offspring.append({
                        'code': code,
                        'fitness': None,
                        'metrics': None,
                        'parent': parent
                    })
                except CompilationError:
                    # 编译失败,复制父代
                    offspring.append(parent)

            self.population = offspring

            # 记录历史
            best = max(self.population, key=lambda x: x['fitness'] or -1e9)
            self.fitness_history.append({
                'generation': gen,
                'best_fitness': best['fitness'],
                'best_metrics': best['metrics']
            })

            print(f"Best fitness: {best['fitness']:.4f}")
            print(f"Best metrics: {best['metrics']}")

            # 早停检查
            if self.should_stop():
                print("Early stopping triggered")
                break

        return self.get_best_architecture()
```

---

## 3. Round-by-Round 执行计划

### Round 1: 内循环验证 (Week 1-2)

#### 目标
- 实现完整的内循环验证器
- 验证编译成功率从5%→95%
- 找到首个涌现案例

#### Week 1 任务分解

**Day 1-2: 基础架构搭建**
```bash
# 1. 创建项目结构
mkdir -p src/{inner_loop,outer_loop,evaluator,architectures,data,utils,baselines}
mkdir -p configs experiments scripts results logs

# 2. 创建初始文件
touch src/__init__.py
touch requirements.txt

# 3. 提交初始代码
git add .
git commit -m "Initial project structure"
```

**Day 3-4: LLM Backend实现**
```python
# src/utils/llm_backend.py

class LLMBackend:
    """统一LLM接口"""

    def __init__(self, provider='aliyun', model='qwen-max', api_key=None):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.environ.get('ALIYUN_API_KEY')

    def generate(self, prompt, temperature=0.7, max_tokens=2000):
        """生成代码"""
        if self.provider == 'aliyun':
            return self._call_aliyun(prompt, temperature, max_tokens)
        elif self.provider == 'deepseek':
            return self._call_deepseek(prompt, temperature, max_tokens)
        # ...

    def _call_aliyun(self, prompt, temperature, max_tokens):
        import openai
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/api/v1"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a PyTorch expert. Generate executable neural network code."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content
```

**Day 5-7: 内循环实现**
- `syntax_validator.py`: 使用`ast`模块验证Python语法
- `shape_verifier.py`: 创建dummy tensor验证前向传播
- `error_repair.py`: 构建错误反馈prompt

**关键代码片段**:
```python
# src/inner_loop/syntax_validator.py
import ast

class SyntaxValidator:
    def check(self, code):
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

# src/inner_loop/shape_verifier.py
class ShapeVerifier:
    def verify(self, code, api_contract):
        """
        验证代码输入输出形状
        """
        try:
            # 创建沙盒环境
            namespace = {}
            exec(code, namespace)

            # 获取模型类
            ModelClass = namespace.get('DynamicFusionModel')
            if not ModelClass:
                return False, "Model class not found"

            # 创建模型实例
            model = ModelClass(**api_contract.get('model_kwargs', {}))

            # 创建dummy输入
            dummy_inputs = {}
            for name, shape in api_contract['inputs'].items():
                dummy_inputs[name] = torch.randn(*shape)

            # 前向传播
            with torch.no_grad():
                output = model(**dummy_inputs)

            # 验证输出形状
            expected_shape = api_contract['output_shape']
            if output.shape != expected_shape:
                return False, f"Output shape mismatch: {output.shape} vs {expected_shape}"

            return True, None

        except Exception as e:
            return False, str(e)
```

#### Week 2 任务分解

**Day 8-10: 整合测试**
```bash
# 运行内循环测试
python experiments/run_round1.py --config configs/round1_inner_loop.yaml
```

**配置文件** (`configs/round1_inner_loop.yaml`):
```yaml
experiment:
  name: "round1_inner_loop_validation"
  output_dir: "results/round1"
  seed: 42

inner_loop:
  max_retries: 3
  llm:
    provider: "aliyun"
    model: "qwen-max"
    temperature: 0.7

api_contract:
  inputs:
    vision: [2, 576, 1024]  # [batch, seq, dim]
    audio: [2, 400, 512]
    text: [2, 77, 768]
  output_shape: [2, 10]  # 10分类
  model_kwargs:
    hidden_dim: 256

dataset:
  name: "mosei_toy"
  data_path: "data/mosei_toy.pkl"
  num_samples: 1000  # 只用10%数据

evaluation:
  num_epochs: 5
  batch_size: 16
```

**Day 11-12: 涌现案例发现**
- 分析成功生成的代码
- 寻找条件分支模式 (`if confidence < 0.3`)
- 记录第一个涌现案例

**Day 13-14: 结果整理**
- 生成编译成功率曲线图
- 撰写Round 1报告
- 准备Round 2所需数据

#### Round 1 交付物

| 交付物 | 位置 | 验收标准 |
|--------|------|----------|
| 内循环代码 | `src/inner_loop/` | 100%单测通过 |
| 编译成功率曲线 | `results/round1/compile_rate.png` | 5%→95%趋势 |
| 首个涌现案例 | `results/round1/emergent_case_1.py` | 包含条件分支 |
| Round 1报告 | `results/round1/report.md` | 详细记录 |

---

### Round 2: 主实验+消融 (Week 3-5)

#### Week 3: CMU-MOSEI完整实验

**实验矩阵**:
```
方法 × 缺失率 × 种子
= 7方法 × 3缺失率(0%, 25%, 50%) × 5种子
= 105个实验配置
```

**并行策略**:
```bash
# GPU 0: EAS + 主要基线
CUDA_VISIBLE_DEVICES=0 python experiments/run_round2_main.py \
    --dataset mosei --method eas --seed 42

# GPU 1: 其他基线
CUDA_VISIBLE_DEVICES=1 python experiments/run_round2_main.py \
    --dataset mosei --method darts --seed 42

# ... 其他GPU
```

**自动化脚本** (`scripts/run_all_baselines.sh`):
```bash
#!/bin/bash

METHODS=("eas" "darts" "llmatic" "evo_prompting" "dynmm" "fdsnet" "admn" "centaur")
SEEDS=(42 123 456 789 1024)
DROP_PROBS=(0.0 0.25 0.5)

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        for drop in "${DROP_PROBS[@]}"; do
            python experiments/run_round2_main.py \
                --dataset mosei \
                --method $method \
                --seed $seed \
                --drop_prob $drop \
                --gpu 0 &

            # 控制并发数
            if (( $(jobs -r | wc -l) >= 4 )); then
                wait -n
            fi
        done
    done
done
wait
```

#### Week 4: VQA-v2 + IEMOCAP实验

重复相同流程,换数据集配置:
```yaml
dataset:
  name: "vqa_v2"
  # ...
```

#### Week 5: 消融实验

**消融配置** (`configs/round2_ablation.yaml`):
```yaml
ablations:
  - name: "w/o_inner_loop"
    inner_loop:
      enabled: false

  - name: "w/o_outer_loop"
    outer_loop:
      evolver: "random"  # 随机搜索替代CMA-ES

  - name: "w/o_llm"
    mutation:
      use_llm: false  # 传统变异算子

  - name: "fixed_arch"
    search:
      mode: "fixed"  # 固定架构不调参
```

#### Round 2 交付物

| 交付物 | 位置 | 说明 |
|--------|------|------|
| Table 2 | `results/round2/table2_main_results.csv` | 主结果 |
| Table 3 | `results/round2/table3_ablation.csv` | 消融结果 |
| 训练曲线 | `results/round2/learning_curves/` | 每个实验的训练日志 |

---

### Round 3: 可解释性+迁移 (Week 6-7)

#### Week 6: AST分析

**代码分析脚本**:
```python
# experiments/run_round3_analysis.py

import ast
import json
from collections import Counter

def analyze_architecture(code):
    """分析架构代码的AST特征"""
    tree = ast.parse(code)

    features = {
        'num_if_statements': 0,
        'num_residual_connections': 0,
        'num_attention_heads': 0,
        'has_modality_gate': False,
        'has_early_exit': False,
        'operators': Counter()
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            features['num_if_statements'] += 1
            # 检查是否为模态门控
            if 'confidence' in ast.dump(node) or 'modality' in ast.dump(node):
                features['has_modality_gate'] = True

        elif isinstance(node, ast.Call):
            func_name = ast.dump(node.func)
            if 'residual' in func_name or 'skip' in func_name:
                features['num_residual_connections'] += 1
            if 'MultiheadAttention' in func_name:
                features['num_attention_heads'] += 1

    return features

# 批量分析
results = []
for code_file in glob('results/round2/successful_architectures/*.py'):
    with open(code_file) as f:
        code = f.read()

    features = analyze_architecture(code)
    features['reward'] = load_reward(code_file)
    results.append(features)

# 相关性分析
df = pd.DataFrame(results)
correlation = df.corr()['reward'].sort_values(ascending=False)
print(correlation)
```

#### Week 7: 跨模态迁移

**迁移实验**:
```bash
# 在MOSEI上搜索的架构,零样本应用到VQA
python experiments/run_round3_transfer.py \
    --source mosei \
    --target vqa_v2 \
    --arch_path results/round2/best_arch_mosei.py
```

---

### Round 4: 部署+统计 (Week 8)

#### 边缘部署模拟

```python
# src/evaluator/edge_simulator.py

class EdgeSimulator:
    """模拟边缘设备上的部署性能"""

    def __init__(self, device='cuda'):
        self.device = device

    def measure_latency(self, model, input_shape, num_runs=100):
        """测量推理延迟"""
        dummy_input = torch.randn(*input_shape).to(self.device)

        # 预热
        for _ in range(10):
            _ = model(dummy_input)

        torch.cuda.synchronize()

        # 测量
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        latencies = []
        for _ in range(num_runs):
            start.record()
            _ = model(dummy_input)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        return {
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'p99': np.percentile(latencies, 99)
        }

    def measure_energy(self, model, input_shape):
        """估算能耗 (使用nvidia-smi)"""
        # ...
```

#### 图表生成

```bash
# 生成所有论文图表
python scripts/generate_figures.py --results_dir results/ --output_dir figures/
```

---

## 4. 风险控制与备份策略

### API限流处理

```python
# src/utils/llm_backend.py

from tenacity import retry, stop_after_attempt, wait_exponential

class LLMBackend:
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((RateLimitError, TimeoutError))
    )
    def generate_with_retry(self, prompt):
        return self.generate(prompt)
```

### 实验中断恢复

```python
# src/utils/checkpoint.py

class ExperimentCheckpoint:
    """实验检查点管理"""

    def save(self, generation, population, fitness_history):
        checkpoint = {
            'generation': generation,
            'population': population,
            'fitness_history': fitness_history,
            'timestamp': time.time()
        }
        torch.save(checkpoint, f'checkpoints/gen_{generation}.pt')

    def load(self, path):
        return torch.load(path)
```

### 数据备份

```bash
# 定时备份脚本 (crontab)
0 */6 * * * rsync -avz results/ /backup/eas_results_$(date +\%Y\%m\%d)/
```

---

## 5. 每日工作流程

### 晨会检查清单

```bash
# 1. 检查GPU状态
make status

# 2. 检查运行中的实验
ps aux | grep python | grep -v grep

# 3. 查看最新结果
tail -f logs/latest.log

# 4. 同步代码
git pull origin main
git push origin main
```

### 实验记录模板

每个实验运行后记录:
```markdown
## 实验记录: [ID]

**日期**: 2026-03-XX
**配置**: `configs/xxx.yaml`
**GPU**: CUDA 0
**时长**: X小时

**结果**:
- mAcc: X.XX
- mRob: X.XX
- GFLOPs: X.X

**观察**:
- [关键发现]
- [异常情况]

**下一步**:
- [待办]
```

---

## 6. 交付时间表

| 周次 | 主要交付物 | 论文写作 |
|------|-----------|----------|
| W1 | 内循环基础代码 | 开始写Method部分 |
| W2 | 编译率曲线+涌现案例 | 完成Method初稿 |
| W3 | CMU-MOSEI完整结果 | 开始Experiments |
| W4 | VQA+IEMOCAP结果 | 继续Experiments |
| W5 | 消融实验完成 | 完成Experiments |
| W6 | AST分析结果 | 开始Analysis |
| W7 | 迁移实验完成 | 完成Analysis |
| W8 | 所有图表+完整论文 | 完成Conclusion |

---

**下一步行动**:
1. 立即配置GPU43环境
2. 下载CMU-MOSEI数据集
3. 开始Day 1任务: 项目结构搭建

*文档版本: v1.0*
*更新日期: 2026-03-06*

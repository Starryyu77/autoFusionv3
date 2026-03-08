# EAS 实验控制协议

**文档目的**: 确保所有实验的可重复性、可比性和严谨性
**版本**: v1.1
**最后更新**: 2026-03-06

**变更记录**:
- v1.1: 固定使用 `kimi-k2.5` 作为主模型 (用户确认)
- v1.0: 初始版本

---

## 1. 实验设计原则

### 1.1 控制变量总览

```
自变量 (Independent Variables):
├── 搜索方法: {EAS, DARTS, LLMatic, EvoPrompting, DynMM, TFN, ADMN, Centaur}
├── 模态缺失率: {0%, 25%, 50%}
└── 数据集: {CMU-MOSEI, VQA-v2, IEMOCAP}

因变量 (Dependent Variables):
├── mAcc (平均准确率)
├── mRob (模态鲁棒性)
├── GFLOPs (计算量)
└── Latency (推理延迟)

控制变量 (Controlled Variables):
├── 随机种子 (固定5个: 42, 123, 456, 789, 1024)
├── LLM API (统一使用Aliyun Bailian)
├── 评估协议 (统一的few-shot设置)
├── 硬件环境 (统一GPU型号和驱动)
├── 软件版本 (PyTorch, CUDA等)
├── 训练超参 (epochs, batch_size, lr)
└── 缺失模拟策略 (统一的dropout实现)
```

---

## 2. API 统一调用规范

### 2.1 主API配置 (强制统一)

**提供商**: Aliyun Bailian ( dashscope )
**理由**:
- 在AutoFusion 1.0/2.0中已验证稳定性
- 支持多种模型 (kimi-k2.5, glm-5, qwen-max)
- 成本可控

```yaml
# configs/api_config.yaml (全局统一配置)
api:
  provider: "aliyun"
  base_url: "https://dashscope.aliyuncs.com/api/v1"
  api_key: "${ALIYUN_API_KEY}"  # 从环境变量读取

  # 主实验使用模型 (固定)
  default_model: "kimi-k2.5"

  # 备用模型 (主模型超限时切换)
  fallback_models: ["glm-5", "qwen-max"]

  # 统一参数 (所有实验必须一致)
  temperature: 0.7  # 控制创造性，固定
  max_tokens: 2048  # 代码生成足够用
  top_p: 0.95

  # 重试策略
  retry:
    max_attempts: 5
    backoff_factor: 2  # 指数退避
    initial_wait: 4    # 首次等待4秒
    max_wait: 60       # 最大等待60秒
```

### 2.2 API调用封装 (强制使用)

所有实验必须通过以下统一接口调用LLM：

```python
# src/utils/llm_backend.py

import os
import time
import yaml
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

class UnifiedLLMBackend:
    """
    统一LLM后端 - 所有实验必须此类
    确保API调用参数完全一致
    """

    _instance = None  # 单例模式

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return

        # 加载配置
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "../../configs/api_config.yaml"
            )

        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.api_config = self.config['api']
        self.model = self.api_config['default_model']
        self.api_key = os.environ.get('ALIYUN_API_KEY')

        if not self.api_key:
            raise ValueError("ALIYUN_API_KEY environment variable not set")

        # 统计信息
        self.call_count = 0
        self.total_tokens = 0
        self.errors = []

        self._initialized = True

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(
            multiplier=2,
            min=4,
            max=60
        ),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()
    )
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        统一的代码生成接口

        Args:
            prompt: 输入prompt
            temperature: 覆盖默认温度 (不推荐)
            max_tokens: 覆盖默认max_tokens (不推荐)
            model: 覆盖默认模型 (仅fallback时使用)

        Returns:
            {
                'code': str,  # 生成的代码
                'metadata': {  # 调用元数据
                    'model': str,
                    'temperature': float,
                    'prompt_tokens': int,
                    'completion_tokens': int,
                    'latency_ms': float,
                    'timestamp': str
                }
            }
        """
        import openai

        # 使用统一配置 (不允许随意修改)
        temp = temperature or self.api_config['temperature']
        tokens = max_tokens or self.api_config['max_tokens']
        use_model = model or self.model

        client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_config['base_url']
        )

        start_time = time.time()

        try:
            response = client.chat.completions.create(
                model=use_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a PyTorch expert. Generate executable neural network code."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=temp,
                max_tokens=tokens,
                top_p=self.api_config['top_p']
            )

            latency = (time.time() - start_time) * 1000

            result = {
                'code': response.choices[0].message.content,
                'metadata': {
                    'model': use_model,
                    'temperature': temp,
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens,
                    'latency_ms': latency,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }

            # 更新统计
            self.call_count += 1
            self.total_tokens += response.usage.total_tokens

            return result

        except Exception as e:
            self.errors.append({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': str(e),
                'prompt_length': len(prompt)
            })
            raise

    def get_stats(self) -> Dict[str, Any]:
        """获取API调用统计"""
        return {
            'total_calls': self.call_count,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_call': self.total_tokens / max(1, self.call_count),
            'error_count': len(self.errors),
            'error_rate': len(self.errors) / max(1, self.call_count)
        }

    def reset_stats(self):
        """重置统计 (新实验开始时调用)"""
        self.call_count = 0
        self.total_tokens = 0
        self.errors = []


# 使用示例
if __name__ == "__main__":
    # 所有实验统一使用此方式
    llm = UnifiedLLMBackend()

    result = llm.generate("Generate a simple neural network...")
    print(result['code'])
    print(f"Latency: {result['metadata']['latency_ms']}ms")

    # 查看统计
    print(llm.get_stats())
```

### 2.3 API使用约束

**严格禁止**:
- ❌ 不同实验使用不同temperature
- ❌ 随意修改max_tokens
- ❌ 混用不同API提供商 (除非主API故障)
- ❌ 不记录API调用元数据

**强制要求**:
- ✅ 所有实验使用`UnifiedLLMBackend`
- ✅ 每次实验记录完整的metadata
- ✅ 实验结果包含API调用统计
- ✅ 失败调用必须记录并重试

---

## 3. 硬件环境控制

### 3.1 服务器规格 (NTU GPU43)

```yaml
# configs/hardware_config.yaml
hardware:
  server: "NTU GPU43"
  host: "gpu43.dynip.ntu.edu.sg"

  gpu:
    model: "NVIDIA RTX A5000"
    count: 4
    memory: "24GB per GPU"
    driver_version: "535.104.05"  # 固定
    cuda_version: "12.2"  # 固定

  cpu:
    model: "AMD EPYC 7313"
    cores: 32
    memory: "256GB RAM"

  storage:
    project_path: "/usr1/home/s125mdg43_10/AutoFusion_v3"
    data_path: "/usr1/home/s125mdg43_10/data"
    min_free_space: "100GB"
```

### 3.2 GPU分配策略

```bash
# 实验并行分配规则

# Round 1: 内循环验证 (单GPU足够)
CUDA_VISIBLE_DEVICES=0 python experiments/run_round1.py

# Round 2: 主实验 (4GPU并行)
# GPU 0: EAS + DARTS
CUDA_VISIBLE_DEVICES=0 python experiments/run_round2_main.py --methods eas,darts &
# GPU 1: LLMatic + EvoPrompting
CUDA_VISIBLE_DEVICES=1 python experiments/run_round2_main.py --methods llmatic,evo_prompting &
# GPU 2: DynMM + TFN
CUDA_VISIBLE_DEVICES=2 python experiments/run_round2_main.py --methods dynmm,tfn &
# GPU 3: ADMN + Centaur
CUDA_VISIBLE_DEVICES=3 python experiments/run_round2_main.py --methods admn,centaur &

wait
```

### 3.3 环境隔离

```bash
# 每个实验使用独立conda环境 (防止依赖冲突)
conda create -n eas python=3.10 -y
conda activate eas

# 安装固定版本依赖
pip install -r requirements.txt

# 锁定依赖版本
pip freeze > requirements_locked.txt
```

---

## 4. 软件版本控制

### 4.1 核心依赖版本 (固定)

```txt
# requirements.txt - 所有实验必须使用此版本

# PyTorch生态
torch==2.1.0
torchvision==0.16.0
torchaudio==2.1.0
pytorch-cuda==11.8

# Transformers
transformers==4.35.0
datasets==2.14.0
accelerate==0.24.0
tokenizers==0.14.1

# 科学计算
numpy==1.24.3
scipy==1.11.0
scikit-learn==1.3.0
pandas==2.0.3

# 可视化
matplotlib==3.8.0
seaborn==0.13.0

# 优化
optuna==3.4.0
cma==3.3.0

# API
openai==1.3.0
httpx==0.25.0

# 工具
pyyaml==6.0.1
tqdm==4.66.0
tensorboard==2.15.0
wandb==0.16.0

# 代码分析 (Round 3)
ast-decompiler==0.7.0
tree-sitter==0.20.2

# 测试
pytest==7.4.3
pytest-cov==4.1.0
```

### 4.2 版本验证脚本

```python
# scripts/verify_environment.py

import sys
import torch
import transformers

REQUIRED_VERSIONS = {
    'python': '3.10',
    'torch': '2.1.0',
    'transformers': '4.35.0',
    'cuda': '11.8'
}

def verify():
    """验证环境版本是否符合要求"""
    errors = []

    # Python版本
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    if py_version != REQUIRED_VERSIONS['python']:
        errors.append(f"Python版本错误: {py_version}, 需要: {REQUIRED_VERSIONS['python']}")

    # PyTorch版本
    if torch.__version__ != REQUIRED_VERSIONS['torch']:
        errors.append(f"PyTorch版本错误: {torch.__version__}, 需要: {REQUIRED_VERSIONS['torch']}")

    # CUDA版本
    cuda_version = torch.version.cuda
    if cuda_version != REQUIRED_VERSIONS['cuda']:
        errors.append(f"CUDA版本错误: {cuda_version}, 需要: {REQUIRED_VERSIONS['cuda']}")

    # GPU可用性
    if not torch.cuda.is_available():
        errors.append("CUDA不可用")
    else:
        gpu_count = torch.cuda.device_count()
        print(f"✓ 检测到 {gpu_count} 个GPU")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {name} ({mem:.1f}GB)")

    if errors:
        print("\n❌ 环境验证失败:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✅ 环境验证通过")

if __name__ == "__main__":
    verify()
```

---

## 5. 训练超参控制

### 5.1 统一训练配置

```yaml
# configs/training_config.yaml
training:
  # 所有实验固定
  epochs: 15  # 参考AutoFusion 2.0验证结果
  batch_size: 32
  learning_rate: 0.001
  optimizer: "AdamW"
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_epochs: 2

  # 早停
  early_stop:
    enabled: true
    patience: 5
    min_delta: 0.001

  # 随机性控制
  deterministic: true  # 使用确定性算法
  benchmark: false     # 禁用cudnn benchmark
```

### 5.2 评估协议

```yaml
# configs/evaluation_config.yaml
evaluation:
  # Few-shot评估设置
  num_shots: 64  # 统一使用64-shot

  # 模态缺失模拟
  modality_dropout:
    modes: ['random', 'burst', 'progressive']
    probabilities: [0.0, 0.25, 0.5]

  # 评估指标
  metrics:
    - accuracy
    - f1_score
    - mrob  # 模态鲁棒性

  # 重复次数
  num_seeds: 5
  seeds: [42, 123, 456, 789, 1024]
```

---

## 6. 模态缺失模拟控制

### 6.1 统一缺失模拟器

```python
# src/data/modality_dropout.py

import torch
import numpy as np
from typing import Dict, Literal

class UnifiedModalityDropout:
    """
    统一的模态缺失模拟器
    所有实验必须使用此类，确保缺失模拟一致
    """

    def __init__(
        self,
        drop_prob: float = 0.5,
        mode: Literal['random', 'burst', 'progressive'] = 'random',
        seed: int = 42
    ):
        self.drop_prob = drop_prob
        self.mode = mode
        self.rng = np.random.RandomState(seed)

        # 缺失模式配置 (固定)
        self.mode_configs = {
            'random': {
                'description': '随机独立缺失',
                'implementation': self._random_dropout
            },
            'burst': {
                'description': '连续时间窗缺失（模拟传感器故障）',
                'burst_length': 5,  # 连续5帧缺失
                'implementation': self._burst_dropout
            },
            'progressive': {
                'description': '渐进衰减（模拟信号弱化）',
                'noise_std': 0.1,
                'implementation': self._progressive_dropout
            }
        }

    def __call__(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """
        应用模态缺失

        Args:
            batch: {
                'vision': [B, T, D] 或 [B, D],
                'audio': [B, T, D] 或 [B, D],
                'text': [B, T, D] 或 [B, D]
            }

        Returns:
            masked_batch: 缺失后的数据
            masks: {modality: binary_mask}
        """
        return self.mode_configs[self.mode]['implementation'](batch)

    def _random_dropout(self, batch):
        """随机独立缺失"""
        batch_size = batch['vision'].shape[0]
        masks = {}

        for modality in ['vision', 'audio', 'text']:
            if modality not in batch:
                continue

            # 为每个样本独立决定是否缺失
            mask = torch.from_numpy(
                self.rng.rand(batch_size) > self.drop_prob
            ).float()

            # 应用mask
            while mask.dim() < batch[modality].dim():
                mask = mask.unsqueeze(-1)

            batch[modality] = batch[modality] * mask.to(batch[modality].device)
            masks[modality] = mask

        return batch, masks

    def _burst_dropout(self, batch):
        """连续时间窗缺失"""
        # 模拟传感器连续故障
        # 实现略...
        pass

    def _progressive_dropout(self, batch):
        """渐进衰减"""
        # 模拟信号逐渐变弱
        # 实现略...
        pass

    def get_config(self) -> Dict:
        """获取配置 (用于实验记录)"""
        return {
            'drop_prob': self.drop_prob,
            'mode': self.mode,
            'mode_description': self.mode_configs[self.mode]['description']
        }
```

### 6.2 缺失模拟验证

```python
# 验证缺失模拟的正确性
def verify_dropout():
    dropout = UnifiedModalityDropout(drop_prob=0.5, mode='random', seed=42)

    # 创建测试数据
    batch = {
        'vision': torch.ones(100, 576, 1024),
        'audio': torch.ones(100, 400, 512),
        'text': torch.ones(100, 77, 768)
    }

    masked, masks = dropout(batch)

    # 验证缺失率
    for mod in ['vision', 'audio', 'text']:
        actual_drop = 1 - masks[mod].mean().item()
        print(f"{mod}: 目标缺失率={0.5}, 实际={actual_drop:.3f}")
        assert abs(actual_drop - 0.5) < 0.1, "缺失率偏差过大"

    print("✅ 缺失模拟验证通过")
```

---

## 7. 随机性控制

### 7.1 种子设置

```python
# src/utils/random_control.py

import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    设置全局随机种子，确保可重复性
    所有实验必须在开始前调用
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 确定性设置
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"✅ 随机种子已设置: {seed}")

# 实验使用的5个种子
EXPERIMENT_SEEDS = [42, 123, 456, 789, 1024]
```

### 7.2 种子验证

```python
def verify_seed_reproducibility():
    """验证相同种子产生相同结果"""
    set_seed(42)

    # 生成随机数
    r1 = torch.rand(10)

    set_seed(42)
    r2 = torch.rand(10)

    assert torch.allclose(r1, r2), "种子不可重复!"
    print("✅ 种子可重复性验证通过")
```

---

## 8. 实验记录规范

### 8.1 必须记录的信息

每个实验运行必须生成以下文件：

```
results/
└── {experiment_id}/
    ├── config.yaml           # 完整配置快照
    ├── metadata.json         # 实验元数据
    ├── logs/
    │   ├── training.log      # 训练日志
    │   └── api_calls.jsonl   # API调用记录
    ├── checkpoints/
    │   └── best_model.pt     # 最佳模型
    ├── metrics/
    │   ├── train_metrics.csv # 训练过程指标
    │   └── eval_metrics.csv  # 评估指标
    └── generated/
        └── architecture.py   # 生成的架构代码
```

### 8.2 元数据格式

```json
{
  "experiment_id": "eas_mosei_seed42_drop50",
  "timestamp": "2026-03-06T10:30:00",
  "method": "EAS",
  "dataset": "CMU-MOSEI",
  "seed": 42,
  "modality_dropout": {
    "prob": 0.5,
    "mode": "random"
  },
  "hardware": {
    "gpu": "RTX A5000",
    "cuda_version": "12.2"
  },
  "software": {
    "python": "3.10.12",
    "torch": "2.1.0",
    "transformers": "4.35.0"
  },
  "api_stats": {
    "total_calls": 150,
    "total_tokens": 45000,
    "avg_latency_ms": 1200
  },
  "results": {
    "mAcc": 0.852,
    "mRob": 0.84,
    "GFLOPs": 7.2
  }
}
```

---

## 9. 数据预处理控制

### 9.1 特征提取统一

```python
# 所有数据集使用相同的特征提取器

FEATURE_EXTRACTORS = {
    'vision': {
        'model': 'openai/clip-vit-large-patch14',
        'output_dim': 1024,
        'preprocessing': 'clip_standard'
    },
    'audio': {
        'model': 'facebook/wav2vec2-large-960h',
        'output_dim': 1024,
        'preprocessing': 'wav2vec_standard'
    },
    'text': {
        'model': 'bert-base-uncased',
        'output_dim': 768,
        'preprocessing': 'bert_standard'
    }
}
```

### 9.2 预处理流程

```bash
# 1. 下载原始数据
python scripts/download_data.py --dataset mosei

# 2. 提取特征 (统一使用上述模型)
python scripts/extract_features.py --dataset mosei --config configs/features.yaml

# 3. 验证特征
python scripts/verify_features.py --dataset mosei

# 4. 保存到共享目录
cp -r data/mosei_processed /usr1/home/s125mdg43_10/data/
```

---

## 10. 质量控制检查清单

### 实验开始前

- [ ] 运行 `python scripts/verify_environment.py` 通过
- [ ] 确认GPU驱动版本 535.104.05
- [ ] 确认CUDA版本 12.2
- [ ] 设置环境变量 `export ALIYUN_API_KEY=xxx`
- [ ] 检查磁盘空间 > 100GB
- [ ] 确认使用统一conda环境 `eas`

### 实验运行中

- [ ] API调用通过 `UnifiedLLMBackend`
- [ ] 随机种子正确设置
- [ ] 训练日志正常输出
- [ ] 定期保存checkpoint

### 实验结束后

- [ ] 所有metrics文件完整
- [ ] metadata.json已生成
- [ ] 生成的代码已保存
- [ ] 结果已备份到共享目录
- [ ] 实验记录已更新

---

## 11. 常见错误预防

### 11.1 API相关

| 错误 | 预防 | 处理 |
|------|------|------|
| Rate Limit | 使用指数退避 | 自动重试5次 |
| Timeout | 设置超时30秒 | 切换到fallback模型 |
| Invalid Key | 预检查环境变量 | 报错并退出 |

### 11.2 训练相关

| 错误 | 预防 | 处理 |
|------|------|------|
| OOM | 监控显存使用 | 减小batch_size |
| NaN loss | 梯度裁剪 | 记录并跳过 |
| 过拟合 | 早停机制 | 自动停止 |

### 11.3 数据相关

| 错误 | 预防 | 处理 |
|------|------|------|
| 缺失文件 | 预检查路径 | 自动下载 |
| 特征维度不匹配 | 统一预处理 | 报错提示 |
| 标签缺失 | 数据验证脚本 | 跳过样本 |

---

## 12. 附录：快速启动检查表

```bash
# 1. 登录服务器
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg

# 2. 激活环境
conda activate eas

# 3. 验证环境
python scripts/verify_environment.py

# 4. 设置API密钥
export ALIYUN_API_KEY="your-key"

# 5. 检查数据
ls /usr1/home/s125mdg43_10/data/mosei_processed

# 6. 运行实验
CUDA_VISIBLE_DEVICES=0 python experiments/run_round1.py --config configs/round1.yaml

# 7. 监控实验
tail -f logs/latest.log

# 8. 实验结束后备份
rsync -avz results/ /backup/experiment_$(date +%Y%m%d)/
```

---

**文档维护**: 任何实验流程变更必须更新本文档
**审核周期**: 每周检查一次控制变量执行情况

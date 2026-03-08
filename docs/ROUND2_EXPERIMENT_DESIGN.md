# Round 2 实验详细设计方案

**文档版本**: v1.0
**创建日期**: 2026-03-07
**状态**: 待确认

---

## 目录

1. [实验架构与流程](#1-实验架构与流程)
2. [最优方案的选择与评估](#2-最优方案的选择与评估)
3. [实验严谨性与调整](#3-实验严谨性与调整)
4. [附录: 技术实现细节](#4-附录技术实现细节)

---

## 1. 实验架构与流程

### 1.1 统一实验框架设计

为了保证公平比较，我们设计了一个**统一的实验框架**，所有方法（包括EAS和基线）都在相同的"骨架"下进行测试：

```
┌─────────────────────────────────────────────────────────────┐
│                    统一实验框架 (Fixed)                       │
├─────────────────────────────────────────────────────────────┤
│  输入层 (Input Layer)                                        │
│  ├── 视觉: CLIP-ViT-L/14 [batch, 576, 768]                  │
│  ├── 音频: wav2vec 2.0 [batch, 400, 1024]                   │
│  └── 文本: BERT-Base [batch, 77, 768]                       │
│                                                              │
│  ↓ 冻结特征提取器 (Frozen)                                    │
│                                                              │
│  投影层 (Projection Layer) - 共享且可训练                     │
│  ├── 视觉投影: Linear(768, 1024) + LayerNorm + GELU         │
│  ├── 音频投影: 无 (已是1024维)                               │
│  └── 文本投影: Linear(768, 1024) + LayerNorm + GELU         │
│                                                              │
│  ↓ 统一维度输出 [batch, seq, 1024]                           │
│                                                              │
│  融合层 (Fusion Layer) ← 【不同方法的核心差异】                │
│  ├── EAS: 动态生成的PyTorch代码                              │
│  ├── DARTS: 可微分架构搜索的cell                              │
│  ├── LLMatic: LLM+QD搜索的架构                               │
│  ├── EvoPrompting: 进化优化的架构                             │
│  ├── DynMM: 动态路由+门控融合                                │
│  ├── TFN: 张量融合网络                                      │
│  ├── ADMN: 层级自适应网络                                    │
│  └── Centaur: 去噪+补全+鲁棒融合                              │
│                                                              │
│  ↓                                                          │
│                                                              │
│  输出层 (Output Layer)                                       │
│  └── 分类头: Linear(1024, 10)                               │
└─────────────────────────────────────────────────────────────┘
```

**关键设计决策**:
- 特征提取器: 全部冻结 (CLIP 768维, wav2vec 1024维, BERT 768维)
- 投影层: **共享且可训练**，将视觉和文本从768维投影到1024维
- 统一维度: 所有模态进入融合层前均为**1024维**
- 分类头: 统一为 Linear(1024, 10)

### 1.2 EAS方法的完整流程

#### 阶段A: 架构搜索 (Architecture Search)

```python
# EAS搜索流程详细说明

初始化:
  - 清空历史记录
  - 初始化LLM后端 (kimi-k2.5)
  - 设置API契约 (输入/输出维度)

for iteration in range(200):  # 200轮搜索

    # Step 1: 构建提示词 (Prompt Construction)
    prompt = f"""
    任务: 设计多模态融合架构
    输入维度: {vision: [B,576,1024], audio: [B,400,1024], text: [B,77,768]}
    输出维度: [B, 10]

    历史最佳: {best_architecture_so_far}
    当前策略: {explore|exploit|refine}  # 根据进度调整

    要求:
    1. 创建nn.Module子类
    2. 处理模态缺失 (输入可能为None)
    3. 使用标准PyTorch操作

    生成完整Python代码:
    """

    # Step 2: LLM生成代码 (Code Generation)
    for attempt in range(5):  # 最多5次尝试
        code = llm.generate(prompt)

        # Step 3: 编译验证 (Inner Loop)
        is_valid, error = compile_and_verify(code)
        if is_valid:
            break
        else:
            # 错误反馈给LLM
            prompt += f"\n【错误】: {error}\n请修复后重新生成。"

    if not is_valid:
        continue  # 跳过这轮

    # Step 4: Few-shot评估 (Proxy Evaluation)
    model = instantiate(code)

    # 在64个样本上训练5个epoch
    train(model, few_shot_data, epochs=5)

    # 评估指标
    acc_full = evaluate(model, val_data, dropout=0.0)
    acc_50 = evaluate(model, val_data, dropout=0.50)
    mrob = acc_50 / acc_full if acc_full > 0 else 0
    flops = count_flops(model)

    # 计算奖励
    reward = 1.0 * acc_full + 2.0 * mrob - 0.5 * (flops / 1e9)

    # Step 5: 记录与反馈
    history.append({
        'iteration': iteration,
        'code': code,
        'accuracy': acc_full,
        'mrob': mrob,
        'flops': flops,
        'reward': reward
    })

    # Step 6: 更新搜索策略 (CMA-ES + LLM feedback)
    if iteration % 10 == 0:
        strategy = update_strategy(history)
        prompt = inject_strategy(prompt, strategy)

# 输出: 奖励最高的架构代码
best_architecture = max(history, key=lambda x: x['reward'])
save_to_file(best_architecture['code'], 'best_eas_arch.py')
```

#### 阶段B: 最优架构全量训练

```python
# 使用找到的最优架构进行完整训练

# 1. 加载最优架构
code = load('best_eas_arch.py')
model = instantiate(code)

# 2. 完整训练 (所有数据)
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, patience=10)

training_history = []
for epoch in range(50):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val_acc = evaluate(model, val_loader)
    scheduler.step(val_acc)

    training_history.append({
        'epoch': epoch,
        'loss': train_loss,
        'val_acc': val_acc
    })

    # 早停检查
    if early_stopping.should_stop(val_acc):
        break

# 3. 最终测试 (3种缺失率 × 5个随机种子)
results = {}
for seed in [42, 123, 456, 789, 999]:
    set_seed(seed)
    for dropout in [0.0, 0.25, 0.50]:
        acc = evaluate(model, test_loader, dropout=dropout)
        results[f'seed{seed}_drop{dropout}'] = acc
```

### 1.3 EAS架构的具体组成

#### 内循环 (Self-Healing Compilation)

```python
class InnerLoop:
    """确保生成代码100%可编译"""

    def compile(self, code, max_retries=5):
        for attempt in range(max_retries):
            # 1. 语法检查
            try:
                ast.parse(code)
            except SyntaxError as e:
                feedback = f"语法错误: {e}"
                continue

            # 2. 执行验证
            try:
                namespace = {}
                exec(code, namespace)
                model_class = find_model_class(namespace)
                model = model_class()
            except Exception as e:
                feedback = f"执行错误: {e}"
                continue

            # 3. 形状验证
            try:
                dummy_input = create_dummy_input()
                output = model(**dummy_input)
                assert output.shape == expected_shape
            except Exception as e:
                feedback = f"形状错误: {e}"
                continue

            # 全部通过
            return True, code, attempt + 1

        return False, None, max_retries
```

#### 外循环 (Performance-Driven Evolution)

```python
class OuterLoop:
    """基于性能反馈优化搜索"""

    def __init__(self):
        self.archive = []  # 存储所有评估过的架构
        self.cma_es = CMAEvolutionStrategy(...)  # CMA-ES优化器

    def search(self, max_iterations=200):
        for iteration in range(max_iterations):
            # 确定当前策略阶段
            phase = self.get_phase(iteration / max_iterations)
            # 0-30%: exploration (探索多样架构)
            # 30-70%: exploitation (利用已知好架构)
            # 70-100%: refinement (精调最优架构)

            # 生成提示词
            prompt = self.build_prompt(phase)

            # 内循环获取可编译代码
            code, attempts = self.inner_loop.compile(prompt)

            # 评估
            metrics = self.evaluate(code)

            # 更新CMA-ES
            self.cma_es.tell(metrics['reward'])

            # 保存到archive
            self.archive.append({
                'code': code,
                'metrics': metrics,
                'iteration': iteration,
                'phase': phase
            })

        # 返回最佳架构
        return max(self.archive, key=lambda x: x['metrics']['reward'])
```

#### 奖励函数设计

```python
def compute_reward(self, metrics):
    """
    奖励函数: 准确率 + 鲁棒性 - 计算成本

    权重设计依据:
    - accuracy (1.0): 基础性能，不能太低
    - mrob (2.0): 核心指标，重中之重
    - flops (0.5): 效率约束，防止过大
    """
    acc = metrics['accuracy']  # 0-1
    mrob = metrics['mrob']     # 0-1 (希望>0.85)
    flops = metrics['flops']   # 实际FLOPs

    # 归一化FLOPs (目标<10G)
    target_flops = 10e9
    flops_penalty = max(0, flops - target_flops) / target_flops

    reward = (1.0 * acc +
              2.0 * mrob -
              0.5 * flops_penalty)

    return reward
```

### 1.4 特征投影层设计 (关键修改)

#### 维度对齐方案 (方案1确认)

| 模态 | 原始维度 | 投影后维度 | 投影层 | 训练状态 |
|------|---------|-----------|--------|---------|
| **视觉 (CLIP)** | 768维 | **1024维** | Linear(768,1024) + LN + GELU | 可训练 ✅ |
| **音频 (wav2vec)** | 1024维 | **1024维** | 无 | - |
| **文本 (BERT)** | 768维 | **1024维** | Linear(768,1024) + LN + GELU | 可训练 ✅ |

#### 投影层实现

```python
class UnifiedFeatureProjection(nn.Module):
    """
    统一特征投影层 - 所有方法共享
    将不同模态投影到相同维度 (1024)
    """

    def __init__(self):
        super().__init__()

        # 视觉投影: CLIP-ViT-L/14 (768→1024)
        self.vision_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 音频无需投影 (已是1024)
        self.audio_proj = nn.Identity()

        # 文本投影: BERT-Base (768→1024)
        self.text_proj = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, vision, audio, text):
        """
        Args:
            vision: [B, 576, 768] from CLIP
            audio: [B, 400, 1024] from wav2vec
            text: [B, 77, 768] from BERT

        Returns:
            vision: [B, 576, 1024]
            audio: [B, 400, 1024]
            text: [B, 77, 1024]
        """
        return {
            'vision': self.vision_proj(vision),
            'audio': self.audio_proj(audio),
            'text': self.text_proj(text)
        }
```

#### 设计说明

1. **为什么投影到1024维？**
   - wav2vec 2.0 Large原生1024维，质量高，不应压缩
   - 768→1024是升维，不会损失信息
   - 三模态统一1024维，便于后续融合

2. **为什么共享投影层？**
   - 所有方法在相同输入空间竞争，公平
   - 投影层作为"特征提取"的一部分，不算方法差异
   - 避免某些方法学习更好的投影而占便宜

3. **投影层是否参与架构搜索？**
   - **不参与**！投影层是固定的预处理
   - EAS搜索的是融合模块，不包括投影层
   - 所有基线方法使用相同的投影层实例

#### 使用方式

```python
# 1. 冻结特征提取
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").freeze()
wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h").freeze()
bert = BertModel.from_pretrained("bert-base-uncased").freeze()

# 2. 共享投影层 (可训练)
projection = UnifiedFeatureProjection()  # 所有方法共享这个实例

# 3. 特征提取流程
raw_vision = clip(images)      # [B, 576, 768]
raw_audio = wav2vec(audio)     # [B, 400, 1024]
raw_text = bert(text)          # [B, 77, 768]

features = projection(
    raw_vision, raw_audio, raw_text
)  # 全部变为 [B, seq, 1024]

# 4. 输入到融合模块 (各方法差异点)
fused = method_fusion(features)  # EAS/DynMM/ADMN等
output = classifier(fused)       # 统一分类头
```

---

## 2. 最优方案的选择与评估

### 2.1 最优架构选择标准

#### 主要标准 (Primary Criteria)

| 排名 | 标准 | 权重 | 说明 |
|------|------|------|------|
| 1 | **mRob@50%** | 40% | 50%模态缺失时性能保持率，核心指标 |
| 2 | **Accuracy** | 30% | 完整模态下的准确率 |
| 3 | **mRob@25%** | 20% | 25%模态缺失时性能 |
| 4 | **FLOPs** | 10% | 计算效率 |

#### 选择流程

```python
def select_best_architecture(archive):
    """
    从搜索历史中选择最优架构
    """
    # 筛选条件
    candidates = [
        arch for arch in archive
        if arch['metrics']['flops'] < 20e9  # 计算成本可接受
    ]

    # 多目标排序
    scored_candidates = []
    for arch in candidates:
        m = arch['metrics']
        score = (
            0.4 * m['mrob_50'] +      # 40%权重
            0.3 * m['accuracy'] +      # 30%权重
            0.2 * m['mrob_25'] +       # 20%权重
            0.1 * (1 - m['flops']/20e9)  # 10%权重
        )
        scored_candidates.append((score, arch))

    # 返回最高分
    best = max(scored_candidates, key=lambda x: x[0])
    return best[1]
```

### 2.2 具体评估标准详解

#### 指标A: 模态鲁棒性 (mRob)

```python
def compute_mrob(model, test_data):
    """
    计算模态鲁棒性
    mRob = Performance_missing / Performance_full
    """
    # 完整模态性能
    acc_full = evaluate(model, test_data, dropout_rate=0.0)

    # 50%缺失性能
    acc_50 = evaluate(model, test_data, dropout_rate=0.5)

    # 鲁棒性指标
    mrob = acc_50 / acc_full if acc_full > 0 else 0

    return mrob

# 评估时的模态缺失模拟
def apply_dropout(batch, dropout_rate=0.5):
    """
    随机缺失模态
    dropout_rate=0.5: 每个模态有50%概率被置零
    """
    batch_size = batch['vision'].shape[0]

    for mod in ['vision', 'audio', 'text']:
        # 为每个样本独立决定是否缺失该模态
        mask = torch.rand(batch_size) > dropout_rate
        batch[mod] = batch[mod] * mask.unsqueeze(1).unsqueeze(2)

    return batch
```

#### 指标B: 完整准确率

```python
def evaluate_accuracy(model, test_loader):
    """
    在完整模态下评估准确率
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(batch['vision'], batch['audio'], batch['text'])
            predictions = outputs.argmax(dim=-1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    return correct / total
```

#### 指标C: FLOPs计算

```python
def count_flops(model, input_dims):
    """
    计算模型FLOPs
    """
    from thop import profile

    dummy_inputs = {
        'vision': torch.randn(1, 576, 1024),
        'audio': torch.randn(1, 400, 1024),
        'text': torch.randn(1, 77, 768)
    }

    flops, params = profile(model, inputs=(dummy_inputs,))

    return flops, params
```

### 2.3 测试流程详细说明

#### 测试流程图

```
┌──────────────────────────────────────────────────────────────┐
│                     完整测试流程                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  步骤1: 模型加载                                              │
│  ├── 加载架构代码 (best_arch.py)                              │
│  ├── 实例化模型                                               │
│  └── 加载训练好的权重                                          │
│                                                              │
│  步骤2: 多种子测试 (5个随机种子)                               │
│  ├── Seed 42                                                  │
│  │   ├── 0%缺失测试 → Accuracy_42_0                           │
│  │   ├── 25%缺失测试 → Accuracy_42_25                         │
│  │   └── 50%缺失测试 → Accuracy_42_50                         │
│  ├── Seed 123                                                 │
│  │   └── ...                                                  │
│  ├── Seed 456                                                 │
│  ├── Seed 789                                                 │
│  └── Seed 999                                                 │
│                                                              │
│  步骤3: 计算统计量                                             │
│  ├── 平均准确率: mean(Accuracy_*_0)                           │
│  ├── 平均mRob@25%: mean(Accuracy_*_25 / Accuracy_*_0)         │
│  └── 平均mRob@50%: mean(Accuracy_*_50 / Accuracy_*_0)         │
│                                                              │
│  步骤4: 置信区间                                               │
│  └── 计算95%置信区间 (5次运行的标准差)                         │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

#### 测试代码示例

```python
class FinalEvaluator:
    """最终评估器"""

    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data

    def run_full_evaluation(self):
        """运行完整评估"""
        results = []

        for seed in [42, 123, 456, 789, 999]:
            set_seed(seed)

            for dropout in [0.0, 0.25, 0.50]:
                acc = self.evaluate_with_dropout(dropout)
                results.append({
                    'seed': seed,
                    'dropout': dropout,
                    'accuracy': acc
                })

        # 汇总
        summary = self.summarize(results)
        return summary

    def summarize(self, results):
        """汇总结果"""
        import numpy as np

        # 按缺失率分组
        full_acc = [r['accuracy'] for r in results if r['dropout'] == 0.0]
        drop25_acc = [r['accuracy'] for r in results if r['dropout'] == 0.25]
        drop50_acc = [r['accuracy'] for r in results if r['dropout'] == 0.50]

        # 计算mRob
        mrob_25 = [d25 / full for full, d25 in zip(full_acc, drop25_acc)]
        mrob_50 = [d50 / full for full, d50 in zip(full_acc, drop50_acc)]

        return {
            'accuracy_mean': np.mean(full_acc),
            'accuracy_std': np.std(full_acc),
            'mrob_25_mean': np.mean(mrob_25),
            'mrob_25_std': np.std(mrob_25),
            'mrob_50_mean': np.mean(mrob_50),
            'mrob_50_std': np.std(mrob_50),
        }
```

---

## 3. 实验严谨性与调整

### 3.1 统一框架设计

为了确保公平比较，我们对所有方法做了以下统一：

#### A. 输入统一

```python
# 所有方法使用相同的输入维度
INPUT_DIMS = {
    'vision': [576, 1024],  # CLIP-ViT-L/14
    'audio': [400, 1024],   # wav2vec 2.0 Large
    'text': [77, 768]       # BERT-Base
}

# 所有方法使用相同的预处理
class UnifiedPreprocessor:
    """统一预处理"""

    def __init__(self):
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 冻结所有特征提取器
        for model in [self.clip, self.wav2vec, self.bert]:
            for param in model.parameters():
                param.requires_grad = False

    def extract_features(self, raw_data):
        """提取特征，所有方法使用相同的特征"""
        return {
            'vision': self.clip(raw_data['image']),
            'audio': self.wav2vec(raw_data['audio']),
            'text': self.bert(raw_data['text'])
        }
```

#### B. 输出层统一

```python
# 所有方法使用相同的输出层
class UnifiedClassifier(nn.Module):
    """统一分类头"""

    def __init__(self, hidden_dim, num_classes=10):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, fused_features):
        return self.classifier(fused_features)
```

#### C. 训练配置统一

```python
# 所有方法使用相同的训练配置
TRAIN_CONFIG = {
    'optimizer': 'Adam',
    'lr': 0.001,
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_patience': 10,
    'batch_size': 32,
    'max_epochs': 50,
    'early_stop_patience': 20,
    'loss_function': 'CrossEntropyLoss'
}
```

### 3.2 控制变量

| 变量 | 控制方式 | 说明 |
|------|---------|------|
| **数据** | 固定train/val/test划分 | 所有方法使用相同的数据分割 |
| **特征** | 预提取并保存 | 避免特征提取的随机性 |
| **种子** | 固定5个随机种子 | 所有方法使用相同的随机种子 |
| **硬件** | 同一台服务器 | NTU GPU43, 相同GPU型号 |
| **软件** | 相同环境 | PyTorch 2.1.0, CUDA 11.8 |
| **评估** | 相同评估代码 | 使用统一的评估脚本 |

### 3.3 基线方法的具体调整

#### 基线1: DARTS

```python
# 原始DARTS问题: 搜索的是cell结构，需要适配多模态

class DARTSAdapter:
    """DARTS适配器"""

    def __init__(self, input_dims, num_classes):
        # 为每个模态对创建搜索空间
        self.search_space = {
            'vision_audio': DARTSCell(input_dims['vision'][1], input_dims['audio'][1]),
            'vision_text': DARTSCell(input_dims['vision'][1], input_dims['text'][1]),
            'audio_text': DARTSCell(input_dims['audio'][1], input_dims['text'][1])
        }

    def search(self, train_data, val_data, iterations=200):
        """运行DARTS搜索"""
        # 标准DARTS双层级优化
        for iteration in range(iterations):
            # 1. 训练架构参数
            train_arch_params(self.search_space, train_data)
            # 2. 更新网络权重
            train_network_weights(self.search_space, val_data)

        # 离散化得到最终架构
        best_arch = discretize(self.search_space)
        return best_arch
```

#### 基线2: DynMM

```python
# 原始DynMM需要调整输入维度

class DynMMAdapter(DynMM):
    """适配后的DynMM"""

    def __init__(self, input_dims, num_classes, hidden_dim=1024):
        # 注意: input_dims已经是投影后的统一维度(1024)
        # 这里不需要再投影，直接使用

        # 动态路由和融合 (输入输出都是1024维)
        self.routing = DynamicRouter(hidden_dim)
        self.fusion = GatedFusion(hidden_dim)

        # 输出处理: 确保返回[B, 1024]
        self.output_pool = nn.AdaptiveAvgPool1d(1)

        # 注意: 分类头在框架层面统一，不在适配器里定义
```

#### 基线3: TFN, ADMN, Centaur

```python
# 这些方法的适配类似

class BaselineAdapter:
    """通用基线适配器 - 输入已经是投影后的1024维"""

    def __init__(self, baseline_class, input_dims, num_classes):
        # 注意: input_dims已经是1024(投影后)
        self.baseline = baseline_class(
            input_dim=1024,  # 统一1024维
            num_classes=num_classes
        )

        # 移除基线自带的分类头(由框架统一提供)
        if hasattr(self.baseline, 'classifier'):
            del self.baseline.classifier

    def forward(self, inputs):
        # inputs: {vision:[B,576,1024], audio:[B,400,1024], text:[B,77,1024]}

        # 基线融合模块处理
        fused = self.baseline.fusion(inputs)  # [B, 1024]

        return fused  # 返回[B, 1024], 由框架统一分类
```

### 3.4 公平性验证清单

```markdown
□ 所有方法使用相同的预提取特征
□ 所有方法的输出层结构相同
□ 所有方法的训练超参数相同
□ 所有方法使用相同的数据加载器
□ 所有方法使用相同的评估脚本
□ 所有方法在相同硬件上运行
□ 所有方法测试时使用的随机种子相同
□ 所有方法的FLOPs计算使用相同工具
```

---

## 4. 附录: 技术实现细节

### 4.1 特征提取命令

```bash
# 预提取CLIP特征
python scripts/extract_features.py \
    --model clip-vit-l14 \
    --input data/mosei/raw_images \
    --output data/mosei/clip_features.npy

# 预提取wav2vec特征
python scripts/extract_features.py \
    --model wav2vec2-large \
    --input data/mosei/raw_audio \
    --output data/mosei/wav2vec_features.npy

# 预提取BERT特征
python scripts/extract_features.py \
    --model bert-base \
    --input data/mosei/raw_text \
    --output data/mosei/bert_features.npy
```

### 4.2 实验运行命令

```bash
# EAS搜索
python experiments/run_round2_main.py \
    --method eas \
    --config configs/round2/eas_mosei.yaml \
    --phase search \
    --output results/round2/eas_search/

# EAS评估
python experiments/run_round2_main.py \
    --method eas \
    --config configs/round2/eas_mosei.yaml \
    --phase evaluate \
    --architecture results/round2/eas_search/best_arch.py \
    --seeds 42,123,456,789,999

# 基线方法
python experiments/run_round2_main.py \
    --method dynmm \
    --config configs/round2/dynmm_mosei.yaml \
    --seeds 42,123,456,789,999
```

### 4.3 结果汇总脚本

```python
# scripts/summarize_results.py

import pandas as pd
import glob

# 收集所有结果
results = []
for method in ['eas', 'darts', 'llmatic', 'evoprompting',
               'dynmm', 'tfn', 'admn', 'centaur']:
    for result_file in glob.glob(f'results/round2/{method}_*.json'):
        data = json.load(open(result_file))
        results.append({
            'method': method,
            **data
        })

# 生成Table 2
df = pd.DataFrame(results)
table2 = df.groupby('method').agg({
    'accuracy': ['mean', 'std'],
    'mrob_50': ['mean', 'std'],
    'flops': 'mean'
})

print(table2.to_latex())
```

---

## 5. 关键修复与更新 (2026-03-07)

基于详细review后的修复记录：

### 5.1 维度统一修复

**修复前问题**: Prompt写768维，实际输入1024维，导致mismatch

**修复后** (已确认):
```python
# 所有地方统一为1024维
INPUT_DIMS = {
    'vision': [576, 1024],  # CLIP 768→1024 (投影后)
    'audio': [400, 1024],   # wav2vec 原生1024
    'text': [77, 1024]      # BERT 768→1024 (投影后)
}

# EAS Prompt模板 (强制要求)
"""
输入维度:
- vision: [B, 576, 1024]
- audio: [B, 400, 1024]
- text: [B, 77, 1024]

输出要求: [B, 1024] (全局平均池化后)
"""
```

### 5.2 模态缺失模拟统一

**修复前问题**: Prompt说"None"，实际用mask

**修复后** (已确认):
```python
# Prompt明确说明
"""
模态缺失处理:
- 不会传入None
- 缺失模态会传入shape相同的全零张量
- 可选: 通过mask参数标识哪些位置是缺失的

示例:
if 'modality_mask' in kwargs:
    # 使用mask处理
else:
    # 直接处理（全零位置自然不产生梯度）
"""

# InnerLoop验证增加缺失模拟测试
def validate_with_dropout(model):
    # 测试完整输入
    test_full(model)
    # 测试50%缺失输入
    test_with_mask(model, dropout=0.5)
```

### 5.3 输出维度强制统一

**修复前问题**: 基线用256，EAS用1024

**修复后** (已确认):
```python
# 所有融合模块强制输出1024维
class AnyFusionModule(nn.Module):
    def forward(self, vision, audio, text):
        # vision: [B, 576, 1024]
        # audio: [B, 400, 1024]
        # text: [B, 77, 1024]

        # 融合逻辑...
        fused = ...  # [B, 1024]

        return fused  # 必须是 [B, 1024]

# 统一分类头 (所有方法共享)
classifier = nn.Linear(1024, 10)  # MOSEI 10类分类
```

### 5.4 基线搜索预算统一

**修复前问题**: 只有EAS跑200轮搜索

**修复后** (已确认):
```python
# 所有NAS方法相同预算
SEARCH_CONFIG = {
    'eas': {'iterations': 200, 'proxy_samples': 256},
    'darts': {'iterations': 200, 'proxy_samples': 256},
    'llmatic': {'iterations': 200, 'proxy_samples': 256},
    'evoprompting': {'iterations': 200, 'proxy_samples': 256},
    'dynmm': {'iterations': 0},  # 固定架构，不搜索
    'tfn': {'iterations': 0},  # 固定架构，不搜索
    'admn': {'iterations': 0},  # 固定架构，不搜索
    'centaur': {'iterations': 0}  # 固定架构，不搜索
}
```

### 5.5 其他优化 (已确认)

| 优化项 | 修复前 | 修复后 | 说明 |
|--------|--------|--------|------|
| Proxy样本数 | 64 | **256** | 提高与全量数据相关性 |
| 奖励归一化 | 无 | **动态归一化** | acc/mrob除以历史最优，防止早期抛弃 |
| FLOPs计算 | thop | **torchprofile+thop** | 双重验证，更准确 |
| Early stop patience | 20 | **15** | 稍微严格，防止过拟合 |
| 搜索进度输出 | 无 | **每50轮输出top-5** | 便于中途干预 |

### 5.6 MOSEI任务确认

- **任务类型**: 10类分类 (sentiment回归值-3~+3分桶)
- **损失函数**: CrossEntropyLoss
- **评估指标**: Accuracy@10
- **备注**: 论文中说明"为便于统一比较，将连续sentiment分桶为10类"

### 5.7 资源与时间预算 (已确认)

- **硬件**: 4× RTX A5000 24GB = 96GB
- **搜索阶段**: 4方法×200轮 ≈ 1-2天 (可并行)
- **训练阶段**: 8方法×50epoch ≈ 2-3天
- **总预算**: 5-7天 (接受夜间/后台长期运行)

---

**修复状态**: ✅ 已完成确认，等待实施

**下一步行动**:
1. 创建统一投影层代码
2. 修复EAS prompt模板
3. 修复InnerLoop形状验证
4. 创建基线适配器 (统一输出1024维)
5. 运行小规模测试验证修复效果

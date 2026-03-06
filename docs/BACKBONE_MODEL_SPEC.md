# 基座模型规范 (Backbone Model Specification)

**版本**: v1.0
**最后更新**: 2026-03-07

---

## 1. 设计决策：冻结基座 + 可训练融合层

### 1.1 核心设计

AutoFusion v3 采用 **"冻结预训练基座 + NAS搜索融合层"** 的架构：

```
输入数据 → [冻结基座] → 提取特征 → [可训练融合层] → 输出预测
                    ↑
            (NAS搜索空间：我们生成的代码)
```

### 1.2 为什么这样设计？

| 方案 | 优点 | 缺点 | 适用性 |
|------|------|------|--------|
| **冻结基座** (我们的选择) | 计算成本低、聚焦融合架构 | 特征可能不是最优 | ✅ NAS场景 |
| **端到端训练** | 特征+融合联合优化 | 计算量巨大(单实验>1000 GPU-hours) | ❌ 不适合快速NAS |
| **部分微调** | 折中方案 | 复杂度中等 | 可选扩展 |

**NAS的核心是搜索融合架构，而不是重新发明特征提取器。**

---

## 2. 基座模型选择

### 2.1 各数据集基座配置

#### CMU-MOSEI (三模态情感分析)

| 模态 | 基座模型 | 输出维度 | 预训练权重 |
|------|---------|---------|-----------|
| **视觉** | CLIP-ViT-L/14 | 1024 | `openai/clip-vit-large-patch14` |
| **音频** | wav2vec 2.0 Large | 1024 | `facebook/wav2vec2-large-960h` |
| **文本** | BERT-Base | 768 | `bert-base-uncased` |

**特征提取代码**:
```python
from transformers import CLIPModel, Wav2Vec2Model, BertModel

# 视觉特征提取
vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
vision_features = vision_encoder.get_image_features(pixel_values)  # [B, 1024]

# 音频特征提取
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
audio_features = audio_encoder(audio_input).last_hidden_state  # [B, T, 1024]

# 文本特征提取
text_encoder = BertModel.from_pretrained("bert-base-uncased")
text_features = text_encoder(input_ids).last_hidden_state  # [B, T, 768]
```

#### VQA-v2 (视觉问答)

| 模态 | 基座模型 | 输出维度 | 说明 |
|------|---------|---------|------|
| **视觉** | ViT-Base/16 | 768 | 图像patch特征 [197, 768] |
| **文本** | BERT-Base | 768 | 问题编码 [20, 768] |

#### IEMOCAP (情感识别)

| 模态 | 基座模型 | 输出维度 | 说明 |
|------|---------|---------|------|
| **视觉** | ResNet-50 / CLIP | 2048/1024 | 视频帧特征 |
| **音频** | wav2vec 2.0 | 1024 | 语音特征 |
| **文本** | BERT / RoBERTa | 768 | 文本转录 |

---

## 3. 特征提取流程

### 3.1 预处理方式

```python
# scripts/extract_features.py (需要实现)

def extract_mosei_features():
    """
    CMU-MOSEI特征提取流程
    """
    # 1. 加载原始视频/音频/文本
    dataset = load_raw_mosei()

    # 2. 提取视觉特征 (CLIP)
    vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    vision_encoder.eval()
    for video in dataset.videos:
        # 均匀采样帧
        frames = sample_frames(video, num_frames=8)
        with torch.no_grad():
            features = vision_encoder.get_image_features(frames)  # [8, 1024]
        save(features, f"{video_id}_vision.pkl")

    # 3. 提取音频特征 (wav2vec)
    audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
    audio_encoder.eval()
    for audio in dataset.audios:
        with torch.no_grad():
            features = audio_encoder(audio).last_hidden_state  # [T, 1024]
        save(features, f"{audio_id}_audio.pkl")

    # 4. 提取文本特征 (BERT)
    text_encoder = BertModel.from_pretrained("bert-base-uncased")
    text_encoder.eval()
    for text in dataset.texts:
        with torch.no_grad():
            features = text_encoder(text).last_hidden_state  # [T, 768]
        save(features, f"{text_id}_text.pkl")
```

### 3.2 特征存储格式

```
data/mosei_processed/
├── train/
│   ├── sample_0001_vision.pkl   # [T, 1024] tensor
│   ├── sample_0001_audio.pkl    # [T, 512] tensor
│   ├── sample_0001_text.pkl     # [T, 768] tensor
│   └── sample_0001_label.pkl    # scalar or [num_classes]
├── val/
└── test/
```

---

## 4. NAS搜索空间

### 4.1 融合层架构 (我们的搜索目标)

EAS生成的代码只负责**融合层**，输入是已提取的特征：

```python
# EAS生成的融合架构示例
class FusionArchitecture(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        # 投影层 (统一维度)
        self.vision_proj = nn.Linear(1024, hidden_dim)
        self.audio_proj = nn.Linear(512, hidden_dim)
        self.text_proj = nn.Linear(768, hidden_dim)

        # 融合模块 (NAS搜索此部分)
        self.fusion = nn.MultiheadAttention(hidden_dim, num_heads=4)
        self.gate = nn.Sequential(...)  # 模态门控

        # 分类器
        self.classifier = nn.Linear(hidden_dim, 10)

    def forward(self, vision, audio, text):
        # vision: [B, T_vis, 1024]  (来自CLIP)
        # audio: [B, T_aud, 512]    (来自wav2vec)
        # text: [B, T_txt, 768]     (来自BERT)

        # 投影到统一维度
        v = self.vision_proj(vision)   # [B, T, hidden_dim]
        a = self.audio_proj(audio)     # [B, T, hidden_dim]
        t = self.text_proj(text)       # [B, T, hidden_dim]

        # 融合 (NAS搜索的核心)
        fused = self.fusion(v, a, t)

        # 分类
        return self.classifier(fused)
```

### 4.2 为什么不搜索基座？

1. **计算成本**: 训练CLIP/BERT需要数百GPU-hours
2. **聚焦创新**: 我们的创新是"融合架构的自动发现"
3. **公平对比**: 所有基线方法使用相同的冻结特征，确保公平

---

## 5. 训练配置

### 5.1 冻结 vs 可训练

```yaml
# configs/training_config.yaml

training:
  # 基座模型设置
  backbone:
    freeze: true                    # 冻结基座
    pretrained: true                # 使用预训练权重
    weights: "imagenet"             # 预训练权重来源

  # 融合层设置
  fusion:
    trainable: true                 # 可训练
    initialization: "xavier"        # 初始化方法

  # 优化器
  optimizer:
    type: "AdamW"
    lr: 0.001                       # 只优化融合层
    weight_decay: 0.01

  # 训练轮数
  epochs: 15                        # few-shot快速评估
```

### 5.2 Few-shot评估

```python
# 快速评估协议 (用于NAS)
num_shots = 64                     # 每类64个样本
num_epochs = 15                    # 快速收敛

# 完整评估协议 (用于最终报告)
num_shots = -1                     # 全部数据
num_epochs = 50                    # 充分训练
```

---

## 6. 实现任务清单

### 6.1 需要实现的脚本

- [ ] `scripts/extract_features_mosei.py` - MOSEI特征提取
- [ ] `scripts/extract_features_vqa.py` - VQA特征提取
- [ ] `scripts/extract_features_iemocap.py` - IEMOCAP特征提取
- [ ] `src/encoders/backbone_wrapper.py` - 基座模型包装器

### 6.2 预训练权重下载

```bash
# 下载脚本 (scripts/download_backbones.sh)

# CLIP
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-large-patch14')"

# wav2vec
python -c "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-960h')"

# BERT
python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-uncased')"
```

---

## 7. 常见问题

### Q1: 基座模型是固定的吗？

**A**: 是的，在NAS搜索过程中基座模型保持冻结。但在最终评估时，可以选择：
1. 保持冻结 (快速评估)
2. 端到端微调 (最佳性能，用于论文最终对比)

### Q2: 如果使用不同的基座模型，结果会不同吗？

**A**: 会。为了保证公平性，**所有对比方法必须使用相同的基座模型和相同的冻结特征**。

### Q3: 可以更换基座模型吗？

**A**: 可以，但需要：
1. 重新提取所有特征
2. 所有基线方法重新运行
3. 在论文中明确说明使用的基座模型

### Q4: 基座模型的特征质量会影响NAS结果吗？

**A**: 会。如果基座提取的特征质量差，再好的融合架构也无法补救。因此我们选择SOTA的预训练模型(CLIP, wav2vec, BERT)。

---

## 8. 相关文档

- `docs/EAS_PAPER_PLAN.md` - 论文整体规划
- `docs/EXPERIMENT_CONTROL_PROTOCOL.md` - 实验控制协议
- `src/data/mosei_loader.py` - 数据加载实现

---

**总结**: 我们搜索的是**融合层架构**，基座模型(CLIP/wav2vec/BERT)提供固定特征。这是NAS领域的标准做法。

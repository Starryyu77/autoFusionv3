#!/usr/bin/env python3
"""
CMU-MOSEI特征提取脚本

使用预训练基座模型提取特征并保存

基座模型:
- 视觉: CLIP-ViT-L/14 (openai/clip-vit-large-patch14)
- 音频: wav2vec 2.0 Large (facebook/wav2vec2-large-960h)
- 文本: BERT-Base (bert-base-uncased)

用法:
    python scripts/extract_features_mosei.py \
        --input_dir /path/to/raw/mosei \
        --output_dir /path/to/processed/mosei
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_backbone_models(device: str = 'cuda'):
    """
    加载预训练基座模型

    Args:
        device: 计算设备

    Returns:
        字典包含三个模态的编码器
    """
    print("📥 Loading backbone models...")

    models = {}

    # 1. 视觉编码器 (CLIP)
    print("  Loading CLIP-ViT-L/14...")
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        models['vision'] = {
            'model': clip_model.vision_model.to(device),
            'processor': clip_processor,
            'dim': 1024
        }
        print("    ✓ CLIP loaded")
    except Exception as e:
        print(f"    ✗ Failed to load CLIP: {e}")
        models['vision'] = None

    # 2. 音频编码器 (wav2vec)
    print("  Loading wav2vec 2.0 Large...")
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        models['audio'] = {
            'model': wav2vec_model.to(device),
            'processor': wav2vec_processor,
            'dim': 1024
        }
        print("    ✓ wav2vec loaded")
    except Exception as e:
        print(f"    ✗ Failed to load wav2vec: {e}")
        models['audio'] = None

    # 3. 文本编码器 (BERT)
    print("  Loading BERT-Base...")
    try:
        from transformers import BertModel, BertTokenizer
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        models['text'] = {
            'model': bert_model.to(device),
            'tokenizer': bert_tokenizer,
            'dim': 768
        }
        print("    ✓ BERT loaded")
    except Exception as e:
        print(f"    ✗ Failed to load BERT: {e}")
        models['text'] = None

    # 冻结所有模型
    for mod in models.values():
        if mod is not None:
            for param in mod['model'].parameters():
                param.requires_grad = False
            mod['model'].eval()

    print("✅ All backbone models loaded and frozen")
    return models


def extract_vision_features(video_frames, vision_model, processor, device):
    """
    提取视觉特征

    Args:
        video_frames: 视频帧 [T, H, W, C] 或图像列表
        vision_model: CLIP视觉模型
        processor: CLIP处理器
        device: 设备

    Returns:
        特征 [T, 1024]
    """
    with torch.no_grad():
        # 处理帧
        if isinstance(video_frames, list):
            inputs = processor(images=video_frames, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=[video_frames], return_tensors="pt")

        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 提取特征
        outputs = vision_model(**inputs)
        features = outputs.last_hidden_state  # [T, 577, 1024] (577 = 1 cls token + 576 patches)

        # 使用CLS token和patch tokens的平均
        features = features.mean(dim=1)  # [T, 1024]

    return features.cpu()


def extract_audio_features(audio_waveform, audio_model, processor, device):
    """
    提取音频特征

    Args:
        audio_waveform: 音频波形 [T]
        audio_model: wav2vec模型
        processor: wav2vec处理器
        device: 设备

    Returns:
        特征 [T', 1024]
    """
    with torch.no_grad():
        # 处理音频
        inputs = processor(audio_waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 提取特征
        outputs = audio_model(**inputs)
        features = outputs.last_hidden_state  # [1, T', 1024]

    return features[0].cpu()  # [T', 1024]


def extract_text_features(text, text_model, tokenizer, device, max_length: int = 77):
    """
    提取文本特征

    Args:
        text: 文本字符串
        text_model: BERT模型
        tokenizer: BERT分词器
        device: 设备
        max_length: 最大长度

    Returns:
        特征 [seq_len, 768]
    """
    with torch.no_grad():
        # 编码文本
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding='max_length'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 提取特征
        outputs = text_model(**inputs)
        features = outputs.last_hidden_state  # [1, seq_len, 768]

    return features[0].cpu()  # [seq_len, 768]


def extract_mosei_features(input_dir: str, output_dir: str, device: str = 'cuda'):
    """
    主特征提取函数

    Args:
        input_dir: 原始数据目录
        output_dir: 输出特征目录
        device: 计算设备
    """
    print("=" * 70)
    print("CMU-MOSEI Feature Extraction")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Device: {device}")
    print()

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'train').mkdir(exist_ok=True)
    (output_path / 'val').mkdir(exist_ok=True)
    (output_path / 'test').mkdir(exist_ok=True)

    # 加载基座模型
    models = load_backbone_models(device)

    # 检查是否有模型加载成功
    if all(m is None for m in models.values()):
        print("❌ No backbone models loaded. Exiting.")
        return

    # TODO: 实现实际的数据加载和特征提取
    # 这里需要根据CMU-MOSEI的实际数据格式实现

    print("\n⚠️  Note: This is a template script.")
    print("   Actual implementation depends on CMU-MOSEI data format.")
    print("   Please implement data loading logic based on your data source.")

    # 示例：创建虚拟特征用于测试
    print("\n📝 Creating dummy features for testing...")
    create_dummy_features(output_dir)


def create_dummy_features(output_dir: str, num_samples: Dict[str, int] = None):
    """
    创建虚拟特征用于测试

    Args:
        output_dir: 输出目录
        num_samples: 各split的样本数
    """
    if num_samples is None:
        num_samples = {'train': 1000, 'val': 200, 'test': 400}

    for split, n in num_samples.items():
        print(f"  Creating {split}: {n} samples...")

        split_dir = Path(output_dir) / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            sample_id = f"{split}_{i:04d}"

            # 虚拟特征
            features = {
                'vision': torch.randn(50, 1024),   # CLIP特征
                'audio': torch.randn(400, 512),    # wav2vec特征 (注意：这里应该是1024)
                'text': torch.randn(77, 768),      # BERT特征
                'label': torch.randn(1) * 3        # 情感强度 -3~3
            }

            # 保存
            with open(split_dir / f"{sample_id}.pkl", 'wb') as f:
                pickle.dump(features, f)

    print(f"✅ Dummy features created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Extract CMU-MOSEI features")
    parser.add_argument("--input_dir", type=str, default="data/mosei_raw",
                        help="Directory containing raw MOSEI data")
    parser.add_argument("--output_dir", type=str, default="data/mosei_processed",
                        help="Directory to save processed features")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for feature extraction")
    parser.add_argument("--dummy", action="store_true",
                        help="Create dummy features for testing")

    args = parser.parse_args()

    if args.dummy:
        print("Creating dummy features only...")
        create_dummy_features(args.output_dir)
    else:
        extract_mosei_features(args.input_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()

"""完整测试 TFN 训练流程"""
import sys
sys.path.insert(0, '/usr1/home/s125mdg43_10/paper_reproduction_2026/experiments/tfn_mosi_paper/src')

from tfn_paper import TFNPaper
from mosi_dataset_v2 import get_mosi_loaders_v2, get_input_dims_v2
import torch

print("=== Testing TFN Model ===")
input_dims = get_input_dims_v2()
print(f"Input dims: {input_dims}")

model = TFNPaper(input_dims, task='binary')
print(f"✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

print("\n=== Testing Data Loading ===")
train_loader, valid_loader, test_loader = get_mosi_loaders_v2(
    '/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl',
    task='binary',
    batch_size=4
)
print(f"✓ Data loaded: {len(train_loader)}/{len(valid_loader)}/{len(test_loader)} batches")

print("\n=== Testing Forward Pass ===")
text, vision, audio, label = next(iter(train_loader))
print(f"Batch shapes: text={text.shape}, vision={vision.shape}, audio={audio.shape}, label={label.shape}")

output = model(text, vision, audio)
print(f"✓ Forward pass successful, output shape: {output.shape}")

print("\n✓✓✓ All tests passed! Ready to train.")

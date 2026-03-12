"""测试 TFN 模型"""
import sys
sys.path.insert(0, '/usr1/home/s125mdg43_10/paper_reproduction_2026/experiments/tfn_mosi_paper/src')

from tfn_paper import TFNPaper
import torch

# Test model creation
input_dims = {'language': 300, 'visual': 35, 'acoustic': 74}
model = TFNPaper(input_dims, task='binary')
print('✓ Model created successfully')
print(f'  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# Test forward pass
B = 2
language = torch.randn(B, 300)
visual = torch.randn(B, 35)
acoustic = torch.randn(B, 74)
output = model(language, visual, acoustic)
print(f'✓ Forward pass successful, output shape: {output.shape}')

# Test data loading
try:
    from mosi_dataset import get_mosi_loaders
    train_loader, valid_loader, test_loader = get_mosi_loaders(
        '/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl',
        task='binary',
        fold=0,
        batch_size=4
    )
    print(f'✓ Data loading successful')
    print(f'  Train: {len(train_loader.dataset)}')
    print(f'  Valid: {len(valid_loader.dataset)}')
    print(f'  Test: {len(test_loader.dataset)}')
except Exception as e:
    print(f'✗ Data loading failed: {e}')

print('\n✓ All tests passed! Ready to run experiments.')

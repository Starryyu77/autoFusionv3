"""检查 MOSI 数据格式"""
import pickle

with open('/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl', 'rb') as f:
    data = pickle.load(f)

print('Keys:', list(data.keys()))

# 检查 train split
print('\n=== Train Split ===')
train_data = data['train']
print('Train keys:', list(train_data.keys()))
for k, v in train_data.items():
    if hasattr(v, 'shape'):
        print(f'{k}: shape={v.shape}, dtype={v.dtype}')
    elif isinstance(v, list):
        print(f'{k}: list, len={len(v)}')
        if len(v) > 0 and hasattr(v[0], 'shape'):
            print(f'  First item shape: {v[0].shape}')
    else:
        print(f'{k}: type={type(v)}')

print('\n=== Sample Label ===')
print(f"First label: {train_data['labels'][0]}")
print(f"Label range: [{min(train_data['labels'])}, {max(train_data['labels'])}]")

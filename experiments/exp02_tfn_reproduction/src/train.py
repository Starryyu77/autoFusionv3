"""
TFN 训练脚本 - 论文原始设置
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
import os
from pathlib import Path
from typing import Dict
from tqdm import tqdm

from tfn_stable import TFNStable
from mosi_dataset_v2 import get_mosi_loaders_v2, get_input_dims_v2


def get_loss_function(task: str):
    """获取损失函数"""
    if task == 'binary':
        # BCEWithLogitsLoss 更稳定，不需要模型输出 sigmoid
        return nn.BCEWithLogitsLoss()
    elif task == '5class':
        return nn.CrossEntropyLoss()
    else:  # regression
        return nn.MSELoss()


def evaluate(model, loader, task, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for language, visual, acoustic, labels in loader:
            language = language.to(device)
            visual = visual.to(device)
            acoustic = acoustic.to(device)
            labels = labels.to(device)

            outputs = model(language, visual, acoustic)

            if task == 'binary':
                # 应用 sigmoid 得到 [0,1]，然后映射到 [-3, 3]
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = probs * 6 - 3
                labels_np = labels.cpu().numpy()
            elif task == '5class':
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                labels_np = labels.cpu().numpy()
            else:  # regression
                # 输出 [-1,1]，映射到 [-3, 3]
                preds = outputs.cpu().numpy() * 3
                labels_np = labels.cpu().numpy() * 3

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if task == 'binary':
        # Binary accuracy (threshold at 0 for preds, 0.5 for labels)
        # preds are in [-3, 3], labels are in [0, 1]
        pred_binary = (all_preds >= 0).astype(int)
        label_binary = (all_labels >= 0.5).astype(int)  # Fix: 0.5 threshold for [0, 1] labels
        accuracy = (pred_binary == label_binary).mean()
        mae = np.abs(all_preds - all_labels).mean()
        return {'accuracy': accuracy * 100, 'mae': mae}

    elif task == '5class':
        accuracy = (all_preds == all_labels).mean()
        return {'accuracy': accuracy * 100}

    else:  # regression
        mae = np.abs(all_preds - all_labels).mean()
        return {'mae': mae}


def train_epoch(model, loader, optimizer, criterion, task, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for language, visual, acoustic, labels in loader:
        language = language.to(device)
        visual = visual.to(device)
        acoustic = acoustic.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(language, visual, acoustic)

        if task == 'binary':
            loss = criterion(outputs, labels)
        elif task == '5class':
            loss = criterion(outputs, labels)
        else:  # regression
            loss = criterion(outputs, labels)

        loss.backward()

        # Gradient clipping (论文使用)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train(args):
    """主训练函数"""
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 获取输入维度
    input_dims = get_input_dims_v2()
    print(f"Input dims: {input_dims}")

    # 5-fold 交叉验证结果
    all_results = []

    # TFN 论文使用预定义的数据划分，不进行 5-fold
    print(f"\n{'='*50}")
    print(f"Training TFN with paper settings")
    print(f"{'='*50}")

    # 数据加载器
    train_loader, valid_loader, test_loader = get_mosi_loaders_v2(
        args.data_path,
        task=args.task,
        batch_size=args.batch_size
    )

    print(f"Train: {len(train_loader.dataset)}, Valid: {len(valid_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # 模型 (使用稳定版本)
    model = TFNStable(
        input_dims=input_dims,
        embed_dim=args.embed_dim,
        fusion_dim=args.fusion_dim,
        hidden_dim=args.hidden_dim,
        task=args.task
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 优化器 (论文使用 Adam)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 损失函数
    criterion = get_loss_function(args.task)

    # 训练
    best_valid_metric = 0 if args.task != 'regression' else float('inf')
    best_test_result = None
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.task, device)

        # 验证
        valid_result = evaluate(model, valid_loader, args.task, device)

        # 选择最佳模型
        if args.task == 'binary':
            current_metric = valid_result['accuracy']
            is_better = current_metric > best_valid_metric
        elif args.task == '5class':
            current_metric = valid_result['accuracy']
            is_better = current_metric > best_valid_metric
        else:  # regression
            current_metric = valid_result['mae']
            is_better = current_metric < best_valid_metric

        if is_better:
            best_valid_metric = current_metric
            best_test_result = evaluate(model, test_loader, args.task, device)
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'tfn_{args.task}_best.pt'))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Valid: {valid_result}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"\nFinal Test Result: {best_test_result}")

    # 汇总结果
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")

    if args.task == 'binary':
        print(f"Binary Accuracy: {best_test_result['accuracy']:.2f}%")
        print(f"MAE: {best_test_result['mae']:.4f}")
    elif args.task == '5class':
        print(f"5-class Accuracy: {best_test_result['accuracy']:.2f}%")
    else:
        print(f"MAE: {best_test_result['mae']:.4f}")

    # 保存结果
    results = {
        'task': args.task,
        'test_result': best_test_result,
        'config': vars(args)
    }

    if args.task == 'binary':
        results['accuracy'] = float(best_test_result['accuracy'])
        results['mae'] = float(best_test_result['mae'])
    elif args.task == '5class':
        results['accuracy'] = float(best_test_result['accuracy'])
    else:
        results['mae'] = float(best_test_result['mae'])

    output_path = Path(args.output_dir) / f'tfn_{args.task}_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TFN on CMU-MOSI')

    # 数据
    parser.add_argument('--data_path', type=str,
                        default='/usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl',
                        help='Path to MOSI data')

    # 任务
    parser.add_argument('--task', type=str, default='binary',
                        choices=['binary', '5class', 'regression'],
                        help='Task type')

    # 模型参数
    parser.add_argument('--embed_dim', type=int, default=32,
                        help='Modality embedding dimension (paper default: 32)')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension (paper default: 128)')
    parser.add_argument('--fusion_dim', type=int, default=128,
                        help='Fusion layer dimension (default: 128)')

    # 训练参数 (论文设置)
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (paper: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (paper: 0.01)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')

    # 其他
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU id')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')

    args = parser.parse_args()
    train(args)

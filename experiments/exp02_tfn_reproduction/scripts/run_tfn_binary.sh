#!/bin/bash
# TFN Binary Classification - 论文原始设置

cd /usr1/home/s125mdg43_10/paper_reproduction_2026

export PYTHONPATH=/usr1/home/s125mdg43_10/paper_reproduction_2026:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo "TFN Binary Classification (E1)"
echo "========================================"
echo "Task: Binary (positive vs negative)"
echo "Expected: 77.1% accuracy, 0.87 MAE"
echo "========================================"

python3 experiments/tfn_mosi_paper/src/train.py \
    --task binary \
    --data_path /usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl \
    --embed_dim 32 \
    --hidden_dim 128 \
    --fusion_dim 128 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0001 \
    --weight_decay 0.01 \
    --patience 20 \
    --gpu 0 \
    --output_dir experiments/tfn_mosi_paper/results

echo "========================================"
echo "Binary classification completed!"
echo "========================================"

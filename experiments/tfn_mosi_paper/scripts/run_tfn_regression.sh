#!/bin/bash
# TFN Regression - 论文原始设置

cd /usr1/home/s125mdg43_10/paper_reproduction_2026

export PYTHONPATH=/usr1/home/s125mdg43_10/paper_reproduction_2026:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

echo "========================================"
echo "TFN Regression (E1)"
echo "========================================"
echo "Task: Sentiment regression [-3, 3]"
echo "Expected: 0.87 MAE"
echo "========================================"

python3 experiments/tfn_mosi_paper/src/train.py \
    --task regression \
    --data_path /usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl \
    --embed_dim 32 \
    --hidden_dim 128 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0005 \
    --weight_decay 0.01 \
    --patience 20 \
    --gpu 0 \
    --output_dir experiments/tfn_mosi_paper/results

echo "========================================"
echo "Regression completed!"
echo "========================================"

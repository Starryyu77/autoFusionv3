#!/bin/bash
# EAS Regression - 与 TFN 对比

cd /usr1/home/s125mdg43_10/paper_reproduction_2026

export PYTHONPATH=/usr1/home/s125mdg43_10/paper_reproduction_2026:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=3

echo "========================================"
echo "EAS Regression"
echo "========================================"
echo "Comparison with TFN: 0.824 MAE"
echo "========================================"

python3 experiments/eas_mosi_paper/src/train.py \
    --task regression \
    --data_path /usr1/home/s125mdg43_10/AutoFusion_v3/data/mosei/mosei_senti_data.pkl \
    --embed_dim 64 \
    --hidden_dim 128 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.0005 \
    --weight_decay 0.01 \
    --patience 20 \
    --gpu 0 \
    --output_dir experiments/eas_mosi_paper/results

echo "========================================"
echo "EAS Regression completed!"
echo "========================================"

#!/bin/bash
# EAS Binary Classification - 与 TFN 对比

cd /usr1/home/s125mdg43_10/paper_reproduction_2026

export PYTHONPATH=/usr1/home/s125mdg43_10/paper_reproduction_2026:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

echo "========================================"
echo "EAS Binary Classification"
echo "========================================"
echo "Comparison with TFN: 71.03%"
echo "========================================"

python3 experiments/eas_mosi_paper/src/train.py \
    --task binary \
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
echo "EAS Binary completed!"
echo "========================================"

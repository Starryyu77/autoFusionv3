#!/bin/bash
# VQA数据集并行实验

cd /usr1/home/s125mdg43_10/AutoFusion_v3
mkdir -p logs results

echo "========================================================================"
echo "  VQA数据集 - 8个并行基线实验"
echo "========================================================================"

# 清理VQA相关日志
rm -f logs/*_vqa*.log

# GPU 0: Mean, Concat
CUDA_VISIBLE_DEVICES=0 nohup python3 -u experiments/run_baseline_specific.py \
    --method mean --dataset vqa --device cuda \
    --output_dir results/baselines_mean > logs/mean_vqa_gpu0.log 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=0 nohup python3 -u experiments/run_baseline_specific.py \
    --method concat --dataset vqa --device cuda \
    --output_dir results/baselines_concat > logs/concat_vqa_gpu0.log 2>&1 &
sleep 1

# GPU 1: Attention, Max
CUDA_VISIBLE_DEVICES=1 nohup python3 -u experiments/run_baseline_specific.py \
    --method attention --dataset vqa --device cuda \
    --output_dir results/baselines_attention > logs/attention_vqa_gpu1.log 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=1 nohup python3 -u experiments/run_baseline_specific.py \
    --method max --dataset vqa --device cuda \
    --output_dir results/baselines_max > logs/max_vqa_gpu1.log 2>&1 &
sleep 1

# GPU 2: DynMM, TFN
CUDA_VISIBLE_DEVICES=2 nohup python3 -u experiments/run_baseline_specific.py \
    --method dynmm --dataset vqa --device cuda \
    --output_dir results/baselines_dynmm > logs/dynmm_vqa_gpu2.log 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=2 nohup python3 -u experiments/run_baseline_specific.py \
    --method tfn --dataset vqa --device cuda \
    --output_dir results/baselines_tfn > logs/tfn_vqa_gpu2.log 2>&1 &
sleep 1

# GPU 3: ADMN, Centaur
CUDA_VISIBLE_DEVICES=3 nohup python3 -u experiments/run_baseline_specific.py \
    --method admn --dataset vqa --device cuda \
    --output_dir results/baselines_admn > logs/admn_vqa_gpu3.log 2>&1 &
sleep 1

CUDA_VISIBLE_DEVICES=3 nohup python3 -u experiments/run_baseline_specific.py \
    --method centaur --dataset vqa --device cuda \
    --output_dir results/baselines_centaur > logs/centaur_vqa_gpu3.log 2>&1 &

echo ""
echo "✅ VQA实验已启动（8个并行）"
echo ""
sleep 2
ps aux | grep run_baseline_specific | grep vqa | grep -v grep | wc -l | xargs echo "运行中的VQA实验数:"

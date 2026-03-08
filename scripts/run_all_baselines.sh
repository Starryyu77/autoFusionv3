#!/bin/bash
# 并行运行所有基线方法
# 在4个GPU上并行运行不同的基线方法

set -e

echo "=========================================="
echo "Starting Baseline Experiments (Parallel)"
echo "=========================================="

# 配置
PROJECT_DIR="/usr1/home/s125mdg43_10/AutoFusion_v3"
RESULTS_DIR="$PROJECT_DIR/results/baselines"
mkdir -p $RESULTS_DIR

# 数据集配置
declare -A DATASETS=(
    ["mosei"]="$PROJECT_DIR/data/mosei_processed/mosei_data.pkl"
    ["iemocap"]="$PROJECT_DIR/data/iemocap_processed/iemocap_data.pkl"
    ["vqa"]="$PROJECT_DIR/data/vqa_processed/vqa_data.pkl"
)

declare -A NUM_CLASSES=(
    ["mosei"]="10"
    ["iemocap"]="10"
    ["vqa"]="10"
)

# 基线方法列表
METHODS=("dynmm" "admn" "centaur" "tfn" "fdsnet")

# GPU分配 - 4个GPU，每个运行一个方法
GPUS=(0 1 2 3)

# 启动函数
run_baseline() {
    local method=$1
    local dataset=$2
    local gpu=$3

    echo "Starting $method on $dataset (GPU $gpu)..."

    cd $PROJECT_DIR
    export PYTHONPATH=$PROJECT_DIR
    export CUDA_VISIBLE_DEVICES=$gpu

    python3 experiments/run_baseline.py \
        --method $method \
        --dataset $dataset \
        --data_path ${DATASETS[$dataset]} \
        --num_classes ${NUM_CLASSES[$dataset]} \
        --device cuda \
        --output_dir results/baselines \
        --epochs 50 \
        > logs/baseline_${method}_${dataset}.log 2>&1 &

    echo "  -> PID: $!, GPU: $gpu, Log: logs/baseline_${method}_${dataset}.log"
}

# 在4个GPU上并行启动不同的基线方法
echo ""
echo "Launching experiments on 4 GPUs..."
echo ""

# GPU 0: DynMM on MOSEI
run_baseline "dynmm" "mosei" 0

# GPU 1: ADMN on MOSEI
run_baseline "admn" "mosei" 1

# GPU 2: Centaur on MOSEI
run_baseline "centaur" "mosei" 2

# GPU 3: TFN on MOSEI
run_baseline "tfn" "mosei" 3

echo ""
echo "First batch launched. Waiting 5 seconds before next batch..."
sleep 5

# 启动第二批 - 剩余的基线方法和数据集
echo ""
echo "Launching second batch..."

# 再次使用GPU 0-3运行其他数据集或方法
# GPU 0: FDSNet on MOSEI
run_baseline "fdsnet" "mosei" 0

echo ""
echo "=========================================="
echo "All baseline experiments launched!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/baseline_*.log"
echo "  nvidia-smi"
echo "  ps aux | grep run_baseline"
echo ""
echo "Results will be saved to: $RESULTS_DIR"

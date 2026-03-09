#!/bin/bash
# =============================================================================
# AutoFusion v3 - 基线实验启动脚本
# 在NTU-GPU43服务器上运行
# =============================================================================

set -e  # 遇到错误立即退出

echo "========================================================================"
echo "  AutoFusion v3 - Baseline Experiments"
echo "========================================================================"
echo ""

# 配置
PROJECT_DIR="/projects/AutoFusion_v3"
CONDA_ENV="autofusion"
RESULTS_DIR="${PROJECT_DIR}/results"

# 检查是否在服务器上
if [[ ! -d "/projects" ]]; then
    echo "❌ 错误: 这个脚本需要在NTU-GPU43服务器上运行!"
    echo "   请先SSH到服务器: ssh tianyu016@10.97.216.128"
    exit 1
fi

echo "✅ 检测到服务器环境"
echo ""

# =============================================================================
# 步骤1: 激活环境
# =============================================================================
echo "步骤1: 激活conda环境..."
source ~/.bashrc
conda activate ${CONDA_ENV} 2>/dev/null || {
    echo "⚠️  环境 ${CONDA_ENV} 不存在，尝试创建..."
    conda create -n ${CONDA_ENV} python=3.10 -y
    conda activate ${CONDA_ENV}
}
echo "✅ 环境激活成功: $(which python)"
echo ""

# =============================================================================
# 步骤2: 检查GPU状态
# =============================================================================
echo "步骤2: 检查GPU状态..."
echo "------------------------------------------------------------------------"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
    --format=csv,noheader | while IFS=, read -r idx name total used free util temp; do
    echo "GPU ${idx}:"
    echo "  型号: ${name}"
    echo "  显存: 已用${used}/总共${total} (空闲${free})"
    echo "  利用率: ${util}"
    echo "  温度: ${temp}"
done
echo "------------------------------------------------------------------------"
echo ""

# =============================================================================
# 步骤3: 选择GPU
# =============================================================================
echo "步骤3: 选择GPU"
echo "请输入要使用的GPU ID (0-3, 推荐选择空闲显存最大的):"
read -p "> " GPU_ID

if ! [[ "$GPU_ID" =~ ^[0-3]$ ]]; then
    echo "❌ 错误: 无效的GPU ID: $GPU_ID"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=$GPU_ID
echo "✅ 使用 GPU $GPU_ID"
echo ""

# =============================================================================
# 步骤4: 选择实验
# =============================================================================
echo "步骤4: 选择实验"
echo ""
echo "请选择要运行的实验:"
echo ""
echo "  [1] 简单基线 (Mean/Concat/Attention/Max)"
echo "  [2] 固定架构基线 (DynMM/TFN/ADMN/Centaur/FDSNet)"
echo "  [3] NAS基线 (DARTS/LLMatic/EvoPrompting) - 需要API Key"
echo "  [4] 所有基线 (全部12个方法)"
echo "  [5] 单个基线"
echo ""
read -p "> " EXP_CHOICE

case $EXP_CHOICE in
    1)
        METHODS=("mean" "concat" "attention" "max")
        echo "✅ 选择: 简单基线"
        ;;
    2)
        METHODS=("dynmm" "tfn" "admn" "centaur" "fdsnet")
        echo "✅ 选择: 固定架构基线"
        ;;
    3)
        METHODS=("darts" "llmatic" "evoprompting")
        echo "✅ 选择: NAS基线"
        echo ""
        if [[ -z "$ALIYUN_API_KEY" ]]; then
            echo "⚠️  需要设置 ALIYUN_API_KEY"
            read -p "请输入API Key: " API_KEY
            export ALIYUN_API_KEY=$API_KEY
        fi
        ;;
    4)
        METHODS=("mean" "concat" "attention" "max" "dynmm" "tfn" "admn" "centaur" "fdsnet" "darts" "llmatic" "evoprompting")
        echo "✅ 选择: 所有基线"
        ;;
    5)
        echo ""
        echo "可用基线方法:"
        echo "  mean, concat, attention, max"
        echo "  dynmm, tfn, admn, centaur, fdsnet"
        echo "  darts, llmatic, evoprompting"
        echo ""
        read -p "请输入方法名: " SINGLE_METHOD
        METHODS=($SINGLE_METHOD)
        ;;
    *)
        echo "❌ 错误: 无效的选择"
        exit 1
        ;;
esac
echo ""

# =============================================================================
# 步骤5: 选择数据集
# =============================================================================
echo "步骤5: 选择数据集"
echo ""
echo "  [1] MOSEI (推荐，数据最完整)"
echo "  [2] IEMOCAP"
echo "  [3] VQA"
echo "  [4] 所有数据集"
echo ""
read -p "> " DATA_CHOICE

case $DATA_CHOICE in
    1)
        DATASETS=("mosei")
        ;;
    2)
        DATASETS=("iemocap")
        ;;
    3)
        DATASETS=("vqa")
        ;;
    4)
        DATASETS=("mosei" "iemocap" "vqa")
        ;;
    *)
        echo "❌ 错误: 无效的选择"
        exit 1
        ;;
esac
echo ""

# =============================================================================
# 步骤6: 确认并开始
# =============================================================================
echo "========================================================================"
echo "  实验配置确认"
echo "========================================================================"
echo "GPU:        $GPU_ID"
echo "方法:       ${METHODS[@]}"
echo "数据集:     ${DATASETS[@]}"
echo "实验总数:   $((${#METHODS[@]} * ${#DATASETS[@]}))"
echo "========================================================================"
echo ""
read -p "确认开始? (y/n) " CONFIRM

if [[ $CONFIRM != "y" && $CONFIRM != "Y" ]]; then
    echo "已取消"
    exit 0
fi

echo ""
echo "🚀 开始实验..."
echo ""

# =============================================================================
# 步骤7: 运行实验
# =============================================================================
cd ${PROJECT_DIR}

TOTAL=$((${#METHODS[@]} * ${#DATASETS[@]}))
COUNT=0

for dataset in "${DATASETS[@]}"; do
    for method in "${METHODS[@]}"; do
        COUNT=$((COUNT + 1))
        echo ""
        echo "========================================================================"
        echo "  [$COUNT/$TOTAL] Running: $method on $dataset"
        echo "========================================================================"
        echo ""

        python experiments/run_baseline_on_server.py \
            --method $method \
            --dataset $dataset \
            --gpu $GPU_ID

        if [ $? -eq 0 ]; then
            echo "✅ 成功: $method on $dataset"
        else
            echo "❌ 失败: $method on $dataset"
            read -p "继续下一个? (y/n) " CONTINUE
            if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
                echo "已停止"
                exit 1
            fi
        fi
    done
done

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "========================================================================"
echo "  所有实验完成!"
echo "========================================================================"
echo ""
echo "结果保存在: ${RESULTS_DIR}/"
echo ""
echo "查看结果:"
ls -lh ${RESULTS_DIR}/baselines_*/*.json 2>/dev/null | head -20
echo ""
echo "生成汇总:"
python -c "
import json, glob
results = {}
for f in glob.glob('${RESULTS_DIR}/baselines_*/*.json'):
    with open(f) as fp:
        data = json.load(fp)
        m, d = data.get('method'), data.get('dataset')
        if m and d:
            results.setdefault(m, {})[d] = data.get('dropout_0_mean', 0)
for m in sorted(results):
    print(f'{m:15}', end='')
    for d in ['mosei', 'iemocap', 'vqa']:
        acc = results[m].get(d, 0)
        print(f'  {d:8}: {acc:.4f}', end='')
    print()
" 2>/dev/null || echo "结果文件不存在"
echo ""

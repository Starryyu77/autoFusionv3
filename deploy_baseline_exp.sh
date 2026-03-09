#!/bin/bash
# 部署基线实验代码到NTU-GPU43服务器

SERVER="tianyu016@10.97.216.128"
REMOTE_DIR="/projects/AutoFusion_v3"

echo "🚀 Deploying baseline experiment code to NTU-GPU43..."

# 创建远程目录
ssh ${SERVER} "mkdir -p ${REMOTE_DIR}"

# 同步代码 (排除大数据文件)
rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='results/' --exclude='logs/' --exclude='checkpoints/' \
    --exclude='data/*.hdf5' --exclude='data/*.pkl' \
    . ${SERVER}:${REMOTE_DIR}/

echo "✅ Deployment complete!"
echo ""
echo "To run experiments on server:"
echo "  ssh ${SERVER}"
echo "  cd ${REMOTE_DIR}"
echo "  python experiments/run_baseline_specific.py --method mean --dataset mosei"

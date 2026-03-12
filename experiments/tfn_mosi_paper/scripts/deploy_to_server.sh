#!/bin/bash
# 部署 TFN 论文复现代码到服务器

set -e

echo "============================================"
echo "Deploying TFN Paper Reproduction to Server"
echo "============================================"

# 服务器信息
SERVER="ntu-gpu43"
REMOTE_DIR="/usr1/home/s125mdg43_10/paper_reproduction_2026"

# 本地路径
LOCAL_DIR="/Users/starryyu/2026/Auto-Fusion-Advanced/autofusionv3/experiments/tfn_mosi_paper"

echo ""
echo "[1/3] Creating remote directory..."
ssh $SERVER "mkdir -p $REMOTE_DIR/experiments/tfn_mosi_paper/{src,scripts,results,logs}"

echo ""
echo "[2/3] Copying source files..."
scp $LOCAL_DIR/src/tfn_paper.py $SERVER:$REMOTE_DIR/experiments/tfn_mosi_paper/src/
scp $LOCAL_DIR/src/mosi_dataset.py $SERVER:$REMOTE_DIR/experiments/tfn_mosi_paper/src/
scp $LOCAL_DIR/src/train.py $SERVER:$REMOTE_DIR/experiments/tfn_mosi_paper/src/

echo ""
echo "[3/3] Copying scripts..."
scp $LOCAL_DIR/scripts/*.sh $SERVER:$REMOTE_DIR/experiments/tfn_mosi_paper/scripts/

echo ""
echo "============================================"
echo "Deployment completed!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. SSH to server: ssh $SERVER"
echo "  2. Run experiments:"
echo "     bash $REMOTE_DIR/experiments/tfn_mosi_paper/scripts/run_all.sh"
echo ""
echo "Or run directly:"
echo "  ssh $SERVER 'bash $REMOTE_DIR/experiments/tfn_mosi_paper/scripts/run_tfn_binary.sh'"
echo "============================================"

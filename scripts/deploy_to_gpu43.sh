#!/bin/bash
# 部署脚本: 将代码部署到NTU GPU43服务器

set -e

echo "🚀 Deploying AutoFusion v3 to GPU43..."

# 配置
SERVER="s125mdg43_10@gpu43.dynip.ntu.edu.sg"
REMOTE_DIR="/usr1/home/s125mdg43_10/AutoFusion_v3"
LOCAL_DIR="/Users/starryyu/2026/Auto-Fusion-Advanced/autofusionv3"

echo "📡 Server: $SERVER"
echo "📁 Remote path: $REMOTE_DIR"

# 1. 创建远程目录
echo "📂 Creating remote directory..."
ssh $SERVER "mkdir -p $REMOTE_DIR"

# 2. 同步代码 (排除不需要的文件)
echo "📤 Syncing code..."
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='checkpoints/' \
    --exclude='data/' \
    $LOCAL_DIR/ \
    $SERVER:$REMOTE_DIR/

# 3. 在服务器上设置环境
echo "🔧 Setting up environment..."
ssh $SERVER << EOF
    cd $REMOTE_DIR

    # 创建conda环境 (如果不存在)
    if ! conda env list | grep -q "eas"; then
        echo "Creating conda environment 'eas'..."
        conda create -n eas python=3.10 -y
    fi

    # 激活环境并安装依赖
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate eas

    echo "Installing dependencies..."
    pip install -r requirements.txt -q

    # 创建必要的目录
    mkdir -p results logs checkpoints data

    echo "✅ Environment setup complete!"
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "  1. SSH to server: ssh $SERVER"
echo "  2. Activate environment: conda activate eas"
echo "  3. Set API key: export ALIYUN_API_KEY='your-key'"
echo "  4. Download data: python scripts/download_data.py"
echo "  5. Run experiments: python experiments/run_round1.py"

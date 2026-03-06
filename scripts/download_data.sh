#!/bin/bash
# 数据集下载脚本

set -e

DATA_DIR="/usr1/home/s125mdg43_10/data"

echo "📥 Downloading datasets for EAS experiments..."
echo "📁 Data directory: $DATA_DIR"
mkdir -p $DATA_DIR

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate eas

# 1. CMU-MOSEI
echo ""
echo "📊 Downloading CMU-MOSEI..."
echo "   Source: http://multicomp.cs.cmu.edu/resources/cmu-mosei-dataset/"
echo "   Note: Using MultiBench preprocessed version"

if [ ! -f "$DATA_DIR/mosei.pkl" ]; then
    python -c "
import sys
sys.path.insert(0, 'src')
from data.mosei_loader import download_mosei
download_mosei('$DATA_DIR')
"
    echo "   ✅ CMU-MOSEI downloaded"
else
    echo "   ✓ CMU-MOSEI already exists"
fi

# 2. VQA-v2 (使用HuggingFace简化版)
echo ""
echo "📊 Downloading VQA-v2..."
echo "   Source: https://visualqa.org/download.html"

if [ ! -d "$DATA_DIR/vqa_v2" ]; then
    python -c "
from datasets import load_dataset
ds = load_dataset('HuggingFaceM4/VQAv2', split='train[:10%]')  # 先下10%测试
ds.save_to_disk('$DATA_DIR/vqa_v2')
"
    echo "   ✅ VQA-v2 downloaded (10% sample)"
else
    echo "   ✓ VQA-v2 already exists"
fi

# 3. IEMOCAP (需要手动申请，提供预处理脚本)
echo ""
echo "📊 IEMOCAP..."
echo "   ⚠️  Requires manual application at https://sail.usc.edu/iemocap/"
echo "   Preprocessing script: src/data/iemocap_loader.py"

if [ ! -f "$DATA_DIR/iemocap_processed.pkl" ]; then
    echo "   ⏭️  Skipping (requires manual download)"
else
    echo "   ✓ IEMOCAP already exists"
fi

echo ""
echo "📊 Dataset status:"
ls -lh $DATA_DIR/

echo ""
echo "✅ Data download complete!"
echo ""
echo "Next: Run feature extraction"
echo "  python scripts/extract_features.py --dataset mosei"

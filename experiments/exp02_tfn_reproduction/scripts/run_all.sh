#!/bin/bash
# 运行所有 TFN 论文复现实验

cd /usr1/home/s125mdg43_10/paper_reproduction_2026

echo "============================================"
echo "TFN Paper Reproduction - All Experiments"
echo "============================================"

# E1: 多模态对比实验
echo ""
echo "[1/3] Running Binary Classification..."
bash experiments/tfn_mosi_paper/scripts/run_tfn_binary.sh > logs/tfn_binary.log 2>&1 &
PID1=$!

echo ""
echo "[2/3] Running 5-class Classification..."
bash experiments/tfn_mosi_paper/scripts/run_tfn_5class.sh > logs/tfn_5class.log 2>&1 &
PID2=$!

echo ""
echo "[3/3] Running Regression..."
bash experiments/tfn_mosi_paper/scripts/run_tfn_regression.sh > logs/tfn_regression.log 2>&1 &
PID3=$!

echo ""
echo "============================================"
echo "All experiments started in parallel!"
echo "PIDs: $PID1 $PID2 $PID3"
echo "============================================"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/tfn_binary.log"
echo "  tail -f logs/tfn_5class.log"
echo "  tail -f logs/tfn_regression.log"
echo ""
echo "Expected duration: ~2-3 hours per experiment"
echo "============================================"

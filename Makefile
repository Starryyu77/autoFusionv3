# AutoFusion v3 - Makefile
# 简化实验执行流程

.PHONY: help deploy setup run-round1 run-round2 run-round3 run-round4 test clean

# 配置
SERVER := s125mdg43_10@gpu43.dynip.ntu.edu.sg
REMOTE_DIR := /usr1/home/s125mdg43_10/AutoFusion_v3
API_KEY := $(ALIYUN_API_KEY)

help:
	@echo "AutoFusion v3 - EAS Experiment Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make deploy      - Deploy code to GPU43"
	@echo "  make setup       - Setup Python environment"
	@echo "  make test        - Run unit tests"
	@echo "  make run-round1  - Run Round 1: Inner Loop Validation"
	@echo "  make run-round2  - Run Round 2: Main Experiments"
	@echo "  make run-round3  - Run Round 3: Analysis"
	@echo "  make run-round4  - Run Round 4: Deployment & Figures"
	@echo "  make download    - Download datasets"
	@echo "  make sync-up     - Sync local code to server"
	@echo "  make sync-down   - Sync results from server"
	@echo "  make status      - Check GPU43 status"
	@echo "  make clean       - Clean local temporary files"

# 部署
deploy:
	@echo "🚀 Deploying to GPU43..."
	@bash scripts/deploy_to_gpu43.sh

# 环境设置
setup:
	@echo "🔧 Setting up environment..."
	@pip install -r requirements.txt
	@mkdir -p results logs checkpoints data

# 单元测试
test:
	@echo "🧪 Running tests..."
	@python -m pytest src/tests/ -v --cov=src

# Round 1: 内循环验证
run-round1:
	@echo "🔄 Running Round 1: Inner Loop Validation"
	@export ALIYUN_API_KEY=$(API_KEY) && \
		python experiments/run_round1.py --config configs/round1_inner_loop.yaml

# Round 2: 主实验
run-round2:
	@echo "🔄 Running Round 2: Main Experiments"
	@export ALIYUN_API_KEY=$(API_KEY) && \
		python experiments/run_round2_main.py --config configs/round2_main_mosei.yaml

# Round 3: 分析
run-round3:
	@echo "🔄 Running Round 3: Analysis"
	@python experiments/run_round3_analysis.py --config configs/round3_analysis.yaml

# Round 4: 部署与图表
run-round4:
	@echo "🔄 Running Round 4: Deployment & Figures"
	@python experiments/run_round4_deployment.py --config configs/round4_deployment.yaml

# 数据下载
download:
	@echo "📥 Downloading datasets..."
	@bash scripts/download_data.sh

# 同步代码到服务器
sync-up:
	@echo "📤 Syncing code to GPU43..."
	@rsync -avz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
		--exclude='results/' --exclude='logs/' --exclude='checkpoints/' \
		--exclude='data/' \
		./ $(SERVER):$(REMOTE_DIR)/

# 同步结果到本地
sync-down:
	@echo "📥 Syncing results from GPU43..."
	@mkdir -p results_from_server
	@rsync -avz $(SERVER):$(REMOTE_DIR)/results/ ./results_from_server/
	@echo "Results downloaded to: ./results_from_server/"

# 检查服务器状态
status:
	@echo "📊 Checking GPU43 status..."
	@ssh $(SERVER) "echo '=== GPU Status ===' && nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used --format=csv && echo '' && echo '=== Disk Usage ===' && df -h /usr1 && echo '' && echo '=== Recent Results ===' && ls -ltr $(REMOTE_DIR)/results 2>/dev/null | tail -5"

# 监控GPU
monitor:
	@echo "📈 Monitoring GPU43..."
	@ssh $(SERVER) "watch -n 2 'nvidia-smi && echo "=== Processes ===" && ps aux | grep python | grep -v grep'"

# 清理本地临时文件
clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned"

# 服务器快捷命令
ssh-server:
	@ssh $(SERVER)

# 在服务器上运行命令
server-cmd:
	@ssh $(SERVER) "cd $(REMOTE_DIR) && $(CMD)"

# 快速测试（本地小样本）
quick-test:
	@echo "🚀 Quick test with toy dataset..."
	@export ALIYUN_API_KEY=$(API_KEY) && \
		python experiments/run_round1.py --config configs/round1_inner_loop.yaml --quick

# ============== 基线特定实验命令 ==============

# 运行基线实验（在服务器上）
# 使用方法: make baseline METHOD=mean DATASET=mosei GPU=0
baseline:
	@echo "🧪 Running $(METHOD) baseline on $(DATASET) (GPU $(GPU))..."
	@python experiments/run_baseline_on_server.py --method $(METHOD) --dataset $(DATASET) --gpu $(GPU)

# 运行MOSEI数据集的所有简单基线
baseline-mosei-simple:
	@echo "🧪 Running all simple baselines on MOSEI..."
	@for method in mean concat attention max; do \
		python experiments/run_baseline_on_server.py --method $$method --dataset mosei --gpu 0; \
	done

# 运行MOSEI数据集的所有固定架构基线
baseline-mosei-fixed:
	@echo "🧪 Running all fixed-architecture baselines on MOSEI..."
	@for method in dynmm tfn admn centaur fdsnet; do \
		python experiments/run_baseline_on_server.py --method $$method --dataset mosei --gpu 0; \
	done

# 运行MOSEI数据集的所有NAS基线（需要API key）
baseline-mosei-nas:
	@echo "🧪 Running all NAS baselines on MOSEI..."
	@export ALIYUN_API_KEY=$(API_KEY) && \
	for method in darts llmatic evoprompting; do \
		python experiments/run_baseline_on_server.py --method $$method --dataset mosei --gpu 0; \
	done

# 运行所有基线（全部3个数据集）
baseline-all:
	@echo "🧪 Running all baselines on all datasets..."
	@for dataset in mosei iemocap vqa; do \
		for method in mean concat attention max dynmm tfn admn centaur fdsnet; do \
			echo "Running $$method on $$dataset..."; \
			python experiments/run_baseline_on_server.py --method $$method --dataset $$dataset --gpu 0; \
		done; \
	done

# 查看基线实验结果
baseline-results:
	@echo "📊 Baseline experiment results:"
	@for dir in results/baselines_*; do \
		if [ -d "$$dir" ]; then \
			echo ""; \
			echo "=== $$dir ==="; \
			ls -1 $$dir/*.json 2>/dev/null | head -5; \
		fi; \
	done

# 汇总基线结果
baseline-summary:
	@echo "📊 Generating baseline summary..."
	@python -c "
import json
import glob
from pathlib import Path

results = {}
for json_file in glob.glob('results/baselines_*/*.json'):
    with open(json_file) as f:
        data = json.load(f)
        method = data.get('method', 'unknown')
        dataset = data.get('dataset', 'unknown')
        if method not in results:
            results[method] = {}
        results[method][dataset] = data

# Print summary
print('='*70)
print('BASELINE EXPERIMENT SUMMARY')
print('='*70)
for method in sorted(results.keys()):
    print(f'\n{method.upper()}:')
    for dataset in sorted(results[method].keys()):
        data = results[method][dataset]
        acc = data.get('dropout_0_mean', 0)
        acc_std = data.get('dropout_0_std', 0)
        mrob50 = data.get('mrob_50_mean', 0)
        print(f'  {dataset}: Acc={acc:.4f}±{acc_std:.4f}, mRob@50={mrob50:.4f}')
print('='*70)
" 2>/dev/null || echo "No results yet. Run some experiments first."

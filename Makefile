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

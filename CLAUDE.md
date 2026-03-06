# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

**AutoFusion v3** is the third iteration of an LLM-Driven Neural Architecture Search (NAS) system for multimodal fusion architectures. This directory is the next evolution after:

1. **phase5_llm_rl/** - AutoFusion 1.0: Template-based NAS (5 fixed templates, 100% compile success)
2. **autofusion2/** - AutoFusion 2.0: Search space-free NAS with dual-loop feedback
3. **autofusionv3/** - AutoFusion 3.0: (Current - to be implemented)

### Predecessor Architectures

**AutoFusion 1.0** (`../phase5_llm_rl/`):
- 5 predefined architecture templates (attention, gated, mlp, hybrid, transformer)
- LLM acts as template selector + parameter optimizer
- 100% compile success via template guarantees
- Best result: Hybrid architecture with reward=3.913 on MMMU

**AutoFusion 2.0** (`../autofusion2/`):
- Search space-free: LLM generates raw PyTorch code directly
- Dual-loop feedback: Inner loop (auto-debugging) + Outer loop (performance evolution)
- Dynamic data adapter: Auto-sniffs data dimensions, generates API contracts
- Three validation scenarios: MMMU (high-dim), VQA-RAD (medical), Edge robotics

## Common Commands (From Predecessors)

### Development Setup

```bash
# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Core dependencies
pip install transformers datasets accelerate pyyaml matplotlib numpy pillow

# LLM API
pip install openai

# API Key (required)
export ALIYUN_API_KEY="your-api-key"
# or: export DEEPSEEK_API_KEY="your-api-key"
```

### Running Experiments (AutoFusion 2.0 Reference)

```bash
cd ../autofusion2

# Run with config file
python src/main.py --config configs/scenario_a_mmmu.yaml

# Run with CLI args
python src/main.py \
    --data_dir ./data/mmmu \
    --scenario high_dim_reasoning \
    --max_iterations 200
```

### GPU Cluster Operations (AutoFusion 2.0 Makefile)

```bash
cd ../autofusion2

# Deploy to NTU GPU43
make deploy

# Run tests on cluster
make test

# Sync code to cluster
make sync-up

# Sync results from cluster
make sync-down

# Check cluster status
make status

# Monitor GPU
make monitor
```

### Legacy Experiments (AutoFusion 1.0)

```bash
cd ../phase5_llm_rl

# Phase 5.5 (template mode)
python src/v2/run_v2.py --config configs/v2/exp_kimi.yaml --output-dir results/exp_kimi

# Phase 5.6 (extended search)
python src/v3/run_v3.py --config configs/v3/exp_kimi_extended.yaml

# Deploy to server
bash scripts/deploy_v2.sh
```

### Generate Figures

```bash
# Generate comparison charts for papers/reports
python ../scripts/generate_figures.py

# Output: ../docs/experiments/figures/
```

## Key Architectural Patterns

### Template Mode (AutoFusion 1.0)

```python
# Predefined templates ensuring 100% compile success
ARCHITECTURE_TEMPLATES = {
    "attention": CrossModalAttention,
    "gated": GatedFusion,
    "mlp": MLPFusion,
    "hybrid": AttentionGatingHybrid,  # Best performing
    "transformer": TransformerFusion,
}

# LLM output: {"template": "hybrid", "params": {"hidden_dim": 32, "num_heads": 1}}
```

### Dual-Loop Feedback (AutoFusion 2.0)

```python
# Inner Loop: Auto-debugging guarantees compile success
class InnerLoopSandbox:
    def self_healing_compile(self, prompt, max_retries=5):
        for attempt in range(max_retries):
            code = self.llm.generate(prompt)
            if self.validate_and_compile(code):
                return code, attempt + 1
            prompt = self.add_error_context(prompt, code, error)

# Outer Loop: Performance evolution
class DualLoopController:
    def search(self):
        for iteration in range(self.max_iterations):
            # Inner loop: Get compilable code
            code, attempts = self.inner_loop.self_healing_compile(prompt)
            # Outer loop: Evaluate and evolve
            metrics = self.proxy_evaluator.evaluate(code)
            reward = self.reward_fn(metrics)
            self.update_strategy(reward)
```

### Dynamic Data Adapter (AutoFusion 2.0)

```python
from src.adapter import DynamicDataAdapter

adapter = DynamicDataAdapter()
dataset, contract = adapter.ingest_folder("./data/mmmu")

# Generates API contract for LLM
print(contract.to_prompt())
# 【API Interface Contract】
# Input Specifications:
#   - visual: Shape [B, 576, 1024], Dtype float32
#   - text: Shape [B, 77, 768], Dtype float32
```

## Server Configuration

**NTU GPU Cluster**:
- **Host**: `gpu43.dynip.ntu.edu.sg`
- **User**: `s125mdg43_10`
- **GPU**: 4 × NVIDIA RTX A5000 (24GB)
- **AutoFusion 1.0 Path**: `/usr1/home/s125mdg43_10/AutoFusion_Advanced/`
- **AutoFusion 2.0 Path**: `/usr1/home/s125mdg43_10/AutoFusion_v2/`

## Configuration Format (YAML)

```yaml
# Reference from AutoFusion 2.0
scenario:
  name: "high_dim_reasoning"
  data_path: "./data/mmmu"

constraints:
  max_flops: 10_000_000
  max_params: 50_000_000
  target_accuracy: 0.45

llm:
  backend: "aliyun"
  model: "kimi-k2.5"
  api_key: "${ALIYUN_API_KEY}"

dual_loop:
  max_iterations: 200
  inner_loop:
    max_retries: 5
    timeout_seconds: 120
  outer_loop:
    early_stop_patience: 20

proxy_evaluator:
  num_shots: 64
  train_epochs: 10
  batch_size: 4
```

## AutoFusion 3.0: Executable Architecture Synthesis (EAS)

**Research Goal**: ICCV/CVPR/NeurIPS 2026 submission on open-space NAS with emergent multimodal robustness

**Core Innovation**:
- Search space = Turing-complete Python code (not discrete DAGs)
- Inner loop: Syntax-Aware Generation + iterative repair (100% compile success)
- Outer loop: Performance-Driven Evolution (CMA-ES + LLM mutation)
- Target: mRob > 0.85 at 50% modality missing (baseline < 0.60)

**Three Key Datasets**:
1. **CMU-MOSEI**: 3-modal sentiment (23K samples)
2. **VQA-v2**: Visual QA (200K+ samples)
3. **IEMOCAP**: Emotion recognition (12h)

**Baseline Papers**:
| Method | Link | Type |
|--------|------|------|
| DARTS | [arXiv:1806.09055](https://arxiv.org/abs/1806.09055) | Traditional NAS |
| LLMatic | [arXiv:2306.01102](https://arxiv.org/abs/2306.01102) | LLM-NAS |
| EvoPrompting | [arXiv:2302.14838](https://arxiv.org/abs/2302.14838) | Code-level NAS |
| DynMM | [arXiv:2204.00102](https://arxiv.org/abs/2204.00102) | Dynamic fusion |
| FDSNet | [Nature](https://www.nature.com/articles/s41598-025-25693-y) | Multimodal dynamic |
| ADMN | [arXiv:2502.07862](https://arxiv.org/abs/2502.07862) | Adaptive network |
| Centaur | [arXiv:2303.04636](https://arxiv.org/abs/2303.04636) | Robust fusion |

**Full Paper Plan**: `docs/EAS_PAPER_PLAN.md`

**Detailed Implementation Plan**: `docs/EXPERIMENT_IMPLEMENTATION_PLAN.md`
- Hardware/software requirements
- Day-by-day task breakdown
- Code architecture design
- Risk management strategies

## Project Evolution Insights

### What Worked in 1.0
- Template mode achieved 100% compile success (vs 0-24% without)
- Hybrid (attention + gating) architecture consistently optimal
- Error feedback loop improved retry success

### What Worked in 2.0
- Dual-loop feedback: Inner loop guarantees compile success without templates
- Dynamic data adapter enables cross-task generalization
- Raw code generation removes search space limitations

### Challenges Addressed in 3.0 (EAS)
- **Open code space**: True Turing-complete search, not fixed templates
- **Emergent robustness**: Architecture learns to skip unreliable modalities
- **Structure dynamics**: Runtime conditional execution (if/else in forward pass)

## Important Notes

1. **PYTHONPATH**: Must include project root when running from subdirectories
2. **GPU Memory**: Each experiment uses ~8GB VRAM; can run 2-3 concurrent on GPU43
3. **API Costs**: ~$50-100 USD for 200-iteration experiment on Aliyun Bailian
4. **Data Location**: Datasets stored in `/usr1/home/s125mdg43_10/data/` on cluster

# Synthesis Collapse: Why Greedy Selection Degrades Iterative LLM Data Synthesis and How to Prevent It

This repository contains the code, experiment results, and paper for our study of **synthesis-phase collapse** in iterative LLM data synthesis.

## Paper

The paper is available in `paper/main.tex` (NeurIPS 2026 format). Key finding: greedy (top-$k$ by quality) selection causes progressive behavioral coverage collapse across three domains (Dialogue, Math, Code), while per-cell deduplication prevents it.

## Key Results

### Code Domain (7B Model, MBPP → HumanEval)

| Strategy | R0 pass@1 | R1 pass@1 | R0 Cells | R1 Cells | Cell Change |
|----------|-----------|-----------|----------|----------|-------------|
| QD | 100.0% | 99.4% | 68 | 68 | 0% |
| Greedy | 100.0% | 98.8% | 29 | 25 | **-13.8%** |
| Simple-Dedup | 100.0% | -- | 68 | -- | -- |

Greedy's cell set is always a **strict subset** of QD's. QD discovers 39+ exclusive behavioral regions that Greedy never reaches.

### Downstream Fine-tuning (8-seed, Wilcoxon p=0.0078)

QD-selected code data achieves **2.0x higher HumanEval pass@1** (68.5% vs 34.5%) with **5x fewer samples** (40 vs 200).

## Repository Structure

```
synthesis-collapse/
├── scripts/            # Core experiment scripts
│   ├── self_synthesis_code.py         # V7 Code self-synthesis (main experiment)
│   ├── self_synthesis_base_reset.py   # Base-reset experiment (isolates LoRA drift)
│   ├── self_synthesis_loop.py         # Iterative collapse dynamics
│   ├── analyze_v7_results.py          # V7 results analysis
│   ├── eval_per_problem_humaneval.py  # Per-problem HumanEval analysis
│   ├── code_downstream_finetune.py    # Code downstream evaluation
│   ├── compute_8seed_stats.py         # 8-seed statistical analysis
│   └── ...
├── results/            # Experiment results (JSON summaries)
│   ├── self_synthesis_v7_code/        # V7 Code iterative results
│   ├── iterative_collapse/            # Collapse dynamics data
│   ├── downstream/                    # Downstream fine-tuning results
│   ├── code_downstream/               # Code domain results
│   ├── cross_domain_eval_v3/          # Cross-domain GSM8K evaluation
│   └── llm_judge/                     # LLM Judge win-rate results
├── paper/              # Paper source (LaTeX)
│   ├── main.tex
│   ├── references.bib
│   └── neurips_2026.sty
├── figures/            # Paper figures (PDF)
└── .github/workflows/  # CI/CD
```

## Models and Data

### Models

We use two open-source models from the Qwen family:

| Model | Purpose | Download |
|-------|---------|----------|
| **Qwen2.5-7B-Instruct** | Self-synthesis generation + evaluation | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) |
| **Qwen2.5-1.5B-Instruct** | Downstream fine-tuning evaluation | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) / [ModelScope](https://modelscope.cn/models/Qwen/Qwen2.5-1.5B-Instruct) |

Download and place under a local path, then update `MODEL_PATH` in the scripts.

### Datasets

| Dataset | Domain | Purpose | Source |
|---------|--------|---------|--------|
| **CCSE-CS** | Dialogue | 542 empathetic dialogues, 18-strategy taxonomy | [GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI) |
| **GSM8K** | Math | Math reasoning benchmark | [HuggingFace](https://huggingface.co/datasets/gsm8k) |
| **MBPP** | Code | Code generation prompt pool (377 problems) | [HuggingFace](https://huggingface.co/datasets/google-research-datasets/mbpp) |
| **HumanEval** | Code | Code evaluation benchmark (164 problems) | [HuggingFace](https://huggingface.co/datasets/openai/openai_humaneval) |
| **all_dialogues_final.json** | Dialogue | Our processed dialogue data | See `data/` instructions below |

### Obtaining the Data

```bash
# Dialogue data (CCSE-CS)
git clone https://github.com/AlibabaResearch/DAMO-ConvAI.git
# Process into all_dialogues_final.json using the descriptor functions in our scripts

# GSM8K, MBPP, HumanEval
pip install datasets
python -c "from datasets import load_dataset; load_dataset('gsm8k', 'main'); load_dataset('openai/openai_humaneval', 'openai_humaneval'); load_dataset('google-research-datasets/mbpp')"
```

## Reproducing the Experiments

### Environment

```bash
conda create -n synth-collapse python=3.10
conda activate synth-collapse
pip install torch transformers peft trl datasets numpy scipy matplotlib
```

### Quick Start: Code Self-Synthesis

```bash
# Run V7 Code self-synthesis (QD strategy, seed=42)
CUDA_VISIBLE_DEVICES=0 STRATEGY=qd SEED=42 GPU_ID=0 \
  python scripts/self_synthesis_code.py

# Run with Greedy strategy for comparison
CUDA_VISIBLE_DEVICES=1 STRATEGY=greedy SEED=42 GPU_ID=1 \
  python scripts/self_synthesis_code.py

# Run with Simple-Dedup baseline
CUDA_VISIBLE_DEVICES=2 STRATEGY=simple_dedup SEED=42 GPU_ID=2 \
  python scripts/self_synthesis_code.py
```

### Analysis

```bash
# Analyze V7 results
python scripts/analyze_v7_results.py

# Per-problem analysis (requires completed experiments)
CUDA_VISIBLE_DEVICES=0 python scripts/eval_per_problem_humaneval.py
```

### Hardware

- **Minimum**: 1x A100-80GB or equivalent (80GB VRAM for 7B model)
- **Recommended**: 3-7 GPUs for parallel strategy comparison
- **Training**: ~2 min per round (LoRA, 3 epochs, bf16)
- **Generation**: ~30 min per round (377 problems x 5 solutions)

## Citation

If you use this code or find our work useful, please cite:

```bibtex
@article{synthesis_collapse_2026,
  title={Synthesis Collapse: Why Greedy Selection Degrades Iterative LLM Data Synthesis and How to Prevent It},
  author={Anonymous},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Qwen team for the open-source models
- CCSE-CS dataset authors
- MAP-Elites framework by Mouret and Clune

# Task-Aware Adaptive Compression (TAAC) for LLM Inference

This repository contains the research code and data for the paper:

**"The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts"**

Submitted to NeurIPS 2026

## Key Findings

### 1. The Perplexity Paradox
Code syntax tokens exhibit **79x higher perplexity** than content words and are preserved under compression, while numerical values in math problems have *lower* perplexity than surrounding text and are pruned despite being task-critical.

### 2. Task-Dependent Compression Thresholds
- **Code generation** tolerates aggressive compression (r >= 0.6) before sharp degradation
- **Chain-of-thought reasoning** degrades gradually across all compression levels
- ANCOVA validates this is a genuine task-type effect: F(5, 2019) = 57.84, p < .001

### 3. Function Identity Collapse (NEW)
At aggressive compression (r <= 0.4), function signatures are stripped despite explicit inclusion:
- **70.9% of failures at r=0.3 are NameError** (function undefined)
- Compression treats function names as "redundant" due to moderate perplexity
- This explains the sharp threshold at r=0.6 for code generation

### 4. TAAC Algorithm
Quality-gated compression achieving **21.8% cost reduction** with **95.6% quality preservation**.

## Repository Structure

```
.
├── taac_algorithm.py              # TAAC implementation
├── perplexity_analysis.py         # Per-token perplexity analysis
├── length_controlled_analysis.py  # ANCOVA analysis
├── mbpp_experiment.py             # MBPP benchmark validation
├── algorithm_comparison.py        # Multi-algorithm comparison
├── config.py                      # Experiment configuration
├── data/                          # Analysis results and data
│   ├── length_controlled_results.json
│   ├── mbpp_experiment_results.jsonl  # MBPP validation data
│   ├── mbpp_analysis_report.md        # Current MBPP findings
│   └── perplexity_analysis/
└── requirements.txt               # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run Length-Controlled Analysis
```bash
python length_controlled_analysis.py --mode run --sample-size 500
```

### 2. Run Perplexity Analysis
```bash
python perplexity_analysis.py --mode run --num-samples 100
```

### 3. Run TAAC Evaluation
```bash
python taac_algorithm.py --mode evaluate
```

### 4. Run MBPP Benchmark
```bash
python mbpp_experiment.py --mode run --models validation --sample-size 300
```

## MBPP Experiment Results (In Progress)

| Compression Ratio | Pass Rate | NameError Rate | Interpretation |
|-------------------|-----------|----------------|----------------|
| r=0.3 | 3.7% | 68.3% | Function signature stripped |
| r=0.4 | 11.3% | 48.0% | Partial recovery |
| r=0.5 | 23.0% | 35.3% | Threshold emerging |
| r=0.6 | TBD | TBD | Expected: ~65%+ |
| r=0.7 | TBD | TBD | Expected: ~80%+ |
| r=1.0 | TBD | TBD | Baseline (~65%) |

## Key Results

| Finding | Value |
|---------|-------|
| Code syntax perplexity vs content words | 79x higher |
| Kept vs removed token perplexity | 71,000x difference |
| ANCOVA Task x Compression interaction | F(5, 2019) = 57.84, p < .001 |
| NameError rate at r=0.3 | 70.9% of failures |
| TAAC cost savings | 21.8% |
| TAAC quality preservation | 95.6% |

## Citation

```bibtex
@article{johnson2026perplexity,
  title={The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts},
  author={Johnson, Warren},
  journal={Proceedings of NeurIPS 2026},
  year={2026}
}
```

## License

MIT License

## Acknowledgments

This research was conducted at Bona Opera Studios with computational resources provided by Microsoft Azure.

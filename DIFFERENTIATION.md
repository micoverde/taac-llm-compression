# TAAC Differentiation from Prior Work

## Overview

This document details how TAAC (Task-Aware Adaptive Compression) differs from two key prior works in task-aware prompt compression:

1. **ATACompressor** (Huang et al., OpenReview 2024)
2. **TACO-RL** (Shi et al., arXiv:2409.13035, 2024)

## Summary Table

| Aspect | ATACompressor | TACO-RL | TAAC (Ours) |
|--------|---------------|---------|-------------|
| **Task Awareness** | End-to-end learned | RL with task rewards | Empirical thresholds |
| **Adaptation Mechanism** | Adaptive controller | REINFORCE policy | Quality-gating |
| **Training Required** | Heavy (full model) | Moderate (RL) | Light (classifier + predictor) |
| **Compression Target** | Dynamic ratio | Policy-determined | Quality floor |
| **Theoretical Basis** | None explicit | Reward optimization | Perplexity paradox |
| **Key Innovation** | Soft/hard prompt combo | Task-specific rewards | Quality gating |

---

## ATACompressor (Huang et al. 2024)

### Method
ATACompressor combines hard and soft prompt paradigms with an adaptive controller that dynamically adjusts compression rates. The system learns task-awareness end-to-end through the training process.

**Key Components:**
- Soft prompt tokens for task encoding
- Hard prompt compression for content
- Adaptive controller for rate adjustment
- End-to-end joint training

### TAAC Differentiation

| ATACompressor | TAAC |
|---------------|------|
| Learns task-awareness implicitly through training gradients | Exploits **empirically-discovered thresholds** from prior work (Johnson 2026) |
| Targets a compression ratio determined by the controller | Targets a **quality floor** - stops when quality would drop |
| Requires full end-to-end training | Uses lightweight task classifier + quality predictor |
| No explicit mechanism for understanding *why* compression works | Provides **mechanistic explanation** via perplexity paradox |

**Key Insight:** ATACompressor learns that different tasks need different compression, but doesn't explain *why*. TAAC exploits the discovery that code tolerates compression at r>=0.6 due to high-perplexity syntax tokens being preserved, while math reasoning fails because low-perplexity numbers are pruned.

---

## TACO-RL (Shi et al. 2024)

### Method
TACO-RL uses reinforcement learning with task-specific reward signals to guide compression. The system learns a compression policy via REINFORCE that maximizes task performance.

**Key Components:**
- REINFORCE algorithm for policy learning
- Task-specific rewards (BLEU for summarization, F1 for QA, etc.)
- Policy network for compression decisions
- RL training loop with reward shaping

### TAAC Differentiation

| TACO-RL | TAAC |
|---------|------|
| Optimizes a reward signal through RL | Uses **direct quality prediction** from embeddings |
| Compression policy is a learned function | Compression is **iterative with quality monitoring** |
| Requires RL training with reward engineering | Uses simple supervised learning for quality predictor |
| No explicit stopping criterion | Explicit **quality floor** stops compression |
| Task-specific reward functions must be designed | Task-agnostic quality prediction |

**Key Insight:** TACO-RL learns *what* compression to do via reward optimization. TAAC directly predicts *whether* a compression will maintain quality, enabling dynamic stopping.

---

## TAAC's Unique Contributions

### 1. Empirically-Derived Task Thresholds

Based on Johnson (2026), we discovered that:
- **Code**: Threshold behavior at r >= 0.6 (quality cliff below)
- **CoT**: Gradual linear degradation with compression
- **Hybrid**: Intermediate behavior

These thresholds are used as starting points, adjusted by information density.

```python
# From taac_algorithm.py
r_code = 0.65     # Conservative, above the 0.6 cliff
r_cot = 0.80      # Minimal compression for reasoning tasks
r_hybrid = 0.72   # Intermediate for mixed tasks
```

### 2. Quality Gating

The key innovation: rather than targeting a fixed ratio, we **stop when predicted quality drops below a user-specified floor**.

```python
# Algorithm 1 from paper
while r_current > r_target:
    x' <- Compress(x, r_current - delta)
    Q_hat <- QualityPredictor(x', tau)
    if Q_hat < Q_min:
        break  # Quality floor reached
    r_current <- r_current - delta
```

This prevents over-compression that would degrade task performance.

### 3. Information Density Adjustment

We use the coefficient of variation (CV) of per-token perplexity:

```
rho(x) = std(PPL(x)) / mean(PPL(x))
```

- High CV = heterogeneous information (some tokens much more important)
- Low CV = uniform importance distribution

This allows more aggressive compression when information is localized.

### 4. Mechanistic Explanation

We provide the first per-token perplexity analysis explaining *why* different tasks respond differently:

**The Perplexity Paradox:**
- Code syntax tokens (def, return, if) have **high perplexity** (unusual to LM)
- These are **preserved** under compression
- Numbers in math problems have **low perplexity** (predictable syntactic position)
- These are **pruned** despite being task-critical

---

## Practical Implications

### Training Efficiency

| Method | Training Requirements |
|--------|----------------------|
| ATACompressor | Full end-to-end training, heavy compute |
| TACO-RL | RL training with reward engineering |
| TAAC | Task classifier (fine-tune DistilBERT) + quality predictor (2-layer MLP) |

### Deployment Overhead

| Method | Inference Overhead |
|--------|-------------------|
| ATACompressor | Adaptive controller inference |
| TACO-RL | Policy network inference |
| TAAC | Classification (<10ms) + density estimation + quality prediction |

### Interpretability

| Method | Interpretability |
|--------|-----------------|
| ATACompressor | Black-box controller decisions |
| TACO-RL | Black-box policy outputs |
| TAAC | Clear reasoning: task type -> threshold -> density adjustment -> quality monitoring |

---

## Experimental Validation

### Ablation Study Design

We validate each component's contribution:

1. **TAAC-full**: Complete algorithm
2. **Task-only**: Fixed thresholds per task (no density, no quality-gating)
3. **Density-only**: Density-adjusted ratio (no task awareness, no quality-gating)
4. **Quality-gate-only**: Quality monitoring only (no task/density awareness)

### Expected Results

Based on theoretical analysis:
- Task classification provides the largest single contribution
- Quality gating prevents catastrophic over-compression
- Density estimation provides fine-grained adjustment

---

## Citation

If you use TAAC, please cite:

```bibtex
@article{johnson2026taac,
  title={The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts},
  author={Johnson, Warren},
  journal={NeurIPS},
  year={2026}
}
```

## References

- Huang et al. (2024). ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLM. OpenReview preprint.
- Shi et al. (2024). TACO-RL: Task Aware Prompt Compression Optimization with Reinforcement Learning. arXiv:2409.13035.
- Johnson, W. (2026). Compress or Route? Task-Dependent Strategies for Cost-Efficient Large Language Model Inference.

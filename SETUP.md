# Compression Algorithm Comparison: VM Setup Guide

This guide covers installation and configuration of all compression algorithms for the TAAC algorithm comparison experiment.

## Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with 16GB+ VRAM (required for LLMLingua-1)
- **RAM**: 32GB recommended
- **Storage**: 50GB free space for model weights

## Quick Start

```bash
# Clone and setup environment
cd /home/warrenjo/src/tmp5/plex-vc-fund-platform/research/paper-v2
python -m venv .venv
source .venv/bin/activate

# Install base dependencies
pip install numpy scipy pandas torch transformers datasets httpx

# Run Phase 1 (quick validation)
python experiments/algorithm_comparison.py --phase phase1_quick_validation --dry-run
```

---

## Algorithm-Specific Installation

### 1. LLMLingua-2 (Primary Method)

**Description**: Trained BERT-based classifier for token importance scoring.

**Characteristics**:
- Fast inference (~100ms per prompt)
- Low GPU memory (~4GB)
- Task-agnostic compression

**Installation**:
```bash
pip install llmlingua

# Verify installation
python -c "from llmlingua import PromptCompressor; print('LLMLingua-2 OK')"
```

**Model Download** (automatic on first use):
- Model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
- Size: ~500MB

**Configuration**:
```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True,
)

result = compressor.compress_prompt(
    prompt,
    rate=0.5,  # Keep 50% of tokens
    force_tokens=["\n", "def", "return"],  # Preserve these tokens
)
```

---

### 2. LLMLingua-1 (Perplexity Baseline)

**Description**: Uses Llama-2 perplexity to identify important tokens.

**Characteristics**:
- Slower inference (~2-5s per prompt)
- High GPU memory (~16GB)
- Original compression method

**Installation**:
```bash
pip install llmlingua accelerate bitsandbytes

# For 4-bit quantization (reduces memory to ~6GB)
pip install bitsandbytes

# Verify installation
python -c "from llmlingua import PromptCompressor; print('LLMLingua-1 OK')"
```

**Model Download**:
- Model: `NousResearch/Llama-2-7b-hf`
- Size: ~14GB (full precision) or ~4GB (4-bit quantized)

**Important**: Requires HuggingFace access token for Llama-2:
```bash
# Login to HuggingFace
huggingface-cli login

# Or set environment variable
export HF_TOKEN="your_token_here"
```

**Configuration**:
```python
from llmlingua import PromptCompressor

# Full precision (requires 16GB VRAM)
compressor = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    use_llmlingua2=False,
    device_map="auto",
)

# 4-bit quantized (requires 6GB VRAM)
compressor = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    use_llmlingua2=False,
    device_map="auto",
    model_config={"load_in_4bit": True},
)

result = compressor.compress_prompt(
    prompt,
    rate=0.5,
    condition_compare=True,
    condition_in_question="none",
)
```

---

### 3. Selective Context (Self-Information)

**Description**: Uses GPT-2 self-information to select most informative tokens.

**Characteristics**:
- Moderate speed (~500ms per prompt)
- Low GPU memory (~4GB)
- Different theoretical basis than LLMLingua

**Installation**:
```bash
pip install transformers torch

# Verify installation
python -c "from transformers import GPT2LMHeadModel; print('Selective Context OK')"
```

**Model Download**:
- Model: `gpt2-large`
- Size: ~1.5GB

**Configuration**:
```python
import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large").cuda().eval()

def selective_context_compress(text: str, ratio: float) -> str:
    """Compress using self-information selection."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)

        # Self-information = -log(P(token|context))
        self_info = []
        for i in range(1, len(inputs["input_ids"][0])):
            prob = probs[i-1, inputs["input_ids"][0][i]].item()
            self_info.append(-np.log(prob + 1e-10))

    # Keep tokens with highest self-information
    n_keep = max(1, int(len(tokens) * ratio))
    indices = [0] + list(np.argsort(self_info)[::-1][:n_keep-1] + 1)
    indices = sorted(set(indices))

    kept_tokens = [tokens[i] for i in indices]
    return tokenizer.convert_tokens_to_string(kept_tokens)
```

**Reference**: Li et al. (2023) "Compressing Context to Enhance Inference Efficiency of Large Language Models"

---

### 4. Random Baseline (Control)

**Description**: Random token selection for establishing compression floor.

**Characteristics**:
- Instant (~1ms per prompt)
- No GPU required
- Control condition

**Installation**:
```bash
# No additional installation needed - uses Python stdlib
```

**Configuration**:
```python
import random
import hashlib

def random_compress(text: str, ratio: float, seed: int = None) -> str:
    """Compress by random token selection."""
    tokens = text.split()
    n_keep = max(1, int(len(tokens) * ratio))

    # Deterministic seed based on input
    if seed is None:
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(tokens)), min(n_keep, len(tokens))))

    return " ".join(tokens[i] for i in indices)
```

---

## API Keys

Set the following environment variables:

```bash
# LLM Providers (for evaluation)
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."

# HuggingFace (for Llama-2 access)
export HF_TOKEN="hf_..."

# Optional: Together AI for Llama-3
export TOGETHER_API_KEY="..."
```

---

## Memory Management

### GPU Memory by Algorithm

| Algorithm | Full Precision | Quantized | CPU Fallback |
|-----------|---------------|-----------|--------------|
| LLMLingua-2 | 4 GB | N/A | Slow but works |
| LLMLingua-1 | 16 GB | 6 GB (4-bit) | Very slow |
| Selective Context | 4 GB | N/A | Slow but works |
| Random | 0 GB | N/A | Native |

### Running on Limited Hardware

For machines with < 16GB VRAM:

```python
# Option 1: Use 4-bit quantized LLMLingua-1
compressor = PromptCompressor(
    model_name="NousResearch/Llama-2-7b-hf",
    use_llmlingua2=False,
    model_config={"load_in_4bit": True},
)

# Option 2: Run algorithms sequentially, clearing GPU between
import torch

# Run LLMLingua-2
compressor_l2 = LLMLingua2Compressor(config)
# ... use it ...
del compressor_l2
torch.cuda.empty_cache()

# Now run Selective Context
compressor_sc = SelectiveContextCompressor(config)
# ... use it ...
```

---

## Experiment Phases

### Phase 1: Quick Validation (~$15, 30 min)
```bash
python experiments/algorithm_comparison.py --phase phase1_quick_validation
```
- Algorithms: LLMLingua-2, Random
- Ratios: 0.5, 0.7, 1.0
- Samples: 10 per condition

**Purpose**: Verify random baseline differs from learned compression.

### Phase 2: Algorithm Pair Comparison (~$80, 2 hours)
```bash
python experiments/algorithm_comparison.py --phase phase2_algorithm_pairs
```
- Algorithms: LLMLingua-2, LLMLingua-1
- Ratios: 0.4-0.8
- Samples: 20 per condition

**Purpose**: Compare perplexity vs. trained classifier thresholds.

### Phase 3: Full Comparison (~$350, 8 hours)
```bash
python experiments/algorithm_comparison.py --phase phase3_full_comparison
```
- All algorithms
- Full ratio sweep
- 25 samples per condition

**Purpose**: Complete cross-algorithm dichotomy analysis.

---

## Analysis

After running experiments:

```bash
# Generate analysis report
python analysis/threshold_analysis.py results/algorithm_comparison/phase1_quick_validation/results.jsonl \
    --output report.json \
    --plot thresholds.png

# Key outputs:
# - Threshold location (r*) for each algorithm
# - Cross-algorithm homogeneity test
# - Algorithm x TaskType interaction test
# - Hypothesis verdict
```

---

## Troubleshooting

### CUDA Out of Memory

```python
# Clear cache between algorithms
import torch
torch.cuda.empty_cache()

# Use smaller batch sizes
# (handled internally, but can adjust if needed)
```

### LLMLingua-1 Slow

LLMLingua-1 uses Llama-2 for perplexity, which is inherently slower:
- Expect 2-5 seconds per prompt
- Consider running Phase 1 first to validate setup
- Use 4-bit quantization to reduce both memory and compute

### Missing HuggingFace Access

```bash
# For Llama-2 models
huggingface-cli login
# Accept license at: https://huggingface.co/meta-llama/Llama-2-7b-hf
```

### Selective Context Numerical Issues

The self-information calculation can produce NaN/Inf for very low probability tokens:

```python
# Add epsilon to prevent log(0)
self_info.append(-np.log(prob + 1e-10))
```

---

## References

1. **LLMLingua-2**: Pan et al. (2024) "LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression"
   - https://arxiv.org/abs/2403.12968

2. **LLMLingua-1**: Jiang et al. (2023) "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
   - https://arxiv.org/abs/2310.05736

3. **Selective Context**: Li et al. (2023) "Compressing Context to Enhance Inference Efficiency of Large Language Models"
   - https://arxiv.org/abs/2310.06201

---

## Cost Estimates

| Phase | API Calls | Est. Cost | Notes |
|-------|-----------|-----------|-------|
| Phase 1 | ~120 | $15 | Quick validation |
| Phase 2 | ~720 | $80 | Pair comparison |
| Phase 3 | ~4,200 | $350 | Full study |
| **Total** | ~5,040 | **$445** | |

Costs assume Claude-3-Haiku ($0.25/$1.25 per 1M tokens) as primary model.

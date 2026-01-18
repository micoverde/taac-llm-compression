# MBPP Benchmark Expansion Experiment Design

**Gap 2 from Research Roadmap**: Validate TAAC findings on MBPP (974 problems)

**Author**: Dr. Michael Park
**Date**: January 2026
**Status**: Ready for Execution

---

## 1. Experiment Overview

### Objective
Validate that the threshold behavior (r >= 0.6) observed in HumanEval generalizes to the MBPP benchmark, which is 6x larger (974 vs 164 problems) and has different prompt characteristics.

### Hypothesis
**H1**: The compression threshold of r >= 0.6 will hold for MBPP, though possibly requiring slight adjustment to r >= 0.65 due to shorter, more formulaic prompts.

**H2**: Model tier effects (Premium > Balanced > Economy) will remain consistent across benchmarks.

**H3**: MBPP's simpler problems may show higher baseline pass rates but similar degradation patterns under aggressive compression.

---

## 2. Experiment Configuration

### Compression Ratios
Matching original HumanEval experiments:
- **r = 0.3** (70% token reduction - aggressive)
- **r = 0.4** (60% token reduction)
- **r = 0.5** (50% token reduction)
- **r = 0.6** (40% token reduction - **threshold**)
- **r = 0.7** (30% token reduction - conservative)
- **r = 1.0** (no compression - baseline)

### Model Tiers

| Tier | Models | Cost/1M Input | Cost/1M Output |
|------|--------|---------------|----------------|
| 1 (Economy) | DeepSeek-Chat, Llama-3.1-8B | $0.14-0.18 | $0.18-0.28 |
| 2 (Balanced) | GPT-4o-mini, Claude-3-Haiku | $0.15-0.25 | $0.60-1.25 |
| 3 (Premium) | Claude-3.5-Sonnet, GPT-4o | $2.50-3.00 | $10.00-15.00 |

### Validation Model Set (Recommended for Initial Run)
- `claude-3-haiku` (Tier 2)
- `deepseek-chat` (Tier 1)
- `gpt-4o-mini` (Tier 2)

---

## 3. Cost and Runtime Estimates

### Validation Run (3 models, 500 problems)
| Metric | Value |
|--------|-------|
| Total API Calls | 9,000 |
| Estimated Cost | $1.52 |
| Estimated Runtime | 6.2 hours |

**Cost Breakdown by Model**:
- claude-3-haiku: 3,000 calls, $0.86
- deepseek-chat: 3,000 calls, $0.23
- gpt-4o-mini: 3,000 calls, $0.43

### Full Run (6 models, 500 problems)
| Metric | Value |
|--------|-------|
| Total API Calls | 18,000 |
| Estimated Cost | $19.18 |
| Estimated Runtime | 12.5 hours |

**Cost Breakdown by Model**:
- deepseek-chat: 3,000 calls, $0.23
- llama-3.1-8b: 3,000 calls, $0.19
- gpt-4o-mini: 3,000 calls, $0.43
- claude-3-haiku: 3,000 calls, $0.86
- claude-3.5-sonnet: 3,000 calls, $10.35
- gpt-4o: 3,000 calls, $7.12

---

## 4. MBPP vs HumanEval Prompt Structure Analysis

### Key Differences

| Characteristic | HumanEval | MBPP |
|---------------|-----------|------|
| Problems | 164 | 974 |
| Avg Prompt Length | ~200 tokens | ~100 tokens |
| Prompt Style | Function signature + docstring | Natural language description |
| Test Format | unittest framework | assert statements |
| Complexity | Higher (algorithms) | Lower (basic programming) |

### Threshold Behavior Prediction

**Why r >= 0.6 should still hold:**

1. **Consistent Critical Information Density**: Both benchmarks contain essential information (function name, parameters, behavior description) that compression algorithms should preserve.

2. **LLMLingua-2's Code-Aware Tokens**: The compressor force-preserves Python keywords (`def`, `return`, `if`, `for`, etc.) regardless of ratio.

3. **Prompt Structure Redundancy**: MBPP's formulaic "Write a function to..." prefix is highly compressible, similar to HumanEval's docstring patterns.

**Potential Adjustment Needed:**

MBPP prompts are shorter (~100 tokens vs ~200), so:
- Less redundant content to compress
- Each token carries more relative information
- **May require r >= 0.65 or r >= 0.7 for optimal results**

### Compression Sensitivity Analysis

| Prompt Section | Compressibility | Criticality |
|----------------|-----------------|-------------|
| "Write a function to..." | High | Low |
| Function behavior description | Low | High |
| Example inputs/outputs | Medium | High |
| Edge case hints | Low | High |

---

## 5. Execution Instructions

### Azure VM Setup

```bash
# SSH to Azure VM
ssh azureuser@20.185.221.53

# Activate environment
source ~/taac-env/bin/activate

# Navigate to experiment directory
cd ~/plex-vc-fund-platform/research/paper-v2/experiments

# Install dependencies (if needed)
pip install datasets anthropic openai llmlingua torch
```

### Running the Experiment

#### Step 1: Verify Environment
```bash
python mbpp_experiment.py --mode estimate --models validation
```

#### Step 2: Run Validation Experiment
```bash
# Run with validation models (recommended first)
python mbpp_experiment.py --mode run --models validation --output-dir results/mbpp_validation

# Or with smaller sample for testing
python mbpp_experiment.py --mode run --models validation --sample-size 50
```

#### Step 3: Run Full Experiment
```bash
# Full model set (after validation succeeds)
python mbpp_experiment.py --mode run --models full --output-dir results/mbpp_full
```

#### Step 4: Analyze Results
```bash
python mbpp_experiment.py --mode analyze --output-dir results/mbpp_validation
```

### Environment Variables Required

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
export DEEPSEEK_API_KEY="your-key"
export TOGETHER_API_KEY="your-key"
```

---

## 6. Expected Outcomes

### Primary Metrics

1. **Pass@1 by Compression Ratio**
   - Expected: Sharp degradation at r < 0.6
   - Expected: Stable performance at r >= 0.6

2. **Threshold Validation**
   - If MBPP confirms r >= 0.6: Strong external validity
   - If MBPP requires r >= 0.7: Important benchmark-specific finding

3. **Model Tier Analysis**
   - Premium models expected to maintain performance at lower ratios
   - Economy models expected to degrade faster

### Analysis Outputs

The experiment generates:
- `results.jsonl` - Raw trial data
- `checkpoint.json` - Resume state
- Summary statistics by ratio, model, and tier

### Comparison Framework

```python
# Example analysis code
from mbpp_experiment import analyze_results, compare_with_humaneval

mbpp_analysis = analyze_results(Path("results/mbpp_validation/results.jsonl"))

# Known HumanEval baseline (from original paper)
humaneval_baseline = {
    "threshold": 0.6,
    "by_ratio": {
        0.3: {"pass_rate": 0.45},
        0.4: {"pass_rate": 0.52},
        0.5: {"pass_rate": 0.58},
        0.6: {"pass_rate": 0.72},
        0.7: {"pass_rate": 0.78},
        1.0: {"pass_rate": 0.82},
    }
}

comparison = compare_with_humaneval(mbpp_analysis, humaneval_baseline)
```

---

## 7. Checkpointing and Resume

The experiment supports automatic checkpointing:
- Saves progress every 50 trials
- Can resume from interruption
- Uses trial keys: `{task_id}|{ratio}|{model}`

```bash
# Resume interrupted experiment
python mbpp_experiment.py --mode run --models validation --output-dir results/mbpp_validation
```

---

## 8. Risk Mitigation

### API Rate Limiting
- Sequential execution by default
- 2.5s average per call includes implicit rate limiting

### Cost Control
- Start with validation models ($1.52)
- Sample size option for testing (--sample-size 50)
- Monitor costs via provider dashboards

### Data Integrity
- JSONL format for append-only writes
- Checkpointing prevents data loss
- Automatic resume capability

---

## 9. Success Criteria

### Minimum Success
- [ ] Complete validation run (9,000 trials)
- [ ] Confirm/adjust r >= 0.6 threshold
- [ ] Document any MBPP-specific findings

### Full Success
- [ ] Complete full run (18,000 trials)
- [ ] Statistical comparison with HumanEval
- [ ] Model tier effects validated
- [ ] Publication-ready results table

---

## 10. Next Steps After MBPP

If MBPP validates the threshold:
1. **Gap 3**: Multi-language validation (MultiPL-E)
2. **Gap 4**: Chain-of-thought reasoning (GSM8K, MATH)
3. **Gap 5**: Long-context tasks (SCROLLS, QMSum)

---

**File**: `/home/warrenjo/src/tmp5/plex-vc-fund-platform/research/paper-v2/experiments/mbpp_experiment.py`
**VM**: `ssh azureuser@20.185.221.53`
**Environment**: `taac-env`

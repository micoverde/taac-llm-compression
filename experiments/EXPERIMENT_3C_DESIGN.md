# Experiment 3C: Perplexity Engineering
## Adversarial Manipulation of Token Importance for Compression Resilience

**Principal Investigator**: Dr. Elena Rodriguez
**Affiliation**: Bona Opera Studios Research
**Date**: January 2026
**GitHub Issue**: #1138 (Article 3 Experiments)
**Hardware Requirement**: GPU (T4 or better) for perplexity computation with GPT-2

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Research Questions](#2-research-questions)
3. [Theoretical Framework](#3-theoretical-framework)
4. [Techniques to Test](#4-techniques-to-test)
5. [Experimental Design](#5-experimental-design)
6. [Metrics and Measurements](#6-metrics-and-measurements)
7. [Statistical Analysis Plan](#7-statistical-analysis-plan)
8. [Expected Results](#8-expected-results)
9. [Implementation Details](#9-implementation-details)
10. [Risk Assessment and Mitigations](#10-risk-assessment-and-mitigations)
11. [References](#11-references)

---

## 1. Executive Summary

### The Perplexity Paradox Problem

LLMLingua-2 and related compression algorithms use perplexity as a proxy for token importance: high-perplexity tokens (surprising to the language model) are retained, while low-perplexity tokens (predictable) are pruned. This creates a fundamental vulnerability:

**Critical tokens with predictable surface forms are systematically pruned.**

For example:
- Function name `calculate_sum` has LOW perplexity (common English words in expected positions)
- The function name is CRITICAL for code generation (required for test execution)
- Result: Function identity collapse at aggressive compression ratios (r <= 0.4)

### This Experiment

We investigate whether we can "trick" compression algorithms into preserving critical tokens by manipulating their perplexity through adversarial techniques. This represents a novel research direction: **perplexity engineering** for compression resilience.

### Key Insight from Information Theory

If perplexity measures linguistic predictability, we can increase the perplexity of critical tokens by:
1. Making them lexically rare (obfuscation)
2. Placing them in surprising contexts (decoy injection)
3. Adding explicit preservation markers (structural hints)
4. Prepending rare prefixes (perplexity boosting)

---

## 2. Research Questions

### Primary Research Questions

**RQ1 (Effectiveness)**: Can perplexity engineering techniques significantly increase the retention rate of critical tokens during compression?

> **Formal Statement**: Let R_baseline denote the retention rate of critical tokens under standard compression, and R_engineered denote the retention rate after applying perplexity engineering. We test:
>
> H0: R_engineered = R_baseline
> H1: R_engineered > R_baseline

**RQ2 (Quality Preservation)**: Do prompts with engineered perplexity maintain or improve downstream task performance compared to unmodified compressed prompts?

> **Formal Statement**: Let Q_baseline denote task quality (Pass@1) for compressed prompts and Q_engineered for perplexity-engineered compressed prompts. We test:
>
> H0: Q_engineered <= Q_baseline
> H1: Q_engineered > Q_baseline

**RQ3 (Technique Comparison)**: Which perplexity engineering technique provides the best trade-off between retention improvement and quality preservation?

> This involves pairwise comparisons among four techniques using repeated measures analysis.

### Secondary Research Questions

**RQ4 (Compression Ratio Interaction)**: How does the effectiveness of perplexity engineering vary across compression ratios (r = 0.3, 0.4, 0.5)?

**RQ5 (Token Type Sensitivity)**: Which critical token types (function names, parameters, return types) benefit most from perplexity engineering?

**RQ6 (Overhead Analysis)**: What is the computational and token-length overhead of each technique?

---

## 3. Theoretical Framework

### 3.1 Information-Theoretic Foundation

#### Perplexity as a Compression Signal

For a token $x_i$ in context $x_{<i}$, perplexity is defined as:

$$\text{PPL}(x_i) = \exp\left(-\log P(x_i | x_{<i})\right) = \frac{1}{P(x_i | x_{<i})}$$

LLMLingua-2 uses perplexity as an importance signal:

$$\text{Importance}(x_i) \approx \text{PPL}(x_i)$$

Tokens with high perplexity are retained; tokens with low perplexity are pruned.

#### The Critical Token Vulnerability

Let $T_{\text{critical}}$ be the set of task-critical tokens (e.g., function names). The Perplexity Paradox states:

$$\mathbb{E}[\text{PPL}(t) | t \in T_{\text{critical}}] < \mathbb{E}[\text{PPL}(t) | t \in T_{\text{all}}]$$

This occurs because critical tokens often use common English words in predictable syntactic positions.

#### Perplexity Engineering Goal

The goal of perplexity engineering is to transform the prompt $\mathbf{x}$ to $\mathbf{x}'$ such that:

$$\text{PPL}(t') > \text{PPL}(t) \quad \forall t \in T_{\text{critical}}$$

where $t' = \text{Transform}(t, \text{context})$ is the engineered version of the critical token.

### 3.2 Compression Dynamics Model

#### Token Retention Probability

Under perplexity-based compression with ratio $r$, the probability that token $x_i$ is retained is:

$$P(\text{retained}(x_i)) = P\left(\text{rank}(\text{PPL}(x_i)) \leq \lfloor r \cdot n \rfloor\right)$$

where rank is computed by descending perplexity.

#### Engineering Impact on Retention

If perplexity engineering increases a token's perplexity by factor $\alpha > 1$:

$$\text{PPL}'(x_i) = \alpha \cdot \text{PPL}(x_i)$$

Then the token's rank improves (lower rank = higher retention priority):

$$\text{rank}'(\text{PPL}'(x_i)) < \text{rank}(\text{PPL}(x_i))$$

This increases retention probability.

### 3.3 The Preservation-Semantic Trade-off

#### Semantic Fidelity Constraint

Perplexity engineering must preserve semantic meaning for the downstream LLM:

$$\text{Meaning}(\mathbf{x}') \approx \text{Meaning}(\mathbf{x})$$

This creates a trade-off: aggressive perplexity manipulation may increase retention but degrade semantic clarity.

#### Formal Trade-off Function

We model the combined objective as:

$$\text{Objective} = \lambda \cdot \text{RetentionGain}(\mathbf{x}') + (1-\lambda) \cdot \text{SemanticFidelity}(\mathbf{x}')$$

Different techniques occupy different points on this trade-off curve.

---

## 4. Techniques to Test

We evaluate four perplexity engineering techniques, each exploiting a different mechanism for increasing token perplexity.

### 4.1 Technique A: Lexical Obfuscation

**Mechanism**: Replace critical tokens with high-perplexity alternatives that preserve meaning through context.

**Implementation**:
```python
def lexical_obfuscation(function_name: str) -> str:
    """
    Replace common words with rare synonyms or obfuscated forms.

    Examples:
        calculate_sum -> quantify_aggregate
        get_maximum -> procure_zenith
        is_prime -> verifies_primality
    """
    obfuscation_map = {
        'calculate': 'quantify',
        'compute': 'adjudicate',
        'get': 'procure',
        'find': 'ascertain',
        'check': 'scrutinize',
        'is': 'verifies',
        'sum': 'aggregate',
        'max': 'zenith',
        'min': 'nadir',
        'count': 'enumerate',
        'list': 'manifest',
        'sort': 'ordinate',
        'reverse': 'invert',
        'remove': 'expunge',
        'add': 'append_element',
        'delete': 'obliterate'
    }

    words = function_name.split('_')
    obfuscated = [obfuscation_map.get(w.lower(), w) for w in words]
    return '_'.join(obfuscated)
```

**Perplexity Impact**: Words like "quantify" and "zenith" are rarer in training corpora than "calculate" and "maximum", yielding higher perplexity.

**Expected Perplexity Boost**: 2.5-5x based on word frequency statistics.

**Semantic Risk**: Moderate - LLMs may misinterpret rare vocabulary.

### 4.2 Technique B: Decoy Token Injection

**Mechanism**: Surround critical tokens with rare tokens that don't affect meaning but create a high-perplexity context.

**Implementation**:
```python
def decoy_injection(prompt: str, critical_tokens: list) -> str:
    """
    Inject rare tokens around critical values without changing semantics.

    Example:
        "def calculate_sum(a, b):"
        -> "def [PRESERVE:calculate_sum] (a, b):"

    The [PRESERVE:...] wrapper has high perplexity due to unusual syntax.
    """
    DECOY_PREFIX = "[PRESERVE:"
    DECOY_SUFFIX = "]"

    for token in critical_tokens:
        wrapped = f"{DECOY_PREFIX}{token}{DECOY_SUFFIX}"
        prompt = prompt.replace(token, wrapped, 1)  # Replace first occurrence

    return prompt
```

**Alternative Decoys**:
```python
DECOY_VARIANTS = [
    ("[CRITICAL:{token}]", "Explicit preservation marker"),
    ("<<<{token}>>>", "Rare delimiter pattern"),
    ("__KEEP__{token}__KEEP__", "Underscore emphasis"),
    ("/*MUST_RETAIN*/{token}", "Comment-style marker"),
    ("{token}|ESSENTIAL|", "Pipe-delimited flag")
]
```

**Perplexity Impact**: Square brackets in non-standard patterns and uppercase markers have very high perplexity in natural text.

**Expected Perplexity Boost**: 5-15x for the marker tokens, spillover effect on adjacent tokens.

**Semantic Risk**: Low-Moderate - markers are explicit, but may confuse some LLMs.

### 4.3 Technique C: Structural Preservation Markers

**Mechanism**: Add explicit instructions that create a high-perplexity context while semantically requesting preservation.

**Implementation**:
```python
def structural_markers(prompt: str, critical_info: dict) -> str:
    """
    Prepend a high-perplexity structural preservation block.

    This creates an unusual but semantically clear instruction section.
    """
    marker_block = f"""
<<<COMPRESSION_RESISTANT_BLOCK>>>
CRITICAL_FUNCTION: {critical_info['function_name']}
REQUIRED_PARAMS: {', '.join(critical_info['parameters'])}
RETURN_TYPE: {critical_info.get('return_type', 'inferred')}
<<<END_BLOCK>>>

{prompt}
"""
    return marker_block
```

**Perplexity Impact**: The unusual delimiter pattern (<<<...>>>) and uppercase keywords have extreme perplexity in NL-trained models.

**Expected Perplexity Boost**: 10-50x for marker tokens due to syntactic novelty.

**Semantic Risk**: Low - explicit structure aids LLM comprehension.

### 4.4 Technique D: Perplexity Prefix Boosting

**Mechanism**: Prepend rare token sequences to the entire prompt, elevating baseline perplexity context.

**Implementation**:
```python
def perplexity_prefix(prompt: str, boost_level: str = 'medium') -> str:
    """
    Add rare token prefixes that prime high perplexity.

    The prefixes use rare Unicode, technical jargon, or nonsense sequences
    that don't affect task semantics but elevate perplexity context.
    """
    BOOST_PREFIXES = {
        'low': "TASK_ID:0x7F3A | ",
        'medium': "ZXQV_PROTOCOL_ALPHA | RETAIN_SEMANTIC_CORE | ",
        'high': "<<<ZETA_OMEGA_PRESERVE>>> | AXIOM_0x4B2C | QED_MARKER | ",
        'extreme': ("<<<XENOMORPHIC_PRESERVATION_MATRIX>>> | "
                    "AXIOM_OMEGA_0x7F4B | QUINTESSENTIAL_SEMANTIC_ANCHOR | "
                    "PERPLEXITY_ELEVATION_PROTOCOL_V2 | ")
    }

    prefix = BOOST_PREFIXES[boost_level]
    return prefix + prompt
```

**Perplexity Impact**: Rare character sequences, hexadecimal patterns, and invented terminology have extremely high perplexity.

**Expected Perplexity Boost**: 3-20x depending on boost level, affects entire prompt baseline.

**Semantic Risk**: Variable - higher boost levels may confuse LLMs.

---

## 5. Experimental Design

### 5.1 Design Type: Within-Subjects Repeated Measures

We use a **within-subjects design** where each MBPP task is evaluated under all conditions. This controls for task difficulty variance and maximizes statistical power.

### 5.2 Independent Variables

| Variable | Levels | Type |
|----------|--------|------|
| **Technique** | Control, Lexical, Decoy, Structural, Prefix | Categorical (5 levels) |
| **Compression Ratio** | 0.3, 0.4, 0.5 | Categorical (3 levels) |

**Full Factorial Design**: 5 techniques x 3 ratios = 15 conditions per task

### 5.3 Dependent Variables

| Variable | Type | Measurement |
|----------|------|-------------|
| **Perplexity Delta** | Continuous | PPL(engineered) - PPL(original) for critical tokens |
| **Retention Rate** | Proportion | % of critical tokens retained after compression |
| **Pass@1** | Binary | Task execution success (0/1) |
| **Token Overhead** | Integer | Additional tokens introduced by technique |
| **Semantic Drift** | Ordinal | Manual rating of meaning preservation (1-5) |

### 5.4 Sample Size Calculation

**Power Analysis for Primary Outcome (Retention Rate)**:
- Expected effect size: Cohen's d = 0.5 (medium effect)
- Desired power: 0.80
- Alpha: 0.05 (Bonferroni-corrected for 10 pairwise comparisons: 0.005)

Using G*Power for repeated measures ANOVA:
$$n = \frac{2(z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{d^2}$$

With Bonferroni correction and within-subjects correlation of 0.5:
- **Minimum sample: 150 MBPP tasks**
- **Recommended sample: 300 MBPP tasks** (for robustness)

### 5.5 Counterbalancing Protocol

To control for order effects, we use **Latin Square counterbalancing**:

```
Task Block 1: Control -> Lexical -> Decoy -> Structural -> Prefix
Task Block 2: Lexical -> Decoy -> Structural -> Prefix -> Control
Task Block 3: Decoy -> Structural -> Prefix -> Control -> Lexical
Task Block 4: Structural -> Prefix -> Control -> Lexical -> Decoy
Task Block 5: Prefix -> Control -> Lexical -> Decoy -> Structural
```

Tasks are assigned to blocks via modulo: `block = task_id % 5`

### 5.6 Control Condition

The **Control** condition applies standard compression without any perplexity engineering:

```python
def control_condition(prompt: str, ratio: float) -> str:
    """Baseline: compress without modification."""
    return llmlingua2_compress(prompt, target_ratio=ratio)
```

### 5.7 Experimental Procedure

```
For each task in MBPP[1:300]:
    1. Extract critical tokens (function name, parameters)
    2. Calculate baseline perplexity for critical tokens

    For each technique in [Control, Lexical, Decoy, Structural, Prefix]:
        3. Apply perplexity engineering transformation
        4. Calculate post-engineering perplexity for critical tokens

        For each ratio in [0.3, 0.4, 0.5]:
            5. Compress engineered prompt
            6. Measure critical token retention
            7. Generate code using compressed prompt
            8. Execute tests, record Pass@1
            9. Log all metrics
```

---

## 6. Metrics and Measurements

### 6.1 Primary Metrics

#### 6.1.1 Perplexity Delta (Delta PPL)

The change in perplexity for critical tokens after engineering:

$$\Delta \text{PPL}(t) = \text{PPL}_{\text{engineered}}(t) - \text{PPL}_{\text{original}}(t)$$

**Aggregation**: Mean across all critical tokens in a task.

$$\overline{\Delta \text{PPL}}_{\text{task}} = \frac{1}{|T_{\text{critical}}|} \sum_{t \in T_{\text{critical}}} \Delta \text{PPL}(t)$$

#### 6.1.2 Retention Rate

Proportion of critical tokens retained after compression:

$$\text{RetentionRate} = \frac{|\{t \in T_{\text{critical}} : t \in \mathbf{x}'\}|}{|T_{\text{critical}}|}$$

where $\mathbf{x}'$ is the compressed prompt.

**Calculation Method**: Token-level substring matching with fuzzy matching for obfuscated tokens.

#### 6.1.3 Pass@1

Binary indicator of test execution success:

$$\text{Pass@1} = \begin{cases} 1 & \text{if all test cases pass} \\ 0 & \text{otherwise} \end{cases}$$

#### 6.1.4 Retention Improvement Factor (RIF)

Normalized improvement over control:

$$\text{RIF} = \frac{\text{RetentionRate}_{\text{technique}} - \text{RetentionRate}_{\text{control}}}{\text{RetentionRate}_{\text{control}}}$$

### 6.2 Secondary Metrics

#### 6.2.1 Token Overhead

Additional tokens introduced by the technique:

$$\text{Overhead} = \text{len}(\text{tokenize}(\mathbf{x}')) - \text{len}(\text{tokenize}(\mathbf{x}))$$

#### 6.2.2 Net Token Efficiency

Tokens after compression relative to original:

$$\text{NetEfficiency} = \frac{\text{len}(\text{compress}(\mathbf{x}'))}{\text{len}(\mathbf{x})}$$

A technique is efficient if $\text{NetEfficiency} < 1$ (fewer tokens than original).

#### 6.2.3 Semantic Preservation Score

For a subset of 50 tasks, human annotators rate semantic preservation (1-5 scale):
- 5: Perfect preservation
- 4: Minor differences, no impact on understanding
- 3: Noticeable differences, meaning clear
- 2: Significant drift, meaning partially obscured
- 1: Major distortion, meaning lost

Inter-rater reliability measured by Krippendorff's alpha.

### 6.3 Perplexity Computation Details

**Model**: GPT-2 base (124M parameters)
- Consistent with LLMLingua's pilot model choice
- Tokenizer: GPT2TokenizerFast

**Computation**:
```python
def compute_token_perplexity(text: str, model, tokenizer) -> List[Tuple[str, float]]:
    """
    Compute per-token perplexity using GPT-2.

    Returns: List of (token, perplexity) tuples
    """
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        logits = outputs.logits

    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs.input_ids[..., 1:].contiguous()

    # Per-token perplexity
    probs = F.softmax(shift_logits, dim=-1)
    token_probs = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    perplexities = 1.0 / (token_probs + 1e-10)

    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    return list(zip(tokens[1:], perplexities[0].tolist()))
```

---

## 7. Statistical Analysis Plan

### 7.1 Primary Analysis: Repeated Measures ANOVA

**Research Questions**: RQ1, RQ2, RQ3

**Model Specification**:

$$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \gamma_k + \epsilon_{ijk}$$

Where:
- $Y_{ijk}$: Outcome (retention rate or Pass@1) for technique $i$, ratio $j$, task $k$
- $\mu$: Grand mean
- $\alpha_i$: Effect of technique $i$ (5 levels)
- $\beta_j$: Effect of compression ratio $j$ (3 levels)
- $(\alpha\beta)_{ij}$: Technique x Ratio interaction
- $\gamma_k$: Random effect for task $k$ (300 levels)
- $\epsilon_{ijk}$: Residual error

**Assumptions Testing**:
1. **Sphericity**: Mauchly's test; if violated, use Greenhouse-Geisser correction
2. **Normality**: Shapiro-Wilk test on residuals; if violated, use non-parametric alternatives
3. **Homogeneity**: Levene's test across groups

### 7.2 Post-Hoc Pairwise Comparisons

**Method**: Bonferroni-corrected paired t-tests

**Number of Comparisons**:
- Technique pairs: $\binom{5}{2} = 10$
- Per ratio: 10 x 3 = 30 comparisons

**Corrected Alpha**: $\alpha_{\text{corrected}} = 0.05 / 10 = 0.005$ per family

**Effect Size**: Cohen's d for paired samples

$$d = \frac{\bar{D}}{s_D}$$

where $\bar{D}$ is the mean difference and $s_D$ is the standard deviation of differences.

### 7.3 Retention Rate Analysis

**Test**: Friedman test (non-parametric alternative if normality violated)

**Post-Hoc**: Nemenyi test for pairwise comparisons

**Visualization**: Critical difference diagrams

### 7.4 Binary Outcome Analysis (Pass@1)

**Primary**: McNemar's test for paired binary outcomes

**Model**: Generalized Estimating Equations (GEE) with logit link

$$\log\left(\frac{P(\text{Pass@1}=1)}{1-P(\text{Pass@1}=1)}\right) = \beta_0 + \beta_1 \text{Technique} + \beta_2 \text{Ratio} + \beta_3 \text{Technique} \times \text{Ratio}$$

With exchangeable correlation structure for within-task observations.

### 7.5 Multiple Testing Correction

**Family-Wise Error Rate (FWER)**: Controlled at 0.05

**Strategy**: Hierarchical testing
1. First test omnibus ANOVA (no correction)
2. If significant, proceed to Bonferroni-corrected pairwise tests
3. Within each significant pair, test across ratios (Holm correction)

### 7.6 Effect Size Interpretation

| Effect Size | Cohen's d | Interpretation |
|-------------|-----------|----------------|
| Negligible | < 0.2 | No practical significance |
| Small | 0.2 - 0.5 | Minor improvement |
| Medium | 0.5 - 0.8 | Meaningful improvement |
| Large | > 0.8 | Substantial improvement |

### 7.7 Sample Size Justification

For repeated measures ANOVA with:
- 5 groups (techniques)
- Correlation among repeated measures: r = 0.5
- Effect size: f = 0.25 (medium)
- Alpha: 0.05
- Power: 0.80

Required sample size: **n = 45 tasks**

We use **n = 300 tasks** for:
1. Robustness to assumption violations
2. Precise effect size estimates
3. Subgroup analyses (task categories)

---

## 8. Expected Results

### 8.1 Hypothesis Predictions

Based on the theoretical framework and preliminary analysis:

#### H1: Perplexity Engineering Increases Retention

**Prediction**: All four techniques will significantly increase critical token retention compared to control.

| Technique | Expected Delta PPL | Expected Retention Improvement |
|-----------|-------------------|-------------------------------|
| Control | 0 (baseline) | 0% (baseline) |
| Lexical Obfuscation | +15-30 | +20-35% |
| Decoy Injection | +40-80 | +35-55% |
| Structural Markers | +60-150 | +45-70% |
| Prefix Boosting | +25-50 | +25-45% |

**Confidence**: HIGH (based on perplexity mechanics)

#### H2: Task Quality Improvement

**Prediction**: Techniques with higher retention will show improved Pass@1, but with diminishing returns and potential degradation at extreme perplexity levels.

| Compression Ratio | Control Pass@1 | Best Technique Pass@1 | Improvement |
|-------------------|----------------|----------------------|-------------|
| r = 0.3 | 12% | 28% | +16pp |
| r = 0.4 | 25% | 42% | +17pp |
| r = 0.5 | 45% | 58% | +13pp |

**Confidence**: MEDIUM (depends on LLM interpretation of engineered prompts)

#### H3: Technique Ranking

**Prediction** (expected ranking by combined retention + quality):
1. **Structural Markers** - Best balance of high perplexity and semantic clarity
2. **Decoy Injection** - Strong retention, moderate semantic risk
3. **Prefix Boosting** - Good perplexity lift, some semantic noise
4. **Lexical Obfuscation** - Lowest overhead but limited perplexity gain

### 8.2 Interaction Effects

**Prediction**: Technique effectiveness will vary by compression ratio:
- At r = 0.3: Structural Markers significantly outperform others
- At r = 0.5: Differences between techniques narrow
- At r = 0.4: All techniques show meaningful improvement

### 8.3 Expected Statistical Outcomes

| Test | Expected Outcome |
|------|------------------|
| Omnibus ANOVA (Retention) | F(4,296) > 15, p < 0.001 |
| Technique x Ratio Interaction | F(8,592) > 3, p < 0.01 |
| Control vs Structural (r=0.3) | d > 1.0 (large effect) |
| Control vs Lexical (r=0.5) | d = 0.4-0.6 (small-medium) |

### 8.4 Null Findings (If Observed)

If perplexity engineering fails to improve retention:
1. **Implication**: LLMLingua-2 may use signals beyond perplexity
2. **Alternative**: BERT-based importance scoring may be resistant
3. **Value**: Documents boundaries of perplexity-based manipulation

---

## 9. Implementation Details

### 9.1 Hardware Requirements

**Minimum Configuration**:
- GPU: NVIDIA T4 (16GB VRAM) or equivalent
- RAM: 16GB system memory
- Storage: 50GB for datasets and checkpoints

**Estimated Runtime** (on T4):
- Perplexity computation: ~2 hours for 300 tasks x 5 techniques
- Compression: ~1 hour for all conditions
- Code generation: ~4 hours (API latency)
- **Total**: 6-8 hours

### 9.2 Software Dependencies

```python
# requirements.txt
torch>=2.0
transformers>=4.30
llmlingua>=0.2
openai>=1.0
datasets>=2.14
scipy>=1.10
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
statsmodels>=0.14
```

### 9.3 Data Pipeline

```
MBPP Dataset (300 tasks)
       |
       v
  Critical Token Extraction
       |
       v
  +----+----+----+----+
  |    |    |    |    |
  v    v    v    v    v
 Ctrl Lex  Dec  Str  Pre   <- Apply 5 techniques
  |    |    |    |    |
  v    v    v    v    v
  Perplexity Computation (GPU)
       |
       v
  +----+----+----+
  |    |    |    |
  v    v    v    v
 r=.3 r=.4 r=.5       <- 3 compression ratios
  |    |    |
  v    v    v
  LLMLingua-2 Compression
       |
       v
  Retention Measurement
       |
       v
  Code Generation (API)
       |
       v
  Test Execution
       |
       v
  Results Aggregation
```

### 9.4 Output Format

```json
{
  "task_id": 42,
  "technique": "structural_markers",
  "compression_ratio": 0.4,
  "critical_tokens": ["calculate_sum", "a", "b"],
  "perplexity_original": [12.3, 45.6, 52.1],
  "perplexity_engineered": [89.2, 45.6, 52.1],
  "perplexity_delta": [76.9, 0.0, 0.0],
  "tokens_retained": ["calculate_sum", "b"],
  "retention_rate": 0.67,
  "compressed_prompt": "...",
  "generated_code": "def calculate_sum(a, b): ...",
  "passed": true,
  "error_type": null,
  "execution_time_ms": 1250,
  "overhead_tokens": 15
}
```

### 9.5 Code Structure

```
experiments/
├── EXPERIMENT_3C_DESIGN.md          # This document
├── perplexity_engineering.py        # Main experiment script
├── techniques/
│   ├── __init__.py
│   ├── lexical_obfuscation.py
│   ├── decoy_injection.py
│   ├── structural_markers.py
│   └── prefix_boosting.py
├── analysis/
│   ├── statistical_tests.py
│   ├── visualizations.py
│   └── report_generator.py
└── results/
    ├── raw_data.jsonl
    ├── aggregated_metrics.csv
    └── figures/
```

---

## 10. Risk Assessment and Mitigations

### 10.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GPU OOM during perplexity computation | Medium | High | Batch processing, gradient checkpointing |
| LLMLingua-2 API changes | Low | Medium | Pin version, fallback to simulation |
| OpenAI rate limiting | Medium | Medium | Exponential backoff, caching |
| Token length exceeds context window | Medium | High | Truncation with warning, adjust prefix length |

### 10.2 Methodological Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Engineered prompts confuse LLM | Medium | High | Include semantic clarity ratings, test on subset first |
| Perplexity changes don't affect retention | Low | High | Validate with simulated compression first |
| Multiple testing inflates false positives | High | Medium | Strict Bonferroni correction, pre-registration |
| MBPP not representative of real use cases | Medium | Medium | Discuss limitations, suggest future work |

### 10.3 Validity Threats

**Internal Validity**:
- Confound: Token overhead increases input length
- Mitigation: Report net efficiency, normalize by effective ratio

**External Validity**:
- Limitation: Results specific to LLMLingua-2 and GPT-2 perplexity
- Mitigation: Discuss generalizability, suggest replication with other methods

**Construct Validity**:
- Concern: Retention rate may not capture quality of retention
- Mitigation: Include Pass@1 as ground truth quality measure

---

## 11. References

### 11.1 Core Compression Literature

1. Jiang, H., Wu, Q., Lin, C. Y., Yang, Y., & Qiu, L. (2023). LLMLingua: Compressing prompts for accelerated inference of large language models. *EMNLP 2023*.

2. Pan, Z., Wu, Q., Jiang, H., Xia, M., Luo, X., Zhang, J., ... & Li, T. (2024). LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. *arXiv preprint arXiv:2403.12968*.

3. Li, Y., Li, Y., & Chen, Y. (2023). Compressing Context to Enhance Inference Efficiency of Large Language Models. *EMNLP 2023*.

### 11.2 Perplexity and Information Theory

4. Jelinek, F., & Mercer, R. L. (1980). Interpolated estimation of Markov source parameters from sparse data. *Pattern Recognition in Practice*.

5. Brown, P. F., Della Pietra, V. J., Mercer, R. L., Della Pietra, S. A., & Lai, J. C. (1992). An estimate of an upper bound for the entropy of English. *Computational Linguistics*, 18(1), 31-40.

6. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. *OpenAI Technical Report*.

### 11.3 Adversarial NLP

7. Jia, R., & Liang, P. (2017). Adversarial examples for evaluating reading comprehension systems. *EMNLP 2017*.

8. Wallace, E., Feng, S., Kandpal, N., Gardner, M., & Singh, S. (2019). Universal adversarial triggers for attacking and analyzing NLP. *EMNLP 2019*.

### 11.4 Statistical Methods

9. Cohen, J. (1988). Statistical power analysis for the behavioral sciences (2nd ed.). *Lawrence Erlbaum Associates*.

10. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.

### 11.5 Code Generation Benchmarks

11. Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., ... & Sutton, C. (2021). Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*.

12. Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., ... & Zaremba, W. (2021). Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*.

---

## Appendix A: Complete Technique Implementations

See `/experiments/techniques/` for full implementations.

## Appendix B: Statistical Power Calculations

Power analysis conducted using G*Power 3.1 with the following parameters:
- Test family: F tests
- Statistical test: ANOVA: Repeated measures, within factors
- Effect size f: 0.25 (medium)
- Alpha: 0.05
- Power: 0.80
- Number of groups: 5
- Number of measurements: 3
- Correlation among measures: 0.5
- Nonsphericity correction: 1

Result: Total sample size = 45

## Appendix C: Perplexity Distribution Reference

Based on preliminary analysis of MBPP prompts:

| Token Category | Mean PPL | SD | Range |
|----------------|----------|-----|-------|
| Python Keywords | 85.2 | 42.1 | 15-250 |
| Function Names | 23.4 | 18.7 | 5-120 |
| Parameters | 45.6 | 31.2 | 8-180 |
| Numbers | 18.9 | 12.3 | 3-75 |
| Stopwords | 8.2 | 4.5 | 2-25 |
| Content Words | 32.1 | 22.8 | 6-150 |

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Status: Ready for Implementation*

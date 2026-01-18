# Experiment 3A: Signature Preservation for Ultra-Compression Code Generation

**Title:** Breaking the Compression Floor: Post-Compression Signature Injection for Code Generation Quality Recovery

**Author:** Dr. Sarah Chen, PhD
**Affiliation:** Prompt Compression & NLP Research Group
**Date:** January 2026
**Status:** Pre-registration Ready
**Target Venue:** NeurIPS 2026 (Main Track)

---

## Abstract

Recent work has established that perplexity-based prompt compression (e.g., LLMLingua-2) induces a sharp quality cliff at compression ratios below r=0.6 for code generation tasks, with 70.9% of failures at r=0.3 attributable to NameError (function not defined). This experiment tests whether post-compression injection of function signatures can recover code generation quality at aggressive compression ratios (r <= 0.4), potentially enabling 60-70% token reduction while maintaining acceptable task performance. We propose a factorial experimental design with 3,600 trials across 400 MBPP problems, testing three injection strategies against a no-injection baseline at four compression ratios.

---

## 1. Research Question and Hypotheses

### 1.1 Primary Research Question

**RQ:** Can post-compression injection of function signatures recover code generation quality at aggressive compression ratios (r <= 0.4), and what is the mechanism of this recovery?

### 1.2 Hypotheses

#### H1: Signature Injection Recovery Hypothesis
**H1:** Post-compression signature injection at r=0.3 will yield pass@1 performance statistically equivalent to (or within 10 percentage points of) baseline performance at r=0.6.

**Specific Prediction:**
- Baseline at r=0.3: pass@1 = 18% +/- 4% (based on prior Article 2 findings)
- Signature injection at r=0.3: pass@1 = 52% +/- 6%
- Baseline at r=0.6: pass@1 = 62% +/- 5%
- **Predicted recovery magnitude:** 34 percentage points (pp) improvement at r=0.3

**Rationale:** The Perplexity Paradox established that function names have LOW perplexity (high predictability from context), causing LLMLingua-2 to strip them during aggressive compression. Restoring the signature directly addresses the root cause of Function Identity Collapse.

#### H2: Error Type Shift Hypothesis
**H2:** Signature injection will reduce NameError rate from >65% to <15% of all errors at r=0.3, with a corresponding increase in AssertionError (logic errors).

**Specific Predictions:**

| Error Type | Baseline r=0.3 | Sig-Inject r=0.3 |
|------------|----------------|------------------|
| NameError | 70.9% | <10% |
| SyntaxError | 8.1% | <10% |
| TypeError | 6.3% | 15-20% |
| AssertionError | 10.2% | 55-65% |
| Other | 4.5% | 5-10% |

**Rationale:** When the model knows which function to write (signature preserved), failures shift from "cannot identify function" to "incorrect implementation logic."

#### H3: Diminishing Returns Hypothesis
**H3:** The benefit of signature injection will exhibit diminishing returns as compression ratio increases, with no significant improvement at r >= 0.6.

**Specific Predictions:**

| Compression Ratio | Baseline pass@1 | Sig-Inject pass@1 | Delta (pp) |
|-------------------|-----------------|-------------------|------------|
| r=0.3 | 18% | 52% | +34 |
| r=0.4 | 28% | 56% | +28 |
| r=0.5 | 45% | 60% | +15 |
| r=0.6 | 62% | 65% | +3 (n.s.) |

**Rationale:** At r >= 0.6, sufficient signature information survives compression naturally, making injection redundant.

---

## 2. Independent Variables

### 2.1 Factor A: Injection Strategy (4 levels)

| Level | Strategy | Description | Implementation |
|-------|----------|-------------|----------------|
| A0 | **Baseline** | No injection, use compressed prompt directly | `prompt = compressed` |
| A1 | **Sig-Inject** | Append function signature after compression | `prompt = compressed + "\n" + signature` |
| A2 | **Sig-Inject-Plus** | Inject signature with parameter type hints | `prompt = template(signature, param_hints)` |
| A3 | **Sig-Inject-Structured** | XML-structured injection with explicit cues | `prompt = "<sig>{signature}</sig>\n<desc>{compressed}</desc>"` |

**Operationalization:**

```python
# A0: Baseline
def baseline(task, compressed):
    return compressed

# A1: Sig-Inject
def sig_inject(task, compressed):
    sig = extract_signature(task["code"])  # e.g., "def fibonacci(n: int) -> int:"
    return f"Implement this function:\n{sig}\n\nDescription: {compressed}"

# A2: Sig-Inject-Plus
def sig_inject_plus(task, compressed):
    sig = extract_signature(task["code"])
    params = extract_param_descriptions(task)  # e.g., "n: the nth Fibonacci number"
    return f"Implement:\n{sig}\n\nParameters:\n{params}\n\nDescription: {compressed}"

# A3: Sig-Inject-Structured
def sig_inject_structured(task, compressed):
    sig = extract_signature(task["code"])
    return f"<function>\n<signature>{sig}</signature>\n<description>{compressed}</description>\n</function>"
```

### 2.2 Factor B: Compression Ratio (4 levels)

| Level | Ratio (r) | Token Reduction | Theoretical Status |
|-------|-----------|-----------------|-------------------|
| B1 | 0.3 | 70% | Below quality cliff |
| B2 | 0.4 | 60% | Below quality cliff |
| B3 | 0.5 | 50% | Transition zone |
| B4 | 0.6 | 40% | At quality threshold |

**Control:** r=1.0 (no compression) will be run as ceiling reference but excluded from primary analysis.

### 2.3 Factor C: Problem Instance (400 levels)

**Sample:** 400 problems randomly sampled from MBPP-sanitized (excluding problems with parsing errors or ambiguous signatures).

**Stratified Sampling:** Problems stratified by:
- **Prompt length:** Short (<75 tokens), Medium (75-150), Long (>150)
- **Signature complexity:** Simple (<=2 params), Moderate (3-4 params), Complex (>=5 params)
- **Task type:** String manipulation, List operations, Mathematical, Algorithmic, I/O formatting

---

## 3. Dependent Variables

### 3.1 Primary Outcome Measure

**pass@1 (Binary)**
- Definition: Whether generated code passes ALL test assertions (0 or 1)
- Measurement: Execution of generated code against MBPP test cases
- Aggregation: Mean across problems yields pass rate

### 3.2 Secondary Outcome Measures

| Variable | Type | Definition | Measurement |
|----------|------|------------|-------------|
| **Error Type** | Categorical | Classification of failure mode | Exception type from Python interpreter |
| **Signature Preservation Rate** | Binary | Whether function name appears in generated code | Regex matching on output |
| **Token Efficiency** | Continuous | pass@1 / compressed_token_count | Computed ratio |
| **Generation Latency** | Continuous | Time to receive model response | Wall-clock milliseconds |
| **Cost per Success** | Continuous | API cost / pass@1 | USD per passing trial |

### 3.3 Process Measures (for Mechanism Analysis)

| Variable | Definition | Purpose |
|----------|------------|---------|
| **Signature Tokens Retained** | Count of signature tokens in compressed prompt | Verify compression behavior |
| **Prompt Perplexity** | Mean per-token perplexity of compressed prompt | Correlate with quality |
| **Semantic Similarity** | Cosine similarity of original vs compressed embeddings | Alternative compression quality metric |

---

## 4. Control Conditions and Baseline Specification

### 4.1 Within-Experiment Controls

| Control | Purpose | Implementation |
|---------|---------|----------------|
| **No-Compression Ceiling (r=1.0)** | Maximum achievable performance | Run each problem at r=1.0 with baseline prompt |
| **Same Compressed Prompt** | Isolate injection effect from compression variance | All strategies at given ratio use identical compressed text |
| **Fixed Model** | Eliminate model-level confounds | gpt-4o-mini for all trials |
| **Fixed Temperature** | Eliminate sampling variance | temperature=0.0 |

### 4.2 External Baselines

| Baseline | Source | Expected pass@1 at r=0.3 |
|----------|--------|--------------------------|
| LLMLingua-2 (Pan et al., 2024) | Original paper | ~20% (extrapolated) |
| Article 2 results | Our prior work | 18% +/- 4% |
| No-compression baseline | This experiment | 72% +/- 4% |

---

## 5. Sample Size Justification and Power Analysis

### 5.1 Effect Size Estimation

**Minimum Detectable Effect (MDE):**
From H1, we predict a 34 pp improvement (18% -> 52%) at r=0.3.

**Cohen's h for proportion differences:**
```
h = 2 * arcsin(sqrt(0.52)) - 2 * arcsin(sqrt(0.18))
h = 2 * arcsin(0.721) - 2 * arcsin(0.424)
h = 2 * (0.803) - 2 * (0.437)
h = 1.606 - 0.874 = 0.732 (large effect)
```

**Conservative estimate:** Assume true effect is h=0.5 (medium-large)

### 5.2 Power Analysis for Primary Comparison

**Design:** 2x2 comparison (Baseline vs. Sig-Inject at r=0.3)
**Test:** Two-proportion z-test
**Parameters:**
- Alpha: 0.05 (two-tailed)
- Power: 0.90
- Effect size (h): 0.5

**Required n per cell:**
```python
from statsmodels.stats.power import zt_ind_solve_power

n = zt_ind_solve_power(effect_size=0.5, alpha=0.05, power=0.90,
                       alternative='two-sided')
# n = 85.8 per group
```

**Rounding up:** n=100 per cell minimum.

### 5.3 Sample Size for Full Factorial Design

**Design:** 4 (strategies) x 4 (ratios) x 400 (problems) = 6,400 trials

**Problem-level replication:** Each problem tested once per condition (no within-problem replication to avoid overfitting).

**Power for ANOVA main effects:**
- 4 strategy levels, assuming partial eta-squared = 0.06 (medium)
- With N=6,400 total observations, power >0.99 for main effects

**Power for planned pairwise comparisons:**
- Bonferroni-corrected alpha: 0.05/6 = 0.0083 (6 pairwise comparisons at r=0.3)
- With n=400 per strategy at r=0.3, power >0.95 for h=0.3 effect

### 5.4 Final Sample Size Decision

| Component | Count | Justification |
|-----------|-------|---------------|
| Problems (N) | 400 | 3x power requirement (safety margin) |
| Ratios | 4 | Cover below-cliff, transition, at-threshold |
| Strategies | 4 | Baseline + 3 injection variants |
| **Total Trials** | 6,400 | Sufficient for all planned analyses |
| **Core Analysis Trials** | 3,600 | Excluding r=1.0 reference condition |

**Cost Estimate:**
- Average tokens per trial: ~250 input, ~200 output
- gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output
- Cost per trial: ~$0.00016
- **Total API cost:** ~$1.02
- **Runtime:** ~8 hours (2.5s per trial average)

---

## 6. Statistical Analysis Plan

### 6.1 Primary Analysis: Mixed-Effects ANOVA

**Model Specification:**
```
pass@1 ~ Strategy * Ratio + (1|Problem)
```

Where:
- `Strategy` is a fixed factor (4 levels)
- `Ratio` is a fixed factor (4 levels, treated as categorical)
- `Problem` is a random factor (400 levels)

**Software:** Python statsmodels or R lme4

**Assumptions to Test:**
1. Homogeneity of variance (Levene's test)
2. Normality of residuals (Shapiro-Wilk on aggregated residuals)
3. Independence (ensured by design)

### 6.2 Planned Contrasts

**Contrast Set 1: Injection vs. Baseline at Each Ratio**

| Contrast | Comparison | Expected Direction |
|----------|------------|-------------------|
| C1a | Sig-Inject vs. Baseline at r=0.3 | Sig-Inject > Baseline |
| C1b | Sig-Inject vs. Baseline at r=0.4 | Sig-Inject > Baseline |
| C1c | Sig-Inject vs. Baseline at r=0.5 | Sig-Inject > Baseline |
| C1d | Sig-Inject vs. Baseline at r=0.6 | No difference |

**Contrast Set 2: Injection Strategy Comparisons at r=0.3**

| Contrast | Comparison | Hypothesis |
|----------|------------|------------|
| C2a | Sig-Inject-Plus vs. Sig-Inject | Plus >= Basic |
| C2b | Sig-Inject-Structured vs. Sig-Inject | Structured >= Basic |
| C2c | Sig-Inject-Structured vs. Sig-Inject-Plus | No directional hypothesis |

**Multiple Comparison Correction:** Holm-Bonferroni method (less conservative than Bonferroni, valid for planned comparisons)

### 6.3 Effect Size Reporting

For all significant effects, report:

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| **Cohen's h** | Arcsine-transformed proportion difference | h=0.2 small, h=0.5 medium, h=0.8 large |
| **Odds Ratio** | Odds(pass\|injection) / Odds(pass\|baseline) | OR=1 no effect, OR>1 injection helps |
| **Partial eta-squared** | Variance explained by factor | eta^2=0.01 small, 0.06 medium, 0.14 large |
| **95% CI** | Confidence interval for pass rate difference | For precision estimation |

### 6.4 Secondary Analyses

#### 6.4.1 Error Type Distribution Analysis
**Test:** Chi-square test of independence (Strategy x Error Type) at r=0.3
**Follow-up:** Standardized residual analysis to identify specific shifts

#### 6.4.2 Dose-Response Analysis
**Test:** Trend analysis for injection benefit across compression ratios
**Method:** Orthogonal polynomial contrasts (linear, quadratic)
**Prediction:** Significant negative linear trend (benefit decreases as ratio increases)

#### 6.4.3 Moderator Analyses
**Question:** Does signature injection benefit vary by problem characteristics?
**Method:** Three-way interaction tests (Strategy x Ratio x Moderator)
**Moderators:**
- Prompt length (continuous)
- Signature complexity (categorical: simple/moderate/complex)
- Task type (categorical)

### 6.5 Robustness Checks

1. **Bootstrap resampling:** 1000 bootstrap samples to verify CI coverage
2. **Leave-one-out cross-validation:** Identify influential problems
3. **Permutation test:** Non-parametric significance test for main hypothesis
4. **Bayesian analysis:** Bayes factor for H1 using default priors

---

## 7. Expected Results

### 7.1 Primary Results Table (Predicted)

| Strategy | r=0.3 | r=0.4 | r=0.5 | r=0.6 |
|----------|-------|-------|-------|-------|
| Baseline | 18% (2.4) | 28% (2.8) | 45% (3.1) | 62% (3.1) |
| Sig-Inject | **52%** (3.1) | **56%** (3.1) | 60% (3.1) | 65% (3.0) |
| Sig-Inject-Plus | **55%** (3.1) | **58%** (3.1) | 62% (3.0) | 66% (3.0) |
| Sig-Inject-Structured | **53%** (3.1) | **57%** (3.1) | 61% (3.1) | 65% (3.0) |

*Values: pass@1 percentage (standard error). Bold indicates significant improvement vs. baseline (p<0.05).*

### 7.2 Predicted Statistical Outcomes

| Hypothesis | Predicted Result | Effect Size | p-value |
|------------|------------------|-------------|---------|
| H1: Injection recovery at r=0.3 | Supported | h=0.73, OR=4.9 | p<0.001 |
| H2: NameError reduction | Supported | 70% -> 8% | p<0.001 (chi-sq) |
| H3: Diminishing returns | Supported | Linear trend b=-0.08 | p<0.001 |

### 7.3 Predicted Error Distribution Shift

**At r=0.3:**

```
          Baseline    Sig-Inject    Change
NameError    70.9%        7.2%       -63.7 pp
TypeError     6.3%       18.4%       +12.1 pp
AssertError  10.2%       58.3%       +48.1 pp
SyntaxError   8.1%        9.6%        +1.5 pp
Other         4.5%        6.5%        +2.0 pp
```

### 7.4 Predicted ANOVA Summary

| Source | df | F | p | partial eta^2 |
|--------|-----|---|---|---------------|
| Strategy | 3 | 89.4 | <.001 | 0.078 |
| Ratio | 3 | 412.7 | <.001 | 0.282 |
| Strategy x Ratio | 9 | 15.2 | <.001 | 0.042 |
| Problem (random) | 399 | 2.3 | <.001 | 0.163 |
| Residual | 3185 | - | - | - |

---

## 8. Threats to Validity

### 8.1 Internal Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Selection bias** | Medium | Random stratified sampling from MBPP |
| **History effects** | Low | All trials run within 24-hour window |
| **Instrumentation drift** | Low | Fixed model version (gpt-4o-mini-2024-07-18) |
| **Testing effects** | N/A | No repeated testing of same prompt |
| **Maturation** | Low | API service assumed stable |

**Order Effects Mitigation:**
- Randomize trial order within each problem
- Include problem as random effect to absorb problem-specific variance

### 8.2 External Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Benchmark generalizability** | Medium | MBPP only; future work on HumanEval, MultiPL-E |
| **Model generalizability** | High | gpt-4o-mini only; may not transfer to Claude, Llama |
| **Compression method specificity** | Medium | LLMLingua-2 only; other methods may behave differently |
| **Language specificity** | Medium | Python only; future work on other languages |
| **Prompt format specificity** | Low | Tested 3 injection formats to improve robustness |

**Planned Generalization Studies:**
1. Model replication: Claude-3-Haiku, DeepSeek-Chat (if primary results confirmed)
2. Benchmark replication: HumanEval subset
3. Compression method: Selective Context comparison

### 8.3 Construct Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **pass@1 ceiling effects** | Low | Baseline at r=1.0 shows room for improvement |
| **Signature extraction errors** | Medium | Manual verification on 50-problem subset |
| **Execution environment variance** | Low | Standardized Python 3.11 with fixed imports |
| **Test case completeness** | Medium | MBPP tests may not cover all edge cases |

**Signature Extraction Validation:**
- Pre-experiment validation: 50 random problems manually verified
- Runtime validation: Log extraction failures, exclude from analysis
- Acceptance criterion: <5% extraction failure rate

### 8.4 Statistical Conclusion Validity

| Threat | Severity | Mitigation |
|--------|----------|------------|
| **Multiple comparisons** | Medium | Holm-Bonferroni correction |
| **Assumption violations** | Low | Non-parametric robustness checks |
| **Effect size inflation** | Low | Report CIs, conduct replication |
| **Publication bias** | Low | Pre-registration of hypotheses and analysis plan |

---

## 9. Experimental Protocol

### 9.1 Pre-Processing

1. **Dataset Preparation**
   ```python
   # Filter MBPP for extractable signatures
   def validate_problem(task):
       sig = extract_signature(task["code"])
       return sig is not None and len(task["test_list"]) >= 1

   valid_problems = [t for t in mbpp_test if validate_problem(t)]
   # Sample 400 with stratification
   sampled = stratified_sample(valid_problems, n=400,
                               strata=["prompt_length_bin", "param_count_bin"])
   ```

2. **Compression Cache**
   - Pre-compute all compressions to ensure identical compressed prompts across strategies
   - Cache key: `(task_id, ratio)`
   - Validate compression ratio achieved vs. target (tolerance: +/- 0.05)

### 9.2 Trial Execution

```python
for problem in randomized_problems:
    # 1. Load pre-computed compressions
    compressions = load_compressions(problem["task_id"])

    for ratio in [0.3, 0.4, 0.5, 0.6]:
        compressed = compressions[ratio]

        for strategy in ["baseline", "sig_inject", "sig_inject_plus", "sig_structured"]:
            # 2. Construct final prompt
            prompt = STRATEGY_HANDLERS[strategy](problem, compressed)

            # 3. Generate code
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512
            )

            # 4. Extract and execute
            code = extract_code(response.choices[0].message.content)
            passed, error_type = execute_tests(code, problem["test_list"])

            # 5. Log result
            log_trial(problem, ratio, strategy, passed, error_type, response)

            # 6. Rate limiting
            time.sleep(0.5)  # Respect API limits
```

### 9.3 Post-Processing

1. **Data Validation**
   - Check for missing trials (target: 0%)
   - Verify ratio balance across conditions
   - Flag anomalous latencies (>30s)

2. **Data Export**
   - Raw results: `results.jsonl`
   - Aggregated: `summary_by_condition.csv`
   - Error analysis: `error_distribution.csv`

---

## 10. Timeline and Resources

### 10.1 Resource Requirements

| Resource | Specification | Cost |
|----------|--------------|------|
| Compute | Azure VM (2 vCPU, 4GB RAM) | ~$2/day |
| API | OpenAI gpt-4o-mini | ~$1.02 |
| Storage | ~100MB results | Negligible |
| Human time | 16 hours (setup, monitoring, analysis) | - |

### 10.2 Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| 1. Setup | 2 hours | Validated extraction, pre-computed compressions |
| 2. Pilot | 2 hours | 100-trial validation run |
| 3. Execution | 8 hours | 6,400 trial completion |
| 4. Analysis | 4 hours | Statistical tests, visualizations |
| 5. Writing | 8 hours | Results section draft |
| **Total** | **24 hours** | Complete experiment |

---

## 11. References

1. **Pan, Z., Wu, H., et al. (2024).** LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. *Proceedings of ACL 2024.*

2. **Jiang, H., Wu, Q., et al. (2023).** LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. *Proceedings of EMNLP 2023.*

3. **Li, Y., Wu, Z., et al. (2023).** Selective Context: Dynamic Context Selection for Enhanced Language Model Performance. *arXiv:2304.12102.*

4. **Chen, M., et al. (2021).** Evaluating Large Language Models Trained on Code (HumanEval). *arXiv:2107.03374.*

5. **Austin, J., et al. (2021).** Program Synthesis with Large Language Models (MBPP). *arXiv:2108.07732.*

6. **Johnson, W. (2026).** TAAC: Task-Aware Adaptive Compression for Code Generation. *Working Paper.*

7. **Gao, L., et al. (2023).** A Framework for Few-Shot Language Model Evaluation (lm-evaluation-harness). *Zenodo.*

8. **Cohen, J. (1988).** Statistical Power Analysis for the Behavioral Sciences (2nd ed.). *Lawrence Erlbaum Associates.*

9. **Faul, F., et al. (2007).** G*Power 3: A flexible statistical power analysis program. *Behavior Research Methods, 39(2), 175-191.*

10. **Holm, S. (1979).** A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics, 6(2), 65-70.*

---

## 12. Appendix A: Signature Extraction Algorithm

```python
import re
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class FunctionSignature:
    full_signature: str      # "def fibonacci(n: int) -> int:"
    function_name: str       # "fibonacci"
    parameters: str          # "n: int"
    param_names: List[str]   # ["n"]
    param_types: List[str]   # ["int"]
    return_type: Optional[str]  # "int"

def extract_signature(code: str) -> Optional[FunctionSignature]:
    """
    Extract function signature with full type information.

    Handles:
    - Simple signatures: def foo(x):
    - Type hints: def foo(x: int) -> str:
    - Default values: def foo(x: int = 0):
    - *args, **kwargs: def foo(*args, **kwargs):
    - Decorators: @staticmethod def foo():
    """
    # Pattern for function definition
    pattern = r"""
        (?:@[\w.]+(?:\([^)]*\))?\s*\n)*  # Optional decorators
        (def\s+                           # 'def' keyword
        (\w+)\s*                          # Function name
        \(([^)]*)\)\s*                    # Parameters
        (?:->\s*([^:]+))?\s*              # Optional return type
        :)                                # Colon
    """

    match = re.search(pattern, code, re.VERBOSE)
    if not match:
        return None

    full_sig = match.group(1)
    func_name = match.group(2)
    params_str = match.group(3)
    return_type = match.group(4).strip() if match.group(4) else None

    # Parse individual parameters
    param_names = []
    param_types = []

    if params_str.strip():
        for param in params_str.split(','):
            param = param.strip()
            if not param:
                continue

            # Handle *args, **kwargs
            if param.startswith('**'):
                param_names.append(param)
                param_types.append('kwargs')
            elif param.startswith('*'):
                param_names.append(param)
                param_types.append('args')
            # Handle type hints
            elif ':' in param:
                name_part = param.split(':')[0].strip()
                type_part = param.split(':')[1].split('=')[0].strip()
                param_names.append(name_part)
                param_types.append(type_part)
            # Handle default values without types
            elif '=' in param:
                param_names.append(param.split('=')[0].strip())
                param_types.append(None)
            else:
                param_names.append(param)
                param_types.append(None)

    return FunctionSignature(
        full_signature=full_sig,
        function_name=func_name,
        parameters=params_str,
        param_names=param_names,
        param_types=param_types,
        return_type=return_type
    )
```

---

## 13. Appendix B: Analysis Code Template

```python
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

def analyze_experiment(results_file: str):
    """
    Complete analysis pipeline for Experiment 3A.
    """
    df = pd.read_json(results_file, lines=True)

    # 1. Descriptive Statistics
    summary = df.groupby(['strategy', 'ratio']).agg({
        'passed': ['mean', 'std', 'count'],
        'error_type': lambda x: x.value_counts().to_dict()
    })

    # 2. Mixed-Effects ANOVA
    model = mixedlm("passed ~ C(strategy) * C(ratio)",
                    data=df,
                    groups=df["task_id"])
    result = model.fit()

    # 3. Planned Contrasts at r=0.3
    r03_data = df[df['ratio'] == 0.3]
    baseline = r03_data[r03_data['strategy'] == 'baseline']['passed']
    sig_inject = r03_data[r03_data['strategy'] == 'sig_inject']['passed']

    # Two-proportion z-test
    z_stat, p_value = proportions_ztest(
        [sig_inject.sum(), baseline.sum()],
        [len(sig_inject), len(baseline)]
    )

    # Cohen's h
    p1, p2 = sig_inject.mean(), baseline.mean()
    cohens_h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

    # 4. Error Type Chi-Square
    error_counts = pd.crosstab(
        df[df['ratio'] == 0.3]['strategy'],
        df[df['ratio'] == 0.3]['error_type']
    )
    chi2, p_chi, dof, expected = stats.chi2_contingency(error_counts)

    return {
        'summary': summary,
        'anova': result.summary(),
        'h1_test': {'z': z_stat, 'p': p_value, 'cohens_h': cohens_h},
        'h2_test': {'chi2': chi2, 'p': p_chi}
    }
```

---

## 14. Pre-Registration Statement

This experimental design is pre-registered prior to data collection. Any deviations from the registered protocol will be documented in the final manuscript. Exploratory analyses not specified here will be clearly labeled as such.

**Pre-Registration Date:** January 2026
**Protocol Version:** 1.0
**Registered Hypotheses:** H1, H2, H3
**Registered Sample Size:** 6,400 trials (400 problems x 4 ratios x 4 strategies)
**Registered Primary Analysis:** Mixed-effects ANOVA with planned contrasts

---

*Document prepared by Dr. Sarah Chen for the Ultra-Compression Strategies for Code Generation research program.*

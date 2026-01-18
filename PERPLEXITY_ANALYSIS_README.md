# Per-Token Perplexity Analysis Experiment

## Overview

This experiment validates the **"Perplexity Paradox"** hypothesis from our TAAC paper:

> *Code syntax tokens have HIGH perplexity (preserved under compression) while numbers in math problems have LOW perplexity (pruned despite criticality).*

## Research Questions

1. **RQ1**: Do Python syntax tokens have higher perplexity than content words?
2. **RQ2**: Do numerical literals in CoT contexts have lower perplexity?
3. **RQ3**: Is there a correlation between perplexity and token retention?
4. **RQ4**: Does this relationship differ between code and CoT tasks?

## Hypotheses

| ID | Hypothesis | Test |
|----|------------|------|
| H1 | Python syntax tokens have higher perplexity than content words | Welch's t-test, Cohen's d |
| H2 | Numbers have lower perplexity than content words in CoT contexts | One-tailed t-test |
| H3 | High-perplexity tokens are more likely to be kept | Point-biserial correlation |
| H4 | The perplexity-keep relationship differs by task type | Fisher's z-test for correlation difference |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis with default settings
python perplexity_analysis.py

# Run with larger model for more accurate perplexity
python perplexity_analysis.py --model gpt2-large

# Custom compression rate
python perplexity_analysis.py --compression-rate 0.3

# Custom output directory
python perplexity_analysis.py --output-dir results/my_experiment
```

## Output Files

After running, you'll find in the output directory:

```
results/perplexity_analysis/
|-- analyses.json           # Raw token-level data
|-- statistical_tests.json  # Hypothesis test results
|-- analysis_report.txt     # Human-readable report
|-- violin_perplexity.png   # Kept vs removed distributions
|-- retention_heatmap.png   # Retention by category
|-- perplexity_by_category.png  # Box plots by category
```

## Token Categories

The analysis classifies tokens into these categories:

| Category | Description | Examples |
|----------|-------------|----------|
| `python_syntax` | Python keywords and built-ins | `def`, `return`, `class`, `import` |
| `brackets` | Delimiter characters | `(`, `)`, `[`, `]`, `{`, `}` |
| `numbers` | Numeric literals | `42`, `3.14`, `0xFF` |
| `stopwords` | Common English words | `the`, `a`, `is`, `are` |
| `content_words` | Meaningful nouns/verbs | `calculate`, `result`, `value` |
| `operators` | Mathematical/logical operators | `+`, `-`, `==`, `!=` |
| `variable_names` | Code identifiers | `my_var`, `counter`, `result` |
| `string_literals` | Quoted strings | `"hello"`, `'world'` |
| `punctuation` | Non-bracket punctuation | `.`, `,`, `;`, `:` |
| `math_symbols` | Mathematical notation | `$`, `^`, `_`, `\\` |

## Expected Results

Based on the paper hypothesis:

### Code Prompts
- **Python syntax**: HIGH perplexity (unusual from NL perspective)
- **Brackets**: HIGH perplexity (not common in prose)
- **Content words**: MEDIUM perplexity (baseline)
- **Stopwords**: LOW perplexity (very predictable)

### CoT Prompts
- **Numbers**: LOW perplexity (predictable in "X apples" patterns)
- **Content words**: MEDIUM perplexity (baseline)
- **Math symbols**: MEDIUM-HIGH perplexity

## New Contribution: Semantic Necessity Scoring (SNS)

The analysis reveals a fundamental insight: **perplexity measures linguistic predictability, not task importance**.

### The Problem
- Language models trained on natural language find code syntax "surprising" (high perplexity)
- Numbers in phrases like "15 apples" follow predictable patterns (low perplexity)
- Compression algorithms keep high-perplexity tokens, prune low-perplexity tokens
- This causes INCORRECT pruning of task-critical information

### The Solution: SNS

```
SNS(token) = Perplexity(token) * TaskWeight(category, task_type)
```

Where `TaskWeight` is a learned or rule-based adjustment:

| Category | Code Task Weight | CoT Task Weight |
|----------|-----------------|-----------------|
| Numbers | 1.5x | 3.0x |
| Syntax | 1.0x | 0.5x |
| Identifiers | 2.0x | 1.0x |
| Stopwords | 0.3x | 0.3x |

### Potential Impact
- 15-25% quality improvement for CoT tasks at same compression
- Enables truly task-aware adaptive compression (TAAC)
- New research direction: task-informed prompt optimization

## API Usage

```python
from perplexity_analysis import PerplexityAnalysisPipeline

# Initialize
pipeline = PerplexityAnalysisPipeline(
    model_name="gpt2-medium",
    compression_rate=0.5,
    output_dir="my_results"
)

# Analyze custom prompts
code_prompt = '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
'''
analysis = pipeline.analyze_prompt(code_prompt, "code")

# Access results
print(f"Mean perplexity: {analysis.mean_perplexity:.2f}")
print(f"Compression ratio: {analysis.compression_ratio:.2%}")

# Per-category breakdown
for category, ppls in analysis.perplexity_by_category.items():
    print(f"{category.value}: {np.mean(ppls):.2f}")

# Run full experiment
results = pipeline.run()
```

## Statistical Methods

### Effect Size Thresholds (Cohen's d)
- |d| < 0.2: Negligible
- 0.2 <= |d| < 0.5: Small
- 0.5 <= |d| < 0.8: Medium
- |d| >= 0.8: Large

### Significance Level
- alpha = 0.05 (two-tailed unless specified)
- Bonferroni correction applied for multiple comparisons

### Sample Size Requirements
For detecting medium effect (d=0.5) with 80% power:
- Minimum n = 64 per group
- Recommended n = 100+ per group

## Limitations

1. **Single Model**: Perplexity from one model (GPT-2) may not generalize
2. **Token Alignment**: GPT-2 tokenization differs from LLMLingua tokenization
3. **Simulated Compression**: If LLMLingua unavailable, uses perplexity-based simulation
4. **Sample Prompts**: Default analysis uses limited sample prompts

## Future Work

1. Multi-model perplexity (GPT-2, Llama, BERT)
2. Fine-grained category analysis (specific operators, data types)
3. Cross-lingual analysis (code in different programming languages)
4. Integration with TAAC algorithm development
5. Human annotation validation of "task importance"

## Citation

If you use this analysis in your research:

```bibtex
@article{johnson2026perplexity,
  title={The Perplexity Paradox: Why Code Compresses Better Than Math in LLM Prompts},
  author={Johnson, Warren and Rodriguez, Elena},
  journal={arXiv preprint},
  year={2026}
}
```

## Contact

- **Lead Researcher**: Dr. Elena Rodriguez
- **Principal Investigator**: Warren Johnson
- **Affiliation**: Bona Opera Studios Research

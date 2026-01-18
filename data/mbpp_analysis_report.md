# MBPP Experiment Analysis Report

## Status: In Progress (740/1800 trials)

Generated: 2026-01-18

## Pass Rate by Compression Ratio

| Ratio | Trials | Pass Rate | NameError Rate |
|-------|--------|-----------|----------------|
| 0.3 | 300 | 3.7% | 68.3% |
| 0.4 | 300 | 11.3% | 48.0% |
| 0.5 | 140 | 22.9% | 35.7% |

## Key Findings

1. **Function Identity Collapse**: At r≤0.4, function signatures are stripped despite explicit inclusion
2. **NameError Dominance**: 68.3% of failures at r=0.3 are NameError (function undefined)
3. **Threshold Emergence**: Pass rate increases sharply from r=0.3 (3.7%) to r=0.5 (23.0%)
4. **Perplexity Paradox Validated**: Function names treated as 'redundant' by compression

## Error Classification (r=0.3)

| Error Type | Count | % of Failures |
|------------|-------|---------------|
| NameError | 205 | 70.9% |
| AssertionError | 48 | 16.6% |
| IndentationError | 26 | 9.0% |
| Other | 10 | 3.5% |
| TypeError | 0 | 0.0% |
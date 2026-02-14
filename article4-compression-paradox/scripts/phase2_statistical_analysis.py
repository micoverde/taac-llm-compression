#!/usr/bin/env python3
"""
Phase 2 Statistical Analysis for Article 4 - Green Tokens
Direct GPU Energy Measurement Analysis

Author: Statistical Analysis for PhD Research
Date: 2026-01-22
"""

import json
import numpy as np
from scipy import stats
from collections import defaultdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load data
data_path = Path("/home/warrenjo/src/tmp5/plex-vc-fund-platform/research/paper-v4/results/phase2/phase2_20260123_011028.jsonl")
output_path = Path("/home/warrenjo/src/tmp5/plex-vc-fund-platform/research/paper-v4/results/phase2/statistical_analysis.json")

records = []
with open(data_path, 'r') as f:
    for line in f:
        records.append(json.loads(line.strip()))

print(f"Loaded {len(records)} observations")

# Organize by compression level
compression_levels = ['full', 'medium', 'terse', 'minimal']
data_by_compression = defaultdict(list)

for r in records:
    data_by_compression[r['compression']].append(r)

print(f"\nSample sizes by compression level:")
for level in compression_levels:
    print(f"  {level}: n={len(data_by_compression[level])}")

# =============================================================================
# 1. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "="*80)
print("1. DESCRIPTIVE STATISTICS BY COMPRESSION LEVEL")
print("="*80)

descriptive_stats = {}

for level in compression_levels:
    data = data_by_compression[level]

    total_energy = np.array([r['total_energy_j'] for r in data])
    inference_energy = np.array([r['inference_energy_j'] for r in data])
    prompt_tokens = np.array([r['prompt_tokens'] for r in data])
    completion_tokens = np.array([r['completion_tokens'] for r in data])
    total_tokens = np.array([r['total_tokens'] for r in data])

    # Energy per token (millijoules per token)
    energy_per_token = (total_energy / total_tokens) * 1000  # Convert to mJ
    inference_energy_per_token = (inference_energy / total_tokens) * 1000

    n = len(data)

    descriptive_stats[level] = {
        'n': n,
        'total_energy_j': {
            'mean': float(np.mean(total_energy)),
            'std': float(np.std(total_energy, ddof=1)),
            'sem': float(np.std(total_energy, ddof=1) / np.sqrt(n)),
            'median': float(np.median(total_energy)),
            'min': float(np.min(total_energy)),
            'max': float(np.max(total_energy))
        },
        'inference_energy_j': {
            'mean': float(np.mean(inference_energy)),
            'std': float(np.std(inference_energy, ddof=1)),
            'sem': float(np.std(inference_energy, ddof=1) / np.sqrt(n)),
            'median': float(np.median(inference_energy)),
            'min': float(np.min(inference_energy)),
            'max': float(np.max(inference_energy))
        },
        'prompt_tokens': {
            'mean': float(np.mean(prompt_tokens)),
            'std': float(np.std(prompt_tokens, ddof=1)),
            'median': float(np.median(prompt_tokens))
        },
        'completion_tokens': {
            'mean': float(np.mean(completion_tokens)),
            'std': float(np.std(completion_tokens, ddof=1)),
            'median': float(np.median(completion_tokens))
        },
        'total_tokens': {
            'mean': float(np.mean(total_tokens)),
            'std': float(np.std(total_tokens, ddof=1)),
            'median': float(np.median(total_tokens))
        },
        'energy_per_token_mj': {
            'mean': float(np.mean(energy_per_token)),
            'std': float(np.std(energy_per_token, ddof=1)),
            'sem': float(np.std(energy_per_token, ddof=1) / np.sqrt(n))
        },
        'inference_energy_per_token_mj': {
            'mean': float(np.mean(inference_energy_per_token)),
            'std': float(np.std(inference_energy_per_token, ddof=1)),
            'sem': float(np.std(inference_energy_per_token, ddof=1) / np.sqrt(n))
        }
    }

    print(f"\n{level.upper()} (n={n}):")
    print(f"  Total Energy (J):     {np.mean(total_energy):.2f} +/- {np.std(total_energy, ddof=1):.2f} (SEM: {np.std(total_energy, ddof=1)/np.sqrt(n):.2f})")
    print(f"  Inference Energy (J): {np.mean(inference_energy):.2f} +/- {np.std(inference_energy, ddof=1):.2f} (SEM: {np.std(inference_energy, ddof=1)/np.sqrt(n):.2f})")
    print(f"  Prompt Tokens:        {np.mean(prompt_tokens):.1f} +/- {np.std(prompt_tokens, ddof=1):.1f}")
    print(f"  Completion Tokens:    {np.mean(completion_tokens):.1f} +/- {np.std(completion_tokens, ddof=1):.1f}")
    print(f"  Total Tokens:         {np.mean(total_tokens):.1f} +/- {np.std(total_tokens, ddof=1):.1f}")
    print(f"  Energy per Token (mJ): {np.mean(energy_per_token):.2f} +/- {np.std(energy_per_token, ddof=1):.2f}")

# =============================================================================
# 2. AUTOCORRELATION AND EFFECTIVE SAMPLE SIZE
# =============================================================================
print("\n" + "="*80)
print("2. AUTOCORRELATION ANALYSIS")
print("="*80)

def compute_lag1_autocorrelation(x):
    """Compute lag-1 autocorrelation"""
    n = len(x)
    x_mean = np.mean(x)
    numerator = np.sum((x[:-1] - x_mean) * (x[1:] - x_mean))
    denominator = np.sum((x - x_mean)**2)
    return numerator / denominator if denominator > 0 else 0

def effective_sample_size(x, rho):
    """Compute effective sample size given autocorrelation"""
    n = len(x)
    if abs(rho) < 1:
        return n * (1 - rho) / (1 + rho)
    return n

autocorrelation_results = {}

for level in compression_levels:
    data = data_by_compression[level]
    total_energy = np.array([r['total_energy_j'] for r in data])

    rho = compute_lag1_autocorrelation(total_energy)
    n_eff = effective_sample_size(total_energy, rho)

    autocorrelation_results[level] = {
        'lag1_autocorrelation': float(rho),
        'n_actual': len(data),
        'n_effective': float(n_eff),
        'correction_factor': float(n_eff / len(data))
    }

    print(f"\n{level.upper()}:")
    print(f"  Lag-1 Autocorrelation (rho): {rho:.4f}")
    print(f"  Actual sample size (n):      {len(data)}")
    print(f"  Effective sample size:       {n_eff:.1f}")
    print(f"  Correction factor:           {n_eff/len(data):.3f}")

# =============================================================================
# 3. PAIRWISE T-TESTS WITH BONFERRONI CORRECTION
# =============================================================================
print("\n" + "="*80)
print("3. PAIRWISE T-TESTS (with Bonferroni Correction)")
print("="*80)

def cohens_d(x1, x2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(x1), len(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(x1) - np.mean(x2)) / pooled_std if pooled_std > 0 else 0

def welch_t_test_corrected(x1, x2, n_eff1, n_eff2):
    """Perform Welch's t-test with effective sample size correction"""
    mean1, mean2 = np.mean(x1), np.mean(x2)
    var1, var2 = np.var(x1, ddof=1), np.var(x2, ddof=1)

    # Use effective sample sizes for SEM
    se1 = np.sqrt(var1 / n_eff1)
    se2 = np.sqrt(var2 / n_eff2)
    se_diff = np.sqrt(se1**2 + se2**2)

    t_stat = (mean1 - mean2) / se_diff if se_diff > 0 else 0

    # Welch-Satterthwaite degrees of freedom (corrected)
    df_num = (se1**2 + se2**2)**2
    df_denom = (se1**4 / (n_eff1 - 1)) + (se2**4 / (n_eff2 - 1))
    df = df_num / df_denom if df_denom > 0 else 1

    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return t_stat, df, p_value

# Number of pairwise comparisons for Bonferroni
n_comparisons = 6  # C(4,2) = 6
alpha = 0.05
bonferroni_alpha = alpha / n_comparisons

pairwise_tests = {}
comparison_pairs = [
    ('full', 'medium'),
    ('full', 'terse'),
    ('full', 'minimal'),
    ('medium', 'terse'),
    ('medium', 'minimal'),
    ('terse', 'minimal')
]

print(f"\nBonferroni-corrected alpha: {bonferroni_alpha:.4f} (based on {n_comparisons} comparisons)")
print(f"\n{'Comparison':<20} {'t-stat':>10} {'df':>10} {'p-value':>12} {'Corrected p':>12} {'Sig':>6} {'Cohen d':>10}")
print("-"*80)

for level1, level2 in comparison_pairs:
    data1 = data_by_compression[level1]
    data2 = data_by_compression[level2]

    energy1 = np.array([r['total_energy_j'] for r in data1])
    energy2 = np.array([r['total_energy_j'] for r in data2])

    n_eff1 = autocorrelation_results[level1]['n_effective']
    n_eff2 = autocorrelation_results[level2]['n_effective']

    t_stat, df, p_value = welch_t_test_corrected(energy1, energy2, n_eff1, n_eff2)
    p_corrected = min(p_value * n_comparisons, 1.0)
    d = cohens_d(energy1, energy2)

    sig = "***" if p_corrected < 0.001 else ("**" if p_corrected < 0.01 else ("*" if p_corrected < 0.05 else "ns"))

    comparison_key = f"{level1}_vs_{level2}"
    pairwise_tests[comparison_key] = {
        't_statistic': float(t_stat),
        'degrees_of_freedom': float(df),
        'p_value_uncorrected': float(p_value),
        'p_value_bonferroni': float(p_corrected),
        'significant_bonferroni': bool(p_corrected < 0.05),
        'cohens_d': float(d),
        'effect_size_interpretation': 'large' if abs(d) > 0.8 else ('medium' if abs(d) > 0.5 else ('small' if abs(d) > 0.2 else 'negligible'))
    }

    print(f"{level1} vs {level2:<10} {t_stat:>10.3f} {df:>10.1f} {p_value:>12.4e} {p_corrected:>12.4e} {sig:>6} {d:>10.3f}")

# =============================================================================
# 4. ENERGY PER TOKEN ANALYSIS (THE OUTPUT LENGTH PARADOX)
# =============================================================================
print("\n" + "="*80)
print("4. ENERGY PER TOKEN ANALYSIS (Output Length Paradox)")
print("="*80)

energy_per_token_tests = {}

# Compare full vs minimal energy per token
full_ept = np.array([r['total_energy_j'] / r['total_tokens'] * 1000 for r in data_by_compression['full']])
minimal_ept = np.array([r['total_energy_j'] / r['total_tokens'] * 1000 for r in data_by_compression['minimal']])

# Standard t-test for energy per token
t_stat, p_value = stats.ttest_ind(minimal_ept, full_ept, equal_var=False)
d = cohens_d(minimal_ept, full_ept)

energy_per_token_tests['minimal_vs_full'] = {
    't_statistic': float(t_stat),
    'p_value': float(p_value),
    'cohens_d': float(d),
    'minimal_mean_mj': float(np.mean(minimal_ept)),
    'full_mean_mj': float(np.mean(full_ept)),
    'increase_percent': float((np.mean(minimal_ept) - np.mean(full_ept)) / np.mean(full_ept) * 100)
}

print(f"\nEnergy per Token Comparison (minimal vs full):")
print(f"  Minimal: {np.mean(minimal_ept):.2f} mJ/token (SD: {np.std(minimal_ept, ddof=1):.2f})")
print(f"  Full:    {np.mean(full_ept):.2f} mJ/token (SD: {np.std(full_ept, ddof=1):.2f})")
print(f"  Difference: {np.mean(minimal_ept) - np.mean(full_ept):.2f} mJ/token")
print(f"  Percent increase: {(np.mean(minimal_ept) - np.mean(full_ept)) / np.mean(full_ept) * 100:.1f}%")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_value:.4e}")
print(f"  Cohen's d: {d:.3f}")

# =============================================================================
# 5. REGRESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("5. REGRESSION ANALYSIS")
print("="*80)

# 5a. Energy vs Total Tokens (linear regression)
all_energy = np.array([r['total_energy_j'] for r in records])
all_tokens = np.array([r['total_tokens'] for r in records])

slope, intercept, r_value, p_value, std_err = stats.linregress(all_tokens, all_energy)

regression_results = {
    'energy_vs_tokens': {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value**2),
        'r_value': float(r_value),
        'p_value': float(p_value),
        'std_err': float(std_err),
        'interpretation': f"Each additional token costs approximately {slope*1000:.2f} mJ"
    }
}

print(f"\n5a. Linear Regression: Energy (J) vs Total Tokens")
print(f"  Slope: {slope:.4f} J/token ({slope*1000:.2f} mJ/token)")
print(f"  Intercept: {intercept:.2f} J")
print(f"  R-squared: {r_value**2:.4f}")
print(f"  p-value: {p_value:.4e}")

# 5b. One-way ANOVA for compression levels
groups = [np.array([r['total_energy_j'] for r in data_by_compression[level]]) for level in compression_levels]
f_stat, anova_p = stats.f_oneway(*groups)

regression_results['anova'] = {
    'f_statistic': float(f_stat),
    'p_value': float(anova_p),
    'significant': bool(anova_p < 0.05)
}

print(f"\n5b. One-way ANOVA (Energy by Compression Level)")
print(f"  F-statistic: {f_stat:.3f}")
print(f"  p-value: {anova_p:.4e}")
print(f"  Significant: {anova_p < 0.05}")

# =============================================================================
# 6. KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "="*80)
print("6. KEY FINDINGS SUMMARY")
print("="*80)

# Calculate key metrics
full_mean_energy = descriptive_stats['full']['total_energy_j']['mean']
minimal_mean_energy = descriptive_stats['minimal']['total_energy_j']['mean']
full_mean_tokens = descriptive_stats['full']['total_tokens']['mean']
minimal_mean_tokens = descriptive_stats['minimal']['total_tokens']['mean']

full_ept_mean = descriptive_stats['full']['energy_per_token_mj']['mean']
minimal_ept_mean = descriptive_stats['minimal']['energy_per_token_mj']['mean']

key_findings = {
    'output_length_paradox': {
        'description': 'Minimal prompts use MORE energy per token despite shorter prompts',
        'full_energy_per_token_mj': full_ept_mean,
        'minimal_energy_per_token_mj': minimal_ept_mean,
        'increase_percent': (minimal_ept_mean - full_ept_mean) / full_ept_mean * 100,
        'statistical_significance': {
            'p_value': energy_per_token_tests['minimal_vs_full']['p_value'],
            'significant': bool(energy_per_token_tests['minimal_vs_full']['p_value'] < 0.05)
        }
    },
    'token_savings': {
        'full_mean_total_tokens': full_mean_tokens,
        'minimal_mean_total_tokens': minimal_mean_tokens,
        'token_reduction_percent': (full_mean_tokens - minimal_mean_tokens) / full_mean_tokens * 100
    },
    'energy_savings': {
        'full_mean_energy_j': full_mean_energy,
        'minimal_mean_energy_j': minimal_mean_energy,
        'energy_difference_j': full_mean_energy - minimal_mean_energy,
        'energy_change_percent': (full_mean_energy - minimal_mean_energy) / full_mean_energy * 100
    },
    'most_efficient_level': 'minimal' if minimal_mean_energy < full_mean_energy else 'full',
    'anova_significant': bool(regression_results['anova']['significant'])
}

print(f"\nThe Output Length Paradox:")
print(f"  - Full prompts: {full_ept_mean:.2f} mJ/token")
print(f"  - Minimal prompts: {minimal_ept_mean:.2f} mJ/token")
print(f"  - Minimal uses {(minimal_ept_mean - full_ept_mean) / full_ept_mean * 100:.1f}% MORE energy per token")

print(f"\nToken Efficiency:")
print(f"  - Full prompts average: {full_mean_tokens:.1f} tokens")
print(f"  - Minimal prompts average: {minimal_mean_tokens:.1f} tokens")
print(f"  - Token reduction: {(full_mean_tokens - minimal_mean_tokens) / full_mean_tokens * 100:.1f}%")

print(f"\nAbsolute Energy:")
print(f"  - Full prompts average: {full_mean_energy:.2f} J")
print(f"  - Minimal prompts average: {minimal_mean_energy:.2f} J")
print(f"  - Energy change: {(full_mean_energy - minimal_mean_energy) / full_mean_energy * 100:.1f}%")

# =============================================================================
# 7. PAPER-READY SUMMARY TABLE
# =============================================================================
print("\n" + "="*80)
print("7. SUMMARY TABLE FOR PAPER")
print("="*80)

print("\n| Compression | n | Total Energy (J) | Inference Energy (J) | Total Tokens | Energy/Token (mJ) |")
print("|-------------|-----|------------------|---------------------|--------------|-------------------|")
for level in compression_levels:
    s = descriptive_stats[level]
    print(f"| {level:<11} | {s['n']:>3} | {s['total_energy_j']['mean']:>6.2f} +/- {s['total_energy_j']['std']:>5.2f} | {s['inference_energy_j']['mean']:>7.2f} +/- {s['inference_energy_j']['std']:>5.2f} | {s['total_tokens']['mean']:>6.1f} +/- {s['total_tokens']['std']:>4.1f} | {s['energy_per_token_mj']['mean']:>6.2f} +/- {s['energy_per_token_mj']['std']:>4.2f} |")

# =============================================================================
# COMPILE AND SAVE RESULTS
# =============================================================================
results = {
    'metadata': {
        'data_file': str(data_path),
        'n_observations': len(records),
        'analysis_date': '2026-01-22',
        'bonferroni_alpha': bonferroni_alpha,
        'n_comparisons': n_comparisons
    },
    'descriptive_statistics': descriptive_stats,
    'autocorrelation_analysis': autocorrelation_results,
    'pairwise_t_tests': pairwise_tests,
    'energy_per_token_analysis': energy_per_token_tests,
    'regression_analysis': regression_results,
    'key_findings': key_findings
}

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print(f"Results saved to: {output_path}")
print(f"{'='*80}")

#!/usr/bin/env python3
"""
Length-Controlled Causal Experiment Analysis
============================================

Research Question: Does the observed dichotomy between code and CoT compression
tolerance persist when controlling for prompt length?

Gap 1 Analysis: Reviewers may argue that compression tolerance differences are
due to prompt length differences, not task structure.

This script implements:
1. Experiment 1A: Length-matched sampling from existing data
2. Statistical validation via KS test for length distribution matching
3. Two-way ANOVA for Task × Compression interaction
4. Effect size calculation (η²)

Author: Dr. Sarah Chen (AI Research Team, Bona Opera Studios)
Date: 2026-01-17
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev, variance
from typing import Any, Dict, List, Optional, Tuple
import math


# =============================================================================
# STATISTICAL FUNCTIONS (Pure Python - No External Dependencies)
# =============================================================================

def ks_test_2samp(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Two-sample Kolmogorov-Smirnov test.

    Returns:
        (D-statistic, approximate p-value)

    The KS test measures the maximum distance between two empirical CDFs.
    H0: The two samples are drawn from the same distribution.
    """
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0

    # Combine and sort all values
    all_values = sorted(set(sample1 + sample2))

    # Compute empirical CDFs
    def ecdf(sample, x):
        return sum(1 for v in sample if v <= x) / len(sample)

    # Find maximum distance
    d_stat = 0.0
    for x in all_values:
        cdf1 = ecdf(sample1, x)
        cdf2 = ecdf(sample2, x)
        d_stat = max(d_stat, abs(cdf1 - cdf2))

    # Approximate p-value using asymptotic distribution
    # For large samples, D * sqrt(n1*n2/(n1+n2)) ~ Kolmogorov distribution
    n_eff = (n1 * n2) / (n1 + n2)
    lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d_stat

    # Kolmogorov distribution approximation
    if lambda_val < 0.01:
        p_value = 1.0
    else:
        # Series approximation for Kolmogorov distribution
        p_value = 0.0
        for k in range(1, 101):
            p_value += 2 * ((-1) ** (k - 1)) * math.exp(-2 * k * k * lambda_val * lambda_val)
        p_value = max(0.0, min(1.0, p_value))

    return d_stat, p_value


def f_distribution_cdf(x: float, df1: int, df2: int) -> float:
    """
    Approximate CDF of F-distribution using beta function relationship.
    F(x; d1, d2) relates to incomplete beta function.
    """
    if x <= 0:
        return 0.0

    # Transform to beta distribution
    u = df1 * x / (df1 * x + df2)

    # Regularized incomplete beta function approximation
    # Using continued fraction expansion
    a, b = df1 / 2, df2 / 2

    if u == 0:
        return 0.0
    if u == 1:
        return 1.0

    # Simple numerical integration for beta CDF
    n_steps = 1000
    integral = 0.0
    for i in range(n_steps):
        t = i / n_steps * u
        dt = u / n_steps
        if t > 0 and t < 1:
            try:
                integrand = (t ** (a - 1)) * ((1 - t) ** (b - 1))
                integral += integrand * dt
            except (OverflowError, ValueError):
                pass

    # Normalize by beta function B(a,b)
    # B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
    # Approximation using Stirling's formula for gamma
    def log_gamma(z):
        if z <= 0:
            return float('inf')
        # Stirling's approximation
        return 0.5 * math.log(2 * math.pi / z) + z * (math.log(z + 1 / (12 * z - 1 / (10 * z))) - 1)

    try:
        log_beta = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
        beta_val = math.exp(log_beta)
        cdf = integral / beta_val if beta_val > 0 else 0.5
    except (OverflowError, ValueError):
        cdf = 0.5

    return max(0.0, min(1.0, cdf))


def one_way_anova(groups: List[List[float]]) -> Tuple[float, float]:
    """
    One-way ANOVA test.

    Returns:
        (F-statistic, p-value)
    """
    k = len(groups)  # Number of groups
    if k < 2:
        return 0.0, 1.0

    # Filter out empty groups
    groups = [g for g in groups if len(g) > 0]
    k = len(groups)
    if k < 2:
        return 0.0, 1.0

    n_total = sum(len(g) for g in groups)
    grand_mean = sum(sum(g) for g in groups) / n_total

    # Between-group sum of squares (SSB)
    ssb = sum(len(g) * (mean(g) - grand_mean) ** 2 for g in groups)

    # Within-group sum of squares (SSW)
    ssw = sum(sum((x - mean(g)) ** 2 for x in g) for g in groups)

    # Degrees of freedom
    df_between = k - 1
    df_within = n_total - k

    if df_within <= 0 or ssw == 0:
        return float('inf') if ssb > 0 else 0.0, 0.0 if ssb > 0 else 1.0

    # Mean squares
    msb = ssb / df_between
    msw = ssw / df_within

    # F-statistic
    f_stat = msb / msw if msw > 0 else float('inf')

    # p-value from F-distribution
    p_value = 1 - f_distribution_cdf(f_stat, df_between, df_within)

    return f_stat, p_value


def two_way_anova(data: Dict[Tuple[str, str], List[float]]) -> Dict[str, Any]:
    """
    Two-way ANOVA for Task × Compression interaction.

    Args:
        data: Dict mapping (task_type, compression_level) to list of quality scores

    Returns:
        Dictionary with F-statistics, p-values, and effect sizes for:
        - Main effect of Task
        - Main effect of Compression
        - Task × Compression interaction
    """
    # Extract factor levels
    tasks = sorted(set(k[0] for k in data.keys()))
    compressions = sorted(set(k[1] for k in data.keys()))

    # Calculate means
    grand_mean = mean([v for vals in data.values() for v in vals])
    n_total = sum(len(vals) for vals in data.values())

    # Cell means
    cell_means = {k: mean(v) if v else 0 for k, v in data.items()}

    # Marginal means
    task_means = {}
    for t in tasks:
        task_data = [v for (task, _), vals in data.items() if task == t for v in vals]
        task_means[t] = mean(task_data) if task_data else 0

    comp_means = {}
    for c in compressions:
        comp_data = [v for (_, comp), vals in data.items() if comp == c for v in vals]
        comp_means[c] = mean(comp_data) if comp_data else 0

    # Sum of squares
    ss_task = sum(
        sum(len(data.get((t, c), [])) for c in compressions) * (task_means[t] - grand_mean) ** 2
        for t in tasks
    )

    ss_comp = sum(
        sum(len(data.get((t, c), [])) for t in tasks) * (comp_means[c] - grand_mean) ** 2
        for c in compressions
    )

    # Interaction SS
    ss_interaction = 0
    for t in tasks:
        for c in compressions:
            cell_data = data.get((t, c), [])
            if cell_data:
                expected = task_means[t] + comp_means[c] - grand_mean
                ss_interaction += len(cell_data) * (cell_means[(t, c)] - expected) ** 2

    # Error SS
    ss_error = sum(
        sum((x - cell_means.get(k, 0)) ** 2 for x in vals)
        for k, vals in data.items()
    )

    # Total SS
    ss_total = ss_task + ss_comp + ss_interaction + ss_error

    # Degrees of freedom
    df_task = len(tasks) - 1
    df_comp = len(compressions) - 1
    df_interaction = df_task * df_comp
    df_error = n_total - len(tasks) * len(compressions)

    # Ensure positive df_error
    df_error = max(df_error, 1)

    # Mean squares and F-statistics
    ms_task = ss_task / max(df_task, 1)
    ms_comp = ss_comp / max(df_comp, 1)
    ms_interaction = ss_interaction / max(df_interaction, 1)
    ms_error = ss_error / df_error

    f_task = ms_task / ms_error if ms_error > 0 else float('inf')
    f_comp = ms_comp / ms_error if ms_error > 0 else float('inf')
    f_interaction = ms_interaction / ms_error if ms_error > 0 else float('inf')

    # P-values
    p_task = 1 - f_distribution_cdf(f_task, max(df_task, 1), df_error)
    p_comp = 1 - f_distribution_cdf(f_comp, max(df_comp, 1), df_error)
    p_interaction = 1 - f_distribution_cdf(f_interaction, max(df_interaction, 1), df_error)

    # Effect sizes (η² = SS_effect / SS_total)
    eta_sq_task = ss_task / ss_total if ss_total > 0 else 0
    eta_sq_comp = ss_comp / ss_total if ss_total > 0 else 0
    eta_sq_interaction = ss_interaction / ss_total if ss_total > 0 else 0

    # Partial η² = SS_effect / (SS_effect + SS_error)
    partial_eta_task = ss_task / (ss_task + ss_error) if (ss_task + ss_error) > 0 else 0
    partial_eta_comp = ss_comp / (ss_comp + ss_error) if (ss_comp + ss_error) > 0 else 0
    partial_eta_interaction = ss_interaction / (ss_interaction + ss_error) if (ss_interaction + ss_error) > 0 else 0

    return {
        'task_effect': {
            'F': f_task,
            'p': p_task,
            'df': (df_task, df_error),
            'eta_squared': eta_sq_task,
            'partial_eta_squared': partial_eta_task,
            'SS': ss_task
        },
        'compression_effect': {
            'F': f_comp,
            'p': p_comp,
            'df': (df_comp, df_error),
            'eta_squared': eta_sq_comp,
            'partial_eta_squared': partial_eta_comp,
            'SS': ss_comp
        },
        'interaction': {
            'F': f_interaction,
            'p': p_interaction,
            'df': (df_interaction, df_error),
            'eta_squared': eta_sq_interaction,
            'partial_eta_squared': partial_eta_interaction,
            'SS': ss_interaction
        },
        'error': {
            'df': df_error,
            'SS': ss_error,
            'MS': ms_error
        },
        'total': {
            'SS': ss_total,
            'n': n_total
        }
    }


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size for two independent groups.
    """
    if len(group1) < 2 or len(group2) < 2:
        return 0.0

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = mean(group1), mean(group2)
    var1, var2 = variance(group1), variance(group2)

    # Pooled standard deviation
    pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 1

    return (mean1 - mean2) / pooled_std


# =============================================================================
# LENGTH-MATCHED SAMPLING
# =============================================================================

@dataclass
class MatchedSample:
    """Results from length-matched sampling procedure."""
    code_trials: List[Dict]
    cot_trials: List[Dict]
    code_lengths: List[int]
    cot_lengths: List[int]
    ks_statistic: float
    ks_pvalue: float
    overlap_range: Tuple[int, int]
    matching_method: str


def create_length_matched_samples(
    code_trials: List[Dict],
    cot_trials: List[Dict],
    target_ks_pvalue: float = 0.05,
    random_seed: int = 42,
    max_iterations: int = 1000
) -> Optional[MatchedSample]:
    """
    Create length-matched samples where KS test p-value > target.

    Strategy:
    1. Find overlapping length range
    2. Bin trials by length
    3. Sample equal numbers from each bin
    4. Verify with KS test
    """
    random.seed(random_seed)

    # Get lengths
    code_lengths = [(t, t['original_tokens']) for t in code_trials]
    cot_lengths = [(t, t['original_tokens']) for t in cot_trials]

    # Find overlap range
    code_min, code_max = min(l for _, l in code_lengths), max(l for _, l in code_lengths)
    cot_min, cot_max = min(l for _, l in cot_lengths), max(l for _, l in cot_lengths)

    overlap_min = max(code_min, cot_min)
    overlap_max = min(code_max, cot_max)

    if overlap_min > overlap_max:
        print(f"WARNING: No overlapping length range!")
        print(f"  Code: {code_min}-{code_max}, CoT: {cot_min}-{cot_max}")
        return None

    print(f"Overlapping length range: {overlap_min}-{overlap_max} tokens")

    # Filter to overlap range
    code_in_range = [(t, l) for t, l in code_lengths if overlap_min <= l <= overlap_max]
    cot_in_range = [(t, l) for t, l in cot_lengths if overlap_min <= l <= overlap_max]

    print(f"Code trials in range: {len(code_in_range)}")
    print(f"CoT trials in range: {len(cot_in_range)}")

    if len(code_in_range) < 10 or len(cot_in_range) < 10:
        print("WARNING: Insufficient samples in overlap range")
        # Try stratified sampling approach
        return create_stratified_matched_samples(code_trials, cot_trials, random_seed)

    # Bin-based matching
    # Create bins of length ranges
    bin_size = 5
    bins = defaultdict(lambda: {'code': [], 'cot': []})

    for t, l in code_in_range:
        bin_idx = l // bin_size
        bins[bin_idx]['code'].append(t)

    for t, l in cot_in_range:
        bin_idx = l // bin_size
        bins[bin_idx]['cot'].append(t)

    # Sample from bins with both code and cot
    matched_code = []
    matched_cot = []

    for bin_idx, bin_data in bins.items():
        n_code = len(bin_data['code'])
        n_cot = len(bin_data['cot'])

        if n_code > 0 and n_cot > 0:
            # Sample equal numbers
            n_sample = min(n_code, n_cot)
            matched_code.extend(random.sample(bin_data['code'], n_sample))
            matched_cot.extend(random.sample(bin_data['cot'], n_sample))

    if len(matched_code) < 10:
        print("WARNING: Bin matching produced insufficient samples")
        return create_stratified_matched_samples(code_trials, cot_trials, random_seed)

    # Verify with KS test
    code_matched_lengths = [t['original_tokens'] for t in matched_code]
    cot_matched_lengths = [t['original_tokens'] for t in matched_cot]

    ks_stat, ks_p = ks_test_2samp(code_matched_lengths, cot_matched_lengths)

    print(f"\nMatched sample sizes: Code={len(matched_code)}, CoT={len(matched_cot)}")
    print(f"KS test: D={ks_stat:.4f}, p={ks_p:.4f}")

    return MatchedSample(
        code_trials=matched_code,
        cot_trials=matched_cot,
        code_lengths=code_matched_lengths,
        cot_lengths=cot_matched_lengths,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        overlap_range=(overlap_min, overlap_max),
        matching_method='bin_matching'
    )


def create_stratified_matched_samples(
    code_trials: List[Dict],
    cot_trials: List[Dict],
    random_seed: int = 42
) -> MatchedSample:
    """
    Alternative approach: Stratified sampling by compression ratio.

    When direct length matching isn't possible due to non-overlapping
    distributions, we control by:
    1. Sampling equal N per compression level
    2. Normalizing length effects in analysis
    """
    random.seed(random_seed)

    # Group by compression ratio
    code_by_comp = defaultdict(list)
    cot_by_comp = defaultdict(list)

    for t in code_trials:
        comp = round(t.get('compression_ratio', 0.5), 1)
        code_by_comp[comp].append(t)

    for t in cot_trials:
        comp = round(t.get('compression_ratio', 0.5), 1)
        cot_by_comp[comp].append(t)

    # Common compression levels
    common_comps = set(code_by_comp.keys()) & set(cot_by_comp.keys())

    matched_code = []
    matched_cot = []

    for comp in sorted(common_comps):
        n_code = len(code_by_comp[comp])
        n_cot = len(cot_by_comp[comp])
        n_sample = min(n_code, n_cot, 50)  # Cap at 50 per cell

        matched_code.extend(random.sample(code_by_comp[comp], n_sample))
        matched_cot.extend(random.sample(cot_by_comp[comp], n_sample))

    code_lengths = [t['original_tokens'] for t in matched_code]
    cot_lengths = [t['original_tokens'] for t in matched_cot]

    ks_stat, ks_p = ks_test_2samp(code_lengths, cot_lengths)

    print(f"\nStratified sample sizes: Code={len(matched_code)}, CoT={len(matched_cot)}")
    print(f"KS test (lengths NOT matched): D={ks_stat:.4f}, p={ks_p:.4f}")
    print("Note: Using stratified design with length as covariate")

    return MatchedSample(
        code_trials=matched_code,
        cot_trials=matched_cot,
        code_lengths=code_lengths,
        cot_lengths=cot_lengths,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        overlap_range=(0, 0),  # No overlap in this method
        matching_method='stratified_by_compression'
    )


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_compression_tolerance(
    matched_sample: MatchedSample,
    compression_levels: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> Dict[str, Any]:
    """
    Analyze compression tolerance patterns in length-matched samples.

    Tests whether the code vs CoT dichotomy persists when controlling for length.
    """
    results = {
        'sample_info': {
            'n_code': len(matched_sample.code_trials),
            'n_cot': len(matched_sample.cot_trials),
            'matching_method': matched_sample.matching_method,
            'ks_statistic': matched_sample.ks_statistic,
            'ks_pvalue': matched_sample.ks_pvalue,
            'length_matched': matched_sample.ks_pvalue > 0.05
        },
        'descriptive': {},
        'anova': {},
        'effect_sizes': {},
        'compression_curves': {}
    }

    # Descriptive statistics by task and compression
    anova_data = {}

    for task_name, trials in [('code', matched_sample.code_trials), ('cot', matched_sample.cot_trials)]:
        results['descriptive'][task_name] = {}
        results['compression_curves'][task_name] = {}

        for comp in compression_levels:
            # Get trials at this compression level
            subset = [t for t in trials if abs(t.get('compression_ratio', 0) - comp) < 0.05]

            if subset:
                qualities = [t.get('quality_score', 0) for t in subset]
                successes = [1 if t.get('task_success') else 0 for t in subset]

                results['descriptive'][task_name][f'comp_{comp}'] = {
                    'n': len(subset),
                    'quality_mean': mean(qualities),
                    'quality_std': stdev(qualities) if len(qualities) > 1 else 0,
                    'success_rate': mean(successes) * 100
                }

                results['compression_curves'][task_name][comp] = {
                    'quality': mean(qualities),
                    'success_rate': mean(successes) * 100
                }

                # Prepare data for ANOVA
                comp_key = f'{comp}'
                anova_data[(task_name, comp_key)] = qualities

    # Two-way ANOVA
    results['anova'] = two_way_anova(anova_data)

    # Calculate Cohen's d for each compression level
    results['effect_sizes']['by_compression'] = {}
    for comp in compression_levels:
        code_at_comp = [t.get('quality_score', 0) for t in matched_sample.code_trials
                       if abs(t.get('compression_ratio', 0) - comp) < 0.05]
        cot_at_comp = [t.get('quality_score', 0) for t in matched_sample.cot_trials
                      if abs(t.get('compression_ratio', 0) - comp) < 0.05]

        if code_at_comp and cot_at_comp:
            d = cohens_d(code_at_comp, cot_at_comp)
            results['effect_sizes']['by_compression'][comp] = {
                'cohens_d': d,
                'interpretation': interpret_cohens_d(d)
            }

    # Overall effect size (collapsed across compression)
    all_code_quality = [t.get('quality_score', 0) for t in matched_sample.code_trials]
    all_cot_quality = [t.get('quality_score', 0) for t in matched_sample.cot_trials]

    results['effect_sizes']['overall'] = {
        'cohens_d': cohens_d(all_code_quality, all_cot_quality),
        'code_mean': mean(all_code_quality),
        'cot_mean': mean(all_cot_quality)
    }

    return results


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return "negligible"
    elif d_abs < 0.5:
        return "small"
    elif d_abs < 0.8:
        return "medium"
    else:
        return "large"


def interpret_eta_squared(eta_sq: float) -> str:
    """Interpret eta-squared effect size."""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_report(results: Dict[str, Any], matched_sample: MatchedSample) -> str:
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 80)
    report.append("LENGTH-CONTROLLED CAUSAL EXPERIMENT: ANALYSIS REPORT")
    report.append("=" * 80)
    report.append("")

    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 40)

    length_matched = results['sample_info']['length_matched']
    interaction_p = results['anova']['interaction']['p']
    interaction_eta = results['anova']['interaction']['eta_squared']

    if length_matched:
        report.append(f"Length distributions successfully matched (KS p = {results['sample_info']['ks_pvalue']:.4f})")
    else:
        report.append(f"Length distributions differ (KS p = {results['sample_info']['ks_pvalue']:.4f})")
        report.append(f"Using {matched_sample.matching_method} design with length as covariate")

    report.append("")
    if interaction_p < 0.05:
        report.append(f"FINDING: Significant Task × Compression interaction (p = {interaction_p:.4f})")
        report.append(f"The dichotomy PERSISTS when controlling for length.")
        report.append(f"Effect size: η² = {interaction_eta:.4f} ({interpret_eta_squared(interaction_eta)})")
    else:
        report.append(f"FINDING: No significant Task × Compression interaction (p = {interaction_p:.4f})")
        report.append(f"The dichotomy may be partially explained by length differences.")

    report.append("")
    report.append("")

    # Sample Information
    report.append("1. SAMPLE INFORMATION")
    report.append("-" * 40)
    report.append(f"Matching method: {results['sample_info']['matching_method']}")
    report.append(f"Code trials: n = {results['sample_info']['n_code']}")
    report.append(f"CoT trials: n = {results['sample_info']['n_cot']}")
    report.append(f"KS test statistic: D = {results['sample_info']['ks_statistic']:.4f}")
    report.append(f"KS test p-value: p = {results['sample_info']['ks_pvalue']:.4f}")
    report.append("")

    # Length Statistics
    report.append("Length Statistics (after matching):")
    report.append(f"  Code: Mean = {mean(matched_sample.code_lengths):.1f}, "
                 f"Std = {stdev(matched_sample.code_lengths) if len(matched_sample.code_lengths) > 1 else 0:.1f}")
    report.append(f"  CoT:  Mean = {mean(matched_sample.cot_lengths):.1f}, "
                 f"Std = {stdev(matched_sample.cot_lengths) if len(matched_sample.cot_lengths) > 1 else 0:.1f}")
    report.append("")
    report.append("")

    # Compression Curves
    report.append("2. COMPRESSION TOLERANCE CURVES")
    report.append("-" * 40)
    report.append("")
    report.append("Quality Score by Compression Level:")
    report.append("")
    report.append(f"{'Compression':<12} {'Code Quality':<15} {'CoT Quality':<15} {'Difference':<15}")
    report.append("-" * 57)

    for comp in sorted(results['compression_curves'].get('code', {}).keys()):
        code_q = results['compression_curves']['code'].get(comp, {}).get('quality', 0)
        cot_q = results['compression_curves']['cot'].get(comp, {}).get('quality', 0)
        diff = code_q - cot_q
        report.append(f"{comp:<12.1f} {code_q:<15.3f} {cot_q:<15.3f} {diff:<+15.3f}")

    report.append("")
    report.append("Success Rate by Compression Level:")
    report.append("")
    report.append(f"{'Compression':<12} {'Code Success':<15} {'CoT Success':<15} {'Difference':<15}")
    report.append("-" * 57)

    for comp in sorted(results['compression_curves'].get('code', {}).keys()):
        code_s = results['compression_curves']['code'].get(comp, {}).get('success_rate', 0)
        cot_s = results['compression_curves']['cot'].get(comp, {}).get('success_rate', 0)
        diff = code_s - cot_s
        report.append(f"{comp:<12.1f} {code_s:<15.1f}% {cot_s:<15.1f}% {diff:<+15.1f}pp")

    report.append("")
    report.append("")

    # ANOVA Results
    report.append("3. TWO-WAY ANOVA RESULTS")
    report.append("-" * 40)
    report.append("")

    anova = results['anova']

    report.append("Main Effect of Task Type:")
    task = anova['task_effect']
    sig = "*" if task['p'] < 0.05 else ""
    report.append(f"  F({task['df'][0]}, {task['df'][1]}) = {task['F']:.2f}, p = {task['p']:.4f}{sig}")
    report.append(f"  η² = {task['eta_squared']:.4f} ({interpret_eta_squared(task['eta_squared'])})")
    report.append(f"  Partial η² = {task['partial_eta_squared']:.4f}")
    report.append("")

    report.append("Main Effect of Compression Level:")
    comp = anova['compression_effect']
    sig = "*" if comp['p'] < 0.05 else ""
    report.append(f"  F({comp['df'][0]}, {comp['df'][1]}) = {comp['F']:.2f}, p = {comp['p']:.4f}{sig}")
    report.append(f"  η² = {comp['eta_squared']:.4f} ({interpret_eta_squared(comp['eta_squared'])})")
    report.append(f"  Partial η² = {comp['partial_eta_squared']:.4f}")
    report.append("")

    report.append("Task × Compression Interaction:")
    inter = anova['interaction']
    sig = "*" if inter['p'] < 0.05 else ""
    report.append(f"  F({inter['df'][0]}, {inter['df'][1]}) = {inter['F']:.2f}, p = {inter['p']:.4f}{sig}")
    report.append(f"  η² = {inter['eta_squared']:.4f} ({interpret_eta_squared(inter['eta_squared'])})")
    report.append(f"  Partial η² = {inter['partial_eta_squared']:.4f}")
    report.append("")
    report.append("")

    # Effect Sizes
    report.append("4. EFFECT SIZES (Cohen's d)")
    report.append("-" * 40)
    report.append("")
    report.append("Effect size by compression level (positive = code > CoT):")
    report.append("")

    for comp, effect in sorted(results['effect_sizes']['by_compression'].items()):
        report.append(f"  Compression {comp}: d = {effect['cohens_d']:+.3f} ({effect['interpretation']})")

    overall = results['effect_sizes']['overall']
    report.append("")
    report.append(f"Overall effect: d = {overall['cohens_d']:+.3f}")
    report.append(f"  Code mean quality: {overall['code_mean']:.3f}")
    report.append(f"  CoT mean quality: {overall['cot_mean']:.3f}")
    report.append("")
    report.append("")

    # Conclusions
    report.append("5. CONCLUSIONS AND RECOMMENDATIONS")
    report.append("-" * 40)
    report.append("")

    if interaction_p < 0.05:
        report.append("KEY FINDING: The Task × Compression interaction is STATISTICALLY SIGNIFICANT.")
        report.append("")
        report.append("This means:")
        report.append("  - Code and CoT tasks respond DIFFERENTLY to compression")
        report.append("  - This differential response is NOT explained by length differences alone")
        report.append("  - The finding supports the paper's core thesis about structural differences")
        report.append("")
        report.append("For paper revision:")
        report.append("  1. Add this analysis as Supplementary Experiment 1A")
        report.append("  2. Report: 'Length-controlled analysis confirms Task × Compression interaction'")
        report.append(f"     F = {inter['F']:.2f}, p = {inter['p']:.4f}, η² = {inter['eta_squared']:.4f}")
        report.append("  3. This strengthens the causal claim about task structure")
    else:
        report.append("KEY FINDING: The Task × Compression interaction is NOT significant after length control.")
        report.append("")
        report.append("This means:")
        report.append("  - Length differences may partially explain the observed dichotomy")
        report.append("  - Additional experiments needed to isolate structural effects")
        report.append("")
        report.append("For paper revision:")
        report.append("  1. Acknowledge length as a potential confound")
        report.append("  2. Implement Experiment 1B: Synthetic length-extended prompts")
        report.append("  3. Consider propensity score matching for stronger causal claims")

    report.append("")
    report.append("=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)

    return "\n".join(report)


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def load_data(filepath: str) -> List[Dict]:
    """Load trial data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_length_covariate(code_trials: List[Dict], cot_trials: List[Dict]) -> Dict[str, Any]:
    """
    Analyze whether length explains the task-compression interaction.

    Uses residualization approach: regress quality on length, then analyze residuals.
    This is equivalent to ANCOVA controlling for length.
    """
    print("\n" + "=" * 70)
    print("ANCOVA-STYLE ANALYSIS: Controlling for Length as Covariate")
    print("=" * 70)

    # Combine all trials with labels
    all_trials = []
    for t in code_trials:
        all_trials.append({
            'task': 'code',
            'compression': round(t.get('compression_ratio', 0.5), 1),
            'quality': t.get('quality_score', 0),
            'length': t.get('original_tokens', 0),
            'success': 1 if t.get('task_success') else 0
        })
    for t in cot_trials:
        all_trials.append({
            'task': 'cot',
            'compression': round(t.get('compression_ratio', 0.5), 1),
            'quality': t.get('quality_score', 0),
            'length': t.get('original_tokens', 0),
            'success': 1 if t.get('task_success') else 0
        })

    # Check length-quality correlation
    lengths = [t['length'] for t in all_trials]
    qualities = [t['quality'] for t in all_trials]

    # Simple correlation
    n = len(lengths)
    mean_l, mean_q = mean(lengths), mean(qualities)

    cov_lq = sum((lengths[i] - mean_l) * (qualities[i] - mean_q) for i in range(n)) / (n - 1)
    std_l = stdev(lengths)
    std_q = stdev(qualities)

    correlation = cov_lq / (std_l * std_q) if std_l > 0 and std_q > 0 else 0

    print(f"\nLength-Quality Correlation: r = {correlation:.4f}")

    # Residualize quality on length (simple linear regression)
    # quality_residual = quality - predicted_quality
    # where predicted_quality = intercept + slope * length

    slope = cov_lq / (std_l ** 2) if std_l > 0 else 0
    intercept = mean_q - slope * mean_l

    print(f"Regression: quality = {intercept:.4f} + {slope:.4f} * length")

    # Calculate residuals
    for t in all_trials:
        predicted = intercept + slope * t['length']
        t['quality_residual'] = t['quality'] - predicted

    # Now analyze residuals by task and compression
    results = {
        'length_quality_correlation': correlation,
        'regression_slope': slope,
        'regression_intercept': intercept,
        'residual_analysis': {}
    }

    # Group by task and compression for residual analysis
    anova_data_residual = {}
    for t in all_trials:
        key = (t['task'], str(t['compression']))
        if key not in anova_data_residual:
            anova_data_residual[key] = []
        anova_data_residual[key].append(t['quality_residual'])

    # Two-way ANOVA on residuals
    residual_anova = two_way_anova(anova_data_residual)
    results['residual_anova'] = residual_anova

    print("\n" + "-" * 50)
    print("Two-Way ANOVA on LENGTH-ADJUSTED Quality (Residuals)")
    print("-" * 50)

    task_effect = residual_anova['task_effect']
    print(f"\nMain Effect of Task (after length adjustment):")
    print(f"  F({task_effect['df'][0]}, {task_effect['df'][1]}) = {task_effect['F']:.2f}")
    print(f"  p = {task_effect['p']:.6f}")
    print(f"  η² = {task_effect['eta_squared']:.4f}")

    comp_effect = residual_anova['compression_effect']
    print(f"\nMain Effect of Compression:")
    print(f"  F({comp_effect['df'][0]}, {comp_effect['df'][1]}) = {comp_effect['F']:.2f}")
    print(f"  p = {comp_effect['p']:.6f}")
    print(f"  η² = {comp_effect['eta_squared']:.4f}")

    interaction = residual_anova['interaction']
    print(f"\nTask × Compression Interaction (CRITICAL TEST):")
    print(f"  F({interaction['df'][0]}, {interaction['df'][1]}) = {interaction['F']:.2f}")
    print(f"  p = {interaction['p']:.6f}")
    print(f"  η² = {interaction['eta_squared']:.4f}")

    if interaction['p'] < 0.05:
        print("\n  *** SIGNIFICANT: Interaction persists after controlling for length ***")
    else:
        print("\n  Interaction NOT significant after controlling for length")

    # Effect sizes by compression on residuals
    print("\n" + "-" * 50)
    print("Cohen's d on Length-Adjusted Quality by Compression Level")
    print("-" * 50)

    results['effect_sizes_residual'] = {}
    for comp in sorted(set(t['compression'] for t in all_trials)):
        code_residuals = [t['quality_residual'] for t in all_trials
                         if t['task'] == 'code' and t['compression'] == comp]
        cot_residuals = [t['quality_residual'] for t in all_trials
                        if t['task'] == 'cot' and t['compression'] == comp]

        if len(code_residuals) > 1 and len(cot_residuals) > 1:
            d = cohens_d(code_residuals, cot_residuals)
            results['effect_sizes_residual'][comp] = d
            print(f"  Compression {comp}: d = {d:+.3f} ({interpret_cohens_d(d)})")

    return results


def main():
    """Main analysis pipeline."""
    print("=" * 80)
    print("LENGTH-CONTROLLED CAUSAL EXPERIMENT ANALYSIS")
    print("Gap 1: Controlling for prompt length confound")
    print("=" * 80)
    print()

    # Data path - adjust as needed
    data_path = Path("/home/azureuser/plexor-research/experiments/results/all_trials.json")

    # Alternative: try local copy if exists
    local_path = Path("./all_trials.json")
    if local_path.exists():
        data_path = local_path
        print(f"Using local data: {data_path}")
    else:
        print(f"Using remote data: {data_path}")

    # Load data
    print("\n[1] Loading experimental data...")
    try:
        trials = load_data(str(data_path))
        print(f"    Loaded {len(trials)} trials")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure the data file is available.")
        return

    # Separate task types
    code_trials = [t for t in trials if t.get('task_type') == 'code_generation']
    cot_trials = [t for t in trials if t.get('task_type') == 'math_reasoning']

    print(f"    Code trials: {len(code_trials)}")
    print(f"    CoT trials: {len(cot_trials)}")

    # Create length-matched samples
    print("\n[2] Creating length-matched samples...")
    matched_sample = create_length_matched_samples(
        code_trials,
        cot_trials,
        target_ks_pvalue=0.05,
        random_seed=42
    )

    if matched_sample is None:
        print("ERROR: Could not create matched samples")
        return

    # Analyze compression tolerance
    print("\n[3] Analyzing compression tolerance patterns...")
    results = analyze_compression_tolerance(matched_sample)

    # Run ANCOVA-style analysis on full data
    print("\n[4] Running ANCOVA-style analysis (length as covariate)...")
    ancova_results = analyze_length_covariate(code_trials, cot_trials)

    # Generate report
    print("\n[5] Generating analysis report...")
    report = generate_report(results, matched_sample)

    print("\n")
    print(report)

    # Add ANCOVA summary to report
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY: ANCOVA ANALYSIS ON FULL DATASET")
    print("=" * 80)
    print(f"\nThis analysis uses ALL {len(code_trials) + len(cot_trials)} trials")
    print(f"with length statistically controlled via residualization.")
    print(f"\nLength-Quality Correlation: r = {ancova_results['length_quality_correlation']:.4f}")

    ancova_interaction = ancova_results['residual_anova']['interaction']
    if ancova_interaction['p'] < 0.05:
        print(f"\nCRITICAL RESULT: Task x Compression interaction REMAINS SIGNIFICANT")
        print(f"after controlling for length!")
        print(f"  F = {ancova_interaction['F']:.2f}, p = {ancova_interaction['p']:.6f}, η² = {ancova_interaction['eta_squared']:.4f}")
    else:
        print(f"\nCRITICAL RESULT: Task x Compression interaction NO LONGER SIGNIFICANT")
        print(f"after controlling for length.")
        print(f"  F = {ancova_interaction['F']:.2f}, p = {ancova_interaction['p']:.6f}")

    # Save results
    output_dir = Path("./results")
    output_dir.mkdir(exist_ok=True)

    # Save JSON results
    results_file = output_dir / "length_controlled_results.json"
    with open(results_file, 'w') as f:
        # Convert for JSON serialization
        json_results = {
            'sample_info': results['sample_info'],
            'anova': {
                k: {kk: vv for kk, vv in v.items() if not callable(vv)}
                for k, v in results['anova'].items()
            },
            'effect_sizes': results['effect_sizes'],
            'compression_curves': results['compression_curves'],
            'ancova': {
                'length_quality_correlation': ancova_results['length_quality_correlation'],
                'regression_slope': ancova_results['regression_slope'],
                'regression_intercept': ancova_results['regression_intercept'],
                'residual_anova_interaction': {
                    'F': ancova_results['residual_anova']['interaction']['F'],
                    'p': ancova_results['residual_anova']['interaction']['p'],
                    'eta_squared': ancova_results['residual_anova']['interaction']['eta_squared']
                },
                'effect_sizes_residual': ancova_results['effect_sizes_residual']
            }
        }
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    # Save report
    report_file = output_dir / "length_controlled_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")

    return results, matched_sample


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Article 4: Generate Trial Matrix for Energy Reduction Experiments

This script generates the complete trial matrix for the benchmark suite,
outputting trial configurations in JSON format for use by the experiment runner.

Author: PhD Researcher #4 (Benchmark Design)
Date: January 22, 2026
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Optional
from itertools import product
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

RANDOM_SEED = 42

BENCHMARKS = {
    "humaneval": {
        "full_size": 164,
        "sample_size": 164,  # Use full dataset
        "metric": "pass@1",
        "task_type": "code",
        "avg_tokens": 200,
    },
    "mbpp": {
        "full_size": 500,
        "sample_size": 500,  # Sanitized subset
        "metric": "pass@1",
        "task_type": "code",
        "avg_tokens": 300,
    },
    "gsm8k": {
        "full_size": 1319,
        "sample_size": 200,  # Stratified sample
        "metric": "exact_match",
        "task_type": "math",
        "avg_tokens": 150,
    },
    "math": {
        "full_size": 5000,
        "sample_size": 100,  # Balanced across categories
        "metric": "exact_match",
        "task_type": "math",
        "avg_tokens": 250,
    },
    "mmlu": {
        "full_size": 14042,
        "sample_size": 200,  # Balanced across supercategories
        "metric": "accuracy",
        "task_type": "knowledge",
        "avg_tokens": 100,
    },
}

COMPRESSION_RATIOS = [1.0, 0.7, 0.5, 0.3]

MODELS = {
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "tier": "medium",
        "cost_input_per_m": 3.00,
        "cost_output_per_m": 15.00,
        "model_size_params": "175B",  # Estimated
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "tier": "medium",
        "cost_input_per_m": 0.15,
        "cost_output_per_m": 0.60,
        "model_size_params": "8B",  # Estimated
    },
    "deepseek-chat": {
        "provider": "deepseek",
        "tier": "large",
        "cost_input_per_m": 0.14,
        "cost_output_per_m": 0.28,
        "model_size_params": "67B",
    },
}

REPLICATES = 3

# Energy proxy constants (Joules per token)
ENERGY_CONSTANTS = {
    "small": {"input": 0.000018, "output": 0.000036},   # 7B
    "medium": {"input": 0.000090, "output": 0.000180},  # ~50B
    "large": {"input": 0.00018, "output": 0.00036},     # ~70B
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class TrialConfig:
    """Configuration for a single experimental trial."""
    trial_id: str
    benchmark: str
    sample_id: str
    compression_ratio: float
    model: str
    replicate: int
    # Metadata
    task_type: str
    metric: str
    provider: str
    model_tier: str
    # Estimates
    est_input_tokens: int
    est_output_tokens: int
    est_cost_usd: float
    est_energy_joules: float
    # Status tracking
    status: str = "pending"
    scheduled_at: Optional[str] = None


# ============================================================================
# Functions
# ============================================================================

def generate_sample_ids(benchmark: str, sample_size: int) -> List[str]:
    """Generate sample IDs for a benchmark."""
    if benchmark == "humaneval":
        return [f"HumanEval/{i}" for i in range(sample_size)]
    elif benchmark == "mbpp":
        return [f"mbpp_{i:03d}" for i in range(1, sample_size + 1)]
    elif benchmark == "gsm8k":
        return [f"gsm8k_{i:04d}" for i in range(sample_size)]
    elif benchmark == "math":
        # 7 categories, exactly 100 samples total
        # Distribution: 14, 14, 14, 14, 15, 15, 14 = 100
        categories = ["algebra", "counting", "geometry", "number_theory",
                     "prealgebra", "precalculus", "intermediate_algebra"]
        samples_per_cat = [14, 14, 14, 14, 15, 15, 14]  # = 100
        ids = []
        for cat, n_per_cat in zip(categories, samples_per_cat):
            ids.extend([f"math_{cat}_{j:02d}" for j in range(n_per_cat)])
        return ids[:sample_size]
    elif benchmark == "mmlu":
        # 4 supercategories, 50 per category
        categories = ["stem", "humanities", "social_sciences", "other"]
        ids = []
        for cat in categories:
            ids.extend([f"mmlu_{cat}_{i:02d}" for i in range(50)])
        return ids[:sample_size]
    else:
        return [f"{benchmark}_{i:04d}" for i in range(sample_size)]


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model_config: dict
) -> float:
    """Estimate cost in USD for a trial."""
    input_cost = (input_tokens / 1_000_000) * model_config["cost_input_per_m"]
    output_cost = (output_tokens / 1_000_000) * model_config["cost_output_per_m"]
    return round(input_cost + output_cost, 6)


def estimate_energy(
    input_tokens: int,
    output_tokens: int,
    model_tier: str
) -> float:
    """Estimate energy consumption in Joules."""
    constants = ENERGY_CONSTANTS.get(model_tier, ENERGY_CONSTANTS["medium"])
    energy = (input_tokens * constants["input"]) + (output_tokens * constants["output"])
    return round(energy, 6)


def generate_trial_id(
    benchmark: str,
    sample_id: str,
    ratio: float,
    model: str,
    replicate: int
) -> str:
    """Generate unique trial ID."""
    ratio_str = f"{int(ratio * 100):03d}"
    model_short = model.split("-")[0][:3]
    return f"{benchmark}-{ratio_str}-{model_short}-r{replicate}-{sample_id.replace('/', '_')}"


def generate_trial_matrix() -> List[TrialConfig]:
    """Generate the complete trial matrix."""
    random.seed(RANDOM_SEED)
    trials = []

    for benchmark, bench_config in BENCHMARKS.items():
        sample_ids = generate_sample_ids(benchmark, bench_config["sample_size"])
        avg_tokens = bench_config["avg_tokens"]
        avg_output = 200  # Reasonable default for code/answer generation

        for sample_id in sample_ids:
            for ratio in COMPRESSION_RATIOS:
                # Estimate compressed tokens
                compressed_tokens = int(avg_tokens * ratio)

                for model, model_config in MODELS.items():
                    for replicate in range(1, REPLICATES + 1):
                        trial_id = generate_trial_id(
                            benchmark, sample_id, ratio, model, replicate
                        )

                        est_cost = estimate_cost(
                            compressed_tokens, avg_output, model_config
                        )
                        est_energy = estimate_energy(
                            compressed_tokens, avg_output, model_config["tier"]
                        )

                        trial = TrialConfig(
                            trial_id=trial_id,
                            benchmark=benchmark,
                            sample_id=sample_id,
                            compression_ratio=ratio,
                            model=model,
                            replicate=replicate,
                            task_type=bench_config["task_type"],
                            metric=bench_config["metric"],
                            provider=model_config["provider"],
                            model_tier=model_config["tier"],
                            est_input_tokens=compressed_tokens,
                            est_output_tokens=avg_output,
                            est_cost_usd=est_cost,
                            est_energy_joules=est_energy,
                            scheduled_at=datetime.now().isoformat(),
                        )
                        trials.append(trial)

    return trials


def compute_summary(trials: List[TrialConfig]) -> dict:
    """Compute summary statistics for the trial matrix."""
    total_trials = len(trials)
    total_cost = sum(t.est_cost_usd for t in trials)
    total_energy = sum(t.est_energy_joules for t in trials)

    by_benchmark = {}
    for t in trials:
        if t.benchmark not in by_benchmark:
            by_benchmark[t.benchmark] = {"count": 0, "cost": 0.0}
        by_benchmark[t.benchmark]["count"] += 1
        by_benchmark[t.benchmark]["cost"] += t.est_cost_usd

    by_model = {}
    for t in trials:
        if t.model not in by_model:
            by_model[t.model] = {"count": 0, "cost": 0.0}
        by_model[t.model]["count"] += 1
        by_model[t.model]["cost"] += t.est_cost_usd

    by_ratio = {}
    for t in trials:
        ratio_key = f"r={t.compression_ratio}"
        if ratio_key not in by_ratio:
            by_ratio[ratio_key] = {"count": 0}
        by_ratio[ratio_key]["count"] += 1

    return {
        "total_trials": total_trials,
        "total_estimated_cost_usd": round(total_cost, 2),
        "total_estimated_energy_joules": round(total_energy, 2),
        "total_estimated_energy_kwh": round(total_energy / 3_600_000, 4),
        "by_benchmark": by_benchmark,
        "by_model": by_model,
        "by_ratio": by_ratio,
        "unique_conditions": len(BENCHMARKS) * len(COMPRESSION_RATIOS) * len(MODELS),
        "replicates_per_condition": REPLICATES,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("Article 4: Generating Trial Matrix")
    print("=" * 60)

    # Generate trials
    trials = generate_trial_matrix()
    print(f"\nGenerated {len(trials):,} trials")

    # Compute summary
    summary = compute_summary(trials)

    # Print summary
    print("\n--- Summary ---")
    print(f"Total trials: {summary['total_trials']:,}")
    print(f"Estimated cost: ${summary['total_estimated_cost_usd']:,.2f}")
    print(f"Estimated energy: {summary['total_estimated_energy_joules']:,.0f} J ({summary['total_estimated_energy_kwh']:.4f} kWh)")

    print("\nBy Benchmark:")
    for bench, stats in summary["by_benchmark"].items():
        print(f"  {bench}: {stats['count']:,} trials, ${stats['cost']:.2f}")

    print("\nBy Model:")
    for model, stats in summary["by_model"].items():
        print(f"  {model}: {stats['count']:,} trials, ${stats['cost']:.2f}")

    print("\nBy Compression Ratio:")
    for ratio, stats in summary["by_ratio"].items():
        print(f"  {ratio}: {stats['count']:,} trials")

    # Output files
    output_dir = "/home/warrenjo/src/tmp5/plex-vc-fund-platform/research/paper-v4/results"

    # Write trials to JSON Lines
    trials_file = f"{output_dir}/trial_matrix.jsonl"
    with open(trials_file, "w") as f:
        for trial in trials:
            f.write(json.dumps(asdict(trial)) + "\n")
    print(f"\nWrote trials to: {trials_file}")

    # Write summary to JSON
    summary_file = f"{output_dir}/trial_matrix_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary to: {summary_file}")

    # Write a simple batch manifest for execution
    batches = []
    batch_size = 1000
    for i in range(0, len(trials), batch_size):
        batch = {
            "batch_id": f"batch_{i // batch_size + 1:03d}",
            "start_idx": i,
            "end_idx": min(i + batch_size, len(trials)),
            "trial_count": min(batch_size, len(trials) - i),
        }
        batches.append(batch)

    manifest_file = f"{output_dir}/execution_manifest.json"
    manifest = {
        "total_trials": len(trials),
        "batch_size": batch_size,
        "num_batches": len(batches),
        "batches": batches,
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest to: {manifest_file}")

    print("\n" + "=" * 60)
    print("Trial matrix generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

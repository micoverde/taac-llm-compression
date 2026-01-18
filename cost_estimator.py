#!/usr/bin/env python3
"""
Cost Estimator for Algorithm Comparison Experiment
===================================================

Estimates API costs and experiment duration for each phase.

Usage:
    python cost_estimator.py --phase phase1_quick_validation
    python cost_estimator.py --all
"""

import argparse
from dataclasses import dataclass

# Import phase definitions
from algorithm_comparison import EXPERIMENT_PHASES, BENCHMARKS, ALGORITHM_CONFIGS


@dataclass
class CostEstimate:
    """Detailed cost estimate for an experiment phase."""
    phase_name: str
    total_trials: int
    total_api_calls: int

    # Token estimates
    avg_input_tokens_per_trial: int
    avg_output_tokens_per_trial: int
    total_input_tokens: int
    total_output_tokens: int

    # Cost by model
    cost_by_model: dict

    # Time estimates
    compression_time_hours: float
    inference_time_hours: float
    total_time_hours: float

    # Summary
    total_cost_usd: float
    cost_per_trial: float


# Model pricing (per 1M tokens)
MODEL_PRICING = {
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
    "claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

# Average token estimates by benchmark type
TOKEN_ESTIMATES = {
    "code": {
        "avg_input_uncompressed": 800,
        "avg_output": 300,
    },
    "cot": {
        "avg_input_uncompressed": 2000,  # CoT prompts tend to be longer
        "avg_output": 400,
    },
}


def estimate_phase_cost(phase_name: str) -> CostEstimate:
    """Generate detailed cost estimate for a phase."""
    phase = EXPERIMENT_PHASES[phase_name]

    # Calculate total trials
    total_trials = (
        len(phase.algorithms) *
        len(phase.ratios) *
        len(phase.benchmarks) *
        len(phase.models) *
        phase.samples_per_condition
    )

    # Calculate weighted average input tokens based on benchmarks
    code_benchmarks = [b for b in phase.benchmarks if BENCHMARKS[b]["type"].value == "code"]
    cot_benchmarks = [b for b in phase.benchmarks if BENCHMARKS[b]["type"].value == "cot"]

    if len(phase.benchmarks) > 0:
        code_weight = len(code_benchmarks) / len(phase.benchmarks)
        cot_weight = len(cot_benchmarks) / len(phase.benchmarks)
    else:
        code_weight = cot_weight = 0.5

    avg_uncompressed = (
        code_weight * TOKEN_ESTIMATES["code"]["avg_input_uncompressed"] +
        cot_weight * TOKEN_ESTIMATES["cot"]["avg_input_uncompressed"]
    )
    avg_output = (
        code_weight * TOKEN_ESTIMATES["code"]["avg_output"] +
        cot_weight * TOKEN_ESTIMATES["cot"]["avg_output"]
    )

    # Account for compression ratios
    avg_compression_ratio = sum(phase.ratios) / len(phase.ratios)
    avg_input_tokens = int(avg_uncompressed * avg_compression_ratio)

    total_input_tokens = avg_input_tokens * total_trials
    total_output_tokens = int(avg_output) * total_trials

    # Cost by model
    cost_by_model = {}
    trials_per_model = total_trials / len(phase.models)

    for model in phase.models:
        pricing = MODEL_PRICING.get(model, {"input": 0.50, "output": 2.00})

        model_input_tokens = int(avg_input_tokens * trials_per_model)
        model_output_tokens = int(avg_output * trials_per_model)

        input_cost = model_input_tokens * pricing["input"] / 1_000_000
        output_cost = model_output_tokens * pricing["output"] / 1_000_000

        cost_by_model[model] = {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
            "input_tokens": model_input_tokens,
            "output_tokens": model_output_tokens,
        }

    total_cost = sum(m["total_cost"] for m in cost_by_model.values())

    # Time estimates
    # Compression time: varies by algorithm
    avg_compression_ms = 0
    for alg in phase.algorithms:
        config = ALGORITHM_CONFIGS[alg]
        if "llmlingua1" in alg.value:
            avg_compression_ms += 3000  # 3 seconds for LLMLingua-1
        elif "llmlingua2" in alg.value:
            avg_compression_ms += 150  # 150ms for LLMLingua-2
        elif "selective" in alg.value:
            avg_compression_ms += 500  # 500ms for Selective Context
        else:
            avg_compression_ms += 5  # Random is instant

    avg_compression_ms /= len(phase.algorithms)
    compression_time_hours = (avg_compression_ms * total_trials) / 1000 / 3600

    # Inference time: ~2 seconds average per call
    avg_inference_ms = 2000
    inference_time_hours = (avg_inference_ms * total_trials) / 1000 / 3600

    # Add rate limiting overhead (~200ms per request)
    rate_limit_hours = (200 * total_trials) / 1000 / 3600

    total_time_hours = compression_time_hours + inference_time_hours + rate_limit_hours

    return CostEstimate(
        phase_name=phase.name,
        total_trials=total_trials,
        total_api_calls=total_trials,
        avg_input_tokens_per_trial=avg_input_tokens,
        avg_output_tokens_per_trial=int(avg_output),
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        cost_by_model=cost_by_model,
        compression_time_hours=compression_time_hours,
        inference_time_hours=inference_time_hours,
        total_time_hours=total_time_hours,
        total_cost_usd=total_cost,
        cost_per_trial=total_cost / total_trials if total_trials > 0 else 0,
    )


def print_estimate(estimate: CostEstimate) -> None:
    """Print formatted cost estimate."""
    print("\n" + "="*60)
    print(f"COST ESTIMATE: {estimate.phase_name}")
    print("="*60)

    print(f"\n--- Trial Summary ---")
    print(f"  Total trials: {estimate.total_trials:,}")
    print(f"  API calls: {estimate.total_api_calls:,}")

    print(f"\n--- Token Estimates ---")
    print(f"  Avg input tokens/trial: {estimate.avg_input_tokens_per_trial:,}")
    print(f"  Avg output tokens/trial: {estimate.avg_output_tokens_per_trial:,}")
    print(f"  Total input tokens: {estimate.total_input_tokens:,}")
    print(f"  Total output tokens: {estimate.total_output_tokens:,}")

    print(f"\n--- Cost by Model ---")
    for model, costs in estimate.cost_by_model.items():
        print(f"  {model}:")
        print(f"    Input:  ${costs['input_cost']:.2f} ({costs['input_tokens']:,} tokens)")
        print(f"    Output: ${costs['output_cost']:.2f} ({costs['output_tokens']:,} tokens)")
        print(f"    Total:  ${costs['total_cost']:.2f}")

    print(f"\n--- Time Estimates ---")
    print(f"  Compression time: {estimate.compression_time_hours:.2f} hours")
    print(f"  Inference time: {estimate.inference_time_hours:.2f} hours")
    print(f"  Total time: {estimate.total_time_hours:.2f} hours")

    print(f"\n--- Total Cost ---")
    print(f"  Total: ${estimate.total_cost_usd:.2f}")
    print(f"  Per trial: ${estimate.cost_per_trial:.4f}")

    # Recommendations
    print(f"\n--- Recommendations ---")
    if estimate.total_cost_usd > 100:
        print("  ! Consider running Phase 1 first to validate setup")
        print("  ! Use checkpointing to resume if interrupted")
    if estimate.total_time_hours > 4:
        print("  ! Consider running overnight or on VM")
        print("  ! Increase parallel workers if API rate limits allow")


def print_all_phases_summary() -> None:
    """Print summary table of all phases."""
    print("\n" + "="*80)
    print("ALGORITHM COMPARISON EXPERIMENT: COST SUMMARY")
    print("="*80)

    print("\n{:<25} {:>10} {:>12} {:>12} {:>12}".format(
        "Phase", "Trials", "Est. Cost", "Est. Time", "Per Trial"
    ))
    print("-"*80)

    total_trials = 0
    total_cost = 0
    total_time = 0

    for phase_name in EXPERIMENT_PHASES:
        estimate = estimate_phase_cost(phase_name)
        print("{:<25} {:>10,} {:>12} {:>12} {:>12}".format(
            phase_name[:24],
            estimate.total_trials,
            f"${estimate.total_cost_usd:.2f}",
            f"{estimate.total_time_hours:.1f}h",
            f"${estimate.cost_per_trial:.4f}",
        ))
        total_trials += estimate.total_trials
        total_cost += estimate.total_cost_usd
        total_time += estimate.total_time_hours

    print("-"*80)
    print("{:<25} {:>10,} {:>12} {:>12}".format(
        "TOTAL",
        total_trials,
        f"${total_cost:.2f}",
        f"{total_time:.1f}h",
    ))

    print("\n--- Phase Descriptions ---")
    for phase_name, phase in EXPERIMENT_PHASES.items():
        print(f"\n  {phase_name}:")
        print(f"    {phase.description}")
        print(f"    Algorithms: {', '.join(a.value for a in phase.algorithms)}")
        print(f"    Benchmarks: {', '.join(phase.benchmarks)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cost Estimator for Algorithm Comparison Experiment"
    )
    parser.add_argument(
        "--phase",
        choices=list(EXPERIMENT_PHASES.keys()),
        help="Specific phase to estimate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show summary of all phases",
    )

    args = parser.parse_args()

    if args.all or (not args.phase):
        print_all_phases_summary()
    else:
        estimate = estimate_phase_cost(args.phase)
        print_estimate(estimate)


if __name__ == "__main__":
    main()

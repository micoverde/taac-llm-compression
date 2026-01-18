#!/usr/bin/env python3
"""
TAAC Evaluation and Comparison Script
======================================

Compares TAAC against baseline compression strategies:
1. Fixed ratio (r=0.6 for all prompts)
2. Task-based fixed (r=0.6 for code, r=0.8 for CoT)
3. TAAC (adaptive per prompt with quality gating)

Metrics:
- Quality preservation (task performance relative to uncompressed)
- Cost savings (token reduction)
- Pareto efficiency (cost vs quality tradeoff)
- Latency overhead

Author: Dr. Amanda Foster, Bona Opera Studios
Date: January 2026
"""

import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import defaultdict
import os
import sys

import numpy as np

# Import TAAC and baselines
from taac_algorithm import (
    TAAC, TAACConfig, TaskType,
    FixedRatioCompressor, TaskBasedFixedCompressor,
    CompressionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluating a compression strategy."""
    strategy: str
    prompt_id: str
    task_type: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    baseline_quality: float
    quality_preserved: float
    cost_savings: float
    latency_ms: float
    quality_gated: bool = False


@dataclass
class StrategyStats:
    """Aggregated statistics for a compression strategy."""
    strategy: str
    n_samples: int
    mean_compression_ratio: float
    std_compression_ratio: float
    mean_quality_preserved: float
    std_quality_preserved: float
    mean_cost_savings: float
    std_cost_savings: float
    mean_latency_ms: float
    pareto_optimal_fraction: float  # Fraction of samples on Pareto frontier


class CompressionEvaluator:
    """Evaluates and compares compression strategies."""

    def __init__(self, config: Optional[TAACConfig] = None):
        self.config = config or TAACConfig()

        # Initialize strategies
        self.strategies = {
            "fixed_r60": FixedRatioCompressor(ratio=0.6),
            "fixed_r70": FixedRatioCompressor(ratio=0.7),
            "task_based_fixed": TaskBasedFixedCompressor(self.config),
            "taac": TAAC(self.config),
        }

        self.results: List[EvaluationResult] = []

    def add_strategy(self, name: str, strategy):
        """Add a custom compression strategy."""
        self.strategies[name] = strategy

    def evaluate_prompt(
        self,
        prompt: str,
        prompt_id: str,
        task_type: str,
        baseline_quality: float,
    ) -> List[EvaluationResult]:
        """Evaluate all strategies on a single prompt."""
        results = []

        for strategy_name, compressor in self.strategies.items():
            start_time = time.perf_counter()

            try:
                result = compressor.compress(prompt)
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Extract compression info
                if isinstance(result, CompressionResult):
                    ratio = result.compression_ratio
                    original_tokens = result.original_tokens
                    compressed_tokens = result.compressed_tokens
                    quality_gated = result.quality_gated
                else:
                    ratio = getattr(result, 'compression_ratio', 0.7)
                    original_tokens = len(prompt.split())
                    compressed_tokens = int(original_tokens * ratio)
                    quality_gated = False

                # Simulate quality (in real evaluation, would use actual model)
                quality_score = self._simulate_quality(
                    task_type, ratio, baseline_quality
                )

                eval_result = EvaluationResult(
                    strategy=strategy_name,
                    prompt_id=prompt_id,
                    task_type=task_type,
                    original_tokens=original_tokens,
                    compressed_tokens=compressed_tokens,
                    compression_ratio=ratio,
                    quality_score=quality_score,
                    baseline_quality=baseline_quality,
                    quality_preserved=quality_score / baseline_quality if baseline_quality > 0 else 0,
                    cost_savings=1 - ratio,
                    latency_ms=latency_ms,
                    quality_gated=quality_gated,
                )
                results.append(eval_result)

            except Exception as e:
                logger.warning(f"Error evaluating {strategy_name} on {prompt_id}: {e}")

        return results

    def _simulate_quality(
        self,
        task_type: str,
        compression_ratio: float,
        baseline_quality: float,
    ) -> float:
        """Simulate quality based on empirical curves from Johnson (2026)."""
        r = compression_ratio

        if task_type == "code":
            if r >= 0.6:
                degradation = 0.05 * (1 - r) / 0.4
            else:
                degradation = 0.05 + 0.8 * ((0.6 - r) / 0.6) ** 1.5
        elif task_type in ["cot", "reasoning", "math"]:
            degradation = 0.4 * (1 - r)
        else:
            code_deg = 0.05 * (1 - r) / 0.4 if r >= 0.6 else 0.05 + 0.8 * ((0.6 - r) / 0.6) ** 1.5
            cot_deg = 0.4 * (1 - r)
            degradation = 0.5 * code_deg + 0.5 * cot_deg

        degradation += np.random.normal(0, 0.02)
        degradation = max(0, min(1, degradation))

        return baseline_quality * (1 - degradation)

    def run_evaluation(
        self,
        prompts: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> List[EvaluationResult]:
        """Run evaluation on all prompts."""
        logger.info(f"Evaluating {len(prompts)} prompts across {len(self.strategies)} strategies")

        all_results = []

        for i, prompt_data in enumerate(prompts):
            prompt = prompt_data.get("prompt", "")
            prompt_id = prompt_data.get("id", f"prompt_{i}")
            task_type = prompt_data.get("task_type", "hybrid")
            baseline_quality = prompt_data.get("baseline_quality", 0.95)

            results = self.evaluate_prompt(prompt, prompt_id, task_type, baseline_quality)
            all_results.extend(results)

            if (i + 1) % 20 == 0:
                logger.info(f"Evaluated {i + 1}/{len(prompts)} prompts")

        self.results = all_results

        if output_path:
            self.save_results(output_path)

        return all_results

    def save_results(self, output_path: str):
        """Save results to JSONL file."""
        with open(output_path, 'w') as f:
            for result in self.results:
                f.write(json.dumps(asdict(result)) + '\n')
        logger.info(f"Saved {len(self.results)} results to {output_path}")

    def compute_pareto_frontier(self, results: List[EvaluationResult]) -> set:
        """Identify Pareto-optimal results (cost vs quality).

        A result is Pareto-optimal if no other result has both:
        - Higher quality preservation AND higher cost savings
        """
        pareto_set = set()

        for i, r1 in enumerate(results):
            is_dominated = False
            for j, r2 in enumerate(results):
                if i == j:
                    continue
                # r2 dominates r1 if r2 is better in both dimensions
                if (r2.quality_preserved >= r1.quality_preserved and
                    r2.cost_savings >= r1.cost_savings and
                    (r2.quality_preserved > r1.quality_preserved or
                     r2.cost_savings > r1.cost_savings)):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_set.add(i)

        return pareto_set

    def compute_stats(self) -> Dict[str, StrategyStats]:
        """Compute aggregated statistics per strategy."""
        by_strategy = defaultdict(list)
        for result in self.results:
            by_strategy[result.strategy].append(result)

        stats = {}

        for strategy, results in by_strategy.items():
            # Compute Pareto optimality per prompt
            by_prompt = defaultdict(list)
            for r in results:
                by_prompt[r.prompt_id].append(r)

            pareto_count = 0
            total_prompts = 0

            for prompt_id, prompt_results in by_prompt.items():
                # Find all results for this prompt across strategies
                all_prompt_results = [r for r in self.results if r.prompt_id == prompt_id]
                pareto_indices = self.compute_pareto_frontier(all_prompt_results)

                # Check if this strategy's result is Pareto-optimal
                for i, r in enumerate(all_prompt_results):
                    if r.strategy == strategy and i in pareto_indices:
                        pareto_count += 1
                        break
                total_prompts += 1

            pareto_fraction = pareto_count / total_prompts if total_prompts > 0 else 0

            stats[strategy] = StrategyStats(
                strategy=strategy,
                n_samples=len(results),
                mean_compression_ratio=np.mean([r.compression_ratio for r in results]),
                std_compression_ratio=np.std([r.compression_ratio for r in results]),
                mean_quality_preserved=np.mean([r.quality_preserved for r in results]),
                std_quality_preserved=np.std([r.quality_preserved for r in results]),
                mean_cost_savings=np.mean([r.cost_savings for r in results]),
                std_cost_savings=np.std([r.cost_savings for r in results]),
                mean_latency_ms=np.mean([r.latency_ms for r in results]),
                pareto_optimal_fraction=pareto_fraction,
            )

        return stats

    def compute_stats_by_task(self) -> Dict[str, Dict[str, StrategyStats]]:
        """Compute statistics stratified by task type."""
        by_strategy_task = defaultdict(lambda: defaultdict(list))

        for result in self.results:
            by_strategy_task[result.strategy][result.task_type].append(result)

        stats = {}

        for strategy, by_task in by_strategy_task.items():
            stats[strategy] = {}
            for task_type, results in by_task.items():
                stats[strategy][task_type] = StrategyStats(
                    strategy=f"{strategy}_{task_type}",
                    n_samples=len(results),
                    mean_compression_ratio=np.mean([r.compression_ratio for r in results]),
                    std_compression_ratio=np.std([r.compression_ratio for r in results]),
                    mean_quality_preserved=np.mean([r.quality_preserved for r in results]),
                    std_quality_preserved=np.std([r.quality_preserved for r in results]),
                    mean_cost_savings=np.mean([r.cost_savings for r in results]),
                    std_cost_savings=np.std([r.cost_savings for r in results]),
                    mean_latency_ms=np.mean([r.latency_ms for r in results]),
                    pareto_optimal_fraction=0,  # Would need per-task Pareto analysis
                )

        return stats

    def print_comparison(self):
        """Print comparison table."""
        stats = self.compute_stats()

        print("\n" + "=" * 90)
        print("COMPRESSION STRATEGY COMPARISON")
        print("=" * 90)
        print(f"\n{'Strategy':<20} {'Ratio':>8} {'Quality':>10} {'Savings':>10} {'Latency':>12} {'Pareto':>10}")
        print("-" * 90)

        for strategy in sorted(stats.keys()):
            s = stats[strategy]
            print(
                f"{strategy:<20} "
                f"{s.mean_compression_ratio:>7.3f} "
                f"{s.mean_quality_preserved:>9.1%} "
                f"{s.mean_cost_savings:>9.1%} "
                f"{s.mean_latency_ms:>10.1f}ms "
                f"{s.pareto_optimal_fraction:>9.1%}"
            )

        print("-" * 90)

        # Highlight TAAC improvements
        if 'taac' in stats and 'fixed_r60' in stats:
            taac = stats['taac']
            fixed = stats['fixed_r60']

            quality_improvement = (taac.mean_quality_preserved - fixed.mean_quality_preserved) / fixed.mean_quality_preserved
            savings_change = (taac.mean_cost_savings - fixed.mean_cost_savings) / fixed.mean_cost_savings

            print(f"\nTAAC vs Fixed r=0.6:")
            print(f"  Quality improvement: {quality_improvement:+.1%}")
            print(f"  Cost savings change: {savings_change:+.1%}")

        if 'taac' in stats and 'task_based_fixed' in stats:
            taac = stats['taac']
            task_fixed = stats['task_based_fixed']

            quality_improvement = (taac.mean_quality_preserved - task_fixed.mean_quality_preserved) / task_fixed.mean_quality_preserved
            savings_change = (taac.mean_cost_savings - task_fixed.mean_cost_savings) / task_fixed.mean_cost_savings

            print(f"\nTAAC vs Task-Based Fixed:")
            print(f"  Quality improvement: {quality_improvement:+.1%}")
            print(f"  Cost savings change: {savings_change:+.1%}")

        # By task type breakdown
        print("\n" + "=" * 90)
        print("BREAKDOWN BY TASK TYPE")
        print("=" * 90)

        task_stats = self.compute_stats_by_task()

        for task_type in ['code', 'cot', 'hybrid']:
            print(f"\n{task_type.upper()} Tasks:")
            print(f"{'Strategy':<20} {'Ratio':>8} {'Quality':>10} {'Savings':>10}")
            print("-" * 50)

            for strategy in sorted(task_stats.keys()):
                if task_type in task_stats[strategy]:
                    s = task_stats[strategy][task_type]
                    print(
                        f"{strategy:<20} "
                        f"{s.mean_compression_ratio:>7.3f} "
                        f"{s.mean_quality_preserved:>9.1%} "
                        f"{s.mean_cost_savings:>9.1%}"
                    )


def generate_evaluation_prompts(
    n_code: int = 100,
    n_cot: int = 100,
    n_hybrid: int = 50,
) -> List[Dict[str, Any]]:
    """Generate prompts for evaluation."""
    prompts = []

    # Code prompts
    for i in range(n_code):
        operation = ['sum', 'product', 'maximum', 'minimum', 'average'][i % 5]
        prompts.append({
            "id": f"code_{i:04d}",
            "prompt": f"""def compute_{operation}(data: list) -> float:
    '''
    Compute the {operation} of all elements in the input list.

    Args:
        data: A list of numerical values (integers or floats)

    Returns:
        The {operation} of all elements

    Examples:
        >>> compute_{operation}([1, 2, 3, 4, 5])
        {[15, 120, 5, 1, 3.0][i % 5]}

    Note:
        - The list is guaranteed to be non-empty
        - Handle both integers and floats
    '''
    # Implement the function
""",
            "task_type": "code",
            "baseline_quality": 0.88 + np.random.uniform(0, 0.1),
        })

    # CoT prompts
    for i in range(n_cot):
        problem_type = ['geometry', 'algebra', 'probability', 'combinatorics', 'calculus'][i % 5]
        prompts.append({
            "id": f"cot_{i:04d}",
            "prompt": f"""Solve the following {problem_type} problem step by step.

Problem {i + 1}:
A company has {50 + i * 5} employees. They want to form teams for a project.
Each team must have exactly {3 + (i % 4)} members.
If {10 + (i % 20)} employees are on vacation and cannot participate,
how many complete teams can be formed?

Show all your work:
1. First, identify the total number of available employees
2. Then, determine how many complete teams can be formed
3. Calculate any remaining employees
4. Verify your answer

Solution:
""",
            "task_type": "cot",
            "baseline_quality": 0.82 + np.random.uniform(0, 0.12),
        })

    # Hybrid prompts
    for i in range(n_hybrid):
        prompts.append({
            "id": f"hybrid_{i:04d}",
            "prompt": f"""Task: Implement a mathematical algorithm

Background:
The Fibonacci sequence is defined mathematically as:
F(0) = 0, F(1) = 1
F(n) = F(n-1) + F(n-2) for n > 1

The ratio F(n+1)/F(n) converges to the golden ratio phi = (1 + sqrt(5))/2.

Part 1: Mathematical derivation
Derive the closed-form expression for F(n) using the characteristic equation.

Part 2: Implementation
Write a Python function that efficiently computes F(n) for large n (up to 10^6).

def fibonacci_efficient(n: int) -> int:
    '''
    Compute the nth Fibonacci number efficiently.

    Args:
        n: Index in Fibonacci sequence (0-indexed)

    Returns:
        The nth Fibonacci number

    Time complexity: O(log n)
    Space complexity: O(1)
    '''
    # Your implementation here
""",
            "task_type": "hybrid",
            "baseline_quality": 0.85 + np.random.uniform(0, 0.1),
        })

    return prompts


def visualize_pareto_frontier(
    results: List[EvaluationResult],
    output_path: Optional[str] = None,
):
    """Visualize Pareto frontier of cost vs quality."""
    try:
        import matplotlib.pyplot as plt

        by_strategy = defaultdict(list)
        for r in results:
            by_strategy[r.strategy].append((r.cost_savings, r.quality_preserved))

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        colors = plt.cm.tab10(np.linspace(0, 1, len(by_strategy)))

        for (strategy, points), color in zip(by_strategy.items(), colors):
            savings = [p[0] for p in points]
            quality = [p[1] for p in points]
            ax.scatter(savings, quality, c=[color], label=strategy, alpha=0.6, s=30)

            # Draw mean point
            ax.scatter(
                [np.mean(savings)], [np.mean(quality)],
                c=[color], marker='X', s=200, edgecolors='black', linewidths=2
            )

        ax.set_xlabel('Cost Savings', fontsize=12)
        ax.set_ylabel('Quality Preserved', fontsize=12)
        ax.set_title('Compression Strategy Comparison: Cost vs Quality Tradeoff', fontsize=14)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        # Mark ideal point
        ax.annotate(
            'Ideal\n(high savings,\nhigh quality)',
            xy=(0.5, 0.98), fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved plot to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.warning("matplotlib not available for visualization")


def main():
    parser = argparse.ArgumentParser(description="TAAC Evaluation and Comparison")

    parser.add_argument(
        "--data",
        type=str,
        help="Path to prompts data (JSON or JSONL)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic prompts for evaluation",
    )
    parser.add_argument(
        "--n-code",
        type=int,
        default=100,
        help="Number of synthetic code prompts",
    )
    parser.add_argument(
        "--n-cot",
        type=int,
        default=100,
        help="Number of synthetic CoT prompts",
    )
    parser.add_argument(
        "--n-hybrid",
        type=int,
        default=50,
        help="Number of synthetic hybrid prompts",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.jsonl",
        help="Output path for results",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Path to save Pareto frontier plot",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load or generate prompts
    if args.data:
        with open(args.data) as f:
            if args.data.endswith('.jsonl'):
                prompts = [json.loads(line) for line in f]
            else:
                prompts = json.load(f)
        logger.info(f"Loaded {len(prompts)} prompts from {args.data}")
    elif args.synthetic:
        prompts = generate_evaluation_prompts(args.n_code, args.n_cot, args.n_hybrid)
        logger.info(f"Generated {len(prompts)} synthetic prompts")
    else:
        # Default: moderate synthetic set
        prompts = generate_evaluation_prompts(50, 50, 25)
        logger.info(f"Generated {len(prompts)} prompts (default set)")

    # Run evaluation
    evaluator = CompressionEvaluator()
    results = evaluator.run_evaluation(prompts, args.output)

    # Print comparison
    evaluator.print_comparison()

    # Visualize
    if args.plot:
        visualize_pareto_frontier(results, args.plot)

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

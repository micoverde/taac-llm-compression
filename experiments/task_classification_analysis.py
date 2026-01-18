#!/usr/bin/env python3
"""
Experiment 3B: Task Classification Analysis
============================================

Analyzes existing MBPP experiment results to identify task categories
that tolerate ultra-compression better than others.

Hypothesis: Certain task types (string manipulation, I/O formatting)
can tolerate r ≤ 0.4 while algorithmic tasks require r ≥ 0.6.

Author: Dr. Michael Park
Date: January 2026
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Task category definitions with keyword patterns
TASK_CATEGORIES = {
    "string_manipulation": {
        "keywords": [
            "string", "reverse", "split", "join", "replace", "format",
            "substring", "concatenate", "capitalize", "lower", "upper",
            "strip", "trim", "char", "character", "letter", "word"
        ],
        "code_patterns": [
            r"\.split\(", r"\.join\(", r"\.replace\(", r"\.strip\(",
            r"\.upper\(", r"\.lower\(", r"\[::-1\]", r"\.find\("
        ],
        "description": "String manipulation and text processing"
    },
    "list_operations": {
        "keywords": [
            "list", "array", "sort", "filter", "map", "reduce",
            "find", "search", "index", "element", "append", "remove",
            "insert", "pop", "slice", "flatten", "unique", "duplicate"
        ],
        "code_patterns": [
            r"\.sort\(", r"\.append\(", r"\.remove\(", r"\.pop\(",
            r"\.index\(", r"sorted\(", r"filter\(", r"map\("
        ],
        "description": "List/array manipulation"
    },
    "mathematical": {
        "keywords": [
            "prime", "factorial", "fibonacci", "gcd", "lcm", "power",
            "square", "cube", "sum", "product", "average", "mean",
            "divisor", "multiple", "even", "odd", "number", "digit",
            "calculate", "compute", "math"
        ],
        "code_patterns": [
            r"math\.", r"%\s*==\s*0", r"\*\*", r"//",
            r"range\(.*,.*\)", r"factorial", r"sqrt"
        ],
        "description": "Mathematical computations"
    },
    "algorithmic": {
        "keywords": [
            "binary", "search", "traverse", "dynamic", "recursive",
            "graph", "tree", "node", "path", "depth", "breadth",
            "backtrack", "permutation", "combination", "subsequence",
            "longest", "shortest", "maximum", "minimum", "optimal"
        ],
        "code_patterns": [
            r"def\s+\w+\([^)]*\):[^}]*\w+\([^)]*\)",  # Recursive calls
            r"visited", r"queue", r"stack", r"memo", r"dp\["
        ],
        "description": "Complex algorithms (DP, graphs, recursion)"
    },
    "io_formatting": {
        "keywords": [
            "print", "display", "output", "format", "convert",
            "parse", "read", "input", "extract", "tuple", "dict",
            "dictionary", "key", "value", "pair"
        ],
        "code_patterns": [
            r"print\(", r"\.format\(", r"f\"", r"str\(",
            r"int\(", r"float\(", r"dict\(", r"tuple\("
        ],
        "description": "I/O and data formatting"
    }
}


def classify_task(task: dict) -> str:
    """Classify a task into a category based on prompt and code.

    Args:
        task: Task dict with 'prompt' and optionally 'code' keys

    Returns:
        Category name or 'other'
    """
    prompt = task.get("prompt", "").lower()
    code = task.get("code", "").lower() if "code" in task else ""

    scores = defaultdict(int)

    for category, config in TASK_CATEGORIES.items():
        # Check keywords in prompt
        for keyword in config["keywords"]:
            if keyword in prompt:
                scores[category] += 2
            if keyword in code:
                scores[category] += 1

        # Check code patterns
        for pattern in config["code_patterns"]:
            if re.search(pattern, code, re.IGNORECASE):
                scores[category] += 3

    if not scores:
        return "other"

    # Return category with highest score
    return max(scores.items(), key=lambda x: x[1])[0]


def load_results(results_file: Path) -> list[dict]:
    """Load experiment results from JSONL file."""
    results = []
    with open(results_file) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_mbpp_tasks() -> dict[int, dict]:
    """Load MBPP task metadata."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
        return {
            item["task_id"]: {
                "task_id": item["task_id"],
                "prompt": item["text"],
                "code": item["code"]
            }
            for item in ds
        }
    except Exception as e:
        print(f"Warning: Could not load MBPP dataset: {e}")
        return {}


def analyze_results_by_category(
    results: list[dict],
    tasks: dict[int, dict]
) -> dict[str, dict[float, dict]]:
    """Analyze pass rates by task category and compression ratio.

    Returns:
        Dict mapping category -> ratio -> {passed, total, rate}
    """
    # Group results by category and ratio
    by_category_ratio = defaultdict(lambda: defaultdict(list))

    for r in results:
        task_id = r.get("task_id")
        ratio = r.get("compression_ratio")
        passed = r.get("passed", False)

        if task_id in tasks:
            category = classify_task(tasks[task_id])
        else:
            category = "unknown"

        by_category_ratio[category][ratio].append(passed)

    # Calculate statistics
    analysis = {}
    for category, ratios in by_category_ratio.items():
        analysis[category] = {}
        for ratio, results_list in sorted(ratios.items()):
            total = len(results_list)
            passed = sum(results_list)
            rate = passed / total if total > 0 else 0
            analysis[category][ratio] = {
                "passed": passed,
                "total": total,
                "rate": rate
            }

    return analysis


def find_ultra_compressible_tasks(
    analysis: dict[str, dict[float, dict]],
    threshold_ratio: float = 0.4,
    min_quality: float = 0.3
) -> list[str]:
    """Identify categories that maintain quality at aggressive compression.

    Args:
        analysis: Output from analyze_results_by_category
        threshold_ratio: Compression ratio to check
        min_quality: Minimum pass rate to consider "acceptable"

    Returns:
        List of category names that tolerate ultra-compression
    """
    ultra_compressible = []

    for category, ratios in analysis.items():
        if threshold_ratio in ratios:
            rate = ratios[threshold_ratio]["rate"]
            if rate >= min_quality:
                ultra_compressible.append(category)

    return ultra_compressible


def print_analysis_table(analysis: dict[str, dict[float, dict]]):
    """Print formatted analysis table."""
    # Get all ratios
    all_ratios = set()
    for ratios in analysis.values():
        all_ratios.update(ratios.keys())
    all_ratios = sorted(all_ratios)

    # Header
    header = f"{'Category':<20}"
    for ratio in all_ratios:
        header += f" | r={ratio:<5}"
    print(header)
    print("-" * len(header))

    # Rows
    for category in sorted(analysis.keys()):
        row = f"{category:<20}"
        for ratio in all_ratios:
            if ratio in analysis[category]:
                rate = analysis[category][ratio]["rate"] * 100
                row += f" | {rate:>5.1f}%"
            else:
                row += f" | {'N/A':>6}"
        print(row)


def print_ultra_compression_recommendations(
    analysis: dict[str, dict[float, dict]]
):
    """Print recommendations for ultra-compression routing."""
    print("\n" + "=" * 60)
    print("ULTRA-COMPRESSION ROUTING RECOMMENDATIONS")
    print("=" * 60)

    # Find best ratio for each category (maintaining >50% quality)
    target_quality = 0.5

    recommendations = []
    for category, ratios in analysis.items():
        best_ratio = None
        for ratio in sorted(ratios.keys()):
            if ratios[ratio]["rate"] >= target_quality:
                best_ratio = ratio
                break

        if best_ratio is not None:
            recommendations.append({
                "category": category,
                "recommended_ratio": best_ratio,
                "expected_quality": ratios[best_ratio]["rate"]
            })

    # Sort by recommended ratio (most aggressive first)
    recommendations.sort(key=lambda x: x["recommended_ratio"])

    print(f"\nTarget: >{target_quality*100:.0f}% pass rate")
    print("-" * 50)
    print(f"{'Category':<20} | {'Recommended r':>12} | {'Expected Quality':>15}")
    print("-" * 50)

    for rec in recommendations:
        print(
            f"{rec['category']:<20} | "
            f"{rec['recommended_ratio']:>12.1f} | "
            f"{rec['expected_quality']*100:>14.1f}%"
        )

    # Calculate potential savings
    print("\n" + "-" * 50)
    print("POTENTIAL SAVINGS (vs uniform r=0.6):")

    baseline_ratio = 0.6
    total_tasks = sum(
        sum(r["total"] for r in ratios.values())
        for ratios in analysis.values()
    ) // len(list(analysis.values())[0])  # Approximate unique tasks

    tokens_saved = 0
    for rec in recommendations:
        cat_tasks = sum(
            analysis[rec["category"]][r]["total"]
            for r in analysis[rec["category"]]
        ) // len(analysis[rec["category"]])

        if rec["recommended_ratio"] < baseline_ratio:
            savings = (baseline_ratio - rec["recommended_ratio"]) * cat_tasks
            tokens_saved += savings
            print(f"  {rec['category']}: {savings:.0f} tokens saved per batch")

    print(f"\nTotal additional savings: {tokens_saved:.0f} tokens")
    print(f"Improvement over uniform r=0.6: {tokens_saved/total_tasks*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3B: Task Classification Analysis"
    )
    parser.add_argument(
        "--results-file", type=Path,
        default=Path("results/mbpp_results.jsonl"),
        help="Path to MBPP experiment results"
    )
    parser.add_argument(
        "--output-file", type=Path,
        default=None,
        help="Output file for analysis (JSON)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.4,
        help="Compression ratio threshold for ultra-compression"
    )

    args = parser.parse_args()

    # Load results and tasks
    print("Loading MBPP experiment results...")
    if not args.results_file.exists():
        print(f"Error: Results file not found: {args.results_file}")
        print("Run the MBPP experiment first or specify correct path.")
        return

    results = load_results(args.results_file)
    print(f"Loaded {len(results)} trial results")

    print("\nLoading MBPP task metadata...")
    tasks = load_mbpp_tasks()
    print(f"Loaded {len(tasks)} task definitions")

    # Analyze
    print("\n" + "=" * 60)
    print("TASK CLASSIFICATION ANALYSIS")
    print("=" * 60)

    analysis = analyze_results_by_category(results, tasks)

    print("\nPass Rate by Category and Compression Ratio:")
    print("-" * 50)
    print_analysis_table(analysis)

    # Find ultra-compressible categories
    ultra = find_ultra_compressible_tasks(
        analysis,
        threshold_ratio=args.threshold,
        min_quality=0.2
    )

    print(f"\n\nCategories tolerating r={args.threshold} (>20% quality):")
    for cat in ultra:
        rate = analysis[cat][args.threshold]["rate"] * 100
        print(f"  - {cat}: {rate:.1f}% pass rate")

    # Recommendations
    print_ultra_compression_recommendations(analysis)

    # Save output
    if args.output_file:
        output = {
            "analysis": {
                cat: {str(r): stats for r, stats in ratios.items()}
                for cat, ratios in analysis.items()
            },
            "ultra_compressible_at_0.4": ultra,
            "task_categories": {
                cat: config["description"]
                for cat, config in TASK_CATEGORIES.items()
            }
        }
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nAnalysis saved to: {args.output_file}")


if __name__ == "__main__":
    main()

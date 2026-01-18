#!/usr/bin/env python3
"""
Experiment 3A: Signature Preservation
=====================================

Tests whether post-compression injection of function signatures
recovers code generation quality at aggressive compression ratios (r ≤ 0.4).

Hypothesis: Function Identity Collapse at r=0.3 (70.9% NameError) can be
mitigated by re-injecting the function signature after compression.

Author: Dr. Sarah Chen
Date: January 2026
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class SignaturePreservationResult:
    """Result from a single trial."""
    task_id: int
    compression_ratio: float
    condition: str  # 'baseline', 'sig_inject', 'sig_inject_plus'
    original_prompt: str
    compressed_prompt: str
    final_prompt: str
    generated_code: str
    passed: bool
    error_type: Optional[str]
    execution_time_ms: float
    cost_usd: float


# =============================================================================
# Signature Extraction
# =============================================================================

def extract_function_signature(code: str) -> str:
    """Extract function signature from Python code.

    Args:
        code: Python code string

    Returns:
        Function signature (e.g., "def calculate_sum(a, b):")
    """
    # Match function definition
    match = re.search(r"def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:", code)
    if match:
        return match.group(0)
    return "def solve():"


def extract_function_name(code: str) -> str:
    """Extract just the function name."""
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if match:
        return match.group(1)
    return "solve"


def extract_parameters(code: str) -> list[str]:
    """Extract parameter names from function signature."""
    match = re.search(r"def\s+\w+\s*\(([^)]*)\)", code)
    if match:
        params_str = match.group(1)
        if not params_str.strip():
            return []
        # Parse parameters (handle type hints, defaults)
        params = []
        for p in params_str.split(","):
            p = p.strip()
            # Remove type hints
            if ":" in p:
                p = p.split(":")[0].strip()
            # Remove default values
            if "=" in p:
                p = p.split("=")[0].strip()
            if p:
                params.append(p)
        return params
    return []


# =============================================================================
# Prompt Engineering
# =============================================================================

def create_baseline_prompt(task: dict, compressed_prompt: str) -> str:
    """Baseline: use compressed prompt directly."""
    return compressed_prompt


def create_signature_inject_prompt(task: dict, compressed_prompt: str) -> str:
    """Inject function signature after compression."""
    signature = extract_function_signature(task["code"])

    prompt = f"""You are an expert Python programmer.

Required Function Signature:
{signature}

Problem (compressed):
{compressed_prompt}

Write ONLY the Python function implementation starting with the exact signature above.
Do not include any explanations, comments, or test cases.

Function:
"""
    return prompt


def create_signature_inject_plus_prompt(task: dict, compressed_prompt: str) -> str:
    """Inject signature with parameter hints."""
    signature = extract_function_signature(task["code"])
    func_name = extract_function_name(task["code"])
    params = extract_parameters(task["code"])

    param_hints = ""
    if params:
        param_hints = f"\nParameters: {', '.join(params)}"

    prompt = f"""You are an expert Python programmer.

Required Function:
- Name: {func_name}
- Signature: {signature}{param_hints}

Problem (compressed):
{compressed_prompt}

Write ONLY the Python function implementation starting with the exact signature above.
Do not include any explanations, comments, or test cases.

Function:
"""
    return prompt


CONDITION_HANDLERS = {
    "baseline": create_baseline_prompt,
    "sig_inject": create_signature_inject_prompt,
    "sig_inject_plus": create_signature_inject_plus_prompt,
}


# =============================================================================
# Compression (using LLMLingua-2 if available, else simulation)
# =============================================================================

class Compressor:
    """Wrapper for prompt compression."""

    def __init__(self):
        self.llmlingua = None
        self._try_load_llmlingua()

    def _try_load_llmlingua(self):
        """Try to load LLMLingua-2."""
        try:
            from llmlingua import PromptCompressor
            self.llmlingua = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu"  # CPU for current VM
            )
            logger.info("LLMLingua-2 loaded successfully")
        except Exception as e:
            logger.warning(f"LLMLingua-2 not available: {e}")
            logger.warning("Using simulated compression")

    def compress(self, text: str, ratio: float) -> str:
        """Compress text to target ratio."""
        if ratio >= 1.0:
            return text

        if self.llmlingua:
            result = self.llmlingua.compress_prompt(
                text,
                rate=ratio,
                force_tokens=["def", "return", "if", "for", "while", "class"]
            )
            return result["compressed_prompt"]
        else:
            # Simulated compression: keep ratio of tokens
            tokens = text.split()
            keep_count = max(1, int(len(tokens) * ratio))
            # Keep first and last portions (heuristic)
            first = keep_count // 2
            last = keep_count - first
            kept = tokens[:first] + tokens[-last:] if last > 0 else tokens[:first]
            return " ".join(kept)


# =============================================================================
# Code Generation
# =============================================================================

def generate_code(prompt: str, model: str = "gpt-4o-mini") -> tuple[str, float]:
    """Generate code using OpenAI API.

    Returns:
        Tuple of (generated_code, cost_usd)
    """
    try:
        from openai import OpenAI
        client = OpenAI()

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0
        )

        code = response.choices[0].message.content

        # Calculate cost (gpt-4o-mini pricing)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

        return code, cost

    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return "", 0.0


# =============================================================================
# Code Execution
# =============================================================================

def extract_code_block(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code block
    if "```python" in response:
        match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif "```" in response:
        match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Return as-is if no code block
    return response.strip()


def execute_tests(code: str, test_cases: list[str]) -> tuple[bool, Optional[str]]:
    """Execute test cases against generated code.

    Returns:
        Tuple of (passed, error_type)
    """
    code = extract_code_block(code)

    # Create test script
    setup = """
import sys
import math
from collections import *
from itertools import *
from functools import *
from typing import *
"""

    full_code = setup + "\n" + code + "\n"

    # Add test assertions
    for test in test_cases:
        full_code += f"\n{test}"

    try:
        exec(full_code, {"__builtins__": __builtins__})
        return True, None
    except NameError as e:
        return False, "NameError"
    except TypeError as e:
        return False, "TypeError"
    except AssertionError as e:
        return False, "AssertionError"
    except Exception as e:
        return False, type(e).__name__


# =============================================================================
# Experiment Runner
# =============================================================================

def load_mbpp_tasks(sample_size: int = 300) -> list[dict]:
    """Load MBPP tasks."""
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
        tasks = []
        for i, item in enumerate(ds):
            if i >= sample_size:
                break
            tasks.append({
                "task_id": item["task_id"],
                "prompt": item["text"],
                "code": item["code"],
                "tests": item["test_list"]
            })
        return tasks
    except Exception as e:
        logger.error(f"Failed to load MBPP: {e}")
        return []


def run_experiment(
    sample_size: int = 300,
    ratios: list[float] = None,
    conditions: list[str] = None,
    output_dir: str = "results/signature_preservation",
    model: str = "gpt-4o-mini"
):
    """Run the signature preservation experiment."""

    if ratios is None:
        ratios = [0.3, 0.4, 0.5]
    if conditions is None:
        conditions = ["baseline", "sig_inject", "sig_inject_plus"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results_file = output_path / "results.jsonl"

    # Load tasks
    tasks = load_mbpp_tasks(sample_size)
    if not tasks:
        logger.error("No tasks loaded")
        return

    logger.info(f"Loaded {len(tasks)} tasks")

    # Initialize compressor
    compressor = Compressor()

    # Calculate total trials
    total_trials = len(tasks) * len(ratios) * len(conditions)
    logger.info(f"Total trials: {total_trials}")

    trial_num = 0
    total_cost = 0.0
    results_by_condition = defaultdict(lambda: defaultdict(list))

    with open(results_file, "a") as f:
        for task in tasks:
            for ratio in ratios:
                # Compress prompt once per ratio
                compressed = compressor.compress(task["prompt"], ratio)

                for condition in conditions:
                    trial_num += 1
                    start_time = time.time()

                    # Create final prompt based on condition
                    handler = CONDITION_HANDLERS[condition]
                    final_prompt = handler(task, compressed)

                    # Generate code
                    generated, cost = generate_code(final_prompt, model)
                    total_cost += cost

                    # Execute tests
                    passed, error_type = execute_tests(generated, task["tests"])

                    elapsed_ms = (time.time() - start_time) * 1000

                    result = {
                        "task_id": task["task_id"],
                        "compression_ratio": ratio,
                        "condition": condition,
                        "passed": passed,
                        "error_type": error_type,
                        "execution_time_ms": elapsed_ms,
                        "cost_usd": cost,
                        "prompt_tokens": len(final_prompt.split()),
                        "timestamp": datetime.now().isoformat()
                    }

                    f.write(json.dumps(result) + "\n")
                    f.flush()

                    # Track results
                    results_by_condition[condition][ratio].append(passed)

                    status = "PASS" if passed else f"FAIL ({error_type})"
                    logger.info(
                        f"[{trial_num}/{total_trials}] Task {task['task_id']} | "
                        f"r={ratio} | {condition} | {status} | ${cost:.4f}"
                    )

    # Print summary
    print("\n" + "=" * 70)
    print("SIGNATURE PRESERVATION EXPERIMENT: RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nTotal trials: {total_trials}")
    print(f"Total cost: ${total_cost:.2f}")

    print("\nPass Rate by Condition and Ratio:")
    print("-" * 50)
    print(f"{'Condition':<20} | {'r=0.3':>8} | {'r=0.4':>8} | {'r=0.5':>8}")
    print("-" * 50)

    for condition in conditions:
        rates = []
        for ratio in ratios:
            results = results_by_condition[condition][ratio]
            rate = sum(results) / len(results) * 100 if results else 0
            rates.append(f"{rate:.1f}%")
        print(f"{condition:<20} | {rates[0]:>8} | {rates[1]:>8} | {rates[2]:>8}")

    print("-" * 50)

    # Calculate recovery rate
    print("\nRecovery vs Baseline:")
    for ratio in ratios:
        baseline_rate = sum(results_by_condition["baseline"][ratio]) / len(results_by_condition["baseline"][ratio]) * 100
        for condition in ["sig_inject", "sig_inject_plus"]:
            cond_rate = sum(results_by_condition[condition][ratio]) / len(results_by_condition[condition][ratio]) * 100
            recovery = cond_rate - baseline_rate
            print(f"  r={ratio} {condition}: {recovery:+.1f}pp recovery")

    print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3A: Signature Preservation"
    )
    parser.add_argument(
        "--sample-size", type=int, default=300,
        help="Number of MBPP tasks to test"
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=[0.3, 0.4, 0.5],
        help="Compression ratios to test"
    )
    parser.add_argument(
        "--conditions", type=str, nargs="+",
        default=["baseline", "sig_inject", "sig_inject_plus"],
        help="Conditions to test"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/signature_preservation",
        help="Output directory"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="Model to use for code generation"
    )

    args = parser.parse_args()

    run_experiment(
        sample_size=args.sample_size,
        ratios=args.ratios,
        conditions=args.conditions,
        output_dir=args.output_dir,
        model=args.model
    )


if __name__ == "__main__":
    main()

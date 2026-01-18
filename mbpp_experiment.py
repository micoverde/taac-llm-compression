#!/usr/bin/env python3
"""
MBPP Benchmark Expansion Experiment
====================================
Gap 2 from Research Roadmap: Validate TAAC findings on MBPP (974 problems)

This experiment extends our HumanEval-based compression analysis to the
Mostly Basic Python Problems (MBPP) benchmark to validate:
1. Threshold behavior at r >= 0.6 generalizes across benchmarks
2. Model tier effects are consistent
3. Compression method robustness on different prompt structures

Author: Dr. Michael Park
Date: January 2026
VM Target: ssh azureuser@20.185.221.53 (taac-env virtual environment)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

# =============================================================================
# MBPP-Specific Configuration
# =============================================================================

# Compression ratios matching original HumanEval experiments
COMPRESSION_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]

# Model tiers from original experiment (matching paper Table 1)
MODELS = {
    # Tier 1: Economy
    "deepseek-chat": {
        "tier": 1,
        "provider": "deepseek",
        "model_id": "deepseek-chat",
        "cost_input": 0.14,  # per 1M tokens
        "cost_output": 0.28,
    },
    "llama-3.1-8b": {
        "tier": 1,
        "provider": "together",
        "model_id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "cost_input": 0.18,
        "cost_output": 0.18,
    },
    # Tier 2: Balanced
    "gpt-4o-mini": {
        "tier": 2,
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "cost_input": 0.15,
        "cost_output": 0.60,
    },
    "claude-3-haiku": {
        "tier": 2,
        "provider": "anthropic",
        "model_id": "claude-3-haiku-20240307",
        "cost_input": 0.25,
        "cost_output": 1.25,
    },
    # Tier 3: Premium
    "claude-3.5-sonnet": {
        "tier": 3,
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "cost_input": 3.00,
        "cost_output": 15.00,
    },
    "gpt-4o": {
        "tier": 3,
        "provider": "openai",
        "model_id": "gpt-4o",
        "cost_input": 2.50,
        "cost_output": 10.00,
    },
}

# Subset for cost-effective validation
# Using just gpt-4o-mini for initial validation (VM only has OPENAI_API_KEY)
MODELS_VALIDATION = ["gpt-4o-mini"]
MODELS_FULL = list(MODELS.keys())

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("mbpp_experiment.log"),
    ],
)
logger = logging.getLogger(__name__)

# =============================================================================
# MBPP Prompt Templates
# =============================================================================

# Match HumanEval format for consistency
MBPP_PROMPT_TEMPLATE = """You are an expert Python programmer. Write a Python function that solves the following problem.

Problem Description:
{description}

Write ONLY the Python function implementation. Do not include any explanations, comments, or test cases.
The function should be complete and ready to execute.

Function:
"""

# Extended template with examples (for ablation)
MBPP_PROMPT_WITH_EXAMPLES = """You are an expert Python programmer. Write a Python function that solves the following problem.

Problem Description:
{description}

Example test cases (DO NOT include these in your solution):
{test_examples}

Write ONLY the Python function implementation. Do not include any explanations, comments, or test cases.
The function should be complete and ready to execute.

Function:
"""


@dataclass
class MBPPProblem:
    """Single MBPP problem."""

    task_id: int
    description: str  # 'text' field in original
    code: str  # Reference solution
    test_list: list[str]  # Test assertions

    @property
    def prompt(self) -> str:
        """Generate prompt for this problem."""
        return MBPP_PROMPT_TEMPLATE.format(description=self.description)

    @property
    def prompt_with_examples(self) -> str:
        """Generate prompt with example tests."""
        examples = "\n".join(self.test_list[:2])  # First 2 tests as examples
        return MBPP_PROMPT_WITH_EXAMPLES.format(
            description=self.description, test_examples=examples
        )


@dataclass
class TrialResult:
    """Result of a single trial."""

    task_id: int
    compression_ratio: float
    model: str
    model_tier: int

    # Compression metrics
    original_tokens: int
    compressed_tokens: int
    actual_ratio: float
    compression_time_ms: float

    # Generation metrics
    generation_time_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float

    # Evaluation metrics
    passed: bool
    error: Optional[str] = None
    execution_time_ms: float = 0.0

    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentConfig:
    """Configuration for MBPP experiment."""

    name: str = "MBPP Benchmark Expansion"
    compression_ratios: list[float] = field(
        default_factory=lambda: COMPRESSION_RATIOS
    )
    models: list[str] = field(default_factory=lambda: MODELS_VALIDATION)
    trials_per_problem: int = 1  # MBPP has 974 problems, so 1 trial each
    sample_size: Optional[int] = None  # None = full 974, or subset for testing
    random_seed: int = 42
    checkpoint_every: int = 50
    parallel_workers: int = 1  # Sequential for reproducibility


# =============================================================================
# MBPP Data Loader
# =============================================================================


class MBPPLoader:
    """Load MBPP benchmark from HuggingFace."""

    def __init__(self, split: str = "test"):
        self.split = split
        self._data: Optional[list[MBPPProblem]] = None

    def load(self, sample_size: Optional[int] = None) -> list[MBPPProblem]:
        """Load MBPP problems."""
        if self._data is not None:
            if sample_size:
                return self._data[:sample_size]
            return self._data

        try:
            from datasets import load_dataset

            logger.info("Loading MBPP dataset from HuggingFace...")
            ds = load_dataset("google-research-datasets/mbpp", split=self.split)

            self._data = []
            for row in ds:
                problem = MBPPProblem(
                    task_id=row["task_id"],
                    description=row["text"],
                    code=row["code"],
                    test_list=row["test_list"],
                )
                self._data.append(problem)

            logger.info(f"Loaded {len(self._data)} MBPP problems")

        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            raise

        if sample_size:
            return self._data[:sample_size]
        return self._data

    @property
    def size(self) -> int:
        """Return dataset size."""
        if self._data is None:
            self.load()
        return len(self._data)


# =============================================================================
# Compression Engine
# =============================================================================


class CompressionEngine:
    """LLMLingua-2 compression for MBPP prompts."""

    def __init__(self, method: str = "llmlingua2"):
        self.method = method
        self._compressor = None

    def load(self):
        """Lazy load the compressor."""
        if self._compressor is not None:
            return

        if self.method == "llmlingua2":
            from llmlingua import PromptCompressor

            self._compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu",  # Use CPU since VM has no GPU
            )
            logger.info("Loaded LLMLingua-2 compressor")
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def compress(self, prompt: str, ratio: float) -> dict:
        """Compress a prompt to target ratio."""
        self.load()

        if ratio >= 1.0:
            return {
                "compressed_prompt": prompt,
                "actual_ratio": 1.0,
                "original_tokens": len(prompt.split()),
                "compressed_tokens": len(prompt.split()),
                "compression_time_ms": 0.0,
            }

        start = time.time()

        # Force preserve important tokens for code generation
        result = self._compressor.compress_prompt(
            prompt,
            rate=ratio,
            force_tokens=[
                "\n",
                "def",
                "return",
                "if",
                "for",
                "while",
                "class",
                "import",
                "from",
                ":",
                "(",
                ")",
                "[",
                "]",
            ],
        )

        compression_time = (time.time() - start) * 1000

        compressed = result["compressed_prompt"]
        orig_tokens = result.get("origin_tokens", len(prompt.split()))
        comp_tokens = result.get("compressed_tokens", len(compressed.split()))
        actual_ratio = comp_tokens / orig_tokens if orig_tokens > 0 else 1.0

        return {
            "compressed_prompt": compressed,
            "actual_ratio": actual_ratio,
            "original_tokens": orig_tokens,
            "compressed_tokens": comp_tokens,
            "compression_time_ms": compression_time,
        }


# =============================================================================
# Model Client
# =============================================================================


class ModelClient:
    """Unified client for LLM providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = MODELS[model_name]
        self._client = None

    def load(self):
        """Initialize API client."""
        if self._client is not None:
            return

        provider = self.config["provider"]

        if provider == "anthropic":
            from anthropic import Anthropic

            self._client = Anthropic()
        elif provider == "openai":
            from openai import OpenAI

            self._client = OpenAI()
        elif provider == "deepseek":
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
            )
        elif provider == "together":
            from openai import OpenAI

            self._client = OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1",
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"Initialized client for {self.model_name}")

    def generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> dict:
        """Generate completion."""
        self.load()

        provider = self.config["provider"]
        model_id = self.config.get("model_id", self.model_name)

        start = time.time()

        try:
            if provider == "anthropic":
                response = self._client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
            else:
                # OpenAI-compatible API
                response = self._client.chat.completions.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            latency = (time.time() - start) * 1000

            # Calculate cost
            cost = (
                input_tokens * self.config["cost_input"] / 1_000_000
                + output_tokens * self.config["cost_output"] / 1_000_000
            )

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency,
                "cost_usd": cost,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Generation error: {e}")
            return {
                "content": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": (time.time() - start) * 1000,
                "cost_usd": 0,
                "success": False,
                "error": str(e),
            }


# =============================================================================
# Code Evaluator
# =============================================================================


class MBPPEvaluator:
    """Evaluate generated code against MBPP tests."""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def extract_code(self, output: str) -> str:
        """Extract code from model output."""
        # Try markdown code blocks
        if "```python" in output:
            start = output.find("```python") + 9
            end = output.find("```", start)
            if end > start:
                return output[start:end].strip()
        elif "```" in output:
            start = output.find("```") + 3
            end = output.find("```", start)
            if end > start:
                return output[start:end].strip()

        # Return as-is
        return output.strip()

    def evaluate(self, generated_code: str, problem: MBPPProblem) -> dict:
        """Evaluate generated code against problem test cases."""
        code = self.extract_code(generated_code)

        # Combine code with test assertions
        test_code = "\n".join(problem.test_list)
        full_code = f"{code}\n\n{test_code}"

        start = time.time()

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False
            ) as f:
                f.write(full_code)
                f.flush()

                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

            os.unlink(f.name)
            exec_time = (time.time() - start) * 1000

            passed = result.returncode == 0
            return {
                "passed": passed,
                "execution_time_ms": exec_time,
                "stdout": result.stdout[:500] if result.stdout else None,
                "stderr": result.stderr[:500] if result.stderr else None,
                "error": None if passed else result.stderr[:200],
            }

        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "execution_time_ms": self.timeout * 1000,
                "error": "Timeout",
            }
        except Exception as e:
            return {
                "passed": False,
                "execution_time_ms": (time.time() - start) * 1000,
                "error": str(e),
            }


# =============================================================================
# Experiment Runner
# =============================================================================


class MBPPExperimentRunner:
    """Main experiment orchestrator."""

    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "results/mbpp_experiment",
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "results.jsonl"

        self.completed_trials: set[str] = set()
        self._load_checkpoint()

        # Initialize components
        self.loader = MBPPLoader()
        self.compressor = CompressionEngine()
        self.evaluator = MBPPEvaluator()

    def _load_checkpoint(self):
        """Load completed trials from checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                self.completed_trials = set(data.get("completed", []))
            logger.info(
                f"Loaded checkpoint: {len(self.completed_trials)} completed trials"
            )

    def _save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_file, "w") as f:
            json.dump(
                {
                    "completed": list(self.completed_trials),
                    "timestamp": datetime.now().isoformat(),
                },
                f,
            )

    def _trial_key(self, task_id: int, ratio: float, model: str) -> str:
        """Generate unique key for trial."""
        return f"{task_id}|{ratio}|{model}"

    def estimate_cost_and_calls(self, use_known_size: bool = True) -> dict:
        """Estimate total API calls and cost.

        Args:
            use_known_size: If True and dataset can't load, use known MBPP size (974).
        """
        try:
            problems = self.loader.load(self.config.sample_size)
            n_problems = len(problems)
        except Exception as e:
            if use_known_size:
                # MBPP test split has 500 problems, but full has 974
                # Use sample_size if specified, otherwise known full size
                n_problems = self.config.sample_size or 500
                logger.warning(f"Dataset not available, using known size: {n_problems}")
            else:
                raise
        n_ratios = len(self.config.compression_ratios)
        n_models = len(self.config.models)

        total_calls = n_problems * n_ratios * n_models

        # Estimate tokens per problem (avg MBPP prompt ~100 tokens)
        avg_input_tokens = 150
        avg_output_tokens = 200

        # Calculate cost per model
        cost_breakdown = {}
        total_cost = 0.0

        for model_name in self.config.models:
            cfg = MODELS[model_name]
            calls = n_problems * n_ratios
            input_cost = (avg_input_tokens * calls / 1_000_000) * cfg["cost_input"]
            output_cost = (avg_output_tokens * calls / 1_000_000) * cfg["cost_output"]
            model_cost = input_cost + output_cost
            cost_breakdown[model_name] = {
                "calls": calls,
                "estimated_cost_usd": round(model_cost, 2),
            }
            total_cost += model_cost

        # Estimate runtime (avg 2 seconds per call including compression)
        avg_time_per_call_sec = 2.5
        total_runtime_hours = (total_calls * avg_time_per_call_sec) / 3600

        return {
            "total_problems": n_problems,
            "compression_ratios": self.config.compression_ratios,
            "models": self.config.models,
            "total_api_calls": total_calls,
            "estimated_cost_usd": round(total_cost, 2),
            "estimated_runtime_hours": round(total_runtime_hours, 2),
            "cost_by_model": cost_breakdown,
        }

    def run_trial(
        self,
        problem: MBPPProblem,
        ratio: float,
        model_name: str,
        client: ModelClient,
    ) -> TrialResult:
        """Run a single trial."""
        # Compress prompt
        compression_result = self.compressor.compress(problem.prompt, ratio)

        # Generate code
        generation_result = client.generate(
            compression_result["compressed_prompt"]
        )

        # Evaluate if generation succeeded
        if generation_result["success"]:
            eval_result = self.evaluator.evaluate(
                generation_result["content"], problem
            )
        else:
            eval_result = {"passed": False, "error": "Generation failed"}

        return TrialResult(
            task_id=problem.task_id,
            compression_ratio=ratio,
            model=model_name,
            model_tier=MODELS[model_name]["tier"],
            original_tokens=compression_result["original_tokens"],
            compressed_tokens=compression_result["compressed_tokens"],
            actual_ratio=compression_result["actual_ratio"],
            compression_time_ms=compression_result["compression_time_ms"],
            generation_time_ms=generation_result["latency_ms"],
            input_tokens=generation_result["input_tokens"],
            output_tokens=generation_result["output_tokens"],
            cost_usd=generation_result["cost_usd"],
            passed=eval_result.get("passed", False),
            error=eval_result.get("error"),
            execution_time_ms=eval_result.get("execution_time_ms", 0),
        )

    def run(self, dry_run: bool = False):
        """Run the full experiment."""
        problems = self.loader.load(self.config.sample_size)

        logger.info(f"Starting MBPP Experiment: {self.config.name}")
        logger.info(f"Problems: {len(problems)}")
        logger.info(f"Compression ratios: {self.config.compression_ratios}")
        logger.info(f"Models: {self.config.models}")

        # Estimate
        estimate = self.estimate_cost_and_calls()
        logger.info(f"Total API calls: {estimate['total_api_calls']}")
        logger.info(f"Estimated cost: ${estimate['estimated_cost_usd']}")
        logger.info(f"Estimated runtime: {estimate['estimated_runtime_hours']} hours")

        if dry_run:
            logger.info("DRY RUN - no trials will be executed")
            return estimate

        trial_count = 0
        start_time = time.time()

        for model_name in self.config.models:
            client = ModelClient(model_name)

            for ratio in self.config.compression_ratios:
                for problem in problems:
                    trial_key = self._trial_key(
                        problem.task_id, ratio, model_name
                    )

                    if trial_key in self.completed_trials:
                        continue

                    try:
                        result = self.run_trial(
                            problem, ratio, model_name, client
                        )

                        # Save result
                        with open(self.results_file, "a") as f:
                            f.write(json.dumps(asdict(result)) + "\n")

                        self.completed_trials.add(trial_key)
                        trial_count += 1

                        # Log progress
                        status = "PASS" if result.passed else "FAIL"
                        logger.info(
                            f"[{trial_count}] Task {problem.task_id} "
                            f"| r={ratio} | {model_name} | {status} "
                            f"| ${result.cost_usd:.6f}"
                        )

                        if trial_count % self.config.checkpoint_every == 0:
                            self._save_checkpoint()
                            elapsed = time.time() - start_time
                            rate = trial_count / elapsed * 3600
                            logger.info(
                                f"Checkpoint: {trial_count} trials "
                                f"({rate:.0f}/hour)"
                            )

                    except Exception as e:
                        logger.error(f"Trial failed: {trial_key}: {e}")
                        continue

        self._save_checkpoint()
        total_time = (time.time() - start_time) / 3600
        logger.info(f"Experiment complete: {trial_count} trials in {total_time:.2f} hours")


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_results(results_file: Path) -> dict:
    """Analyze experiment results."""
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    if not results:
        return {}

    # Convert to structured analysis
    analysis = {
        "total_trials": len(results),
        "total_cost_usd": sum(r["cost_usd"] for r in results),
        "by_ratio": {},
        "by_model": {},
        "by_tier": {},
        "threshold_analysis": {},
    }

    # Group by ratio
    for ratio in COMPRESSION_RATIOS:
        ratio_results = [r for r in results if r["compression_ratio"] == ratio]
        if ratio_results:
            n = len(ratio_results)
            passes = sum(1 for r in ratio_results if r["passed"])
            analysis["by_ratio"][ratio] = {
                "n": n,
                "pass_rate": passes / n,
                "avg_cost": sum(r["cost_usd"] for r in ratio_results) / n,
            }

    # Group by model
    for model in MODELS.keys():
        model_results = [r for r in results if r["model"] == model]
        if model_results:
            n = len(model_results)
            passes = sum(1 for r in model_results if r["passed"])
            analysis["by_model"][model] = {
                "n": n,
                "pass_rate": passes / n,
                "tier": MODELS[model]["tier"],
            }

    # Threshold analysis: r >= 0.6 vs r < 0.6
    above_threshold = [r for r in results if r["compression_ratio"] >= 0.6]
    below_threshold = [r for r in results if r["compression_ratio"] < 0.6]

    if above_threshold:
        analysis["threshold_analysis"]["above_0.6"] = {
            "n": len(above_threshold),
            "pass_rate": sum(1 for r in above_threshold if r["passed"])
            / len(above_threshold),
        }
    if below_threshold:
        analysis["threshold_analysis"]["below_0.6"] = {
            "n": len(below_threshold),
            "pass_rate": sum(1 for r in below_threshold if r["passed"])
            / len(below_threshold),
        }

    return analysis


def compare_with_humaneval(mbpp_analysis: dict, humaneval_baseline: dict) -> dict:
    """Compare MBPP results with HumanEval baseline."""
    comparison = {
        "benchmark_sizes": {
            "humaneval": 164,
            "mbpp": 974,
        },
        "threshold_behavior": {
            "humaneval_threshold": humaneval_baseline.get("threshold", 0.6),
            "mbpp_threshold": None,  # To be determined
        },
        "pass_rate_comparison": {},
    }

    # Compare pass rates at each ratio
    for ratio in COMPRESSION_RATIOS:
        if ratio in mbpp_analysis.get("by_ratio", {}):
            mbpp_rate = mbpp_analysis["by_ratio"][ratio]["pass_rate"]
            he_rate = humaneval_baseline.get("by_ratio", {}).get(ratio, {}).get(
                "pass_rate"
            )
            comparison["pass_rate_comparison"][ratio] = {
                "mbpp": mbpp_rate,
                "humaneval": he_rate,
                "delta": mbpp_rate - he_rate if he_rate else None,
            }

    return comparison


# =============================================================================
# MBPP Prompt Structure Analysis
# =============================================================================


def analyze_mbpp_prompt_structure() -> dict:
    """
    Analyze MBPP prompt characteristics to predict threshold behavior.

    Key hypothesis: MBPP prompts are shorter and more formulaic than HumanEval,
    which may affect compression threshold behavior.
    """
    loader = MBPPLoader()
    problems = loader.load()

    # Analyze prompt characteristics
    prompt_lengths = []
    has_docstring = []
    complexity_indicators = []

    for p in problems:
        prompt = p.prompt
        prompt_lengths.append(len(prompt.split()))

        # Check for docstring patterns
        has_docstring.append('"""' in p.description or "'''" in p.description)

        # Check for complexity indicators
        complex_keywords = ["nested", "recursive", "binary", "tree", "graph"]
        has_complex = any(kw in p.description.lower() for kw in complex_keywords)
        complexity_indicators.append(has_complex)

    analysis = {
        "total_problems": len(problems),
        "prompt_length_stats": {
            "mean": np.mean(prompt_lengths),
            "std": np.std(prompt_lengths),
            "min": min(prompt_lengths),
            "max": max(prompt_lengths),
            "median": np.median(prompt_lengths),
        },
        "docstring_rate": sum(has_docstring) / len(has_docstring),
        "complex_problem_rate": sum(complexity_indicators) / len(complexity_indicators),
        "threshold_prediction": {
            "expected_threshold": 0.6,
            "rationale": (
                "MBPP prompts are shorter (avg ~100 tokens vs HumanEval ~200), "
                "so less context is available to compress. The r >= 0.6 threshold "
                "should still hold or may even need to be higher (r >= 0.7) for MBPP."
            ),
            "compression_sensitivity": (
                "MBPP's formulaic 'Write a function to...' structure provides "
                "consistent but less redundant prompts, making aggressive compression "
                "more likely to remove essential information."
            ),
        },
    }

    return analysis


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="MBPP Benchmark Expansion Experiment"
    )
    parser.add_argument(
        "--mode",
        choices=["estimate", "run", "analyze", "prompt-analysis"],
        required=True,
        help="Experiment mode",
    )
    parser.add_argument(
        "--models",
        choices=["validation", "full"],
        default="validation",
        help="Model set to use",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit to N problems (default: all 974)",
    )
    parser.add_argument(
        "--output-dir",
        default="results/mbpp_experiment",
        help="Output directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running",
    )

    args = parser.parse_args()

    # Configure experiment
    models = MODELS_FULL if args.models == "full" else MODELS_VALIDATION
    config = ExperimentConfig(
        models=models,
        sample_size=args.sample_size,
    )

    if args.mode == "estimate":
        runner = MBPPExperimentRunner(config, args.output_dir)
        estimate = runner.estimate_cost_and_calls()
        print("\n" + "=" * 60)
        print("MBPP EXPERIMENT COST ESTIMATE")
        print("=" * 60)
        print(f"Problems: {estimate['total_problems']}")
        print(f"Compression ratios: {estimate['compression_ratios']}")
        print(f"Models: {estimate['models']}")
        print(f"\nTotal API calls: {estimate['total_api_calls']:,}")
        print(f"Estimated cost: ${estimate['estimated_cost_usd']:.2f}")
        print(f"Estimated runtime: {estimate['estimated_runtime_hours']:.1f} hours")
        print("\nCost by model:")
        for model, stats in estimate["cost_by_model"].items():
            print(f"  {model}: {stats['calls']} calls, ${stats['estimated_cost_usd']}")

    elif args.mode == "run":
        runner = MBPPExperimentRunner(config, args.output_dir)
        runner.run(dry_run=args.dry_run)

    elif args.mode == "analyze":
        results_file = Path(args.output_dir) / "results.jsonl"
        if not results_file.exists():
            print(f"No results file found: {results_file}")
            return

        analysis = analyze_results(results_file)
        print("\n" + "=" * 60)
        print("MBPP EXPERIMENT RESULTS")
        print("=" * 60)
        print(f"Total trials: {analysis['total_trials']}")
        print(f"Total cost: ${analysis['total_cost_usd']:.2f}")
        print("\nPass rate by compression ratio:")
        for ratio, stats in analysis.get("by_ratio", {}).items():
            print(f"  r={ratio}: {stats['pass_rate']:.1%} (n={stats['n']})")
        print("\nThreshold analysis (r >= 0.6 hypothesis):")
        for key, stats in analysis.get("threshold_analysis", {}).items():
            print(f"  {key}: {stats['pass_rate']:.1%} (n={stats['n']})")

    elif args.mode == "prompt-analysis":
        analysis = analyze_mbpp_prompt_structure()
        print("\n" + "=" * 60)
        print("MBPP PROMPT STRUCTURE ANALYSIS")
        print("=" * 60)
        print(f"Total problems: {analysis['total_problems']}")
        print("\nPrompt length statistics (tokens):")
        for stat, val in analysis["prompt_length_stats"].items():
            print(f"  {stat}: {val:.1f}")
        print(f"\nDocstring rate: {analysis['docstring_rate']:.1%}")
        print(f"Complex problem rate: {analysis['complex_problem_rate']:.1%}")
        print("\nThreshold prediction:")
        print(f"  Expected threshold: r >= {analysis['threshold_prediction']['expected_threshold']}")
        print(f"\n  Rationale: {analysis['threshold_prediction']['rationale']}")
        print(f"\n  Compression sensitivity: {analysis['threshold_prediction']['compression_sensitivity']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
TAAC Experiment Runner
======================
Main experiment runner for Task-Aware Adaptive Compression study.

Usage:
    python runner.py --experiment phase1_benchmark_expansion
    python runner.py --experiment phase3_method_comparison --resume
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import EXPERIMENTS, COMPRESSION_METHODS, MODELS, BENCHMARKS

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment.log')
    ]
)
logger = logging.getLogger(__name__)


class CompressionEngine:
    """Handles prompt compression using different methods."""

    def __init__(self, method: str):
        self.method = method
        self.config = COMPRESSION_METHODS[method]
        self._model = None

    def load(self):
        """Lazy load the compression model."""
        if self._model is not None:
            return

        if self.method == "llmlingua2":
            from llmlingua import PromptCompressor
            self._model = PromptCompressor(
                model_name=self.config["model"],
                use_llmlingua2=True,
            )
        elif self.method == "llmlingua1":
            from llmlingua import PromptCompressor
            self._model = PromptCompressor(
                model_name=self.config["model"],
                use_llmlingua2=False,
            )
        elif self.method == "selective_context":
            # Implement selective context using GPT-2
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self._tokenizer = GPT2Tokenizer.from_pretrained(self.config["model"])
            self._model = GPT2LMHeadModel.from_pretrained(self.config["model"])
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

        logger.info(f"Loaded compression model: {self.method}")

    def compress(self, prompt: str, ratio: float) -> dict:
        """
        Compress a prompt to the target ratio.

        Returns:
            dict with keys: compressed_prompt, actual_ratio, token_info
        """
        self.load()

        if ratio >= 1.0:
            return {
                "compressed_prompt": prompt,
                "actual_ratio": 1.0,
                "original_tokens": len(prompt.split()),
                "compressed_tokens": len(prompt.split()),
                "token_info": None,
            }

        start_time = time.time()

        if self.method in ["llmlingua2", "llmlingua1"]:
            result = self._model.compress_prompt(
                prompt,
                rate=ratio,
                force_tokens=["\n", "def", "return", "if", "for", "while", "class"],
            )
            compressed = result["compressed_prompt"]
            token_info = {
                "origin_tokens": result.get("origin_tokens", 0),
                "compressed_tokens": result.get("compressed_tokens", 0),
            }
        elif self.method == "selective_context":
            compressed, token_info = self._selective_context_compress(prompt, ratio)
        else:
            compressed = prompt
            token_info = None

        compression_time = time.time() - start_time

        original_tokens = len(prompt.split())
        compressed_tokens = len(compressed.split())
        actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

        return {
            "compressed_prompt": compressed,
            "actual_ratio": actual_ratio,
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_time_ms": compression_time * 1000,
            "token_info": token_info,
        }

    def _selective_context_compress(self, prompt: str, ratio: float) -> tuple[str, dict]:
        """Implement Selective Context compression using self-information."""
        import torch

        # Tokenize
        inputs = self._tokenizer(prompt, return_tensors="pt")
        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Compute self-information (negative log probability)
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

            # Self-information for each token
            token_ids = inputs["input_ids"][0]
            self_info = []
            for i in range(1, len(token_ids)):
                prob = probs[i-1, token_ids[i]].item()
                self_info.append(-np.log(prob + 1e-10))

        # Select tokens with highest self-information
        n_keep = int(len(tokens) * ratio)
        if n_keep < 1:
            n_keep = 1

        # Keep first token always, then select by self-information
        indices_to_keep = [0]
        if len(self_info) > 0:
            sorted_indices = np.argsort(self_info)[::-1]  # Highest first
            for idx in sorted_indices[:n_keep-1]:
                indices_to_keep.append(idx + 1)  # +1 because self_info starts at token 1

        indices_to_keep = sorted(indices_to_keep)
        kept_tokens = [tokens[i] for i in indices_to_keep]
        compressed = self._tokenizer.convert_tokens_to_string(kept_tokens)

        token_info = {
            "self_information": self_info,
            "kept_indices": indices_to_keep,
        }

        return compressed, token_info


class ModelClient:
    """Unified client for different LLM providers."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.config = MODELS[model_name]
        self._client = None

    def load(self):
        """Initialize the API client."""
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
                base_url="https://api.deepseek.com/v1"
            )
        elif provider == "together":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                base_url="https://api.together.xyz/v1"
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info(f"Initialized client for {self.model_name}")

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> dict:
        """Generate a completion."""
        self.load()

        provider = self.config["provider"]
        model_id = self.config.get("model_id", self.model_name)

        start_time = time.time()

        try:
            if provider == "anthropic":
                response = self._client.messages.create(
                    model=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
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
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens
                output_tokens = response.usage.completion_tokens

            latency = time.time() - start_time

            # Calculate cost
            cost = (
                input_tokens * self.config["cost_input"] / 1_000_000 +
                output_tokens * self.config["cost_output"] / 1_000_000
            )

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency * 1000,
                "cost_usd": cost,
                "success": True,
                "error": None,
            }

        except Exception as e:
            logger.error(f"Generation error for {self.model_name}: {e}")
            return {
                "content": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "latency_ms": (time.time() - start_time) * 1000,
                "cost_usd": 0,
                "success": False,
                "error": str(e),
            }


class BenchmarkLoader:
    """Load and manage benchmark datasets."""

    def __init__(self, benchmark_name: str):
        self.name = benchmark_name
        self.config = BENCHMARKS[benchmark_name]
        self._data = None

    def load(self, sample_size: int = None) -> list[dict]:
        """Load benchmark problems."""
        if self._data is not None:
            if sample_size:
                return self._data[:sample_size]
            return self._data

        benchmark_type = self.config["type"]

        if self.name == "humaneval":
            self._data = self._load_humaneval()
        elif self.name == "mbpp":
            self._data = self._load_mbpp()
        elif self.name == "gsm8k":
            self._data = self._load_gsm8k()
        elif self.name == "math":
            self._data = self._load_math()
        elif self.name.startswith("multipl_e"):
            lang = self.name.split("_")[-1]
            self._data = self._load_multipl_e(lang)
        else:
            # Placeholder for other benchmarks
            logger.warning(f"Benchmark {self.name} not yet implemented, using dummy data")
            self._data = [{"prompt": "dummy", "expected": "dummy"}] * 10

        logger.info(f"Loaded {len(self._data)} problems from {self.name}")

        if sample_size:
            return self._data[:sample_size]
        return self._data

    def _load_humaneval(self) -> list[dict]:
        """Load HumanEval benchmark."""
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/openai_humaneval", split="test")
            return [
                {
                    "task_id": row["task_id"],
                    "prompt": row["prompt"],
                    "canonical_solution": row["canonical_solution"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                }
                for row in ds
            ]
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            return []

    def _load_mbpp(self) -> list[dict]:
        """Load MBPP benchmark."""
        try:
            from datasets import load_dataset
            ds = load_dataset("google-research-datasets/mbpp", split="test")
            return [
                {
                    "task_id": row["task_id"],
                    "prompt": row["text"],
                    "code": row["code"],
                    "test_list": row["test_list"],
                }
                for row in ds
            ]
        except Exception as e:
            logger.error(f"Failed to load MBPP: {e}")
            return []

    def _load_gsm8k(self) -> list[dict]:
        """Load GSM8K benchmark."""
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split="test")
            return [
                {
                    "question": row["question"],
                    "answer": row["answer"],
                }
                for row in ds
            ]
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            return []

    def _load_math(self) -> list[dict]:
        """Load MATH benchmark."""
        try:
            from datasets import load_dataset
            ds = load_dataset("lighteval/MATH", split="test")
            return [
                {
                    "problem": row["problem"],
                    "solution": row["solution"],
                    "level": row.get("level", "unknown"),
                    "type": row.get("type", "unknown"),
                }
                for row in ds
            ]
        except Exception as e:
            logger.error(f"Failed to load MATH: {e}")
            return []

    def _load_multipl_e(self, language: str) -> list[dict]:
        """Load MultiPL-E benchmark for a specific language."""
        try:
            from datasets import load_dataset
            ds = load_dataset("nuprl/MultiPL-E", f"humaneval-{language}", split="test")
            return [
                {
                    "name": row["name"],
                    "prompt": row["prompt"],
                    "tests": row["tests"],
                }
                for row in ds
            ]
        except Exception as e:
            logger.error(f"Failed to load MultiPL-E {language}: {e}")
            return []


class Evaluator:
    """Evaluate model outputs against benchmarks."""

    def __init__(self, benchmark_name: str):
        self.name = benchmark_name
        self.config = BENCHMARKS[benchmark_name]

    def evaluate(self, output: str, expected: dict) -> dict:
        """Evaluate a single output."""
        metric = self.config["metric"]

        if metric == "pass@1":
            return self._evaluate_code(output, expected)
        elif metric == "exact_match":
            return self._evaluate_exact_match(output, expected)
        elif metric == "accuracy":
            return self._evaluate_accuracy(output, expected)
        else:
            return {"score": 0.0, "correct": False, "error": f"Unknown metric: {metric}"}

    def _evaluate_code(self, output: str, expected: dict) -> dict:
        """Evaluate code generation using execution."""
        import subprocess
        import tempfile

        try:
            # Extract code from output
            code = self._extract_code(output)

            # Combine with test cases
            if "test" in expected:
                full_code = f"{code}\n\n{expected['test']}"
            elif "test_list" in expected:
                test_code = "\n".join(expected["test_list"])
                full_code = f"{code}\n\n{test_code}"
            else:
                return {"score": 0.0, "correct": False, "error": "No tests found"}

            # Execute in isolated environment
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                f.flush()

                result = subprocess.run(
                    ["python", f.name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            passed = result.returncode == 0
            return {
                "score": 1.0 if passed else 0.0,
                "correct": passed,
                "stdout": result.stdout[:500] if result.stdout else None,
                "stderr": result.stderr[:500] if result.stderr else None,
            }

        except subprocess.TimeoutExpired:
            return {"score": 0.0, "correct": False, "error": "Timeout"}
        except Exception as e:
            return {"score": 0.0, "correct": False, "error": str(e)}

    def _extract_code(self, output: str) -> str:
        """Extract code from model output."""
        # Try to find code blocks
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

        # Return as-is if no code blocks found
        return output.strip()

    def _evaluate_exact_match(self, output: str, expected: dict) -> dict:
        """Evaluate using exact match on final answer."""
        # Extract numerical answer from output
        import re

        # Look for boxed answer (common in MATH)
        boxed = re.search(r'\\boxed{([^}]+)}', output)
        if boxed:
            predicted = boxed.group(1).strip()
        else:
            # Look for final number
            numbers = re.findall(r'-?\d+\.?\d*', output)
            predicted = numbers[-1] if numbers else ""

        # Extract expected answer
        if "answer" in expected:
            answer_text = expected["answer"]
            # GSM8K format: "#### 42"
            if "####" in answer_text:
                expected_answer = answer_text.split("####")[-1].strip()
            else:
                numbers = re.findall(r'-?\d+\.?\d*', answer_text)
                expected_answer = numbers[-1] if numbers else ""
        elif "solution" in expected:
            boxed = re.search(r'\\boxed{([^}]+)}', expected["solution"])
            expected_answer = boxed.group(1).strip() if boxed else ""
        else:
            expected_answer = ""

        correct = predicted == expected_answer
        return {
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted": predicted,
            "expected": expected_answer,
        }

    def _evaluate_accuracy(self, output: str, expected: dict) -> dict:
        """Evaluate multiple choice accuracy."""
        # Extract answer letter
        import re

        match = re.search(r'\b([A-D])\b', output.upper())
        predicted = match.group(1) if match else ""

        expected_answer = expected.get("answer", "")
        correct = predicted == expected_answer

        return {
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "predicted": predicted,
            "expected": expected_answer,
        }


class ExperimentRunner:
    """Main experiment orchestrator."""

    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.config = EXPERIMENTS[experiment_name]
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.results_file = self.output_dir / "results.jsonl"

        self.completed_trials = set()
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load completed trials from checkpoint."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                self.completed_trials = set(data.get("completed", []))
            logger.info(f"Loaded checkpoint: {len(self.completed_trials)} completed trials")

    def _save_checkpoint(self):
        """Save checkpoint."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "completed": list(self.completed_trials),
                "timestamp": datetime.now().isoformat(),
            }, f)

    def _trial_key(self, method: str, ratio: float, model: str, benchmark: str, idx: int) -> str:
        """Generate unique key for a trial."""
        return f"{method}|{ratio}|{model}|{benchmark}|{idx}"

    def run(self, dry_run: bool = False):
        """Run the experiment."""
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Total conditions: {self.config.total_conditions}")
        logger.info(f"Total trials: {self.config.total_trials}")

        trial_count = 0

        for method in self.config.compression_methods:
            compressor = CompressionEngine(method)

            for model_name in self.config.models:
                client = ModelClient(model_name)

                for benchmark_name in self.config.benchmarks:
                    loader = BenchmarkLoader(benchmark_name)
                    evaluator = Evaluator(benchmark_name)
                    problems = loader.load()

                    for ratio in self.config.compression_ratios:
                        for idx in range(min(self.config.trials_per_condition, len(problems))):
                            trial_key = self._trial_key(method, ratio, model_name, benchmark_name, idx)

                            if trial_key in self.completed_trials:
                                continue

                            problem = problems[idx]

                            if dry_run:
                                logger.info(f"[DRY RUN] Would run: {trial_key}")
                                continue

                            # Run trial
                            result = self._run_trial(
                                compressor, client, evaluator,
                                method, ratio, model_name, benchmark_name, problem
                            )

                            # Save result
                            result["trial_key"] = trial_key
                            with open(self.results_file, "a") as f:
                                f.write(json.dumps(result) + "\n")

                            self.completed_trials.add(trial_key)
                            trial_count += 1

                            if trial_count % self.config.checkpoint_every == 0:
                                self._save_checkpoint()
                                logger.info(f"Checkpoint saved: {trial_count} trials completed")

        self._save_checkpoint()
        logger.info(f"Experiment complete: {trial_count} new trials")

    def _run_trial(
        self,
        compressor: CompressionEngine,
        client: ModelClient,
        evaluator: Evaluator,
        method: str,
        ratio: float,
        model_name: str,
        benchmark_name: str,
        problem: dict,
    ) -> dict:
        """Run a single trial."""

        # Get prompt from problem
        prompt = problem.get("prompt") or problem.get("question") or problem.get("problem", "")

        # Compress
        compression_result = compressor.compress(prompt, ratio)

        # Generate
        generation_result = client.generate(compression_result["compressed_prompt"])

        # Evaluate
        if generation_result["success"]:
            eval_result = evaluator.evaluate(generation_result["content"], problem)
        else:
            eval_result = {"score": 0.0, "correct": False, "error": "Generation failed"}

        return {
            "timestamp": datetime.now().isoformat(),
            "compression_method": method,
            "compression_ratio_target": ratio,
            "compression_ratio_actual": compression_result["actual_ratio"],
            "model": model_name,
            "benchmark": benchmark_name,
            "original_tokens": compression_result["original_tokens"],
            "compressed_tokens": compression_result["compressed_tokens"],
            "compression_time_ms": compression_result.get("compression_time_ms"),
            "generation_latency_ms": generation_result["latency_ms"],
            "input_tokens": generation_result["input_tokens"],
            "output_tokens": generation_result["output_tokens"],
            "cost_usd": generation_result["cost_usd"],
            "success": generation_result["success"],
            "score": eval_result["score"],
            "correct": eval_result.get("correct", False),
            "eval_details": eval_result,
        }


def main():
    parser = argparse.ArgumentParser(description="TAAC Experiment Runner")
    parser.add_argument("--experiment", required=True, choices=list(EXPERIMENTS.keys()),
                       help="Experiment configuration to run")
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print what would be done without running")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint (default behavior)")

    args = parser.parse_args()

    runner = ExperimentRunner(args.experiment, args.output_dir)
    runner.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()

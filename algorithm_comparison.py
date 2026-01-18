#!/usr/bin/env python3
"""
Algorithm Comparison Experiment for TAAC Study
===============================================

Gap 4: Test if Code/CoT dichotomy generalizes across compression algorithms.

Hypothesis: "The Code/CoT dichotomy is a fundamental property of task types
(generalizes across algorithms) rather than an artifact of LLMLingua-2
(algorithm-specific)."

Compression Methods Tested:
1. LLMLingua-2 (trained classifier) - our current method
2. LLMLingua-1 (perplexity heuristic) - baseline
3. Selective Context (self-information based) - different approach
4. Random baseline (control) - establishes floor

Key Metrics:
- Threshold location (r*) for each algorithm on code tasks
- Algorithm x Task Type interaction effect
- Cross-algorithm consistency of dichotomy pattern

Author: Dr. James Liu
Created: 2026-01-17
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('algorithm_comparison.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class CompressionAlgorithm(Enum):
    """Compression algorithms to compare."""
    LLMLINGUA2 = "llmlingua2"
    LLMLINGUA1 = "llmlingua1"
    SELECTIVE_CONTEXT = "selective_context"
    RANDOM = "random"


class TaskType(Enum):
    """Task categories for dichotomy testing."""
    CODE = "code"
    COT = "cot"


@dataclass
class AlgorithmConfig:
    """Configuration for a compression algorithm."""
    name: str
    algorithm_type: CompressionAlgorithm
    model_name: str
    description: str
    install_command: str
    gpu_required: bool = True
    memory_gb: float = 8.0
    reference: str = ""


ALGORITHM_CONFIGS = {
    CompressionAlgorithm.LLMLINGUA2: AlgorithmConfig(
        name="LLMLingua-2",
        algorithm_type=CompressionAlgorithm.LLMLINGUA2,
        model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        description="Trained classifier for token importance (fast, accurate)",
        install_command="pip install llmlingua",
        gpu_required=True,
        memory_gb=4.0,
        reference="Pan et al. 2024 - LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression",
    ),
    CompressionAlgorithm.LLMLINGUA1: AlgorithmConfig(
        name="LLMLingua-1",
        algorithm_type=CompressionAlgorithm.LLMLINGUA1,
        model_name="NousResearch/Llama-2-7b-hf",
        description="Perplexity-based heuristic compression (slower, original method)",
        install_command="pip install llmlingua accelerate",
        gpu_required=True,
        memory_gb=16.0,
        reference="Jiang et al. 2023 - LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models",
    ),
    CompressionAlgorithm.SELECTIVE_CONTEXT: AlgorithmConfig(
        name="Selective Context",
        algorithm_type=CompressionAlgorithm.SELECTIVE_CONTEXT,
        model_name="gpt2-large",
        description="Self-information based extractive compression",
        install_command="pip install transformers torch",
        gpu_required=True,
        memory_gb=4.0,
        reference="Li et al. 2023 - Compressing Context to Enhance Inference Efficiency of Large Language Models",
    ),
    CompressionAlgorithm.RANDOM: AlgorithmConfig(
        name="Random Baseline",
        algorithm_type=CompressionAlgorithm.RANDOM,
        model_name="none",
        description="Random token selection (control baseline)",
        install_command="# No installation needed",
        gpu_required=False,
        memory_gb=0.1,
        reference="Control condition",
    ),
}


# Compression ratios for threshold analysis
# Dense sampling near expected threshold regions
COMPRESSION_RATIOS = [
    0.30,  # 70% compression
    0.40,  # 60% compression
    0.45,  # 55% compression
    0.50,  # 50% compression - expected code threshold region
    0.55,  # 45% compression
    0.60,  # 40% compression - expected code threshold region
    0.65,  # 35% compression
    0.70,  # 30% compression
    0.80,  # 20% compression
    0.90,  # 10% compression
    1.00,  # No compression (baseline)
]

# Benchmark configurations
BENCHMARKS = {
    "humaneval": {"type": TaskType.CODE, "size": 164, "metric": "pass@1"},
    "mbpp": {"type": TaskType.CODE, "size": 500, "metric": "pass@1"},
    "gsm8k": {"type": TaskType.COT, "size": 1319, "metric": "exact_match"},
    "math": {"type": TaskType.COT, "size": 500, "metric": "exact_match"},
}


# =============================================================================
# Abstract Compressor Interface
# =============================================================================

class BaseCompressor(ABC):
    """Abstract base class for compression algorithms."""

    def __init__(self, config: AlgorithmConfig):
        self.config = config
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the compression model."""
        pass

    @abstractmethod
    def compress(self, text: str, ratio: float) -> dict:
        """
        Compress text to target ratio.

        Args:
            text: Input text to compress
            ratio: Target retention ratio (0.5 = keep 50% of tokens)

        Returns:
            dict with keys:
                - compressed_text: The compressed output
                - original_tokens: Number of tokens in input
                - compressed_tokens: Number of tokens in output
                - actual_ratio: Actual retention ratio achieved
                - compression_time_ms: Time taken for compression
                - metadata: Algorithm-specific metadata
        """
        pass

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class LLMLingua2Compressor(BaseCompressor):
    """LLMLingua-2 trained classifier compression."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self._model = None

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from llmlingua import PromptCompressor
            self._model = PromptCompressor(
                model_name=self.config.model_name,
                use_llmlingua2=True,
            )
            self._loaded = True
            logger.info(f"Loaded LLMLingua-2 model: {self.config.model_name}")
        except ImportError as e:
            raise RuntimeError(
                f"LLMLingua-2 not installed. Run: {self.config.install_command}"
            ) from e

    def compress(self, text: str, ratio: float) -> dict:
        self.load()

        if ratio >= 1.0:
            tokens = len(text.split())
            return {
                "compressed_text": text,
                "original_tokens": tokens,
                "compressed_tokens": tokens,
                "actual_ratio": 1.0,
                "compression_time_ms": 0,
                "metadata": {"skipped": True},
            }

        start = time.perf_counter()

        result = self._model.compress_prompt(
            text,
            rate=ratio,
            force_tokens=["\n", "def", "return", "if", "for", "while", "class", "```"],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        original_tokens = result.get("origin_tokens", len(text.split()))
        compressed_tokens = result.get("compressed_tokens", len(result["compressed_prompt"].split()))

        return {
            "compressed_text": result["compressed_prompt"],
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "actual_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            "compression_time_ms": elapsed_ms,
            "metadata": {
                "target_ratio": ratio,
                "tokens_removed": original_tokens - compressed_tokens,
            },
        }


class LLMLingua1Compressor(BaseCompressor):
    """LLMLingua-1 perplexity-based compression."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self._model = None

    def load(self) -> None:
        if self._loaded:
            return

        try:
            from llmlingua import PromptCompressor
            self._model = PromptCompressor(
                model_name=self.config.model_name,
                use_llmlingua2=False,  # Use original perplexity method
                device_map="auto",
            )
            self._loaded = True
            logger.info(f"Loaded LLMLingua-1 model: {self.config.model_name}")
        except ImportError as e:
            raise RuntimeError(
                f"LLMLingua not installed. Run: {self.config.install_command}"
            ) from e

    def compress(self, text: str, ratio: float) -> dict:
        self.load()

        if ratio >= 1.0:
            tokens = len(text.split())
            return {
                "compressed_text": text,
                "original_tokens": tokens,
                "compressed_tokens": tokens,
                "actual_ratio": 1.0,
                "compression_time_ms": 0,
                "metadata": {"skipped": True},
            }

        start = time.perf_counter()

        result = self._model.compress_prompt(
            text,
            rate=ratio,
            condition_compare=True,
            condition_in_question="none",
            force_tokens=["\n", "def", "return", "if", "for", "while", "class", "```"],
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        original_tokens = result.get("origin_tokens", len(text.split()))
        compressed_tokens = result.get("compressed_tokens", len(result["compressed_prompt"].split()))

        return {
            "compressed_text": result["compressed_prompt"],
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "actual_ratio": compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            "compression_time_ms": elapsed_ms,
            "metadata": {
                "target_ratio": ratio,
                "perplexity_based": True,
            },
        }


class SelectiveContextCompressor(BaseCompressor):
    """Selective Context self-information based compression."""

    def __init__(self, config: AlgorithmConfig):
        super().__init__(config)
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._loaded:
            return

        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            self._tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            self._model = GPT2LMHeadModel.from_pretrained(self.config.model_name)

            if torch.cuda.is_available():
                self._model = self._model.cuda()

            self._model.eval()
            self._loaded = True
            logger.info(f"Loaded Selective Context model: {self.config.model_name}")

        except ImportError as e:
            raise RuntimeError(
                f"Transformers not installed. Run: {self.config.install_command}"
            ) from e

    def compress(self, text: str, ratio: float) -> dict:
        import torch

        self.load()

        if ratio >= 1.0:
            tokens = len(text.split())
            return {
                "compressed_text": text,
                "original_tokens": tokens,
                "compressed_tokens": tokens,
                "actual_ratio": 1.0,
                "compression_time_ms": 0,
                "metadata": {"skipped": True},
            }

        start = time.perf_counter()

        # Tokenize input
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        tokens = self._tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        # Compute self-information for each token
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)

            token_ids = inputs["input_ids"][0]
            self_info = []

            for i in range(1, len(token_ids)):
                prob = probs[i-1, token_ids[i]].item()
                self_info.append(-np.log(prob + 1e-10))

        # Select tokens with highest self-information (most informative)
        n_keep = max(1, int(len(tokens) * ratio))

        # Always keep first token, then select by self-information
        indices_to_keep = [0]
        if len(self_info) > 0:
            sorted_indices = np.argsort(self_info)[::-1]  # Highest first
            for idx in sorted_indices[:n_keep-1]:
                indices_to_keep.append(idx + 1)

        indices_to_keep = sorted(set(indices_to_keep))
        kept_tokens = [tokens[i] for i in indices_to_keep]
        compressed_text = self._tokenizer.convert_tokens_to_string(kept_tokens)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "compressed_text": compressed_text,
            "original_tokens": len(tokens),
            "compressed_tokens": len(kept_tokens),
            "actual_ratio": len(kept_tokens) / len(tokens) if len(tokens) > 0 else 1.0,
            "compression_time_ms": elapsed_ms,
            "metadata": {
                "target_ratio": ratio,
                "avg_self_info": float(np.mean(self_info)) if self_info else 0.0,
                "kept_indices": indices_to_keep[:10],  # Sample for debugging
            },
        }


class RandomCompressor(BaseCompressor):
    """Random token selection baseline."""

    def __init__(self, config: AlgorithmConfig, seed: int = 42):
        super().__init__(config)
        self._rng = random.Random(seed)

    def load(self) -> None:
        self._loaded = True

    def compress(self, text: str, ratio: float) -> dict:
        if ratio >= 1.0:
            tokens = text.split()
            return {
                "compressed_text": text,
                "original_tokens": len(tokens),
                "compressed_tokens": len(tokens),
                "actual_ratio": 1.0,
                "compression_time_ms": 0,
                "metadata": {"skipped": True},
            }

        start = time.perf_counter()

        tokens = text.split()
        n_keep = max(1, int(len(tokens) * ratio))

        # Random selection (but deterministic for reproducibility)
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)

        # Keep tokens at random positions (maintaining some order)
        indices = sorted(rng.sample(range(len(tokens)), min(n_keep, len(tokens))))
        kept_tokens = [tokens[i] for i in indices]
        compressed_text = " ".join(kept_tokens)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "compressed_text": compressed_text,
            "original_tokens": len(tokens),
            "compressed_tokens": len(kept_tokens),
            "actual_ratio": len(kept_tokens) / len(tokens) if len(tokens) > 0 else 1.0,
            "compression_time_ms": elapsed_ms,
            "metadata": {
                "target_ratio": ratio,
                "selection": "random",
                "seed": seed,
            },
        }


def get_compressor(algorithm: CompressionAlgorithm) -> BaseCompressor:
    """Factory function to get appropriate compressor."""
    config = ALGORITHM_CONFIGS[algorithm]

    if algorithm == CompressionAlgorithm.LLMLINGUA2:
        return LLMLingua2Compressor(config)
    elif algorithm == CompressionAlgorithm.LLMLINGUA1:
        return LLMLingua1Compressor(config)
    elif algorithm == CompressionAlgorithm.SELECTIVE_CONTEXT:
        return SelectiveContextCompressor(config)
    elif algorithm == CompressionAlgorithm.RANDOM:
        return RandomCompressor(config)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# =============================================================================
# Experiment Data Structures
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single experimental trial."""
    trial_id: str
    timestamp: str

    # Experimental conditions
    algorithm: str
    compression_ratio_target: float
    compression_ratio_actual: float
    task_type: str
    benchmark: str
    problem_id: str

    # Token metrics
    original_tokens: int
    compressed_tokens: int
    compression_time_ms: float

    # Generation metrics
    model: str
    input_tokens: int
    output_tokens: int
    generation_latency_ms: float
    cost_usd: float

    # Evaluation
    score: float
    correct: bool
    eval_details: dict = field(default_factory=dict)

    # Debug info
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ExperimentPhase:
    """Configuration for an experiment phase."""
    name: str
    algorithms: list[CompressionAlgorithm]
    ratios: list[float]
    benchmarks: list[str]
    models: list[str]
    samples_per_condition: int
    description: str
    estimated_cost_usd: float = 0.0
    estimated_time_hours: float = 0.0


# Phase definitions for incremental experiment
EXPERIMENT_PHASES = {
    "phase1_quick_validation": ExperimentPhase(
        name="Phase 1: Quick Validation",
        algorithms=[CompressionAlgorithm.LLMLINGUA2, CompressionAlgorithm.RANDOM],
        ratios=[0.5, 0.7, 1.0],
        benchmarks=["humaneval", "gsm8k"],
        models=["claude-3-haiku"],
        samples_per_condition=10,
        description="Quick check if random baseline differs from LLMLingua-2",
        estimated_cost_usd=15.0,
        estimated_time_hours=0.5,
    ),
    "phase2_algorithm_pairs": ExperimentPhase(
        name="Phase 2: Algorithm Pair Comparison",
        algorithms=[
            CompressionAlgorithm.LLMLINGUA2,
            CompressionAlgorithm.LLMLINGUA1,
        ],
        ratios=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        benchmarks=["humaneval", "mbpp", "gsm8k"],
        models=["claude-3-haiku", "deepseek-chat"],
        samples_per_condition=20,
        description="Compare LLMLingua-2 vs LLMLingua-1 threshold patterns",
        estimated_cost_usd=80.0,
        estimated_time_hours=2.0,
    ),
    "phase3_full_comparison": ExperimentPhase(
        name="Phase 3: Full Algorithm Comparison",
        algorithms=[
            CompressionAlgorithm.LLMLINGUA2,
            CompressionAlgorithm.LLMLINGUA1,
            CompressionAlgorithm.SELECTIVE_CONTEXT,
            CompressionAlgorithm.RANDOM,
        ],
        ratios=COMPRESSION_RATIOS,
        benchmarks=["humaneval", "mbpp", "gsm8k", "math"],
        models=["claude-3-haiku", "gpt-4o-mini", "deepseek-chat"],
        samples_per_condition=25,
        description="Full comparison of all algorithms across all conditions",
        estimated_cost_usd=350.0,
        estimated_time_hours=8.0,
    ),
}


# =============================================================================
# Experiment Runner
# =============================================================================

class AlgorithmComparisonExperiment:
    """Main experiment orchestrator for algorithm comparison."""

    def __init__(
        self,
        phase: ExperimentPhase,
        output_dir: str = "results/algorithm_comparison",
        seed: int = 42,
    ):
        self.phase = phase
        self.output_dir = Path(output_dir) / phase.name.lower().replace(" ", "_")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Result tracking
        self.results_file = self.output_dir / "results.jsonl"
        self.checkpoint_file = self.output_dir / "checkpoint.json"
        self.completed_trials: set[str] = set()

        # Load checkpoint if exists
        self._load_checkpoint()

        # Compressor cache
        self._compressors: dict[CompressionAlgorithm, BaseCompressor] = {}

        logger.info(f"Initialized experiment: {phase.name}")
        logger.info(f"Output directory: {self.output_dir}")

    def _load_checkpoint(self) -> None:
        """Load checkpoint from disk."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
                self.completed_trials = set(data.get("completed_trials", []))
            logger.info(f"Loaded checkpoint: {len(self.completed_trials)} completed trials")

    def _save_checkpoint(self) -> None:
        """Save checkpoint to disk."""
        with open(self.checkpoint_file, "w") as f:
            json.dump({
                "completed_trials": list(self.completed_trials),
                "timestamp": datetime.now().isoformat(),
                "phase": self.phase.name,
            }, f, indent=2)

    def _trial_key(
        self,
        algorithm: CompressionAlgorithm,
        ratio: float,
        benchmark: str,
        model: str,
        problem_idx: int,
    ) -> str:
        """Generate unique trial key."""
        return f"{algorithm.value}|{ratio}|{benchmark}|{model}|{problem_idx}"

    def _get_compressor(self, algorithm: CompressionAlgorithm) -> BaseCompressor:
        """Get or create compressor (cached)."""
        if algorithm not in self._compressors:
            self._compressors[algorithm] = get_compressor(algorithm)
        return self._compressors[algorithm]

    def _load_benchmark(self, benchmark_name: str) -> list[dict]:
        """Load benchmark dataset."""
        # Use HuggingFace datasets
        try:
            from datasets import load_dataset

            if benchmark_name == "humaneval":
                ds = load_dataset("openai/openai_humaneval", split="test")
                return [
                    {
                        "id": row["task_id"],
                        "prompt": row["prompt"],
                        "canonical_solution": row["canonical_solution"],
                        "test": row["test"],
                        "entry_point": row["entry_point"],
                    }
                    for row in ds
                ]
            elif benchmark_name == "mbpp":
                ds = load_dataset("google-research-datasets/mbpp", split="test")
                return [
                    {
                        "id": str(row["task_id"]),
                        "prompt": row["text"],
                        "code": row["code"],
                        "test_list": row["test_list"],
                    }
                    for row in ds
                ][:self.phase.samples_per_condition * 10]  # Limit size
            elif benchmark_name == "gsm8k":
                ds = load_dataset("openai/gsm8k", "main", split="test")
                return [
                    {
                        "id": f"gsm8k_{i}",
                        "question": row["question"],
                        "answer": row["answer"],
                    }
                    for i, row in enumerate(ds)
                ]
            elif benchmark_name == "math":
                ds = load_dataset("lighteval/MATH", split="test")
                return [
                    {
                        "id": f"math_{i}",
                        "problem": row["problem"],
                        "solution": row["solution"],
                        "level": row.get("level", "unknown"),
                        "type": row.get("type", "unknown"),
                    }
                    for i, row in enumerate(ds)
                ][:500]  # Limit for cost
            else:
                logger.warning(f"Unknown benchmark: {benchmark_name}")
                return []

        except Exception as e:
            logger.error(f"Failed to load benchmark {benchmark_name}: {e}")
            return []

    def _create_prompt(self, problem: dict, benchmark_name: str) -> str:
        """Create prompt for a benchmark problem."""
        task_type = BENCHMARKS[benchmark_name]["type"]

        if task_type == TaskType.CODE:
            if benchmark_name == "humaneval":
                return f"""Complete the following Python function. Only provide the implementation code.

{problem['prompt']}"""
            elif benchmark_name == "mbpp":
                return f"""Write a Python function to solve this problem. Only provide the complete function code.

Problem: {problem['prompt']}

Solution:"""
        else:  # CoT
            if benchmark_name == "gsm8k":
                return f"""Solve this math problem step by step. Show your work and give the final numerical answer.

Question: {problem['question']}

Let's solve this step by step:"""
            elif benchmark_name == "math":
                return f"""Solve this mathematics problem. Show your complete solution and put your final answer in \\boxed{{}}.

Problem: {problem['problem']}

Solution:"""

        return str(problem.get("prompt", problem.get("question", "")))

    async def _call_model(
        self,
        prompt: str,
        model: str,
        max_tokens: int = 1024,
    ) -> dict:
        """Call LLM model for generation."""
        # Import providers dynamically
        import httpx

        provider_configs = {
            "claude-3-haiku": {
                "provider": "anthropic",
                "model_id": "claude-3-haiku-20240307",
                "url": "https://api.anthropic.com/v1/messages",
                "cost_input": 0.25,
                "cost_output": 1.25,
            },
            "gpt-4o-mini": {
                "provider": "openai",
                "model_id": "gpt-4o-mini",
                "url": "https://api.openai.com/v1/chat/completions",
                "cost_input": 0.15,
                "cost_output": 0.60,
            },
            "deepseek-chat": {
                "provider": "deepseek",
                "model_id": "deepseek-chat",
                "url": "https://api.deepseek.com/v1/chat/completions",
                "cost_input": 0.14,
                "cost_output": 0.28,
            },
        }

        config = provider_configs.get(model)
        if not config:
            return {"error": f"Unknown model: {model}", "content": None}

        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                if config["provider"] == "anthropic":
                    headers = {
                        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    }
                    payload = {
                        "model": config["model_id"],
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    response = await client.post(config["url"], headers=headers, json=payload)
                    data = response.json()

                    if response.status_code != 200:
                        return {"error": data.get("error", {}).get("message", str(data)), "content": None}

                    content = data["content"][0]["text"]
                    input_tokens = data["usage"]["input_tokens"]
                    output_tokens = data["usage"]["output_tokens"]

                else:  # OpenAI-compatible
                    api_key_env = "OPENAI_API_KEY" if config["provider"] == "openai" else "DEEPSEEK_API_KEY"
                    headers = {
                        "Authorization": f"Bearer {os.environ.get(api_key_env, '')}",
                        "Content-Type": "application/json",
                    }
                    payload = {
                        "model": config["model_id"],
                        "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    response = await client.post(config["url"], headers=headers, json=payload)
                    data = response.json()

                    if response.status_code != 200:
                        return {"error": data.get("error", {}).get("message", str(data)), "content": None}

                    content = data["choices"][0]["message"]["content"]
                    input_tokens = data["usage"]["prompt_tokens"]
                    output_tokens = data["usage"]["completion_tokens"]

            latency_ms = (time.perf_counter() - start) * 1000
            cost = (
                input_tokens * config["cost_input"] / 1_000_000 +
                output_tokens * config["cost_output"] / 1_000_000
            )

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency_ms": latency_ms,
                "cost_usd": cost,
                "error": None,
            }

        except Exception as e:
            return {"error": str(e), "content": None}

    def _evaluate(self, output: str, problem: dict, benchmark_name: str) -> dict:
        """Evaluate model output."""
        import re
        import subprocess
        import tempfile

        task_type = BENCHMARKS[benchmark_name]["type"]

        if task_type == TaskType.CODE:
            # Code execution evaluation
            try:
                # Extract code
                code = output
                if "```python" in output:
                    start = output.find("```python") + 9
                    end = output.find("```", start)
                    if end > start:
                        code = output[start:end].strip()
                elif "```" in output:
                    start = output.find("```") + 3
                    end = output.find("```", start)
                    if end > start:
                        code = output[start:end].strip()

                # Prepare test code
                if benchmark_name == "humaneval":
                    full_code = f"{problem['prompt']}{code}\n\n{problem['test']}\n\ncheck({problem['entry_point']})"
                else:  # mbpp
                    test_code = "\n".join(problem.get("test_list", []))
                    full_code = f"{code}\n\n{test_code}"

                # Execute
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
                    "stdout": result.stdout[:200] if result.stdout else None,
                    "stderr": result.stderr[:200] if result.stderr else None,
                }

            except subprocess.TimeoutExpired:
                return {"score": 0.0, "correct": False, "error": "timeout"}
            except Exception as e:
                return {"score": 0.0, "correct": False, "error": str(e)}

        else:  # CoT - exact match
            # Extract numerical answer
            if benchmark_name == "gsm8k":
                # Look for final number
                numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', output)
                predicted = numbers[-1].replace(",", "") if numbers else ""

                # Expected answer from GSM8K format
                expected_text = problem["answer"]
                if "####" in expected_text:
                    expected = expected_text.split("####")[-1].strip()
                else:
                    exp_numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', expected_text)
                    expected = exp_numbers[-1].replace(",", "") if exp_numbers else ""

            else:  # math
                # Look for boxed answer
                boxed = re.search(r'\\boxed{([^}]+)}', output)
                predicted = boxed.group(1).strip() if boxed else ""

                exp_boxed = re.search(r'\\boxed{([^}]+)}', problem.get("solution", ""))
                expected = exp_boxed.group(1).strip() if exp_boxed else ""

            # Normalize for comparison
            try:
                pred_num = float(predicted) if predicted else None
                exp_num = float(expected) if expected else None
                correct = pred_num is not None and exp_num is not None and abs(pred_num - exp_num) < 0.01
            except ValueError:
                correct = predicted == expected

            return {
                "score": 1.0 if correct else 0.0,
                "correct": correct,
                "predicted": predicted,
                "expected": expected,
            }

    async def run_trial(
        self,
        algorithm: CompressionAlgorithm,
        ratio: float,
        benchmark_name: str,
        model: str,
        problem: dict,
        problem_idx: int,
    ) -> TrialResult:
        """Run a single experimental trial."""
        trial_id = self._trial_key(algorithm, ratio, benchmark_name, model, problem_idx)

        # Get task type
        task_type = BENCHMARKS[benchmark_name]["type"]

        # Create prompt
        prompt = self._create_prompt(problem, benchmark_name)

        # Compress
        compressor = self._get_compressor(algorithm)
        compression_result = compressor.compress(prompt, ratio)

        # Generate
        generation_result = await self._call_model(
            compression_result["compressed_text"],
            model,
        )

        # Evaluate
        if generation_result.get("content"):
            eval_result = self._evaluate(generation_result["content"], problem, benchmark_name)
        else:
            eval_result = {"score": 0.0, "correct": False, "error": generation_result.get("error")}

        return TrialResult(
            trial_id=trial_id,
            timestamp=datetime.now().isoformat(),
            algorithm=algorithm.value,
            compression_ratio_target=ratio,
            compression_ratio_actual=compression_result["actual_ratio"],
            task_type=task_type.value,
            benchmark=benchmark_name,
            problem_id=str(problem.get("id", problem_idx)),
            original_tokens=compression_result["original_tokens"],
            compressed_tokens=compression_result["compressed_tokens"],
            compression_time_ms=compression_result["compression_time_ms"],
            model=model,
            input_tokens=generation_result.get("input_tokens", 0),
            output_tokens=generation_result.get("output_tokens", 0),
            generation_latency_ms=generation_result.get("latency_ms", 0),
            cost_usd=generation_result.get("cost_usd", 0),
            score=eval_result["score"],
            correct=eval_result.get("correct", False),
            eval_details=eval_result,
            error=generation_result.get("error"),
            metadata=compression_result.get("metadata", {}),
        )

    async def run(self, dry_run: bool = False) -> None:
        """Run the complete experiment phase."""
        logger.info(f"Starting experiment: {self.phase.name}")
        logger.info(f"Description: {self.phase.description}")
        logger.info(f"Estimated cost: ${self.phase.estimated_cost_usd}")
        logger.info(f"Estimated time: {self.phase.estimated_time_hours} hours")

        # Calculate total trials
        total_trials = (
            len(self.phase.algorithms) *
            len(self.phase.ratios) *
            len(self.phase.benchmarks) *
            len(self.phase.models) *
            self.phase.samples_per_condition
        )
        logger.info(f"Total trials: {total_trials}")

        # Load benchmarks
        benchmarks_data = {}
        for benchmark_name in self.phase.benchmarks:
            benchmarks_data[benchmark_name] = self._load_benchmark(benchmark_name)
            logger.info(f"Loaded {len(benchmarks_data[benchmark_name])} problems from {benchmark_name}")

        trial_count = 0
        new_trials = 0

        for algorithm in self.phase.algorithms:
            logger.info(f"Testing algorithm: {algorithm.value}")

            for model in self.phase.models:
                for benchmark_name in self.phase.benchmarks:
                    problems = benchmarks_data[benchmark_name]

                    for ratio in self.phase.ratios:
                        for idx in range(min(self.phase.samples_per_condition, len(problems))):
                            trial_key = self._trial_key(algorithm, ratio, benchmark_name, model, idx)

                            if trial_key in self.completed_trials:
                                trial_count += 1
                                continue

                            if dry_run:
                                logger.info(f"[DRY RUN] Would run: {trial_key}")
                                continue

                            # Run trial
                            problem = problems[idx]
                            result = await self.run_trial(
                                algorithm, ratio, benchmark_name, model, problem, idx
                            )

                            # Save result
                            with open(self.results_file, "a") as f:
                                f.write(json.dumps(asdict(result)) + "\n")

                            self.completed_trials.add(trial_key)
                            trial_count += 1
                            new_trials += 1

                            # Progress and checkpointing
                            if new_trials % 10 == 0:
                                self._save_checkpoint()
                                pct = trial_count / total_trials * 100
                                logger.info(f"Progress: {trial_count}/{total_trials} ({pct:.1f}%)")

                            # Rate limiting
                            await asyncio.sleep(0.2)

        self._save_checkpoint()
        logger.info(f"Experiment complete: {new_trials} new trials, {trial_count} total")

    def get_summary(self) -> dict:
        """Generate experiment summary."""
        if not self.results_file.exists():
            return {"error": "No results found"}

        results = []
        with open(self.results_file) as f:
            for line in f:
                results.append(json.loads(line))

        # Group by algorithm and task type
        summary = {
            "phase": self.phase.name,
            "total_trials": len(results),
            "by_algorithm": {},
            "by_task_type": {},
        }

        for result in results:
            alg = result["algorithm"]
            task = result["task_type"]

            # By algorithm
            if alg not in summary["by_algorithm"]:
                summary["by_algorithm"][alg] = {"total": 0, "correct": 0, "cost": 0}
            summary["by_algorithm"][alg]["total"] += 1
            summary["by_algorithm"][alg]["correct"] += result["correct"]
            summary["by_algorithm"][alg]["cost"] += result["cost_usd"]

            # By task type
            if task not in summary["by_task_type"]:
                summary["by_task_type"][task] = {}
            if alg not in summary["by_task_type"][task]:
                summary["by_task_type"][task][alg] = {"total": 0, "correct": 0}
            summary["by_task_type"][task][alg]["total"] += 1
            summary["by_task_type"][task][alg]["correct"] += result["correct"]

        return summary


# =============================================================================
# CLI Interface
# =============================================================================

def print_phase_info(phase: ExperimentPhase) -> None:
    """Print detailed phase information."""
    print(f"\n{'='*60}")
    print(f"Phase: {phase.name}")
    print(f"{'='*60}")
    print(f"Description: {phase.description}")
    print(f"\nAlgorithms: {', '.join(a.value for a in phase.algorithms)}")
    print(f"Ratios: {phase.ratios}")
    print(f"Benchmarks: {phase.benchmarks}")
    print(f"Models: {phase.models}")
    print(f"Samples per condition: {phase.samples_per_condition}")
    print(f"\nEstimated cost: ${phase.estimated_cost_usd:.2f}")
    print(f"Estimated time: {phase.estimated_time_hours:.1f} hours")

    # Calculate total trials
    total = (
        len(phase.algorithms) *
        len(phase.ratios) *
        len(phase.benchmarks) *
        len(phase.models) *
        phase.samples_per_condition
    )
    print(f"Total trials: {total}")


def print_installation_guide() -> None:
    """Print installation guide for all algorithms."""
    print("\n" + "="*60)
    print("COMPRESSION ALGORITHM INSTALLATION GUIDE")
    print("="*60)

    for alg, config in ALGORITHM_CONFIGS.items():
        print(f"\n{config.name}")
        print("-" * 40)
        print(f"Type: {alg.value}")
        print(f"Model: {config.model_name}")
        print(f"Description: {config.description}")
        print(f"GPU Required: {'Yes' if config.gpu_required else 'No'}")
        print(f"Memory: {config.memory_gb} GB")
        print(f"Install: {config.install_command}")
        print(f"Reference: {config.reference}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Algorithm Comparison Experiment for TAAC Study"
    )
    parser.add_argument(
        "--phase",
        choices=list(EXPERIMENT_PHASES.keys()),
        default="phase1_quick_validation",
        help="Experiment phase to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print phase information and exit",
    )
    parser.add_argument(
        "--install-guide",
        action="store_true",
        help="Print installation guide for all algorithms",
    )
    parser.add_argument(
        "--output-dir",
        default="results/algorithm_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary of completed experiment",
    )

    args = parser.parse_args()

    if args.install_guide:
        print_installation_guide()
        return

    phase = EXPERIMENT_PHASES[args.phase]

    if args.info:
        print_phase_info(phase)
        return

    experiment = AlgorithmComparisonExperiment(
        phase=phase,
        output_dir=args.output_dir,
    )

    if args.summary:
        summary = experiment.get_summary()
        print(json.dumps(summary, indent=2))
        return

    await experiment.run(dry_run=args.dry_run)


if __name__ == "__main__":
    asyncio.run(main())

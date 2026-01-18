"""
Configuration for TAAC Follow-up Study Experiments
"""

from dataclasses import dataclass
from typing import Literal

# =============================================================================
# Compression Methods
# =============================================================================

COMPRESSION_METHODS = {
    "llmlingua2": {
        "name": "LLMLingua-2",
        "model": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
        "type": "learned",
        "reference": "Pan et al. 2024",
    },
    "llmlingua1": {
        "name": "LLMLingua-1",
        "model": "NousResearch/Llama-2-7b-hf",  # Pilot model for perplexity
        "type": "perplexity",
        "reference": "Jiang et al. 2023",
    },
    "selective_context": {
        "name": "Selective Context",
        "model": "gpt2-large",  # For self-information
        "type": "extractive",
        "reference": "Li et al. 2023",
    },
}

COMPRESSION_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]

# =============================================================================
# Models
# =============================================================================

MODELS = {
    # Tier 1: Economy
    "deepseek-chat": {
        "tier": 1,
        "provider": "deepseek",
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
    # Tier 4: Frontier (optional, expensive)
    "claude-opus": {
        "tier": 4,
        "provider": "anthropic",
        "model_id": "claude-3-opus-20240229",
        "cost_input": 15.00,
        "cost_output": 75.00,
    },
}

# =============================================================================
# Benchmarks
# =============================================================================

BENCHMARKS = {
    # Code Generation
    "humaneval": {
        "type": "code",
        "source": "openai/human-eval",
        "size": 164,
        "metric": "pass@1",
        "language": "python",
    },
    "mbpp": {
        "type": "code",
        "source": "google-research/mbpp",
        "size": 974,
        "metric": "pass@1",
        "language": "python",
    },
    "humaneval_plus": {
        "type": "code",
        "source": "evalplus/humanevalplus",
        "size": 164,
        "metric": "pass@1",
        "language": "python",
    },
    "multipl_e_python": {
        "type": "code",
        "source": "nuprl/MultiPL-E",
        "size": 161,
        "metric": "pass@1",
        "language": "python",
    },
    "multipl_e_javascript": {
        "type": "code",
        "source": "nuprl/MultiPL-E",
        "size": 161,
        "metric": "pass@1",
        "language": "javascript",
    },
    "multipl_e_java": {
        "type": "code",
        "source": "nuprl/MultiPL-E",
        "size": 158,
        "metric": "pass@1",
        "language": "java",
    },
    # Chain-of-Thought Reasoning
    "gsm8k": {
        "type": "cot",
        "source": "openai/gsm8k",
        "size": 1319,
        "metric": "exact_match",
    },
    "math": {
        "type": "cot",
        "source": "hendrycks/math",
        "size": 5000,
        "metric": "exact_match",
    },
    "arc_challenge": {
        "type": "cot",
        "source": "allenai/arc",
        "size": 2590,
        "metric": "accuracy",
    },
    "mmlu_stem": {
        "type": "cot",
        "source": "cais/mmlu",
        "size": 5000,
        "metric": "accuracy",
    },
}

# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run"""
    name: str
    compression_methods: list[str]
    compression_ratios: list[float]
    models: list[str]
    benchmarks: list[str]
    trials_per_condition: int = 25
    random_seed: int = 42
    checkpoint_every: int = 100
    parallel_workers: int = 4

    @property
    def total_conditions(self) -> int:
        return (
            len(self.compression_methods) *
            len(self.compression_ratios) *
            len(self.models) *
            len(self.benchmarks)
        )

    @property
    def total_trials(self) -> int:
        return self.total_conditions * self.trials_per_condition


# Predefined experiment configurations
EXPERIMENTS = {
    # Phase 1: Quick validation on expanded benchmarks
    "phase1_benchmark_expansion": ExperimentConfig(
        name="Phase 1: Benchmark Expansion",
        compression_methods=["llmlingua2"],
        compression_ratios=[0.5, 0.6, 0.7, 1.0],
        models=["claude-3-haiku", "deepseek-chat"],
        benchmarks=["humaneval", "mbpp", "humaneval_plus", "gsm8k", "math"],
        trials_per_condition=20,
    ),

    # Phase 2: Perplexity mechanism study
    "phase2_perplexity_study": ExperimentConfig(
        name="Phase 2: Perplexity Mechanism",
        compression_methods=["llmlingua2", "llmlingua1"],
        compression_ratios=[0.3, 0.5, 0.7],
        models=["claude-3-haiku"],
        benchmarks=["humaneval", "gsm8k"],
        trials_per_condition=50,
    ),

    # Phase 3: Compression method comparison
    "phase3_method_comparison": ExperimentConfig(
        name="Phase 3: Method Comparison",
        compression_methods=["llmlingua2", "llmlingua1", "selective_context"],
        compression_ratios=[0.4, 0.5, 0.6, 0.7, 1.0],
        models=["claude-3-haiku", "gpt-4o-mini", "deepseek-chat"],
        benchmarks=["humaneval", "mbpp", "gsm8k", "math"],
        trials_per_condition=25,
    ),

    # Phase 4: Full scale experiment
    "phase4_full_scale": ExperimentConfig(
        name="Phase 4: Full Scale",
        compression_methods=["llmlingua2", "llmlingua1", "selective_context"],
        compression_ratios=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0],
        models=["deepseek-chat", "llama-3.1-8b", "gpt-4o-mini", "claude-3-haiku",
                "claude-3.5-sonnet", "gpt-4o"],
        benchmarks=["humaneval", "mbpp", "humaneval_plus", "multipl_e_python",
                   "gsm8k", "math", "arc_challenge", "mmlu_stem"],
        trials_per_condition=25,
    ),
}


if __name__ == "__main__":
    # Print experiment summaries
    for name, config in EXPERIMENTS.items():
        print(f"\n{config.name}")
        print(f"  Conditions: {config.total_conditions}")
        print(f"  Total trials: {config.total_trials}")
        print(f"  Estimated cost: ${config.total_trials * 0.05:.0f}-${config.total_trials * 0.15:.0f}")

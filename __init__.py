"""
TAAC: Task-Aware Adaptive Compression
======================================

A quality-gated compression algorithm that dynamically adjusts compression
based on task type, information density, and predicted quality degradation.

Key Components:
- TAAC: Main compression algorithm with quality gating
- TAACConfig: Configuration for thresholds and parameters
- TaskClassifier: Fast task type classification (<10ms)
- InformationDensityEstimator: Perplexity CV estimation
- QualityPredictor: Learned quality prediction

Baselines:
- FixedRatioCompressor: Fixed r for all prompts
- TaskBasedFixedCompressor: Task-specific fixed ratios

Ablation Variants:
- TAACTaskOnly: Task classification without density/quality-gating
- TAACDensityOnly: Density estimation without task/quality-gating
- TAACQualityGateOnly: Quality gating without task/density

Scripts:
- train_quality_predictor.py: Train the quality predictor
- ablation_study.py: Run component ablation study
- evaluate_comparison.py: Compare TAAC vs baselines

Author: Dr. Amanda Foster, Bona Opera Studios
Date: January 2026
"""

from .taac_algorithm import (
    TAAC,
    TAACConfig,
    TaskType,
    CompressionResult,
    TaskClassifier,
    InformationDensityEstimator,
    QualityPredictor,
    CompressionEngine,
    # Ablation variants
    TAACTaskOnly,
    TAACDensityOnly,
    TAACQualityGateOnly,
    # Baselines
    FixedRatioCompressor,
    TaskBasedFixedCompressor,
)

__version__ = "0.1.0"
__author__ = "Dr. Amanda Foster"
__email__ = "amanda.foster@bonaopera.dev"

__all__ = [
    "TAAC",
    "TAACConfig",
    "TaskType",
    "CompressionResult",
    "TaskClassifier",
    "InformationDensityEstimator",
    "QualityPredictor",
    "CompressionEngine",
    "TAACTaskOnly",
    "TAACDensityOnly",
    "TAACQualityGateOnly",
    "FixedRatioCompressor",
    "TaskBasedFixedCompressor",
]

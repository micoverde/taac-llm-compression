#!/usr/bin/env python3
"""
Calibration Protocol: GPU Energy Proxy Validation
==================================================

Dr. Kim - Validation Specialist
Article 4: Green AI Research

This module implements the calibration protocol to validate and correct
our energy proxy model using direct GPU measurements via NVML.

The protocol:
1. Measures actual GPU energy on Llama-3.1-8B via NVML
2. Fits regression model: E = alpha*T_in + beta*T_out
3. Validates with R^2, MAPE, and residual analysis
4. Extrapolates to larger models (70B, API providers)
5. Quantifies uncertainty with confidence intervals

References:
- Patterson et al. (2021) Carbon Emissions and Large Neural Network Training
- Luccioni et al. (2023) Power Hungry Processing
- Strubell et al. (2019) Energy and Policy Considerations for Deep Learning
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable
import warnings

# Scientific computing imports
try:
    import numpy as np
    from scipy import stats
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - statistical analysis will be limited")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Current Proxy Model Constants (to be calibrated)
# =============================================================================

# Original proxy model from Article 4 specification
PROXY_CONSTANTS_ORIGINAL = {
    "small": {  # ~7-8B models
        "E_input": 0.000018,   # J/token
        "E_output": 0.000036,  # J/token
        "model_size_B": 7
    },
    "medium": {  # ~50B models
        "E_input": 0.000090,
        "E_output": 0.000180,
        "model_size_B": 50
    },
    "large": {  # ~70B models
        "E_input": 0.00018,
        "E_output": 0.00036,
        "model_size_B": 70
    },
    "xlarge": {  # ~400B+ models
        "E_input": 0.00102,
        "E_output": 0.00204,
        "model_size_B": 400
    }
}


# =============================================================================
# Data Classes for Calibration
# =============================================================================

@dataclass
class CalibrationSample:
    """Single calibration measurement sample."""
    sample_id: str
    input_tokens: int
    output_tokens: int
    measured_energy_joules: float
    duration_seconds: float
    avg_power_watts: float
    peak_power_watts: float
    idle_power_watts: float
    active_energy_joules: float  # Energy above idle baseline
    gpu_utilization_avg: float
    memory_utilization_avg: float
    temperature_c_avg: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegressionResult:
    """Results from energy regression fitting."""
    alpha: float  # Coefficient for input tokens (J/token)
    beta: float   # Coefficient for output tokens (J/token)
    intercept: float  # Fixed overhead (J)
    alpha_stderr: float
    beta_stderr: float
    intercept_stderr: float
    r_squared: float
    adjusted_r_squared: float
    n_samples: int
    degrees_of_freedom: int

    # Confidence intervals (95%)
    alpha_ci: Tuple[float, float] = (0.0, 0.0)
    beta_ci: Tuple[float, float] = (0.0, 0.0)
    intercept_ci: Tuple[float, float] = (0.0, 0.0)

    # Residual analysis
    residual_mean: float = 0.0
    residual_std: float = 0.0
    durbin_watson: float = 0.0  # Autocorrelation test
    shapiro_wilk_p: float = 0.0  # Normality test

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationMetrics:
    """Validation metrics for the calibrated model."""
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Squared Error
    mae: float   # Mean Absolute Error
    max_error: float
    max_error_percent: float
    bias: float  # Systematic under/over-estimation

    # Per-region metrics (input-heavy vs output-heavy)
    mape_input_heavy: float = 0.0
    mape_output_heavy: float = 0.0
    mape_balanced: float = 0.0

    # Cross-validation scores
    cv_r_squared_mean: float = 0.0
    cv_r_squared_std: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScalingFactors:
    """Factors for extrapolating to different model sizes."""
    base_model_size_B: float
    target_model_size_B: float
    compute_scaling_factor: float  # How compute scales with model size
    memory_bandwidth_factor: float  # Memory transfer energy

    # Derived coefficients for target model
    alpha_scaled: float = 0.0
    beta_scaled: float = 0.0

    # Uncertainty bounds
    scaling_uncertainty_percent: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibratedProxyModel:
    """The final calibrated energy proxy model."""
    model_name: str  # e.g., "Llama-3.1-8B"
    calibration_date: str

    # Core coefficients (from regression)
    alpha: float  # J/input_token
    beta: float   # J/output_token
    intercept: float  # Fixed overhead J

    # Confidence intervals
    alpha_ci_95: Tuple[float, float]
    beta_ci_95: Tuple[float, float]

    # Validation metrics
    r_squared: float
    mape: float

    # Scaling parameters for extrapolation
    model_size_B: float
    scaling_exponent: float  # For E ~ N^scaling_exponent

    # Applicability bounds
    min_input_tokens: int
    max_input_tokens: int
    min_output_tokens: int
    max_output_tokens: int

    # Calibration metadata
    n_calibration_samples: int
    hardware: str  # e.g., "NVIDIA H100"

    def predict(self, input_tokens: int, output_tokens: int) -> float:
        """Predict energy consumption in Joules."""
        return self.intercept + (self.alpha * input_tokens) + (self.beta * output_tokens)

    def predict_with_uncertainty(
        self,
        input_tokens: int,
        output_tokens: int,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Predict energy with confidence interval.

        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        point = self.predict(input_tokens, output_tokens)

        # Propagate uncertainty from coefficients
        # Using linear error propagation
        var_alpha = ((self.alpha_ci_95[1] - self.alpha_ci_95[0]) / 3.92) ** 2
        var_beta = ((self.beta_ci_95[1] - self.beta_ci_95[0]) / 3.92) ** 2

        variance = (input_tokens ** 2) * var_alpha + (output_tokens ** 2) * var_beta
        std = math.sqrt(variance)

        z = 1.96 if confidence == 0.95 else stats.norm.ppf((1 + confidence) / 2)
        return (point, point - z * std, point + z * std)

    def scale_to_model_size(self, target_size_B: float) -> 'CalibratedProxyModel':
        """
        Scale coefficients to a different model size.

        Uses neural scaling laws: E ~ N^scaling_exponent
        where N is model size in parameters.
        """
        scale_factor = (target_size_B / self.model_size_B) ** self.scaling_exponent

        # Add uncertainty for extrapolation
        uncertainty_factor = 1 + 0.1 * abs(math.log(target_size_B / self.model_size_B))

        return CalibratedProxyModel(
            model_name=f"Scaled-{target_size_B}B",
            calibration_date=self.calibration_date,
            alpha=self.alpha * scale_factor,
            beta=self.beta * scale_factor,
            intercept=self.intercept * scale_factor,
            alpha_ci_95=(
                self.alpha_ci_95[0] * scale_factor / uncertainty_factor,
                self.alpha_ci_95[1] * scale_factor * uncertainty_factor
            ),
            beta_ci_95=(
                self.beta_ci_95[0] * scale_factor / uncertainty_factor,
                self.beta_ci_95[1] * scale_factor * uncertainty_factor
            ),
            r_squared=self.r_squared * 0.9,  # Reduced confidence for extrapolation
            mape=self.mape * uncertainty_factor,
            model_size_B=target_size_B,
            scaling_exponent=self.scaling_exponent,
            min_input_tokens=self.min_input_tokens,
            max_input_tokens=self.max_input_tokens,
            min_output_tokens=self.min_output_tokens,
            max_output_tokens=self.max_output_tokens,
            n_calibration_samples=self.n_calibration_samples,
            hardware=self.hardware
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "calibration_date": self.calibration_date,
            "coefficients": {
                "alpha_j_per_input_token": self.alpha,
                "beta_j_per_output_token": self.beta,
                "intercept_j": self.intercept
            },
            "confidence_intervals_95": {
                "alpha": self.alpha_ci_95,
                "beta": self.beta_ci_95
            },
            "validation": {
                "r_squared": self.r_squared,
                "mape_percent": self.mape
            },
            "scaling": {
                "model_size_B": self.model_size_B,
                "scaling_exponent": self.scaling_exponent
            },
            "applicability": {
                "input_tokens": [self.min_input_tokens, self.max_input_tokens],
                "output_tokens": [self.min_output_tokens, self.max_output_tokens]
            },
            "metadata": {
                "n_samples": self.n_calibration_samples,
                "hardware": self.hardware
            }
        }


# =============================================================================
# Experiment Design: Isolating Input vs Output Energy
# =============================================================================

class CalibrationExperimentDesign:
    """
    Design calibration experiments to isolate input vs output energy.

    Key insight: We need to vary input and output tokens independently
    to separate their energy contributions through regression.

    Experiment Matrix:
    - Input-heavy samples: Large input, small output
    - Output-heavy samples: Small input, large output
    - Balanced samples: Similar input and output
    - Extreme corners: (min, min), (max, max)
    """

    # Design parameters
    TOKEN_RANGES = {
        "input": {
            "min": 50,
            "max": 2048,
            "levels": [50, 200, 500, 1000, 2048]
        },
        "output": {
            "min": 10,
            "max": 512,
            "levels": [10, 50, 100, 200, 512]
        }
    }

    REPLICATES_PER_CONDITION = 5  # For variance estimation

    @classmethod
    def generate_factorial_design(cls) -> List[Dict[str, int]]:
        """
        Generate full factorial design for calibration.

        This creates a grid of all input/output combinations
        to enable clean regression coefficient estimation.
        """
        design = []

        for input_level in cls.TOKEN_RANGES["input"]["levels"]:
            for output_level in cls.TOKEN_RANGES["output"]["levels"]:
                for rep in range(cls.REPLICATES_PER_CONDITION):
                    design.append({
                        "input_tokens": input_level,
                        "output_tokens": output_level,
                        "replicate": rep + 1,
                        "condition_type": cls._classify_condition(input_level, output_level)
                    })

        # Shuffle to avoid systematic time effects
        random.shuffle(design)
        return design

    @classmethod
    def generate_latin_hypercube_design(cls, n_samples: int = 50) -> List[Dict[str, int]]:
        """
        Generate Latin Hypercube design for efficient sampling.

        More efficient than factorial for continuous parameters.
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy not available, falling back to random sampling")
            return cls._generate_random_design(n_samples)

        from scipy.stats import qmc

        sampler = qmc.LatinHypercube(d=2)
        samples = sampler.random(n=n_samples)

        # Scale to token ranges
        input_range = cls.TOKEN_RANGES["input"]
        output_range = cls.TOKEN_RANGES["output"]

        design = []
        for i, (u_in, u_out) in enumerate(samples):
            input_tokens = int(input_range["min"] + u_in * (input_range["max"] - input_range["min"]))
            output_tokens = int(output_range["min"] + u_out * (output_range["max"] - output_range["min"]))

            design.append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "replicate": 1,
                "condition_type": cls._classify_condition(input_tokens, output_tokens)
            })

        return design

    @classmethod
    def generate_targeted_design(cls) -> List[Dict[str, int]]:
        """
        Generate targeted design focusing on key regions.

        This design emphasizes:
        1. Corner cases (min/max combinations)
        2. Input-only isolation (max input, min output)
        3. Output-only isolation (min input, max output)
        4. Balanced region for generalization
        """
        design = []

        # Corner cases (5 replicates each)
        corners = [
            (50, 10),    # Min-Min
            (50, 512),   # Min-Max (output isolation)
            (2048, 10),  # Max-Min (input isolation)
            (2048, 512), # Max-Max
        ]

        for input_t, output_t in corners:
            for rep in range(10):  # More replicates at corners
                design.append({
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "replicate": rep + 1,
                    "condition_type": "corner"
                })

        # Input isolation region (fixed min output)
        for input_t in [100, 200, 400, 800, 1200, 1600]:
            for rep in range(5):
                design.append({
                    "input_tokens": input_t,
                    "output_tokens": 20,
                    "replicate": rep + 1,
                    "condition_type": "input_isolation"
                })

        # Output isolation region (fixed min input)
        for output_t in [30, 80, 150, 250, 400]:
            for rep in range(5):
                design.append({
                    "input_tokens": 100,
                    "output_tokens": output_t,
                    "replicate": rep + 1,
                    "condition_type": "output_isolation"
                })

        # Balanced region
        balanced_pairs = [(200, 100), (500, 200), (800, 300), (1200, 400)]
        for input_t, output_t in balanced_pairs:
            for rep in range(5):
                design.append({
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "replicate": rep + 1,
                    "condition_type": "balanced"
                })

        random.shuffle(design)
        return design

    @classmethod
    def _classify_condition(cls, input_tokens: int, output_tokens: int) -> str:
        """Classify a condition as input-heavy, output-heavy, or balanced."""
        ratio = input_tokens / max(output_tokens, 1)

        if ratio > 5:
            return "input_heavy"
        elif ratio < 0.5:
            return "output_heavy"
        else:
            return "balanced"

    @classmethod
    def _generate_random_design(cls, n_samples: int) -> List[Dict[str, int]]:
        """Fallback random design."""
        design = []
        input_range = cls.TOKEN_RANGES["input"]
        output_range = cls.TOKEN_RANGES["output"]

        for _ in range(n_samples):
            design.append({
                "input_tokens": random.randint(input_range["min"], input_range["max"]),
                "output_tokens": random.randint(output_range["min"], output_range["max"]),
                "replicate": 1,
                "condition_type": "random"
            })

        return design


# =============================================================================
# Regression Analysis
# =============================================================================

class EnergyRegressionAnalyzer:
    """
    Fit and validate energy regression model.

    Model: E = alpha * T_in + beta * T_out + intercept

    Where:
    - E = measured energy (Joules)
    - T_in = input tokens
    - T_out = output tokens
    - alpha = energy per input token (J/token)
    - beta = energy per output token (J/token)
    - intercept = fixed overhead (J)
    """

    def __init__(self, samples: List[CalibrationSample]):
        """
        Initialize with calibration samples.

        Args:
            samples: List of CalibrationSample objects from measurements
        """
        self.samples = samples
        self._prepare_data()

    def _prepare_data(self) -> None:
        """Prepare numpy arrays for regression."""
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy required for regression analysis")

        n = len(self.samples)

        # Design matrix: [1, input_tokens, output_tokens] for each sample
        self.X = np.zeros((n, 3))
        self.y = np.zeros(n)

        for i, sample in enumerate(self.samples):
            self.X[i, 0] = 1  # Intercept
            self.X[i, 1] = sample.input_tokens
            self.X[i, 2] = sample.output_tokens
            self.y[i] = sample.measured_energy_joules

        # Store individual columns for convenience
        self.input_tokens = self.X[:, 1]
        self.output_tokens = self.X[:, 2]

    def fit_ols(self) -> RegressionResult:
        """
        Fit Ordinary Least Squares regression.

        Returns:
            RegressionResult with coefficients and statistics
        """
        n, p = self.X.shape

        # OLS solution: beta = (X'X)^-1 X'y
        XtX = self.X.T @ self.X
        Xty = self.X.T @ self.y

        try:
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            beta = np.linalg.lstsq(self.X, self.y, rcond=None)[0]

        intercept, alpha, beta_out = beta

        # Predictions and residuals
        y_pred = self.X @ beta
        residuals = self.y - y_pred

        # Calculate statistics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((self.y - np.mean(self.y)) ** 2)

        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)

        # Standard errors
        dof = n - p
        mse = ss_res / dof
        var_beta = mse * np.linalg.inv(XtX)

        se_intercept = np.sqrt(var_beta[0, 0])
        se_alpha = np.sqrt(var_beta[1, 1])
        se_beta = np.sqrt(var_beta[2, 2])

        # 95% confidence intervals
        t_crit = stats.t.ppf(0.975, dof)
        alpha_ci = (alpha - t_crit * se_alpha, alpha + t_crit * se_alpha)
        beta_ci = (beta_out - t_crit * se_beta, beta_out + t_crit * se_beta)
        intercept_ci = (intercept - t_crit * se_intercept, intercept + t_crit * se_intercept)

        # Residual diagnostics
        durbin_watson = self._durbin_watson_statistic(residuals)

        # Shapiro-Wilk test for normality (on subset if n > 5000)
        test_residuals = residuals[:5000] if len(residuals) > 5000 else residuals
        _, shapiro_p = stats.shapiro(test_residuals)

        return RegressionResult(
            alpha=alpha,
            beta=beta_out,
            intercept=intercept,
            alpha_stderr=se_alpha,
            beta_stderr=se_beta,
            intercept_stderr=se_intercept,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            n_samples=n,
            degrees_of_freedom=dof,
            alpha_ci=alpha_ci,
            beta_ci=beta_ci,
            intercept_ci=intercept_ci,
            residual_mean=np.mean(residuals),
            residual_std=np.std(residuals),
            durbin_watson=durbin_watson,
            shapiro_wilk_p=shapiro_p
        )

    def fit_weighted_ls(self, weights: Optional[np.ndarray] = None) -> RegressionResult:
        """
        Fit Weighted Least Squares regression.

        Useful when variance is heteroscedastic (varies with prediction).

        Args:
            weights: Optional weights. If None, uses inverse variance weighting.
        """
        if weights is None:
            # Estimate variance as function of predicted value
            # First fit unweighted to get predictions
            ols_result = self.fit_ols()
            y_pred = (ols_result.intercept +
                     ols_result.alpha * self.input_tokens +
                     ols_result.beta * self.output_tokens)

            # Residuals
            residuals = self.y - y_pred

            # Estimate variance as proportional to predicted value
            # (common pattern in energy measurements)
            weights = 1.0 / (np.abs(y_pred) + 1e-10)

        # Weighted least squares
        W = np.diag(weights)
        XtWX = self.X.T @ W @ self.X
        XtWy = self.X.T @ W @ self.y

        beta = np.linalg.solve(XtWX, XtWy)
        intercept, alpha, beta_out = beta

        # Weighted residuals
        y_pred = self.X @ beta
        residuals = self.y - y_pred
        weighted_residuals = np.sqrt(weights) * residuals

        n, p = self.X.shape
        ss_res = np.sum(weights * residuals ** 2)
        ss_tot = np.sum(weights * (self.y - np.average(self.y, weights=weights)) ** 2)

        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p)

        # Standard errors for WLS
        dof = n - p
        mse = ss_res / dof
        var_beta = mse * np.linalg.inv(XtWX)

        se_intercept = np.sqrt(var_beta[0, 0])
        se_alpha = np.sqrt(var_beta[1, 1])
        se_beta = np.sqrt(var_beta[2, 2])

        t_crit = stats.t.ppf(0.975, dof)

        return RegressionResult(
            alpha=alpha,
            beta=beta_out,
            intercept=intercept,
            alpha_stderr=se_alpha,
            beta_stderr=se_beta,
            intercept_stderr=se_intercept,
            r_squared=r_squared,
            adjusted_r_squared=adj_r_squared,
            n_samples=n,
            degrees_of_freedom=dof,
            alpha_ci=(alpha - t_crit * se_alpha, alpha + t_crit * se_alpha),
            beta_ci=(beta_out - t_crit * se_beta, beta_out + t_crit * se_beta),
            intercept_ci=(intercept - t_crit * se_intercept, intercept + t_crit * se_intercept),
            residual_mean=np.mean(residuals),
            residual_std=np.std(residuals),
            durbin_watson=self._durbin_watson_statistic(residuals),
            shapiro_wilk_p=0.0  # Skip for WLS
        )

    def fit_robust_regression(self) -> RegressionResult:
        """
        Fit robust regression using iteratively reweighted least squares.

        More resistant to outliers than OLS.
        """
        # Initial OLS fit
        result = self.fit_ols()

        for iteration in range(10):
            # Predict and get residuals
            y_pred = (result.intercept +
                     result.alpha * self.input_tokens +
                     result.beta * self.output_tokens)
            residuals = self.y - y_pred

            # Huber weights
            mad = np.median(np.abs(residuals - np.median(residuals)))
            scale = mad / 0.6745  # Robust scale estimate

            if scale < 1e-10:
                break

            u = residuals / (1.345 * scale)
            weights = np.where(np.abs(u) <= 1, 1.0, 1.0 / np.abs(u))

            # Refit with weights
            result = self.fit_weighted_ls(weights)

        return result

    def _durbin_watson_statistic(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff = np.diff(residuals)
        return np.sum(diff ** 2) / np.sum(residuals ** 2)

    def cross_validate(self, k_folds: int = 5) -> Tuple[float, float]:
        """
        Perform k-fold cross-validation.

        Returns:
            (mean_r_squared, std_r_squared)
        """
        n = len(self.samples)
        indices = np.arange(n)
        np.random.shuffle(indices)

        fold_size = n // k_folds
        r_squared_scores = []

        for fold in range(k_folds):
            # Split indices
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k_folds - 1 else n

            test_idx = indices[test_start:test_end]
            train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

            # Create train analyzer
            train_samples = [self.samples[i] for i in train_idx]
            train_analyzer = EnergyRegressionAnalyzer(train_samples)

            # Fit on train
            result = train_analyzer.fit_ols()

            # Evaluate on test
            test_X = self.X[test_idx]
            test_y = self.y[test_idx]

            y_pred = result.intercept + result.alpha * test_X[:, 1] + result.beta * test_X[:, 2]

            ss_res = np.sum((test_y - y_pred) ** 2)
            ss_tot = np.sum((test_y - np.mean(test_y)) ** 2)

            r_squared = 1 - (ss_res / ss_tot)
            r_squared_scores.append(r_squared)

        return np.mean(r_squared_scores), np.std(r_squared_scores)


# =============================================================================
# Validation Metrics
# =============================================================================

class ModelValidator:
    """
    Validate the calibrated energy model.
    """

    def __init__(
        self,
        model: CalibratedProxyModel,
        validation_samples: List[CalibrationSample]
    ):
        self.model = model
        self.validation_samples = validation_samples

    def compute_metrics(self) -> ValidationMetrics:
        """
        Compute comprehensive validation metrics.
        """
        predictions = []
        actuals = []

        for sample in self.validation_samples:
            pred = self.model.predict(sample.input_tokens, sample.output_tokens)
            predictions.append(pred)
            actuals.append(sample.measured_energy_joules)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        errors = actuals - predictions
        abs_errors = np.abs(errors)
        percent_errors = 100 * abs_errors / np.maximum(actuals, 1e-10)

        # Core metrics
        mape = np.mean(percent_errors)
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(abs_errors)
        max_error = np.max(abs_errors)
        max_error_idx = np.argmax(abs_errors)
        max_error_percent = percent_errors[max_error_idx]
        bias = np.mean(errors)  # Positive = under-prediction

        # Region-specific MAPE
        input_tokens = np.array([s.input_tokens for s in self.validation_samples])
        output_tokens = np.array([s.output_tokens for s in self.validation_samples])
        ratio = input_tokens / np.maximum(output_tokens, 1)

        input_heavy_mask = ratio > 5
        output_heavy_mask = ratio < 0.5
        balanced_mask = ~input_heavy_mask & ~output_heavy_mask

        mape_input_heavy = np.mean(percent_errors[input_heavy_mask]) if np.any(input_heavy_mask) else 0.0
        mape_output_heavy = np.mean(percent_errors[output_heavy_mask]) if np.any(output_heavy_mask) else 0.0
        mape_balanced = np.mean(percent_errors[balanced_mask]) if np.any(balanced_mask) else 0.0

        return ValidationMetrics(
            mape=mape,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            max_error_percent=max_error_percent,
            bias=bias,
            mape_input_heavy=mape_input_heavy,
            mape_output_heavy=mape_output_heavy,
            mape_balanced=mape_balanced
        )

    def residual_analysis(self) -> Dict[str, Any]:
        """
        Perform detailed residual analysis.
        """
        predictions = []
        actuals = []

        for sample in self.validation_samples:
            pred = self.model.predict(sample.input_tokens, sample.output_tokens)
            predictions.append(pred)
            actuals.append(sample.measured_energy_joules)

        predictions = np.array(predictions)
        actuals = np.array(actuals)
        residuals = actuals - predictions

        # Test for heteroscedasticity (Breusch-Pagan style)
        # Regress squared residuals on predictions
        sq_residuals = residuals ** 2
        correlation = np.corrcoef(predictions, sq_residuals)[0, 1]

        # Test for normality
        _, shapiro_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)

        # Test for autocorrelation
        durbin_watson = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)

        # Quantile-quantile data for plotting
        sorted_residuals = np.sort(residuals)
        n = len(residuals)
        theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

        return {
            "residual_stats": {
                "mean": float(np.mean(residuals)),
                "std": float(np.std(residuals)),
                "skewness": float(stats.skew(residuals)),
                "kurtosis": float(stats.kurtosis(residuals)),
                "min": float(np.min(residuals)),
                "max": float(np.max(residuals))
            },
            "heteroscedasticity": {
                "residual_prediction_correlation": float(correlation),
                "likely_heteroscedastic": abs(correlation) > 0.3
            },
            "normality": {
                "shapiro_wilk_p": float(shapiro_p),
                "likely_normal": shapiro_p > 0.05
            },
            "autocorrelation": {
                "durbin_watson": float(durbin_watson),
                "likely_autocorrelated": durbin_watson < 1.5 or durbin_watson > 2.5
            },
            "qq_data": {
                "theoretical_quantiles": theoretical_quantiles.tolist(),
                "sample_quantiles": sorted_residuals.tolist()
            }
        }


# =============================================================================
# Extrapolation Methodology
# =============================================================================

class ModelSizeExtrapolator:
    """
    Extrapolate calibrated coefficients to different model sizes.

    Uses neural scaling laws and empirical observations about
    how compute requirements scale with model parameters.

    Key References:
    - Kaplan et al. (2020) Scaling Laws for Neural Language Models
    - Hoffmann et al. (2022) Training Compute-Optimal LLMs (Chinchilla)
    """

    # Empirical scaling exponents from literature
    SCALING_LAWS = {
        "compute_per_token": 0.7,  # E ~ N^0.7 (flatter than linear)
        "memory_bandwidth": 0.5,  # Memory-bound operations scale slower
        "prefill_vs_decode_ratio": 0.5,  # Prefill uses ~half compute per token vs decode
    }

    # Model size estimates for API providers (approximate)
    PROVIDER_MODEL_SIZES = {
        # Anthropic
        "claude-3-5-opus": 400,
        "claude-3-5-sonnet": 175,
        "claude-3-5-haiku": 35,
        "claude-opus-4-5": 500,

        # OpenAI
        "gpt-4o": 200,
        "gpt-4o-mini": 8,
        "gpt-3.5-turbo": 20,
        "gpt-4-turbo": 220,

        # Google
        "gemini-1.5-pro": 175,
        "gemini-1.5-flash": 35,
        "gemini-2.0-flash": 40,

        # Meta
        "llama-3.1-8b": 8,
        "llama-3.1-70b": 70,
        "llama-3.1-405b": 405,

        # DeepSeek
        "deepseek-chat": 67,
        "deepseek-coder": 33,
    }

    def __init__(self, base_model: CalibratedProxyModel):
        """
        Initialize with a calibrated base model.

        Args:
            base_model: Calibrated model from GPU measurements
        """
        self.base_model = base_model

    def compute_scaling_factor(
        self,
        target_size_B: float,
        method: str = "chinchilla"
    ) -> ScalingFactors:
        """
        Compute scaling factors for extrapolation.

        Args:
            target_size_B: Target model size in billions of parameters
            method: Scaling law to use ("chinchilla", "kaplan", "linear")
        """
        base_size = self.base_model.model_size_B
        size_ratio = target_size_B / base_size

        if method == "chinchilla":
            # Chinchilla optimal scaling
            compute_exponent = 0.7
            memory_exponent = 0.5
        elif method == "kaplan":
            # Original GPT-3 scaling
            compute_exponent = 0.73
            memory_exponent = 0.6
        elif method == "linear":
            # Conservative linear scaling
            compute_exponent = 1.0
            memory_exponent = 1.0
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        compute_factor = size_ratio ** compute_exponent
        memory_factor = size_ratio ** memory_exponent

        # Combine factors (prefill is more memory-bound, decode is compute-bound)
        alpha_scaled = self.base_model.alpha * (0.3 * memory_factor + 0.7 * compute_factor)
        beta_scaled = self.base_model.beta * compute_factor

        # Uncertainty increases with extrapolation distance
        log_ratio = abs(math.log(size_ratio))
        uncertainty = 10 * log_ratio  # 10% uncertainty per e-fold

        return ScalingFactors(
            base_model_size_B=base_size,
            target_model_size_B=target_size_B,
            compute_scaling_factor=compute_factor,
            memory_bandwidth_factor=memory_factor,
            alpha_scaled=alpha_scaled,
            beta_scaled=beta_scaled,
            scaling_uncertainty_percent=min(uncertainty, 50)  # Cap at 50%
        )

    def extrapolate_to_model(self, model_name: str) -> CalibratedProxyModel:
        """
        Extrapolate to a specific named model.

        Args:
            model_name: Model identifier (e.g., "claude-3-5-sonnet")
        """
        if model_name not in self.PROVIDER_MODEL_SIZES:
            raise ValueError(f"Unknown model: {model_name}. Known models: {list(self.PROVIDER_MODEL_SIZES.keys())}")

        target_size = self.PROVIDER_MODEL_SIZES[model_name]
        return self.base_model.scale_to_model_size(target_size)

    def generate_provider_coefficients(self) -> Dict[str, Dict[str, float]]:
        """
        Generate energy coefficients for all known API providers.

        Returns a dictionary suitable for integration with the
        main energy cost model.
        """
        coefficients = {}

        for model_name, size in self.PROVIDER_MODEL_SIZES.items():
            scaled = self.base_model.scale_to_model_size(size)

            coefficients[model_name] = {
                "E_input": scaled.alpha,
                "E_output": scaled.beta,
                "model_size_B": size,
                "uncertainty_percent": scaled.mape,
                "source": f"Extrapolated from {self.base_model.model_name}"
            }

        return coefficients


# =============================================================================
# Confidence Interval Methods
# =============================================================================

class UncertaintyQuantifier:
    """
    Quantify uncertainty in energy proxy estimates.

    Methods:
    1. Parametric CIs from regression coefficients
    2. Bootstrap CIs for non-parametric bounds
    3. Prediction intervals for individual estimates
    """

    def __init__(self, analyzer: EnergyRegressionAnalyzer):
        self.analyzer = analyzer

    def bootstrap_coefficients(
        self,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for coefficients.

        More robust than parametric CIs when assumptions may be violated.
        """
        n_samples = len(self.analyzer.samples)

        alphas = []
        betas = []
        intercepts = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_samples = [self.analyzer.samples[i] for i in indices]

            # Fit regression
            boot_analyzer = EnergyRegressionAnalyzer(boot_samples)
            result = boot_analyzer.fit_ols()

            alphas.append(result.alpha)
            betas.append(result.beta)
            intercepts.append(result.intercept)

        # Compute percentile CIs
        alpha_level = (1 - confidence) / 2
        lower_pct = alpha_level * 100
        upper_pct = (1 - alpha_level) * 100

        return {
            "alpha": (np.percentile(alphas, lower_pct), np.percentile(alphas, upper_pct)),
            "beta": (np.percentile(betas, lower_pct), np.percentile(betas, upper_pct)),
            "intercept": (np.percentile(intercepts, lower_pct), np.percentile(intercepts, upper_pct)),
            "bootstrap_stats": {
                "alpha_mean": np.mean(alphas),
                "alpha_std": np.std(alphas),
                "beta_mean": np.mean(betas),
                "beta_std": np.std(betas),
                "n_bootstrap": n_bootstrap
            }
        }

    def prediction_interval(
        self,
        input_tokens: int,
        output_tokens: int,
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute prediction interval for a new observation.

        This is wider than a confidence interval for the mean
        because it accounts for both parameter uncertainty and
        individual observation variance.

        Returns:
            (point_estimate, lower_bound, upper_bound)
        """
        result = self.analyzer.fit_ols()

        # Point estimate
        point = result.intercept + result.alpha * input_tokens + result.beta * output_tokens

        # Variance of prediction
        x_new = np.array([1, input_tokens, output_tokens])

        # Variance components
        # 1. Parameter uncertainty
        XtX_inv = np.linalg.inv(self.analyzer.X.T @ self.analyzer.X)
        param_var = x_new @ XtX_inv @ x_new * (result.residual_std ** 2)

        # 2. Residual variance
        resid_var = result.residual_std ** 2

        # Total prediction variance
        pred_var = param_var + resid_var
        pred_std = np.sqrt(pred_var)

        # t-critical value
        t_crit = stats.t.ppf((1 + confidence) / 2, result.degrees_of_freedom)

        return (point, point - t_crit * pred_std, point + t_crit * pred_std)


# =============================================================================
# API Provider Adjustment
# =============================================================================

class APIProviderAdjustment:
    """
    Apply calibrated model to API provider energy estimates.

    API providers add overhead not present in direct GPU measurements:
    1. Network latency and data transfer energy
    2. Load balancer and routing energy
    3. Redundancy and reliability overhead
    4. Cooling and datacenter PUE (Power Usage Effectiveness)
    """

    # Provider-specific overhead factors (estimated)
    PROVIDER_OVERHEAD = {
        "anthropic": {
            "pue": 1.1,  # Datacenter efficiency
            "network_overhead": 1.02,  # Network transfer energy
            "safety_overhead": 1.05,  # Additional safety processing
            "total_multiplier": 1.17
        },
        "openai": {
            "pue": 1.12,
            "network_overhead": 1.02,
            "safety_overhead": 1.03,
            "total_multiplier": 1.18
        },
        "google": {
            "pue": 1.08,  # Google's datacenters are very efficient
            "network_overhead": 1.01,
            "safety_overhead": 1.02,
            "total_multiplier": 1.11
        },
        "deepseek": {
            "pue": 1.15,
            "network_overhead": 1.03,
            "safety_overhead": 1.01,
            "total_multiplier": 1.20
        },
        "meta": {  # For self-hosted Llama
            "pue": 1.10,  # Varies by hosting
            "network_overhead": 1.0,  # Direct inference
            "safety_overhead": 1.0,
            "total_multiplier": 1.10
        }
    }

    def __init__(self, calibrated_model: CalibratedProxyModel):
        self.calibrated_model = calibrated_model

    def estimate_api_energy(
        self,
        provider: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int
    ) -> Dict[str, float]:
        """
        Estimate energy for an API call including provider overhead.

        Args:
            provider: Provider name (anthropic, openai, google, deepseek)
            model_name: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dictionary with energy estimates and breakdowns
        """
        # Get model-size-adjusted coefficients
        extrapolator = ModelSizeExtrapolator(self.calibrated_model)

        try:
            scaled_model = extrapolator.extrapolate_to_model(model_name)
        except ValueError:
            # Unknown model - use base model with uncertainty
            scaled_model = self.calibrated_model
            logger.warning(f"Unknown model {model_name}, using base model coefficients")

        # Base energy from scaled model
        base_energy = scaled_model.predict(input_tokens, output_tokens)

        # Apply provider overhead
        provider_lower = provider.lower()
        if provider_lower in self.PROVIDER_OVERHEAD:
            overhead = self.PROVIDER_OVERHEAD[provider_lower]
        else:
            overhead = {"total_multiplier": 1.15}  # Default 15% overhead
            logger.warning(f"Unknown provider {provider}, using default overhead")

        api_energy = base_energy * overhead["total_multiplier"]

        # Get prediction interval
        quantifier = UncertaintyQuantifier(
            EnergyRegressionAnalyzer([])  # Dummy, we use scaled model
        )
        _, lower, upper = scaled_model.predict_with_uncertainty(input_tokens, output_tokens)

        return {
            "base_energy_joules": base_energy,
            "api_energy_joules": api_energy,
            "energy_lower_bound_joules": lower * overhead["total_multiplier"],
            "energy_upper_bound_joules": upper * overhead["total_multiplier"],
            "overhead_multiplier": overhead["total_multiplier"],
            "model_size_B": scaled_model.model_size_B,
            "coefficients": {
                "alpha": scaled_model.alpha,
                "beta": scaled_model.beta
            },
            "provider": provider,
            "model": model_name
        }


# =============================================================================
# Full Calibration Pipeline
# =============================================================================

class CalibrationPipeline:
    """
    Complete calibration pipeline from measurements to validated model.
    """

    def __init__(
        self,
        model_name: str = "Llama-3.1-8B",
        model_size_B: float = 8.0,
        hardware: str = "NVIDIA H100"
    ):
        self.model_name = model_name
        self.model_size_B = model_size_B
        self.hardware = hardware

        self.samples: List[CalibrationSample] = []
        self.regression_result: Optional[RegressionResult] = None
        self.calibrated_model: Optional[CalibratedProxyModel] = None
        self.validation_metrics: Optional[ValidationMetrics] = None

    def add_sample(self, sample: CalibrationSample) -> None:
        """Add a calibration sample."""
        self.samples.append(sample)

    def add_samples_from_measurements(
        self,
        measurements: List[Dict[str, Any]]
    ) -> None:
        """
        Add samples from raw measurement dictionaries.

        Expected keys:
        - input_tokens: int
        - output_tokens: int
        - energy_joules: float
        - duration_seconds: float
        - avg_power_watts: float
        - peak_power_watts: float (optional)
        - idle_power_watts: float (optional)
        """
        for m in measurements:
            sample = CalibrationSample(
                sample_id=m.get("sample_id", str(uuid.uuid4())[:8]),
                input_tokens=m["input_tokens"],
                output_tokens=m["output_tokens"],
                measured_energy_joules=m["energy_joules"],
                duration_seconds=m["duration_seconds"],
                avg_power_watts=m["avg_power_watts"],
                peak_power_watts=m.get("peak_power_watts", m["avg_power_watts"]),
                idle_power_watts=m.get("idle_power_watts", 50.0),
                active_energy_joules=m.get("active_energy_joules", m["energy_joules"]),
                gpu_utilization_avg=m.get("gpu_utilization_avg", 80.0),
                memory_utilization_avg=m.get("memory_utilization_avg", 60.0),
                temperature_c_avg=m.get("temperature_c_avg", 65.0),
                timestamp=m.get("timestamp", datetime.now(timezone.utc).isoformat()),
                metadata=m.get("metadata", {})
            )
            self.samples.append(sample)

    def run_calibration(
        self,
        method: str = "ols",
        train_fraction: float = 0.8
    ) -> CalibratedProxyModel:
        """
        Run the full calibration pipeline.

        Args:
            method: Regression method ("ols", "wls", "robust")
            train_fraction: Fraction of data for training (rest for validation)
        """
        if len(self.samples) < 20:
            raise ValueError(f"Need at least 20 samples for calibration, got {len(self.samples)}")

        logger.info(f"Running calibration with {len(self.samples)} samples")

        # Split data
        n_train = int(len(self.samples) * train_fraction)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_samples = [self.samples[i] for i in train_indices]
        val_samples = [self.samples[i] for i in val_indices]

        logger.info(f"Training on {len(train_samples)} samples, validating on {len(val_samples)}")

        # Fit regression
        analyzer = EnergyRegressionAnalyzer(train_samples)

        if method == "ols":
            self.regression_result = analyzer.fit_ols()
        elif method == "wls":
            self.regression_result = analyzer.fit_weighted_ls()
        elif method == "robust":
            self.regression_result = analyzer.fit_robust_regression()
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"Regression R^2: {self.regression_result.r_squared:.4f}")
        logger.info(f"Alpha (input): {self.regression_result.alpha:.6e} J/token")
        logger.info(f"Beta (output): {self.regression_result.beta:.6e} J/token")

        # Cross-validation
        cv_mean, cv_std = analyzer.cross_validate(k_folds=5)
        logger.info(f"Cross-validation R^2: {cv_mean:.4f} (+/- {cv_std:.4f})")

        # Create calibrated model
        input_tokens_list = [s.input_tokens for s in self.samples]
        output_tokens_list = [s.output_tokens for s in self.samples]

        self.calibrated_model = CalibratedProxyModel(
            model_name=self.model_name,
            calibration_date=datetime.now(timezone.utc).isoformat(),
            alpha=self.regression_result.alpha,
            beta=self.regression_result.beta,
            intercept=self.regression_result.intercept,
            alpha_ci_95=self.regression_result.alpha_ci,
            beta_ci_95=self.regression_result.beta_ci,
            r_squared=self.regression_result.r_squared,
            mape=0.0,  # Will be updated after validation
            model_size_B=self.model_size_B,
            scaling_exponent=0.7,  # Chinchilla scaling
            min_input_tokens=min(input_tokens_list),
            max_input_tokens=max(input_tokens_list),
            min_output_tokens=min(output_tokens_list),
            max_output_tokens=max(output_tokens_list),
            n_calibration_samples=len(train_samples),
            hardware=self.hardware
        )

        # Validate
        if val_samples:
            validator = ModelValidator(self.calibrated_model, val_samples)
            self.validation_metrics = validator.compute_metrics()
            self.calibrated_model.mape = self.validation_metrics.mape

            logger.info(f"Validation MAPE: {self.validation_metrics.mape:.2f}%")
            logger.info(f"Validation RMSE: {self.validation_metrics.rmse:.4f} J")

        return self.calibrated_model

    def compare_with_original_proxy(self) -> Dict[str, Any]:
        """
        Compare calibrated model with original proxy constants.
        """
        if self.calibrated_model is None:
            raise ValueError("Run calibration first")

        # Get original small model constants (for 7B)
        original = PROXY_CONSTANTS_ORIGINAL["small"]

        comparison = {
            "original_proxy": {
                "E_input": original["E_input"],
                "E_output": original["E_output"],
                "model_size_B": original["model_size_B"]
            },
            "calibrated": {
                "E_input": self.calibrated_model.alpha,
                "E_output": self.calibrated_model.beta,
                "model_size_B": self.calibrated_model.model_size_B
            },
            "differences": {
                "E_input_ratio": self.calibrated_model.alpha / original["E_input"],
                "E_output_ratio": self.calibrated_model.beta / original["E_output"],
                "E_input_percent_diff": 100 * (self.calibrated_model.alpha - original["E_input"]) / original["E_input"],
                "E_output_percent_diff": 100 * (self.calibrated_model.beta - original["E_output"]) / original["E_output"]
            },
            "calibrated_model_quality": {
                "r_squared": self.calibrated_model.r_squared,
                "mape": self.calibrated_model.mape,
                "n_samples": self.calibrated_model.n_calibration_samples
            }
        }

        return comparison

    def export_calibration_report(self, output_path: Path) -> None:
        """
        Export comprehensive calibration report.
        """
        if self.calibrated_model is None:
            raise ValueError("Run calibration first")

        report = {
            "metadata": {
                "calibration_date": self.calibrated_model.calibration_date,
                "model_name": self.model_name,
                "model_size_B": self.model_size_B,
                "hardware": self.hardware,
                "n_samples": len(self.samples)
            },
            "calibrated_model": self.calibrated_model.to_dict(),
            "regression_diagnostics": self.regression_result.to_dict() if self.regression_result else None,
            "validation_metrics": self.validation_metrics.to_dict() if self.validation_metrics else None,
            "comparison_with_original": self.compare_with_original_proxy(),
            "provider_coefficients": ModelSizeExtrapolator(self.calibrated_model).generate_provider_coefficients()
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Calibration report saved to {output_path}")

    def export_samples_csv(self, output_path: Path) -> None:
        """Export calibration samples to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="") as f:
            if self.samples:
                writer = csv.DictWriter(f, fieldnames=self.samples[0].to_dict().keys())
                writer.writeheader()
                for sample in self.samples:
                    row = sample.to_dict()
                    row["metadata"] = json.dumps(row["metadata"])
                    writer.writerow(row)

        logger.info(f"Samples exported to {output_path}")


# =============================================================================
# Integration with PowerMonitor
# =============================================================================

def run_calibration_experiment(
    power_monitor,  # PowerMonitor instance
    inference_function: Callable[[str], str],  # Function that runs inference
    tokenizer: Callable[[str], int],  # Function to count tokens
    design: Optional[List[Dict[str, int]]] = None
) -> List[CalibrationSample]:
    """
    Run calibration experiment using PowerMonitor.

    Args:
        power_monitor: Initialized and started PowerMonitor instance
        inference_function: Function that takes prompt and returns response
        tokenizer: Function that counts tokens in a string
        design: Optional experiment design (uses targeted if None)

    Returns:
        List of CalibrationSample objects
    """
    if design is None:
        design = CalibrationExperimentDesign.generate_targeted_design()

    samples = []

    for i, condition in enumerate(design):
        # Generate prompt of target length
        target_input_tokens = condition["input_tokens"]
        target_output_tokens = condition["output_tokens"]

        # Create prompt (actual implementation would use real prompts)
        prompt = f"Generate a response of approximately {target_output_tokens} tokens. " * (target_input_tokens // 20)

        actual_input_tokens = tokenizer(prompt)

        # Run inference with power monitoring
        request_id = power_monitor.begin_request(
            metadata={
                "target_input": target_input_tokens,
                "target_output": target_output_tokens,
                "condition_type": condition["condition_type"]
            }
        )

        try:
            response = inference_function(prompt)
            actual_output_tokens = tokenizer(response)
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            power_monitor.end_request(request_id)
            continue

        attribution = power_monitor.end_request(request_id)

        sample = CalibrationSample(
            sample_id=request_id,
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            measured_energy_joules=attribution.total_energy_joules,
            duration_seconds=attribution.duration_seconds,
            avg_power_watts=attribution.avg_power_watts,
            peak_power_watts=attribution.peak_power_watts,
            idle_power_watts=attribution.idle_baseline_watts,
            active_energy_joules=attribution.active_energy_joules,
            gpu_utilization_avg=attribution.avg_gpu_utilization,
            memory_utilization_avg=attribution.avg_memory_utilization,
            temperature_c_avg=0.0,  # Would come from power monitor if tracked
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "condition_type": condition["condition_type"],
                "replicate": condition["replicate"]
            }
        )

        samples.append(sample)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{len(design)} samples")

    return samples


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_calibration_pipeline():
    """Demonstrate the calibration pipeline with synthetic data."""

    print("=" * 70)
    print("Green AI Calibration Protocol Demo")
    print("=" * 70)

    # Generate synthetic calibration data
    # In real use, this would come from actual GPU measurements
    print("\n1. Generating synthetic calibration data...")

    design = CalibrationExperimentDesign.generate_targeted_design()
    print(f"   Generated {len(design)} experiment conditions")

    # Simulate measurements with known ground truth
    TRUE_ALPHA = 0.000020  # True J/input_token (for 8B model)
    TRUE_BETA = 0.000045   # True J/output_token
    TRUE_INTERCEPT = 0.5   # Fixed overhead
    NOISE_STD = 0.1        # Measurement noise (relative)

    synthetic_measurements = []
    for condition in design:
        input_t = condition["input_tokens"]
        output_t = condition["output_tokens"]

        # True energy
        true_energy = TRUE_INTERCEPT + TRUE_ALPHA * input_t + TRUE_BETA * output_t

        # Add measurement noise
        noise = np.random.normal(0, NOISE_STD * true_energy)
        measured_energy = max(0.01, true_energy + noise)

        # Simulate duration and power
        duration = measured_energy / 150  # Approximate 150W average

        synthetic_measurements.append({
            "input_tokens": input_t,
            "output_tokens": output_t,
            "energy_joules": measured_energy,
            "duration_seconds": duration,
            "avg_power_watts": measured_energy / duration,
            "condition_type": condition["condition_type"]
        })

    # Run calibration pipeline
    print("\n2. Running calibration pipeline...")

    pipeline = CalibrationPipeline(
        model_name="Llama-3.1-8B",
        model_size_B=8.0,
        hardware="NVIDIA H100 (Simulated)"
    )

    pipeline.add_samples_from_measurements(synthetic_measurements)
    calibrated_model = pipeline.run_calibration(method="ols")

    print("\n3. Calibration Results:")
    print(f"   Alpha (input):  {calibrated_model.alpha:.6e} J/token")
    print(f"   Beta (output):  {calibrated_model.beta:.6e} J/token")
    print(f"   Intercept:      {calibrated_model.intercept:.4f} J")
    print(f"   R-squared:      {calibrated_model.r_squared:.4f}")
    print(f"   MAPE:           {calibrated_model.mape:.2f}%")

    print("\n4. Ground Truth Comparison:")
    print(f"   True Alpha:     {TRUE_ALPHA:.6e} J/token")
    print(f"   Fitted Alpha:   {calibrated_model.alpha:.6e} J/token")
    print(f"   Alpha Error:    {100 * (calibrated_model.alpha - TRUE_ALPHA) / TRUE_ALPHA:.2f}%")
    print(f"   True Beta:      {TRUE_BETA:.6e} J/token")
    print(f"   Fitted Beta:    {calibrated_model.beta:.6e} J/token")
    print(f"   Beta Error:     {100 * (calibrated_model.beta - TRUE_BETA) / TRUE_BETA:.2f}%")

    print("\n5. Comparison with Original Proxy Model:")
    comparison = pipeline.compare_with_original_proxy()
    print(f"   Original E_input:     {comparison['original_proxy']['E_input']:.6e}")
    print(f"   Calibrated E_input:   {comparison['calibrated']['E_input']:.6e}")
    print(f"   Difference:           {comparison['differences']['E_input_percent_diff']:.1f}%")
    print(f"   Original E_output:    {comparison['original_proxy']['E_output']:.6e}")
    print(f"   Calibrated E_output:  {comparison['calibrated']['E_output']:.6e}")
    print(f"   Difference:           {comparison['differences']['E_output_percent_diff']:.1f}%")

    print("\n6. Extrapolation to API Providers:")
    extrapolator = ModelSizeExtrapolator(calibrated_model)

    models_to_check = ["claude-3-5-sonnet", "gpt-4o", "llama-3.1-70b"]
    for model_name in models_to_check:
        try:
            scaled = extrapolator.extrapolate_to_model(model_name)
            print(f"\n   {model_name} ({scaled.model_size_B}B):")
            print(f"      E_input:  {scaled.alpha:.6e} J/token")
            print(f"      E_output: {scaled.beta:.6e} J/token")
        except ValueError as e:
            print(f"   {model_name}: {e}")

    print("\n7. Energy Estimate Example:")
    api_adjuster = APIProviderAdjustment(calibrated_model)

    estimate = api_adjuster.estimate_api_energy(
        provider="anthropic",
        model_name="claude-3-5-sonnet",
        input_tokens=1000,
        output_tokens=200
    )

    print(f"   API Call: 1000 input tokens, 200 output tokens")
    print(f"   Provider: Anthropic (claude-3-5-sonnet)")
    print(f"   Base energy: {estimate['base_energy_joules']:.4f} J")
    print(f"   API energy:  {estimate['api_energy_joules']:.4f} J")
    print(f"   Uncertainty: [{estimate['energy_lower_bound_joules']:.4f}, {estimate['energy_upper_bound_joules']:.4f}] J")

    # Export report
    output_dir = Path("/tmp/calibration_demo")
    pipeline.export_calibration_report(output_dir / "calibration_report.json")
    pipeline.export_samples_csv(output_dir / "calibration_samples.csv")

    print(f"\n8. Reports exported to: {output_dir}")
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_calibration_pipeline()

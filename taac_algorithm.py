#!/usr/bin/env python3
"""
TAAC: Task-Aware Adaptive Compression Algorithm
================================================

A quality-gated compression algorithm that dynamically adjusts compression
based on task type, information density, and predicted quality degradation.

KEY DIFFERENTIATION FROM PRIOR WORK:
------------------------------------
1. ATACompressor (Huang et al. 2024):
   - ATACompressor: Learns task-awareness end-to-end via adaptive controller
   - TAAC: Exploits empirically-discovered thresholds (r>=0.6 cliff for code)
   - ATACompressor: Combines hard/soft prompt paradigms
   - TAAC: Uses quality-gating to stop when predicted quality drops below floor

2. TACO-RL (Shi et al. 2024):
   - TACO-RL: Uses REINFORCE with task-specific rewards (BLEU, F1)
   - TAAC: Uses direct quality prediction from learned embeddings
   - TACO-RL: Optimizes compression policy via RL
   - TAAC: Iterative compression with quality floor enforcement

3. TAAC Unique Contributions:
   - Quality-gating: Stops compression when predicted quality < Q_min
   - Information density adjustment via perplexity CV
   - Empirically-derived task-type thresholds from Johnson (2026)
   - Mechanistic explanation through perplexity paradox

Author: Dr. Amanda Foster, Bona Opera Studios
Date: January 2026
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Task types with empirically-derived compression thresholds."""
    CODE = "code"
    COT = "cot"  # Chain-of-thought reasoning
    HYBRID = "hybrid"


@dataclass
class TAACConfig:
    """Configuration for TAAC algorithm.

    Thresholds derived from Johnson (2026) empirical study:
    - Code: Threshold behavior at r >= 0.6 (quality preserved above cliff)
    - CoT: Gradual degradation (conservative compression needed)
    - Hybrid: Intermediate behavior
    """
    # Task-specific compression thresholds (from Johnson 2026)
    r_code: float = 0.65     # Conservative, above the 0.6 cliff
    r_cot: float = 0.80      # Minimal compression for reasoning tasks
    r_hybrid: float = 0.72   # Intermediate for mixed tasks

    # Quality gating parameters
    quality_floor: float = 0.90  # Q_min: Stop compression if quality drops below
    compression_step: float = 0.05  # delta: Compression step size

    # Density adjustment
    density_lambda: float = 0.15  # lambda: Weight for density adjustment

    # Task classifier settings
    classifier_model: str = "distilbert-base-uncased"
    classifier_max_latency_ms: float = 10.0

    # Quality predictor settings
    predictor_embedding_dim: int = 768
    predictor_hidden_dim: int = 256

    # Compression engine settings
    compressor_type: str = "llmlingua2"  # Default compression method


@dataclass
class CompressionResult:
    """Result from TAAC compression."""
    compressed_prompt: str
    original_prompt: str
    compression_ratio: float
    task_type: TaskType
    information_density: float
    predicted_quality: float
    quality_gated: bool  # True if stopped due to quality floor
    compression_time_ms: float
    classification_time_ms: float
    density_time_ms: float
    quality_prediction_time_ms: float

    # Detailed token information
    original_tokens: int = 0
    compressed_tokens: int = 0
    perplexity_stats: dict = field(default_factory=dict)


class TaskClassifier:
    """Fast task classifier using DistilBERT.

    Classifies prompts into task types:
    - code: Programming/code generation tasks
    - cot: Chain-of-thought reasoning (math, logic, science)
    - hybrid: Mixed or unclear task types

    Designed for <10ms latency to minimize overhead.
    """

    def __init__(self, config: TAACConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
        self._classifier = None
        self._loaded = False

        # Code indicators for heuristic fallback
        self._code_keywords = {
            'def ', 'class ', 'import ', 'return ', 'function', 'async ',
            'const ', 'let ', 'var ', 'public ', 'private ', 'void ',
            '```python', '```javascript', '```java', '```cpp', '```rust',
        }
        self._cot_keywords = {
            'calculate', 'solve', 'prove', 'derive', 'step by step',
            'reasoning', 'therefore', 'thus', 'hence', 'equation',
            'probability', 'geometric', 'algebraic', 'mathematical',
        }

    def load(self):
        """Lazy load the classifier model."""
        if self._loaded:
            return

        try:
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
            import torch

            self._tokenizer = DistilBertTokenizer.from_pretrained(
                self.config.classifier_model
            )

            # Check if fine-tuned task classifier exists
            try:
                self._classifier = DistilBertForSequenceClassification.from_pretrained(
                    "taac-task-classifier"  # Would be our trained model
                )
                self._use_trained = True
            except Exception:
                # Fall back to base model + heuristics
                self._classifier = None
                self._use_trained = False
                logger.info("Using heuristic task classifier (no trained model found)")

            self._loaded = True

        except ImportError:
            logger.warning("Transformers not available, using heuristic classifier")
            self._loaded = True
            self._use_trained = False

    def classify(self, prompt: str) -> tuple[TaskType, float]:
        """Classify the prompt's task type.

        Returns:
            tuple of (TaskType, confidence score 0-1)
        """
        self.load()
        start_time = time.perf_counter()

        if self._use_trained and self._classifier is not None:
            result = self._classify_neural(prompt)
        else:
            result = self._classify_heuristic(prompt)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        if elapsed_ms > self.config.classifier_max_latency_ms:
            logger.warning(
                f"Task classification took {elapsed_ms:.2f}ms "
                f"(target: {self.config.classifier_max_latency_ms}ms)"
            )

        return result

    def _classify_neural(self, prompt: str) -> tuple[TaskType, float]:
        """Neural classification using fine-tuned DistilBERT."""
        import torch

        # Truncate for speed (first 512 tokens)
        inputs = self._tokenizer(
            prompt,
            truncation=True,
            max_length=512,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._classifier(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[0]

        # Assuming labels: [code, cot, hybrid]
        idx = probs.argmax().item()
        confidence = probs[idx].item()

        task_map = {0: TaskType.CODE, 1: TaskType.COT, 2: TaskType.HYBRID}
        return task_map[idx], confidence

    def _classify_heuristic(self, prompt: str) -> tuple[TaskType, float]:
        """Heuristic classification based on keyword patterns."""
        prompt_lower = prompt.lower()

        # Count keyword matches
        code_score = sum(1 for kw in self._code_keywords if kw.lower() in prompt_lower)
        cot_score = sum(1 for kw in self._cot_keywords if kw.lower() in prompt_lower)

        # Check for code blocks
        if '```' in prompt or prompt.strip().startswith('def '):
            code_score += 5

        # Check for mathematical notation
        if any(c in prompt for c in ['=', '+', '-', '*', '/', '\\frac', '\\sqrt']):
            cot_score += 1

        total = code_score + cot_score + 1  # +1 to avoid division by zero

        if code_score > cot_score:
            confidence = min(0.95, 0.5 + code_score / total * 0.5)
            return TaskType.CODE, confidence
        elif cot_score > code_score:
            confidence = min(0.95, 0.5 + cot_score / total * 0.5)
            return TaskType.COT, confidence
        else:
            return TaskType.HYBRID, 0.5


class InformationDensityEstimator:
    """Estimates information density using perplexity coefficient of variation.

    High CV (heterogeneous perplexity) = some tokens much more important
    Low CV (uniform perplexity) = importance distributed evenly

    The CV-based approach allows more aggressive compression when information
    is localized (can safely remove low-information tokens).
    """

    def __init__(self, config: TAACConfig, pilot_model: str = "gpt2"):
        self.config = config
        self.pilot_model = pilot_model
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load(self):
        """Lazy load the pilot model for perplexity calculation."""
        if self._loaded:
            return

        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            import torch

            self._tokenizer = GPT2Tokenizer.from_pretrained(self.pilot_model)
            self._model = GPT2LMHeadModel.from_pretrained(self.pilot_model)
            self._model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()

            self._loaded = True
            logger.info(f"Loaded pilot model: {self.pilot_model}")

        except ImportError:
            logger.warning("Transformers not available for density estimation")
            self._loaded = True

    def estimate(self, prompt: str) -> tuple[float, dict]:
        """Estimate information density using perplexity CV.

        Returns:
            tuple of (density_score, perplexity_stats)

        density_score is the coefficient of variation of per-token perplexity:
            CV = std(PPL) / mean(PPL)

        Higher CV means more heterogeneous information distribution.
        """
        self.load()

        if self._model is None:
            # Fallback: estimate based on text statistics
            return self._estimate_heuristic(prompt)

        return self._estimate_neural(prompt)

    def _estimate_neural(self, prompt: str) -> tuple[float, dict]:
        """Neural estimation using per-token perplexity."""
        import torch

        # Tokenize
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits

        # Compute per-token perplexity
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        # Cross-entropy per token
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Convert to perplexity
        token_ppl = torch.exp(token_losses).cpu().numpy()

        # Compute coefficient of variation
        mean_ppl = np.mean(token_ppl)
        std_ppl = np.std(token_ppl)
        cv = std_ppl / mean_ppl if mean_ppl > 0 else 0.0

        # Normalize CV to [0, 1] range (empirically, CV typically 0.5-3.0)
        density_score = min(1.0, cv / 2.0)

        stats = {
            "mean_perplexity": float(mean_ppl),
            "std_perplexity": float(std_ppl),
            "cv": float(cv),
            "min_perplexity": float(np.min(token_ppl)),
            "max_perplexity": float(np.max(token_ppl)),
            "num_tokens": len(token_ppl),
            "token_perplexities": token_ppl.tolist()[:100],  # First 100 for analysis
        }

        return density_score, stats

    def _estimate_heuristic(self, prompt: str) -> tuple[float, dict]:
        """Heuristic estimation based on text statistics."""
        # Use vocabulary diversity as proxy for information density
        words = prompt.lower().split()
        unique_words = set(words)

        if len(words) == 0:
            return 0.5, {"method": "heuristic", "vocabulary_diversity": 0}

        vocab_diversity = len(unique_words) / len(words)

        # Use sentence length variance as additional signal
        sentences = prompt.replace('!', '.').replace('?', '.').split('.')
        sent_lengths = [len(s.split()) for s in sentences if s.strip()]

        if len(sent_lengths) > 1:
            sent_cv = np.std(sent_lengths) / np.mean(sent_lengths)
        else:
            sent_cv = 0.0

        # Combine signals
        density_score = (vocab_diversity * 0.6 + min(1.0, sent_cv) * 0.4)

        stats = {
            "method": "heuristic",
            "vocabulary_diversity": vocab_diversity,
            "sentence_cv": float(sent_cv),
            "num_words": len(words),
            "num_unique_words": len(unique_words),
        }

        return density_score, stats


class QualityPredictor:
    """Predicts compression quality using frozen embeddings + MLP.

    Architecture: sentence_embedding + task_onehot -> MLP(2 layers) -> quality_score

    This is our KEY DIFFERENTIATOR from ATACompressor and TACO-RL:
    - We directly predict quality degradation to enable quality-gating
    - Stops compression when predicted quality < Q_min
    - Uses empirical data from Johnson (2026) experiments for training
    """

    def __init__(self, config: TAACConfig):
        self.config = config
        self._model = None
        self._embedder = None
        self._loaded = False

    def load(self):
        """Load the quality predictor model."""
        if self._loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import torch.nn as nn

            # Frozen sentence embeddings
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')

            # 2-layer MLP for quality prediction
            # Input: embedding_dim + 3 (one-hot task type)
            input_dim = 384 + 3  # MiniLM-L6 has 384 dims

            self._model = nn.Sequential(
                nn.Linear(input_dim, self.config.predictor_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(self.config.predictor_hidden_dim, self.config.predictor_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.config.predictor_hidden_dim // 2, 1),
                nn.Sigmoid(),  # Output in [0, 1]
            )

            # Try to load pretrained weights
            try:
                state_dict = torch.load("taac_quality_predictor.pt")
                self._model.load_state_dict(state_dict)
                logger.info("Loaded pretrained quality predictor")
            except FileNotFoundError:
                logger.info("No pretrained quality predictor found, using untrained model")
                # Initialize with reasonable defaults for quality prediction
                self._init_reasonable_weights()

            self._model.eval()
            self._loaded = True

        except ImportError:
            logger.warning("sentence-transformers not available, using heuristic quality prediction")
            self._loaded = True

    def _init_reasonable_weights(self):
        """Initialize weights to predict high quality by default."""
        import torch.nn as nn

        for layer in self._model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                # Bias the last layer to predict ~0.9 quality
                if layer.out_features == 1:
                    layer.bias.data.fill_(2.0)  # sigmoid(2) ~ 0.88
                else:
                    layer.bias.data.fill_(0.1)

    def predict(self, compressed_prompt: str, task_type: TaskType,
                compression_ratio: float) -> float:
        """Predict quality score for compressed prompt.

        Args:
            compressed_prompt: The compressed text
            task_type: Type of task (code, cot, hybrid)
            compression_ratio: Actual compression ratio achieved

        Returns:
            Predicted quality score in [0, 1]
        """
        self.load()

        if self._model is None or self._embedder is None:
            return self._predict_heuristic(task_type, compression_ratio)

        return self._predict_neural(compressed_prompt, task_type, compression_ratio)

    def _predict_neural(self, compressed_prompt: str, task_type: TaskType,
                        compression_ratio: float) -> float:
        """Neural quality prediction."""
        import torch

        # Get sentence embedding
        embedding = self._embedder.encode(
            compressed_prompt,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Create one-hot task encoding
        task_onehot = torch.zeros(3)
        if task_type == TaskType.CODE:
            task_onehot[0] = 1
        elif task_type == TaskType.COT:
            task_onehot[1] = 1
        else:
            task_onehot[2] = 1

        # Concatenate features
        features = torch.cat([embedding, task_onehot]).unsqueeze(0)

        with torch.no_grad():
            quality = self._model(features).item()

        return quality

    def _predict_heuristic(self, task_type: TaskType, compression_ratio: float) -> float:
        """Heuristic quality prediction based on task-type curves.

        Uses the empirical findings from Johnson (2026):
        - Code: Threshold at r=0.6, quality preserved above, sharp drop below
        - CoT: Gradual linear degradation with compression
        """
        if task_type == TaskType.CODE:
            # Threshold behavior: quality drops sharply below 0.6
            if compression_ratio >= 0.6:
                return 0.95 - 0.05 * (1 - compression_ratio) / 0.4
            else:
                # Sharp degradation below cliff
                return 0.95 * (compression_ratio / 0.6) ** 2

        elif task_type == TaskType.COT:
            # Gradual degradation: approximately linear
            return 0.95 - 0.4 * (1 - compression_ratio)

        else:  # HYBRID
            # Interpolation between code and CoT
            code_q = self._predict_heuristic(TaskType.CODE, compression_ratio)
            cot_q = self._predict_heuristic(TaskType.COT, compression_ratio)
            return 0.5 * code_q + 0.5 * cot_q


class CompressionEngine:
    """Wrapper for underlying compression method (LLMLingua-2, etc.)."""

    def __init__(self, method: str = "llmlingua2"):
        self.method = method
        self._compressor = None
        self._loaded = False

    def load(self):
        """Lazy load the compression model."""
        if self._loaded:
            return

        try:
            if self.method == "llmlingua2":
                from llmlingua import PromptCompressor
                self._compressor = PromptCompressor(
                    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    use_llmlingua2=True,
                )
            elif self.method == "llmlingua1":
                from llmlingua import PromptCompressor
                self._compressor = PromptCompressor(
                    model_name="NousResearch/Llama-2-7b-hf",
                    use_llmlingua2=False,
                )
            else:
                logger.warning(f"Unknown compression method: {self.method}")

            self._loaded = True

        except ImportError:
            logger.warning(f"Compression library not available for {self.method}")
            self._loaded = True

    def compress(self, prompt: str, ratio: float,
                 force_tokens: Optional[list[str]] = None) -> dict:
        """Compress prompt to target ratio."""
        self.load()

        if ratio >= 1.0:
            return {
                "compressed_prompt": prompt,
                "actual_ratio": 1.0,
                "original_tokens": len(prompt.split()),
                "compressed_tokens": len(prompt.split()),
            }

        if self._compressor is None:
            # Fallback: simple truncation
            words = prompt.split()
            n_keep = max(1, int(len(words) * ratio))
            compressed = " ".join(words[:n_keep])
            return {
                "compressed_prompt": compressed,
                "actual_ratio": n_keep / len(words),
                "original_tokens": len(words),
                "compressed_tokens": n_keep,
            }

        # Default force tokens for code
        if force_tokens is None:
            force_tokens = ["\n", "def", "return", "if", "for", "while", "class",
                          "import", "from", "function", "const", "let", "var"]

        result = self._compressor.compress_prompt(
            prompt,
            rate=ratio,
            force_tokens=force_tokens,
        )

        return {
            "compressed_prompt": result["compressed_prompt"],
            "actual_ratio": result.get("ratio", ratio),
            "original_tokens": result.get("origin_tokens", len(prompt.split())),
            "compressed_tokens": result.get("compressed_tokens",
                                           len(result["compressed_prompt"].split())),
        }


class TAAC:
    """Task-Aware Adaptive Compression Algorithm.

    TAAC operates in three stages:
    1. Task Classification: Identify task type (code/cot/hybrid)
    2. Information Density Estimation: Compute perplexity CV
    3. Quality-Gated Compression: Iteratively compress with quality monitoring

    The key innovation is QUALITY-GATING: rather than targeting a fixed ratio,
    we stop compression when predicted quality drops below a user-specified floor.
    This prevents over-compression that would degrade task performance.

    Algorithm (from paper):
    ```
    1. tau <- TaskClassifier(x)
    2. rho <- DensityEstimator(x)
    3. r_target <- r_tau* + lambda * (1 - rho)  # Adjust for density
    4. r_current <- 1.0
    5. while r_current > r_target:
    6.     x' <- Compress(x, r_current - delta)
    7.     Q_hat <- QualityPredictor(x', tau)
    8.     if Q_hat < Q_min: break  # Quality floor reached
    9.     r_current <- r_current - delta
    10. return x', r_current
    ```
    """

    def __init__(self, config: Optional[TAACConfig] = None):
        self.config = config or TAACConfig()

        # Initialize components
        self.task_classifier = TaskClassifier(self.config)
        self.density_estimator = InformationDensityEstimator(self.config)
        self.quality_predictor = QualityPredictor(self.config)
        self.compression_engine = CompressionEngine(self.config.compressor_type)

    def get_task_threshold(self, task_type: TaskType) -> float:
        """Get the task-specific compression threshold."""
        thresholds = {
            TaskType.CODE: self.config.r_code,
            TaskType.COT: self.config.r_cot,
            TaskType.HYBRID: self.config.r_hybrid,
        }
        return thresholds[task_type]

    def compress(self, prompt: str,
                 quality_floor: Optional[float] = None,
                 task_type_override: Optional[TaskType] = None,
                 verbose: bool = False) -> CompressionResult:
        """Apply TAAC compression to a prompt.

        Args:
            prompt: The input prompt to compress
            quality_floor: Minimum acceptable quality (overrides config)
            task_type_override: Force a specific task type (skip classification)
            verbose: Print detailed progress

        Returns:
            CompressionResult with compressed prompt and metadata
        """
        total_start = time.perf_counter()
        q_min = quality_floor or self.config.quality_floor

        # Stage 1: Task Classification
        class_start = time.perf_counter()
        if task_type_override:
            task_type = task_type_override
            task_confidence = 1.0
        else:
            task_type, task_confidence = self.task_classifier.classify(prompt)
        class_time_ms = (time.perf_counter() - class_start) * 1000

        if verbose:
            logger.info(f"Task type: {task_type.value} (confidence: {task_confidence:.2f})")

        # Stage 2: Information Density Estimation
        density_start = time.perf_counter()
        density, perplexity_stats = self.density_estimator.estimate(prompt)
        density_time_ms = (time.perf_counter() - density_start) * 1000

        if verbose:
            logger.info(f"Information density: {density:.3f}")

        # Calculate target ratio with density adjustment
        r_task = self.get_task_threshold(task_type)
        # Higher density (heterogeneous) allows more compression
        r_target = r_task + self.config.density_lambda * (1 - density)
        r_target = max(0.3, min(1.0, r_target))  # Clamp to valid range

        if verbose:
            logger.info(f"Target ratio: {r_target:.3f} (base: {r_task:.3f})")

        # Stage 3: Quality-Gated Compression
        r_current = 1.0
        compressed_prompt = prompt
        quality_gated = False
        predicted_quality = 1.0
        quality_pred_time_ms = 0.0

        delta = self.config.compression_step
        best_result = None

        while r_current > r_target:
            # Compress to next level
            r_next = r_current - delta
            compression_result = self.compression_engine.compress(prompt, r_next)
            candidate_prompt = compression_result["compressed_prompt"]
            actual_ratio = compression_result["actual_ratio"]

            # Predict quality
            pred_start = time.perf_counter()
            predicted_quality = self.quality_predictor.predict(
                candidate_prompt, task_type, actual_ratio
            )
            quality_pred_time_ms += (time.perf_counter() - pred_start) * 1000

            if verbose:
                logger.info(
                    f"  r={actual_ratio:.3f}: predicted quality={predicted_quality:.3f}"
                )

            # Quality gate check
            if predicted_quality < q_min:
                quality_gated = True
                if verbose:
                    logger.info(
                        f"Quality gate triggered at r={actual_ratio:.3f} "
                        f"(quality {predicted_quality:.3f} < floor {q_min:.3f})"
                    )
                break

            # Accept this compression level
            compressed_prompt = candidate_prompt
            r_current = actual_ratio
            best_result = compression_result

        # Use best result if we found one
        if best_result is None:
            best_result = {
                "original_tokens": len(prompt.split()),
                "compressed_tokens": len(compressed_prompt.split()),
            }

        total_time_ms = (time.perf_counter() - total_start) * 1000

        return CompressionResult(
            compressed_prompt=compressed_prompt,
            original_prompt=prompt,
            compression_ratio=r_current,
            task_type=task_type,
            information_density=density,
            predicted_quality=predicted_quality,
            quality_gated=quality_gated,
            compression_time_ms=total_time_ms - class_time_ms - density_time_ms - quality_pred_time_ms,
            classification_time_ms=class_time_ms,
            density_time_ms=density_time_ms,
            quality_prediction_time_ms=quality_pred_time_ms,
            original_tokens=best_result["original_tokens"],
            compressed_tokens=best_result["compressed_tokens"],
            perplexity_stats=perplexity_stats,
        )

    def compress_batch(self, prompts: list[str],
                       quality_floor: Optional[float] = None) -> list[CompressionResult]:
        """Compress multiple prompts."""
        return [self.compress(p, quality_floor) for p in prompts]


# =============================================================================
# Ablation Study Variants
# =============================================================================

class TAACTaskOnly(TAAC):
    """TAAC with only task classification (no density, no quality-gating).

    Uses fixed task-specific thresholds without adaptation.
    For ablation study: measures contribution of task-awareness alone.
    """

    def compress(self, prompt: str,
                 quality_floor: Optional[float] = None,
                 task_type_override: Optional[TaskType] = None,
                 verbose: bool = False) -> CompressionResult:

        total_start = time.perf_counter()

        # Stage 1: Task Classification
        class_start = time.perf_counter()
        if task_type_override:
            task_type = task_type_override
        else:
            task_type, _ = self.task_classifier.classify(prompt)
        class_time_ms = (time.perf_counter() - class_start) * 1000

        # Use fixed threshold (no density adjustment, no quality gating)
        r_target = self.get_task_threshold(task_type)

        # Single compression step
        compression_result = self.compression_engine.compress(prompt, r_target)

        total_time_ms = (time.perf_counter() - total_start) * 1000

        return CompressionResult(
            compressed_prompt=compression_result["compressed_prompt"],
            original_prompt=prompt,
            compression_ratio=compression_result["actual_ratio"],
            task_type=task_type,
            information_density=0.0,  # Not computed
            predicted_quality=1.0,  # Not computed
            quality_gated=False,
            compression_time_ms=total_time_ms - class_time_ms,
            classification_time_ms=class_time_ms,
            density_time_ms=0.0,
            quality_prediction_time_ms=0.0,
            original_tokens=compression_result["original_tokens"],
            compressed_tokens=compression_result["compressed_tokens"],
        )


class TAACDensityOnly(TAAC):
    """TAAC with only density estimation (no task classification, no quality-gating).

    Uses a single base threshold adjusted by density.
    For ablation study: measures contribution of density-awareness alone.
    """

    def compress(self, prompt: str,
                 quality_floor: Optional[float] = None,
                 task_type_override: Optional[TaskType] = None,
                 verbose: bool = False) -> CompressionResult:

        total_start = time.perf_counter()

        # Use a fixed base ratio (average of task thresholds)
        base_ratio = (self.config.r_code + self.config.r_cot + self.config.r_hybrid) / 3

        # Stage 2: Information Density Estimation
        density_start = time.perf_counter()
        density, perplexity_stats = self.density_estimator.estimate(prompt)
        density_time_ms = (time.perf_counter() - density_start) * 1000

        # Adjust target by density
        r_target = base_ratio + self.config.density_lambda * (1 - density)
        r_target = max(0.3, min(1.0, r_target))

        # Single compression step
        compression_result = self.compression_engine.compress(prompt, r_target)

        total_time_ms = (time.perf_counter() - total_start) * 1000

        return CompressionResult(
            compressed_prompt=compression_result["compressed_prompt"],
            original_prompt=prompt,
            compression_ratio=compression_result["actual_ratio"],
            task_type=TaskType.HYBRID,  # Not classified
            information_density=density,
            predicted_quality=1.0,  # Not computed
            quality_gated=False,
            compression_time_ms=total_time_ms - density_time_ms,
            classification_time_ms=0.0,
            density_time_ms=density_time_ms,
            quality_prediction_time_ms=0.0,
            original_tokens=compression_result["original_tokens"],
            compressed_tokens=compression_result["compressed_tokens"],
            perplexity_stats=perplexity_stats,
        )


class TAACQualityGateOnly(TAAC):
    """TAAC with only quality-gating (no task classification, no density).

    Uses a single target ratio but enforces quality floor.
    For ablation study: measures contribution of quality-gating alone.
    """

    def compress(self, prompt: str,
                 quality_floor: Optional[float] = None,
                 task_type_override: Optional[TaskType] = None,
                 verbose: bool = False) -> CompressionResult:

        total_start = time.perf_counter()
        q_min = quality_floor or self.config.quality_floor

        # Use a fixed aggressive target
        r_target = 0.5  # Aggressive target, quality gate will stop if needed

        # Quality-gated compression loop
        r_current = 1.0
        compressed_prompt = prompt
        quality_gated = False
        predicted_quality = 1.0
        quality_pred_time_ms = 0.0
        delta = self.config.compression_step
        best_result = None

        while r_current > r_target:
            r_next = r_current - delta
            compression_result = self.compression_engine.compress(prompt, r_next)
            candidate_prompt = compression_result["compressed_prompt"]
            actual_ratio = compression_result["actual_ratio"]

            # Predict quality (using HYBRID since we don't classify)
            pred_start = time.perf_counter()
            predicted_quality = self.quality_predictor.predict(
                candidate_prompt, TaskType.HYBRID, actual_ratio
            )
            quality_pred_time_ms += (time.perf_counter() - pred_start) * 1000

            if predicted_quality < q_min:
                quality_gated = True
                break

            compressed_prompt = candidate_prompt
            r_current = actual_ratio
            best_result = compression_result

        if best_result is None:
            best_result = {
                "original_tokens": len(prompt.split()),
                "compressed_tokens": len(compressed_prompt.split()),
            }

        total_time_ms = (time.perf_counter() - total_start) * 1000

        return CompressionResult(
            compressed_prompt=compressed_prompt,
            original_prompt=prompt,
            compression_ratio=r_current,
            task_type=TaskType.HYBRID,
            information_density=0.0,
            predicted_quality=predicted_quality,
            quality_gated=quality_gated,
            compression_time_ms=total_time_ms - quality_pred_time_ms,
            classification_time_ms=0.0,
            density_time_ms=0.0,
            quality_prediction_time_ms=quality_pred_time_ms,
            original_tokens=best_result["original_tokens"],
            compressed_tokens=best_result["compressed_tokens"],
        )


# =============================================================================
# Comparison Baselines
# =============================================================================

class FixedRatioCompressor:
    """Fixed ratio compression baseline.

    Applies same compression ratio regardless of task type.
    """

    def __init__(self, ratio: float = 0.6, method: str = "llmlingua2"):
        self.ratio = ratio
        self.engine = CompressionEngine(method)

    def compress(self, prompt: str) -> CompressionResult:
        start_time = time.perf_counter()

        result = self.engine.compress(prompt, self.ratio)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            compressed_prompt=result["compressed_prompt"],
            original_prompt=prompt,
            compression_ratio=result["actual_ratio"],
            task_type=TaskType.HYBRID,
            information_density=0.0,
            predicted_quality=1.0,
            quality_gated=False,
            compression_time_ms=total_time_ms,
            classification_time_ms=0.0,
            density_time_ms=0.0,
            quality_prediction_time_ms=0.0,
            original_tokens=result["original_tokens"],
            compressed_tokens=result["compressed_tokens"],
        )


class TaskBasedFixedCompressor:
    """Task-based fixed ratio compression.

    Uses task-specific ratios but without adaptive adjustment.
    Similar to what one might do with Johnson (2026) findings alone.
    """

    def __init__(self, config: Optional[TAACConfig] = None, method: str = "llmlingua2"):
        self.config = config or TAACConfig()
        self.classifier = TaskClassifier(self.config)
        self.engine = CompressionEngine(method)

    def compress(self, prompt: str) -> CompressionResult:
        start_time = time.perf_counter()

        # Classify task
        class_start = time.perf_counter()
        task_type, _ = self.classifier.classify(prompt)
        class_time_ms = (time.perf_counter() - class_start) * 1000

        # Get task-specific ratio
        ratios = {
            TaskType.CODE: self.config.r_code,
            TaskType.COT: self.config.r_cot,
            TaskType.HYBRID: self.config.r_hybrid,
        }
        target_ratio = ratios[task_type]

        # Compress
        result = self.engine.compress(prompt, target_ratio)

        total_time_ms = (time.perf_counter() - start_time) * 1000

        return CompressionResult(
            compressed_prompt=result["compressed_prompt"],
            original_prompt=prompt,
            compression_ratio=result["actual_ratio"],
            task_type=task_type,
            information_density=0.0,
            predicted_quality=1.0,
            quality_gated=False,
            compression_time_ms=total_time_ms - class_time_ms,
            classification_time_ms=class_time_ms,
            density_time_ms=0.0,
            quality_prediction_time_ms=0.0,
            original_tokens=result["original_tokens"],
            compressed_tokens=result["compressed_tokens"],
        )


# =============================================================================
# Example Usage and Testing
# =============================================================================

def demo():
    """Demonstrate TAAC algorithm."""

    print("=" * 60)
    print("TAAC: Task-Aware Adaptive Compression Demo")
    print("=" * 60)

    # Example prompts
    code_prompt = """
def fibonacci(n):
    '''
    Calculate the nth Fibonacci number using dynamic programming.

    Args:
        n: The position in the Fibonacci sequence (0-indexed)

    Returns:
        The nth Fibonacci number

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    '''
    # Your implementation here
    """

    cot_prompt = """
Solve this step by step:

A farmer has 120 meters of fencing to enclose a rectangular field.
One side of the field is against a river, so no fencing is needed there.
What dimensions will maximize the area of the field?

Think through this problem carefully, showing all your work.
"""

    # Initialize TAAC
    taac = TAAC()

    print("\n--- Code Prompt ---")
    result = taac.compress(code_prompt, verbose=True)
    print(f"\nResult:")
    print(f"  Task type: {result.task_type.value}")
    print(f"  Compression ratio: {result.compression_ratio:.3f}")
    print(f"  Quality gated: {result.quality_gated}")
    print(f"  Predicted quality: {result.predicted_quality:.3f}")
    print(f"  Total time: {result.compression_time_ms + result.classification_time_ms:.1f}ms")

    print("\n--- CoT Prompt ---")
    result = taac.compress(cot_prompt, verbose=True)
    print(f"\nResult:")
    print(f"  Task type: {result.task_type.value}")
    print(f"  Compression ratio: {result.compression_ratio:.3f}")
    print(f"  Quality gated: {result.quality_gated}")
    print(f"  Predicted quality: {result.predicted_quality:.3f}")
    print(f"  Total time: {result.compression_time_ms + result.classification_time_ms:.1f}ms")


if __name__ == "__main__":
    # Run demo if executed directly
    demo()

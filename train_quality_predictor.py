#!/usr/bin/env python3
"""
Quality Predictor Training Pipeline
====================================

Trains the quality predictor component of TAAC algorithm.

Architecture:
    Input: (compressed_prompt, task_type) -> sentence_embedding + task_onehot
    Model: 2-layer MLP with ReLU, dropout
    Output: predicted_quality in [0, 1]

Training Data:
    Uses data from TAAC experiments (Johnson 2026):
    - (compressed_prompt, task_type, actual_quality) tuples
    - Quality measured as pass@1 for code, exact_match for CoT

Key Design Decisions:
    - Frozen sentence embeddings (all-MiniLM-L6-v2) for efficiency
    - Separate from compression model to avoid end-to-end training complexity
    - MSE loss for regression to quality score

Author: Dr. Amanda Foster, Bona Opera Studios
Date: January 2026
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional
import os

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


class QualityPredictorDataset:
    """Dataset for training the quality predictor.

    Loads experimental results and prepares training data:
    - compressed_prompt: the text after compression
    - task_type: code/cot/hybrid (one-hot encoded)
    - compression_ratio: the actual compression ratio achieved
    - quality_score: the measured task performance (0-1)
    """

    def __init__(self, data_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.data_path = Path(data_path)
        self.embedding_model = embedding_model
        self._embedder = None
        self._data = None

    def load(self):
        """Load and preprocess the data."""
        import torch
        from sentence_transformers import SentenceTransformer

        # Load embedder
        logger.info(f"Loading sentence embedder: {self.embedding_model}")
        self._embedder = SentenceTransformer(self.embedding_model)

        # Load experimental data
        self._data = []

        if self.data_path.suffix == ".jsonl":
            with open(self.data_path) as f:
                for line in f:
                    self._data.append(json.loads(line))
        elif self.data_path.suffix == ".json":
            with open(self.data_path) as f:
                self._data = json.load(f)
        else:
            raise ValueError(f"Unknown data format: {self.data_path.suffix}")

        logger.info(f"Loaded {len(self._data)} training samples")

    def prepare_tensors(self):
        """Convert data to PyTorch tensors.

        Returns:
            tuple of (features, labels) tensors
        """
        import torch

        if self._data is None:
            self.load()

        embeddings = []
        task_onehots = []
        qualities = []

        for sample in self._data:
            # Get compressed prompt embedding
            prompt = sample.get("compressed_prompt", sample.get("prompt", ""))
            if not prompt:
                continue

            embedding = self._embedder.encode(
                prompt,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embeddings.append(embedding)

            # Task type one-hot encoding
            task_type = sample.get("task_type", sample.get("benchmark_type", "hybrid"))
            if isinstance(task_type, str):
                task_type = task_type.lower()

            onehot = torch.zeros(3)
            if task_type == "code":
                onehot[0] = 1
            elif task_type in ["cot", "reasoning", "math"]:
                onehot[1] = 1
            else:
                onehot[2] = 1
            task_onehots.append(onehot)

            # Quality score
            quality = sample.get("quality", sample.get("score", 0.0))
            qualities.append(quality)

        # Stack into tensors
        X_embeddings = torch.stack(embeddings)
        X_tasks = torch.stack(task_onehots)
        X = torch.cat([X_embeddings, X_tasks], dim=1)
        y = torch.tensor(qualities, dtype=torch.float32).unsqueeze(1)

        return X, y

    def create_synthetic_data(self, n_samples: int = 10000):
        """Create synthetic training data for demonstration.

        Uses the empirical quality curves from Johnson (2026):
        - Code: Threshold at r=0.6
        - CoT: Gradual linear degradation
        """
        import torch

        logger.info(f"Creating {n_samples} synthetic training samples")

        # Generate random prompts (we'll use random embeddings as proxy)
        embedding_dim = 384  # MiniLM-L6-v2

        embeddings = torch.randn(n_samples, embedding_dim)

        # Random task types
        task_types = torch.randint(0, 3, (n_samples,))
        task_onehots = torch.zeros(n_samples, 3)
        task_onehots.scatter_(1, task_types.unsqueeze(1), 1)

        # Random compression ratios
        ratios = torch.rand(n_samples) * 0.7 + 0.3  # [0.3, 1.0]

        # Quality scores based on empirical curves
        qualities = torch.zeros(n_samples)

        for i in range(n_samples):
            r = ratios[i].item()
            task = task_types[i].item()

            if task == 0:  # Code
                # Threshold behavior
                if r >= 0.6:
                    q = 0.95 - 0.05 * (1 - r) / 0.4
                else:
                    q = 0.95 * (r / 0.6) ** 2
            elif task == 1:  # CoT
                # Gradual degradation
                q = 0.95 - 0.4 * (1 - r)
            else:  # Hybrid
                # Interpolation
                code_q = 0.95 - 0.05 * (1 - r) / 0.4 if r >= 0.6 else 0.95 * (r / 0.6) ** 2
                cot_q = 0.95 - 0.4 * (1 - r)
                q = 0.5 * code_q + 0.5 * cot_q

            # Add noise
            q = max(0, min(1, q + np.random.normal(0, 0.05)))
            qualities[i] = q

        X = torch.cat([embeddings, task_onehots], dim=1)
        y = qualities.unsqueeze(1)

        self._data = [{"synthetic": True}] * n_samples

        return X, y


class QualityPredictorModel:
    """2-layer MLP for quality prediction."""

    def __init__(self, input_dim: int = 387, hidden_dim: int = 256):
        """
        Args:
            input_dim: Embedding dim (384) + task onehot (3) = 387
            hidden_dim: Hidden layer dimension
        """
        import torch
        import torch.nn as nn

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.model(x)

    def save(self, path: str):
        """Save model weights."""
        import torch
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved model to {path}")

    def load(self, path: str):
        """Load model weights."""
        import torch
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Loaded model from {path}")


class Trainer:
    """Training loop for quality predictor."""

    def __init__(
        self,
        model: QualityPredictorModel,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
    ):
        import torch

        self.model = model

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function (MSE for regression)
        self.criterion = torch.nn.MSELoss()

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        epochs: int = 100,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
    ):
        """Train the model."""
        import torch

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        n_samples = X_train.size(0)
        best_val_loss = float('inf')
        patience_counter = 0

        history = {
            'train_loss': [],
            'val_loss': [],
        }

        for epoch in range(epochs):
            self.model.model.train()

            # Shuffle training data
            perm = torch.randperm(n_samples)
            X_train = X_train[perm]
            y_train = y_train[perm]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                self.optimizer.zero_grad()
                predictions = self.model.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)

            # Validation
            if X_val is not None:
                self.model.model.eval()
                with torch.no_grad():
                    val_predictions = self.model.model(X_val)
                    val_loss = self.criterion(val_predictions, y_val).item()

                history['val_loss'].append(val_loss)
                self.scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"train_loss={avg_train_loss:.6f}, val_loss={val_loss:.6f}"
                    )
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.6f}")

        return history

    def evaluate(self, X_test, y_test):
        """Evaluate on test set."""
        import torch

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        self.model.model.eval()
        with torch.no_grad():
            predictions = self.model.model(X_test)
            mse = self.criterion(predictions, y_test).item()
            mae = torch.abs(predictions - y_test).mean().item()

        # Additional metrics
        predictions_np = predictions.cpu().numpy().flatten()
        y_test_np = y_test.cpu().numpy().flatten()

        # Correlation
        correlation = np.corrcoef(predictions_np, y_test_np)[0, 1]

        return {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'correlation': correlation,
        }


def train_from_experiments(
    results_path: str,
    output_path: str = "taac_quality_predictor.pt",
    epochs: int = 100,
    hidden_dim: int = 256,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2,
):
    """Train quality predictor from experiment results.

    Args:
        results_path: Path to experiment results (JSONL or JSON)
        output_path: Path to save trained model
        epochs: Number of training epochs
        hidden_dim: MLP hidden dimension
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
    """
    import torch

    logger.info("Loading training data...")

    dataset = QualityPredictorDataset(results_path)
    X, y = dataset.prepare_tensors()

    # Train/validation split
    n_samples = X.size(0)
    n_val = int(n_samples * validation_split)
    perm = torch.randperm(n_samples)

    X = X[perm]
    y = y[perm]

    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]

    logger.info(f"Training samples: {X_train.size(0)}, Validation samples: {X_val.size(0)}")

    # Initialize model
    model = QualityPredictorModel(input_dim=X.size(1), hidden_dim=hidden_dim)

    # Train
    trainer = Trainer(model, learning_rate=learning_rate)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

    # Evaluate
    metrics = trainer.evaluate(X_val, y_val)
    logger.info(f"Final validation metrics: {metrics}")

    # Save model
    model.save(output_path)

    return history, metrics


def train_synthetic(
    output_path: str = "taac_quality_predictor.pt",
    n_samples: int = 10000,
    epochs: int = 100,
    hidden_dim: int = 256,
):
    """Train on synthetic data for demonstration/testing."""
    import torch

    logger.info("Generating synthetic training data...")

    dataset = QualityPredictorDataset("dummy.json")  # Won't be used
    X, y = dataset.create_synthetic_data(n_samples)

    # Split
    n_val = int(n_samples * 0.2)
    X_train, X_val = X[n_val:], X[:n_val]
    y_train, y_val = y[n_val:], y[:n_val]

    logger.info(f"Training samples: {X_train.size(0)}, Validation samples: {X_val.size(0)}")

    # Train
    model = QualityPredictorModel(input_dim=X.size(1), hidden_dim=hidden_dim)
    trainer = Trainer(model)
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=epochs)

    # Evaluate
    metrics = trainer.evaluate(X_val, y_val)
    logger.info(f"Final validation metrics: {metrics}")

    # Save
    model.save(output_path)

    return history, metrics


def visualize_training(history: dict, output_path: Optional[str] = None):
    """Visualize training curves."""
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        ax.plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            ax.plot(history['val_loss'], label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title('Quality Predictor Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training plot to {output_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        logger.warning("matplotlib not available for visualization")


def main():
    parser = argparse.ArgumentParser(description="Train TAAC Quality Predictor")

    parser.add_argument(
        "--data",
        type=str,
        help="Path to training data (JSONL or JSON file)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data for training (for testing)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="taac_quality_predictor.pt",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="MLP hidden layer dimension",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of synthetic samples (if --synthetic)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        help="Path to save training curve plot",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.synthetic:
        history, metrics = train_synthetic(
            output_path=args.output,
            n_samples=args.n_samples,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
        )
    elif args.data:
        history, metrics = train_from_experiments(
            results_path=args.data,
            output_path=args.output,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
        )
    else:
        parser.error("Either --data or --synthetic is required")

    if args.plot:
        visualize_training(history, args.plot)

    print("\n" + "=" * 50)
    print("Training Complete")
    print("=" * 50)
    print(f"Final Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()

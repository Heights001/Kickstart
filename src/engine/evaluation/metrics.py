"""Probabilistic evaluation metrics: log loss, multiclass Brier, top-label ECE."""

import itertools

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

_EPS = 1e-15


def log_loss(y_true: IntArray, probs: FloatArray) -> float:
    """Mean negative log likelihood of the true class."""
    clipped = np.clip(probs[np.arange(len(y_true)), y_true], _EPS, 1.0)
    return float(-np.mean(np.log(clipped)))


def brier(y_true: IntArray, probs: FloatArray) -> float:
    """Multiclass Brier score: mean squared distance to the one-hot outcome."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def ece(y_true: IntArray, probs: FloatArray, bins: int) -> float:
    """Top-label expected calibration error with equal-width confidence bins."""
    confidence = probs.max(axis=1)
    predicted = probs.argmax(axis=1)
    correct = (predicted == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        # Include the right edge only in the last bin.
        in_bin = (confidence >= lo) & ((confidence < hi) | (hi == 1.0))
        if not in_bin.any():
            continue
        gap = abs(correct[in_bin].mean() - confidence[in_bin].mean())
        total += in_bin.mean() * gap
    return float(total)

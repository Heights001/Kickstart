"""SHAP explanations for Rung 2 (tree model); rungs 0-1 are read off coefficients."""

import numpy as np
import shap

from engine.evaluation.metrics import FloatArray
from engine.models.features import FEATURE_NAMES
from engine.models.rung2 import Rung2Model


def rung2_feature_importance(model: Rung2Model, x: FloatArray) -> dict[str, float]:
    """Mean |SHAP| per feature, averaged over classes and rows."""
    explainer = shap.TreeExplainer(model.booster)
    values = explainer.shap_values(x)
    # list of (n, features) per class, or (n, features, classes)
    stacked = np.stack(values, axis=-1) if isinstance(values, list) else np.asarray(values)
    importance = np.abs(stacked).mean(axis=tuple(i for i in range(stacked.ndim) if i != 1))
    return dict(zip(FEATURE_NAMES, (float(v) for v in importance), strict=True))

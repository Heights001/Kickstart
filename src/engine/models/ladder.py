"""The model ladder: each rung is a drop-in OutcomeModel with its feature extractor.

Rung 2 additionally needs a tuning pass on the validation window before fit;
``needs_tuning`` signals that to the evaluation harness.
"""

from collections.abc import Callable
from dataclasses import dataclass

from engine.core.config import EngineConfig
from engine.evaluation.metrics import FloatArray
from engine.models.base import OutcomeModel
from engine.models.features import FEATURE_NAMES, full_features
from engine.models.rung0 import Rung0Model, rung0_features
from engine.models.rung1 import Rung1Model
from engine.models.rung2 import Rung2Model
from engine.ratings.store import MatchFeatures

RUNG0_FEATURE_NAMES: tuple[str, ...] = ("elo_diff", "neutral", "rest_diff")


@dataclass(frozen=True)
class RungSpec:
    rung: int
    name: str
    extract: Callable[[list[MatchFeatures]], FloatArray]
    make: Callable[[], OutcomeModel]
    feature_names: tuple[str, ...]
    needs_tuning: bool = False


def get_rung(rung: int, config: EngineConfig) -> RungSpec:
    if rung == 0:
        return RungSpec(
            0,
            "rung0",
            rung0_features,
            lambda: Rung0Model(config.models.rung0, config.seed),
            RUNG0_FEATURE_NAMES,
        )
    if rung == 1:
        return RungSpec(
            1,
            "rung1",
            full_features,
            lambda: Rung1Model(config.models.rung1, config.seed),
            FEATURE_NAMES,
        )
    if rung == 2:
        return RungSpec(
            2,
            "rung2",
            full_features,
            lambda: Rung2Model(config.models.rung2, config.seed),
            FEATURE_NAMES,
            needs_tuning=True,
        )
    raise ValueError(f"unknown rung {rung}")

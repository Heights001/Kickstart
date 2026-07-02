"""Promote-only-if-better gate (CLAUDE.md rule 6) and the promoted-rung registry.

A rung ships only if it beats the rung below on mean held-out log loss across all
backtest windows AND is at least as well calibrated (mean ECE). The registry file
records which rung simulation should fit as-of; it never stores a fitted model —
models are as-of-dependent and always retrained at simulation time (rule 3).
"""

import datetime as dt
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from engine.evaluation.walk_forward import WindowResult

REGISTRY_FILE = "model_registry.json"
_ECE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PromotionDecision:
    candidate_rung: int
    incumbent_rung: int
    promoted: bool
    candidate_log_loss: float
    incumbent_log_loss: float
    candidate_ece: float
    incumbent_ece: float


def decide(
    candidate_rung: int,
    candidate: list[WindowResult],
    incumbent_rung: int,
    incumbent: list[WindowResult],
) -> PromotionDecision:
    cand_ll = _mean(candidate, "log_loss")
    inc_ll = _mean(incumbent, "log_loss")
    cand_ece = _mean(candidate, "ece")
    inc_ece = _mean(incumbent, "ece")
    return PromotionDecision(
        candidate_rung=candidate_rung,
        incumbent_rung=incumbent_rung,
        promoted=cand_ll < inc_ll and cand_ece <= inc_ece + _ECE_TOLERANCE,
        candidate_log_loss=cand_ll,
        incumbent_log_loss=inc_ll,
        candidate_ece=cand_ece,
        incumbent_ece=inc_ece,
    )


def promoted_rung(data_dir: Path) -> int:
    """The rung simulation should use; rung 0 until something earns promotion."""
    path = data_dir / "processed" / REGISTRY_FILE
    if not path.exists():
        return 0
    payload = json.loads(path.read_text())
    return int(payload["promoted_rung"])


def record_promotion(data_dir: Path, decision: PromotionDecision) -> Path:
    path = data_dir / "processed" / REGISTRY_FILE
    current = promoted_rung(data_dir)
    new_rung = decision.candidate_rung if decision.promoted else current
    payload = {
        "promoted_rung": new_rung,
        "decided_at": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "last_decision": asdict(decision),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def _mean(results: list[WindowResult], metric: str) -> float:
    return float(sum(getattr(r.model, metric) for r in results)) / len(results)

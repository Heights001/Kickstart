"""MLflow local tracking for rung evaluations and promotion decisions (rule 6)."""

from pathlib import Path

import mlflow

from engine.core.config import MlflowConfig
from engine.evaluation.promotion import PromotionDecision
from engine.evaluation.walk_forward import WindowResult


def log_rung_evaluation(
    config: MlflowConfig,
    rung_name: str,
    results: list[WindowResult],
    decision: PromotionDecision | None = None,
) -> str:
    tracking_dir = Path(config.tracking_dir).absolute()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    # MLflow >=3.14 requires a database backend; local sqlite keeps it self-contained.
    mlflow.set_tracking_uri(f"sqlite:///{tracking_dir / 'mlflow.db'}")
    mlflow.set_experiment(config.experiment)
    with mlflow.start_run(run_name=rung_name) as run:
        mlflow.log_param("rung", rung_name)
        for r in results:
            prefix = r.window
            mlflow.log_metric(f"{prefix}_log_loss", r.model.log_loss)
            mlflow.log_metric(f"{prefix}_brier", r.model.brier)
            mlflow.log_metric(f"{prefix}_ece", r.model.ece)
            mlflow.log_param(f"{prefix}_calibrator", r.calibrator)
        mlflow.log_metric("mean_log_loss", sum(r.model.log_loss for r in results) / len(results))
        mlflow.log_metric("mean_ece", sum(r.model.ece for r in results) / len(results))
        if decision is not None:
            mlflow.log_param("compared_to_rung", decision.incumbent_rung)
            mlflow.log_param("promoted", decision.promoted)
            mlflow.log_metric("incumbent_log_loss", decision.incumbent_log_loss)
            mlflow.log_metric("incumbent_ece", decision.incumbent_ece)
        return str(run.info.run_id)

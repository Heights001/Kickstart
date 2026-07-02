"""Rung 2: LightGBM multiclass with a seeded Optuna search (SCOPE §2.5).

Hyperparameters are tuned on the validation window passed to :meth:`Rung2Model.tune`
— the same window the calibrator later uses, a slight optimism documented in
STATUS.md. All randomness is seeded; Optuna uses a seeded TPE sampler.
"""

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna

from engine.core.config import Rung2Config
from engine.evaluation.metrics import FloatArray, IntArray, log_loss

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Rung2Model:
    def __init__(self, config: Rung2Config, seed: int) -> None:
        self._config = config
        self._seed = seed
        self._params: dict[str, float | int] | None = None
        self._booster: lgb.LGBMClassifier | None = None
        self._val: tuple[FloatArray, IntArray] | None = None

    def tune(
        self, x_train: FloatArray, y_train: IntArray, x_val: FloatArray, y_val: IntArray
    ) -> None:
        """Search hyperparameters on the validation window, then remember them."""
        cfg = self._config

        def objective(trial: optuna.Trial) -> float:
            params = {
                "learning_rate": trial.suggest_float("learning_rate", *cfg.learning_rate, log=True),
                "num_leaves": trial.suggest_int("num_leaves", *cfg.num_leaves),
                "min_child_samples": trial.suggest_int("min_child_samples", *cfg.min_child_samples),
            }
            model = self._make(params)
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_val, y_val)],
                callbacks=[lgb.early_stopping(cfg.early_stopping_rounds, verbose=False)],
            )
            return log_loss(y_val, self._ordered_proba(model, x_val))

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=self._seed)
        )
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
        self._params = dict(study.best_params)
        self._val = (x_val, y_val)

    def fit(self, x: FloatArray, y: IntArray) -> None:
        if self._params is None:
            raise RuntimeError("call tune() before fit()")
        self._booster = self._make(self._params)
        assert self._val is not None
        self._booster.fit(
            x,
            y,
            eval_set=[self._val],
            callbacks=[lgb.early_stopping(self._config.early_stopping_rounds, verbose=False)],
        )

    def predict_proba(self, x: FloatArray) -> FloatArray:
        if self._booster is None:
            raise RuntimeError("model not fitted")
        return self._ordered_proba(self._booster, x)

    @property
    def booster(self) -> lgb.LGBMClassifier:
        if self._booster is None:
            raise RuntimeError("model not fitted")
        return self._booster

    @property
    def best_params(self) -> dict[str, float | int]:
        if self._params is None:
            raise RuntimeError("call tune() first")
        return dict(self._params)

    def _make(self, params: dict[str, Any]) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=self._config.n_estimators,
            random_state=self._seed,
            deterministic=True,
            force_col_wise=True,
            verbose=-1,
            **params,
        )

    @staticmethod
    def _ordered_proba(model: lgb.LGBMClassifier, x: FloatArray) -> FloatArray:
        raw = np.asarray(model.predict_proba(x), dtype=np.float64)
        probs = np.zeros((x.shape[0], 3), dtype=np.float64)
        for column, label in enumerate(model.classes_):
            probs[:, int(label)] = raw[:, column]
        return probs

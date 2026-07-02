# STATUS.md

**Last updated:** 2026-07-02 (end of P3 session)
**Tournament context:** WC2026 live — R32 in progress, final July 19. MVP (P2) shipped;
P3's promoted model now drives the live simulations.

## Current phase

**P3 — complete.** Model ladder built and evaluated; **Rung 2 (LightGBM+Optuna) is
promoted** and `engine simulate` uses it. All four gates green, 104 tests.

Promotion results (walk-forward, WC 2014/18/22, means across windows; MLflow-logged):

| candidate | mean log loss | vs shipped | mean ECE | vs shipped | decision |
|-----------|--------------:|-----------:|---------:|-----------:|----------|
| rung1 (feature LR) | 0.9797 | 0.9880 | 0.1140 | 0.1019 | **NOT promoted** — better log loss, worse calibration (rule 6) |
| rung2 (LightGBM)   | 0.9850 | 0.9880 | 0.0699 | 0.1019 | **PROMOTED** — better on both |

Live simulation with rung 2 (100k runs, as-of 2026-07-02): **France 20.8%, Argentina
19.4%, Spain 11.9%, Brazil 11.4%, Mexico 9.9%, USA 6.8%**. Rung 2 flattens rung 0's
Elo-only hierarchy (Argentina was 28.0%) and rates hosts/USA higher. Rewind
(`--freeze 2026-06-10`) works with the promoted rung: Spain 23.6% ante post.

Shipped this session:
- **Feature store extension:** chronological experience counters (total matches +
  same-tier matches, both strictly pre-match) in `RatingsWalker`/`MatchFeatures`.
- **Rung 1** (`models/rung1.py`): scaled multinomial LR on the full 8-feature set
  (`models/features.py`: elo/form/attack/defence/rest diffs, neutral, log-scaled
  experience diffs).
- **Rung 2** (`models/rung2.py`): LightGBM multiclass, seeded Optuna TPE (25 trials,
  space in config), early stopping, deterministic mode.
- **Ladder + generic evaluation:** `models/ladder.py` RungSpecs; `evaluate_window` and
  `fit_model_asof` take any rung; sampler builds synthetic as-of feature rows per
  pairing, so any rung can drive simulation.
- **Promotion gate** (`evaluation/promotion.py`): better mean log loss AND
  no-worse mean ECE vs the **currently shipped** rung (not blindly N-1 — beating an
  unpromoted middle rung must not ship a regression). `model_registry.json` records
  the promoted rung; simulate reads it. Models are never persisted — they are
  as-of-dependent and retrained at simulation time (rule 3).
- **MLflow** local tracking (sqlite backend — MLflow ≥3.14 dropped the file store) in
  `mlruns/`; each rung evaluation + promotion comparison logged (rule 6).
- **SHAP** (`explain/shap_explain.py`): mean-|SHAP| feature importance for rung 2;
  sanity test confirms elo_diff dominates on synthetic data.

## Decisions made this session

- Promotion compares against the shipped rung, not rung N-1 (see above).
- xG/FIFA-rank enrichment stays out of v1 features (open question #3 unanswered;
  SCOPE treats them as optional-behind-masks anyway).
- numpy pinned <2.5 (numba/shap chain); uv lock limited to arm64-macOS + Linux
  (shap's Intel-macOS constraint drags in unbuildable llvmlite); Python <3.14.
- libomp installed via Homebrew (LightGBM runtime requirement on macOS).

## Honest caveats

- **Optuna tunes on the same window the calibrator uses** — mild optimism in rung 2's
  validation loss. Nested walk-forward tuning is the clean fix if it ever matters.
- Rung 2's promotion margin on log loss is modest (0.9850 vs 0.9880, n_test=192);
  its calibration improvement is the stronger signal. Re-run the comparison after the
  tournament adds data.
- Simulation startup now includes Optuna tuning (~1–2 min) since models retrain as-of.
- P1/P2 caveats stand (wc2014 calibration, attack/defence drift, data lag ~1–2 days).

## Next actions

1. P4: FastAPI endpoints + Streamlit dashboard — the probability-over-time chart has a
   hard deadline (final is July 19) and prob_history already accumulates.
2. `engine ingest --refresh` + `engine ratings` + `engine simulate` daily during the
   knockouts to build out prob_history.
3. Open questions (repo name, football-data.org key, license) still with Daniel.

## Reopened decisions

None.

## Blocked / waiting

Nothing.

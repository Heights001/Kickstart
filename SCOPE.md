# SCOPE.md — Football Prediction Engine (v1: World Cup 2026 pack)

**Status:** Direction confirmed 2026-07-01. Ready for implementation.
**Working name:** `predictor-engine` (repo name is an open item — see Open Questions).
**One-liner:** A competition-agnostic, open-source football prediction engine. Competitions plug in as
**packs** (config + data adapters). The engine trains a match-outcome model, simulates the competition
format via Monte Carlo, and outputs the headline number: **probability of winning the competition**.
v1 ships exactly one pack: **FIFA World Cup 2026**.

---

## 0. Context that shapes v1

The 2026 World Cup is **live right now** (June 11 – July 19, 2026). As of 2026-07-01 the group
stage is complete and the Round of 32 is in progress. This changes the build in three ways:

1. **Live-state ingestion is core MVP, not a stretch goal.** "Predict the World Cup" today means:
   ingest results-to-date, fix them as facts, simulate only the remaining bracket.
2. **The group stage is a free backtest.** Rewind the engine to 2026-06-10, generate pre-tournament
   predictions, and score them against the actual group stage. This is the honest validation story.
3. **Speed matters.** The final is July 19. The phasing below front-loads a working
   champion-probability pipeline (rating-only model) before the heavier ML rungs. The model
   upgrades in place; the product works from week one.

---

## 1. Product definition

**What it does (v1):**
- Ingests historical international match results + live 2026 tournament state.
- Maintains chronological team ratings (Elo family) and engineered features.
- Trains calibrated match-outcome models: P(home win / draw / away win).
- Simulates the WC2026 format (12 groups of 4 → 8 best thirds → 32-team fixed knockout bracket)
  via Monte Carlo from any point in time.
- Outputs per-team probabilities of: winning the tournament, reaching the final, SF, QF, R16, R32
  — with Monte Carlo standard errors and an as-of timestamp.
- Serves results via FastAPI and a Streamlit dashboard.

**What it deliberately does not do (v1):** other competitions (EPL etc.), betting odds ingestion,
player-level modelling, weather, market values, live in-match probabilities. See §10 Deferred.

**North star for the open-source project (post-v1):** anyone can add a league or cup by writing a
pack folder (YAML + optional adapter) with zero engine changes. v1 builds the seams for this but
implements only what WC2026 exercises.

---

## 2. Locked decisions

1. **Engine / pack split.** Engine code never imports pack-specific code. A pack is a folder:
   `format.yaml` (competition structure), `sources.yaml` (data source + column mappings),
   `teams.yaml` (team registry seed / aliases), optional `adapter.py`. The engine loads packs by
   path. No plugin-discovery machinery (entry points, registries) until a second pack exists.
2. **Canonical match schema** (Pydantic): date, competition, season, stage, home, away, neutral
   flag, goals, and a sparse `extras: dict[str, float]` for anything the source happens to have
   (xG, shots, possession). All downstream code consumes only this schema.
3. **Team registry with aliases is a first-class component.** Every ingested team name resolves
   through it ("USA" / "United States" / "USMNT" → one canonical id). Unresolved names fail loudly
   with a suggested-alias report; never silently create duplicate teams.
4. **Chronological integrity everywhere.** Ratings and features are computed strictly as-of match
   date. Validation is walk-forward (train < t, test ≥ t). No random shuffled splits, ever.
5. **Model ladder** (each rung is a drop-in `OutcomeModel`):
   - **Rung 0 — rating-only logistic:** multinomial logistic regression on [elo_diff, neutral flag,
     rest-day diff]. Trains in seconds, works from day one, is the permanent baseline every other
     rung must beat on held-out log loss to earn deployment.
   - **Rung 1 — feature LR:** logistic regression on the full engineered feature set.
   - **Rung 2 — LightGBM + Optuna:** the GBM rung. One GBM family in v1, not three.
   - Calibration layer on top of every rung: fit isotonic and Platt on a walk-forward validation
     window, pick per-model by ECE/Brier, persist with the model.
6. **Ratings in-engine, not scraped.** Compute our own Elo from the results backbone: K scaled by
   competition importance (WC > qualifiers/continental > Nations League > friendlies),
   goal-margin multiplier, home-advantage term suppressed when `neutral=true`. Also maintain
   simple attack/defence ratings (rolling goals for/against vs. opponent-strength-adjusted
   baseline) and a form rating (exponentially weighted recent results). Glicko deferred (§10).
7. **Scoreline layer for group tiebreakers.** A 3-way outcome classifier alone cannot rank a group
   table (tiebreakers are goal-based). Locked approach: sample the outcome from the calibrated
   classifier, then sample a scoreline from a double-Poisson grid (means from attack/defence
   ratings) **conditioned on the sampled outcome**. Knockout draws resolve by renormalising the
   two win probabilities (ET/pens ≈ strength-proportional); refinement deferred.
8. **WC2026 format spec.** `format.yaml` declares: 12 groups of 4, single round-robin; group
   ranking by points → goal difference → goals for → head-to-head → random (fair-play points are
   not simulatable — documented approximation); best-thirds ranking across groups by points → GD
   → GF → random; advancement of 12 winners + 12 runners-up + 8 best thirds into a **fixed
   bracket** using FIFA's published third-place allocation table (shipped in the pack — required
   for rewind backtests even though the live bracket is already determined); then single-elim
   knockout R32 → R16 → QF → SF → Final (third-place match simulated but cosmetic).
9. **Simulation.** Default 100k runs (configurable to 1M), NumPy-vectorised where possible, fully
   seedable. Completed matches are facts; only remaining matches are sampled. Report Monte Carlo
   standard error (√(p(1−p)/N)) alongside every probability, and label outputs with the as-of
   timestamp. UI copy must note that MC error excludes model uncertainty.
10. **Probability-over-time is a first-class output.** Every simulation run appends to a
    `prob_history` table (team, round-reach probs, as_of, model version). The dashboard charts
    champion probability evolving through the tournament — the killer feature of launching live.
11. **Data sources (v1 pack):**
    - **Backbone:** the open "International football results 1872–present" dataset (martj42
      GitHub/Kaggle CSV) — results, competition, neutral flag. CSV adapter.
    - **Live 2026 state:** football-data.org API (free tier covers the World Cup) with a
      lazy-refresh-on-read pattern and rate-limit respect, **plus a manual CSV fallback adapter**
      so the tool never blocks on a third-party API.
    - **Optional enrichment:** FIFA rankings (public scraped datasets) as a feature; StatsBomb
      open event data (free for WC 2018/2022) for xG-based features in backtests. Both behind
      feature-availability masks — models must train and predict cleanly when extras are absent.
    - Cut from v1: betting odds, weather, lineups, market values.
12. **Evaluation protocol.**
    - Match level: log loss (primary), Brier, calibration curves + ECE, on walk-forward splits.
    - Tournament level: freeze data before kickoff of WC 2014, 2018, 2022 (and rewind-2026),
      predict the full tournament, score champion/round-reach probabilities (Brier on realised
      outcomes) vs. the Rung-0 baseline. A rung ships only if it beats the rung below out-of-time.
13. **Stack.** Python 3.12+, uv, pandas + NumPy (Polars optional later), scikit-learn, LightGBM,
    Optuna, MLflow (local tracking), FastAPI + Pydantic v2, Streamlit + Plotly, pytest, **ruff for
    lint and format** (drops Black — one tool; revisit if you object), mypy, Docker, GitHub
    Actions. XGBoost/CatBoost/RandomForest deferred to a post-v1 model bake-off.
14. **Reproducibility.** Fixed seeds threaded through ingestion order, training, and simulation;
    config via YAML in `configs/`; structured logging (JSON) + tqdm progress bars; deterministic
    pipelines re-runnable end-to-end with one command.

---

## 3. Repository layout

```
predictor-engine/
    src/engine/
        core/           # schema.py (Match, Team), registry.py (aliases), config.py
        ingestion/      # base adapter protocol, csv_adapter, football_data_org adapter
        ratings/        # elo.py, attack_defence.py, form.py (chronological updaters)
        features/       # builders + availability masks + as-of feature store
        models/         # ladder (rung0/rung1/rung2), calibration, persistence/versioning
        simulation/     # format interpreter (groups, knockout), monte_carlo.py, state.py
        explain/        # SHAP for rung 2, coefficient/rating-diff views for rungs 0–1
        api/            # FastAPI app
        dashboard/      # Streamlit app
        evaluation/     # walk-forward CV, tournament backtests, metrics
        utils/
    packs/
        world_cup_2026/
            format.yaml
            sources.yaml
            teams.yaml
            bracket_allocation.yaml   # FIFA third-place allocation table
    data/               # raw/ (immutable), processed/ (derived, regenerable)
    configs/            # engine defaults, model configs, simulation configs
    tests/
    notebooks/          # exploration only; nothing production depends on a notebook
    docker/
    docs/
    SCOPE.md  CLAUDE.md  STATUS.md
```

CLI surface (thin wrapper, Typer or argparse):
```
engine ingest   --pack packs/world_cup_2026            # pull/refresh all sources
engine ratings  --pack ...                             # rebuild chronological ratings
engine train    --pack ... --rung 2                    # train + calibrate + log to MLflow
engine simulate --pack ... --as-of now --runs 100000   # -> champion/round-reach table
engine backtest --pack ... --freeze 2026-06-10         # rewind evaluation
engine api / engine dashboard
```

---

## 4. Build phases

**P0 — Skeleton + data backbone.** Repo scaffold (uv, ruff, mypy, pytest, CI), canonical schema,
team registry, CSV adapter for the historical results backbone, raw→processed pipeline.
*Done when:* full history loads to canonical matches; alias collisions surfaced; tests green.

**P1 — Ratings + Rung 0 + calibration.** Chronological Elo / attack-defence / form updaters;
rating-only logistic; calibration layer; walk-forward evaluation harness; backtests vs. 2014/18/22.
*Done when:* Rung 0 beats a naive always-draw/frequency baseline on held-out log loss and is
well-calibrated (ECE reported).

**P2 — Format interpreter + Monte Carlo + live state.** `format.yaml` interpreter (groups with
full tiebreaker chain incl. best-thirds + allocation table; fixed knockout), conditional scoreline
sampler, live ingestion (football-data.org + CSV fallback), simulate-from-now, prob_history table.
*Done when:* `engine simulate --as-of now` prints a champion-probability table for the live
tournament with MC error bars, and `--freeze 2026-06-10` reproduces a pre-tournament run.
**This is the MVP milestone — everything after upgrades quality, not capability.**

**P3 — Feature engineering + ML rungs.** Feature store (form windows, rating diffs, rest days,
WC experience, home/host advantage, optional FIFA-rank and xG features behind masks), Rung 1,
Rung 2 (LightGBM + Optuna), MLflow tracking, model versioning, SHAP explanations,
promote-only-if-better gate vs. Rung 0.

**P4 — API + dashboard.** FastAPI: `/teams`, `/ratings`, `/predict-match`, `/simulate`,
`/champion-probabilities`, `/explain/{match_id}`. Streamlit: live bracket with probabilities,
champion-prob table + **probability-over-time chart**, match explainer, calibration page,
backtest report page, CSV download.

**P5 — Hardening.** Coverage push (unit + integration + simulation determinism tests + API
tests), Docker images (api, dashboard), docs (architecture, methodology incl. Elo/calibration
math, API reference), README worthy of an open-source landing page.

Tests and STATUS.md updates happen in every phase, not just P5.

---

## 5. Key risks & honest caveats

- **Entity resolution is the grind.** Budget real time for the alias registry; it is the #1
  source of silent data corruption in multi-source football data.
- **football-data.org free tier is rate-limited.** Lazy refresh + on-disk cache + CSV fallback
  keeps the tool functional regardless (same pattern as NasSpot's lazy-refresh-on-read).
- **International xG coverage is patchy.** Treat all extras as optional; never let a model rung
  hard-depend on them.
- **Fair-play tiebreaker and penalty shootouts are approximated** (random / strength-proportional).
  Documented in methodology; immaterial at 100k-run resolution for headline probabilities.
- **MC error ≠ model error.** Surface both statements; do not imply false precision. Tournament
  outcomes are single draws from fat-tailed distributions — the backtest protocol (§2.12) is the
  only defensible quality claim.

---

## 6. Deferred (post-v1, in rough order)

1. Second pack: **EPL** (adds `round_robin` season type, two legs, points-table champion) — the
   abstraction stress-test.
2. Third pack: a data-poor league (**Ghana Premier League** fits) — forces ladder auto-selection
   by effective sample size and uncertainty-aware language.
3. **Glicko-2** ratings (rating deviation is the right uncertainty mechanism for sparse packs).
4. Dixon-Coles / bivariate Poisson as a goals-first model family (needed when league tiebreakers
   dominate; validates against the classifier).
5. Plugin/pack discovery via entry points; community pack contribution guide.
6. Model bake-off: XGBoost, CatBoost, RF, stacking ensembles.
7. Odds ingestion + market-comparison page; player-level and injury impact; live in-match probs.
8. Kubernetes/Terraform; GPU training. (Docker + Railway/Vercel-style deploy is sufficient for v1.)

---

## 7. Open questions (answer before or during P0)

1. **Repo/product name** — `predictor-engine` is a placeholder. Your call.
2. **football-data.org API key** — register the free tier, or start CSV-fallback-only for P2?
3. **xG enrichment** — pull StatsBomb open data for 2018/2022 backtests in P3, or cut entirely?
4. **License** — MIT vs. Apache-2.0 for the open-source release.

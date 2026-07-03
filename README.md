# predictor-engine

[![CI](https://github.com/Heights001/Kickstart/actions/workflows/ci.yml/badge.svg)](https://github.com/Heights001/Kickstart/actions/workflows/ci.yml)

**A competition-agnostic football prediction engine, launched live during the
2026 World Cup.** Competitions plug in as *packs* (YAML + data adapters); the
engine builds chronological ratings, trains a calibrated match-outcome model
ladder, and Monte-Carlo-simulates the competition format to answer the headline
question: **who wins, with what probability?**

```
team                      champion     ±SE   final      SF      QF
france                      0.2076  0.0013  0.3539  0.5678  0.7649
argentina                   0.1939  0.0013  0.3398  0.5745  0.8211
spain                       0.1187  0.0010  0.2221  0.3791  0.5724
brazil                      0.1142  0.0010  0.1926  0.3530  0.6485
mexico                      0.0992  0.0009  0.1835  0.3358  0.6005
```
*Live WC2026 champion probabilities, 100k runs, LightGBM rung, as of 2026-07-02.*

## Why it's interesting

- **Live-state simulation.** Played matches are facts; only the remaining
  bracket is sampled. Rewind to any date (`--freeze 2026-06-10`) and the engine
  reproduces a pre-tournament forecast — the group stage becomes a free backtest.
- **Honest modelling.** Strict chronological integrity (no feature ever sees its
  own match), walk-forward evaluation only, and a promote-only-if-better model
  ladder: a fancier model ships only if it beats the shipped one on held-out log
  loss *and* calibration. Rung 1 failed that gate; rung 2 passed it.
- **The format is data.** Groups, tiebreaker chains, best-thirds allocation
  (FIFA's full 495-combination Annex C table) and the knockout tree live in pack
  YAML — a generic interpreter simulates them. Adding a league or cup means
  writing a pack folder, not engine code.
- **Uncertainty is labelled.** Every probability ships with its Monte Carlo
  standard error and an as-of timestamp; MC error explicitly excludes model
  uncertainty.

## Quick start

```bash
uv sync
uv run engine ingest   --pack packs/world_cup_2026    # ~50k matches, 1872-present
uv run engine ratings  --pack packs/world_cup_2026    # chronological Elo/form/attack-defence
uv run engine train    --pack packs/world_cup_2026 --rung 2   # evaluate + promotion gate
uv run engine simulate --pack packs/world_cup_2026    # champion table, 100k seeded runs
uv run engine api      --pack packs/world_cup_2026    # FastAPI service (docs at /docs)
uv run engine dashboard --pack packs/world_cup_2026   # Streamlit: prob-over-time chart
```

During a live tournament: `engine ingest --refresh` pulls new results into
date-stamped raw files, then re-run `ratings` and `simulate` — every run appends
to the probability history the dashboard charts.

Docker images for the API and dashboard live in [docker/](docker/).

## How it works

Raw results (the open [martj42 international results](https://github.com/martj42/international_results)
dataset, 1872–present, plus its shootout companion) are ingested through a team
registry with alias resolution — unresolved names fail loudly with suggestions.
A single chronological walk produces pre-match Elo (importance-tiered K,
goal-margin multiplier, neutral-aware), attack/defence, form, rest and experience
features. The model ladder (rating-only logistic → feature LR → LightGBM+Optuna)
is calibrated per-rung (isotonic/Platt picked by ECE) and gated by walk-forward
backtests against the 2014/18/22 World Cups. The Monte Carlo layer samples
outcomes from the calibrated classifier and scorelines from a conditioned
double-Poisson grid, interprets the full competition format (tiebreakers,
best-thirds allocation, fixed bracket), and reports round-reach probabilities
with standard errors.

Details: [docs/architecture.md](docs/architecture.md) ·
[docs/methodology.md](docs/methodology.md) · [docs/api.md](docs/api.md)

## Project state

v1 targets exactly one pack — the FIFA World Cup 2026, live now. Phases P0–P5
(scaffold → data backbone → ratings/baseline → simulation MVP → ML ladder →
API/dashboard → hardening) are complete; see [SCOPE.md](SCOPE.md) for the design
and [STATUS.md](STATUS.md) for current state. Post-v1 roadmap: an EPL pack (the
abstraction stress test), a data-poor league pack, Glicko-2, Dixon-Coles.

## Development

Python 3.12+, managed by [uv](https://docs.astral.sh/uv/). Four gates must pass:

```bash
uv run pytest && uv run ruff check . && uv run ruff format --check . && uv run mypy src/
```

License: TBD (pre-release).

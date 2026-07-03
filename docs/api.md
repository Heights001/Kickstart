# API reference

Start the service: `uv run engine api --pack packs/world_cup_2026` (default
`127.0.0.1:8000`). Startup fits the promoted rung as-of the data horizon — with
rung 2 that includes the Optuna search, so allow a couple of minutes. Interactive
OpenAPI docs at `/docs`.

Team-name parameters accept any registry alias (`USMNT`, `Korea Republic`, …).
Unresolved names return **404** with the suggested-alias report.

| Endpoint | Params | Returns |
|---|---|---|
| `GET /teams` | — | The pack's 48 teams: id, name, aliases |
| `GET /ratings` | — | As-of Elo/form/attack/defence per team, Elo-descending |
| `GET /predict-match` | `home`, `away` | Calibrated P(home/draw/away); `neutral` reflects the host-venue rule |
| `GET /simulate` | `runs` (100–1,000,000, default 10,000) | Fresh Monte Carlo: champion + round-reach probabilities with MC standard error |
| `GET /champion-probabilities` | — | Latest `prob_history` generation (table) plus the full history for charting |
| `GET /explain` | `home`, `away` | Feature values + per-feature contributions toward the home-win class (SHAP for rung 2, coefficient×value for rungs 0–1; log-odds scale, uncalibrated) |

Notes:

- `/simulate` runs at the service's as-of date only; rewind simulations
  (`--freeze`) are CLI territory.
- Every probability endpoint's `note`/`model` fields state the model version and
  that MC error excludes model uncertainty.
- SCOPE §4 lists `/explain/{match_id}`; canonical matches carry no id scheme yet,
  so v1 explains a pairing instead.

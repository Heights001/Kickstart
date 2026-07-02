# STATUS.md

**Last updated:** 2026-07-02 (end of P4 session)
**Tournament context:** WC2026 live — R32 in progress, final July 19. MVP + promoted
model + API/dashboard all shipped with 17 days of tournament left.

## Current phase

**P4 — complete.** FastAPI service and Streamlit dashboard over the promoted rung-2
model; all four gates green, 114 tests (incl. 10 API tests via TestClient).

- **API** (`engine api --pack packs/world_cup_2026`): `/teams`, `/ratings`,
  `/predict-match`, `/simulate` (runs param, capped), `/champion-probabilities`
  (latest prob_history generation + full history), `/explain`. App context (data,
  registry, promoted-rung sampler fitted as-of the data horizon) is built once in the
  lifespan; unresolved team names return 404 carrying the suggested-alias report.
- **Dashboard** (`engine dashboard --pack packs/world_cup_2026`): champion-odds table
  with CSV download, **probability-over-time Plotly chart** over prob_history,
  tournament-state tab (group results + knockout facts), match explainer (probs +
  feature bar), model report (promotion registry + rung-0 backtest artifacts).

## Decisions made this session

- **`/explain` takes a pairing, not a match id** (SCOPE §4 deviation): canonical
  matches carry no ids yet. Explanations: SHAP class-0 row for rung 2,
  coefficient×value contributions for rungs 0–1 (log-odds scale, uncalibrated).
- API `/simulate` runs at the context's as-of only; rewinds stay CLI territory.
- Dashboard is read-only over processed artifacts except the explainer, which builds
  the cached sampler on first use (rung-2 tuning wait, once per session).
- httpx2 added as a dev dependency (starlette TestClient requirement).

## Honest caveats

- API startup ≈ rung-2 Optuna tuning time (~1–2 min); fine for a long-lived process.
- prob_history mixes model versions (rung0 rows from P2, rung2 rows now); the chart
  carries model in the hover, but a proper per-model filter is P5 polish.
- The bracket tab lists state rather than drawing a visual bracket — P5.
- Earlier caveats stand (Optuna/calibration window overlap, wc2014 calibration,
  attack/defence drift, data lag ~1–2 days).

## Next actions

1. **Daily during knockouts:** `engine ingest --refresh` → `engine ratings` →
   `engine simulate` to grow the probability-over-time chart.
2. P5 hardening: coverage push, Docker images (api, dashboard), docs + README.
3. Open questions (repo name, football-data.org key, license) still with Daniel —
   license blocks any public release.

## Reopened decisions

None.

## Blocked / waiting

Nothing.

# STATUS.md

**Last updated:** 2026-07-02 (end of P5 session)
**Tournament context:** WC2026 live — R32 finishing, final July 19.

## Current phase

**P5 — complete. All SCOPE v1 phases (P0–P5) are done.**

- **Coverage:** 82% → **97%** (pytest-cov; Streamlit entrypoint excluded as UI).
  New end-to-end CLI integration suite drives ingest → ratings → train (rung 0
  gate PASS on strength-ordered synthetic data) → simulate (prob_history append)
  → train rung 1 (promotion decision + sandboxed MLflow sqlite) through
  `engine.cli.main`. 119 tests total.
- **Docker:** `docker/Dockerfile.api`, `docker/Dockerfile.dashboard` (uv
  multi-stage, libgomp for LightGBM), `docker/compose.yaml` mounting host
  `data/`. Both images build locally (1.8GB each; first build ~35 min cold,
  dashboard reuses the shared dependency layers) and the in-container CLI
  entrypoint smoke-tests clean.
- **Docs:** `docs/architecture.md` (engine/pack split, data flow, invariants),
  `docs/methodology.md` (Elo/attack-defence/form math, ladder, calibration,
  promotion gate, simulation, documented approximations), `docs/api.md`.
- **README:** landing page with CI badge, live results table, quick start,
  methodology summary, roadmap. **License deliberately "TBD"** — open question
  §7.4 is Daniel's call and blocks public release.

## Decisions made this session

- Dashboard entrypoint excluded from coverage measurement (manual/UI surface).
- Docker images expect processed `data/` mounted from outside; ingestion is not
  baked into images.

## Honest caveats

- CI does not build the Docker images (local-verified only); add a docker build
  job if images become a release artifact.
- `engine api`/`engine dashboard` subprocess launch paths are the residual
  uncovered CLI lines.
- Earlier caveats stand (Optuna/calibration overlap, wc2014 calibration,
  attack/defence drift, backbone data lag).

## Next actions

1. **Daily tournament loop:** `engine ingest --refresh` → `engine ratings` →
   `engine simulate` to feed the probability-over-time chart through July 19.
2. Answer SCOPE §7 open questions — **license** (blocks release), repo/product
   name, football-data.org key (P2 live adapter slot is ready), xG enrichment.
3. Post-v1 (SCOPE §6): EPL pack as the abstraction stress test.

## Reopened decisions

None.

## Blocked / waiting

Public release blocked only on the license decision (§7.4).

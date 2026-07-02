# STATUS.md

**Last updated:** 2026-07-01 (end of P0 session)
**Tournament context:** WC2026 live — group stage complete, Round of 32 in progress. Final is
July 19. P2 (MVP milestone) before the final is the target that makes this tournament usable
end-to-end; earlier is better for the probability-over-time chart.

## Current phase

**P0 — complete.** Full backbone loads to canonical matches; all four gates green.

- `uv run engine ingest --pack packs/world_cup_2026` ingests **49,484 matches
  (1872-11-30 → 2026-06-30)**, 12 rows skipped for missing scores, 288 names
  auto-registered from the authoritative backbone (all non-FIFA entities — micronations,
  regions), zero alias collisions.
- Shipped: uv/ruff/mypy/pytest scaffold + CI workflow (four gates); canonical `Match`/`Team`
  schema (`src/engine/core/schema.py`); team registry with alias resolution, suggested-alias
  reports, and collision detection (`src/engine/core/registry.py`); adapter protocol + CSV
  adapter + ingest pipeline (`src/engine/ingestion/`); `engine ingest` CLI; `world_cup_2026`
  pack (`teams.yaml` with the 48 qualified teams + aliases, `sources.yaml`); 33 tests.

## Decisions made this session

- **Authoritative-source registration (approved by Daniel):** rule 4 stays strict for all
  sources except one marked `authoritative_names: true` in `sources.yaml` (the martj42
  backbone). Unknown names from that source are explicitly registered as canonical teams —
  collision-checked against existing aliases (collisions still raise) and listed in the
  ingest report. Rationale: the backbone has ~330 teams; teams.yaml seeds only the 48
  qualified, but P1 Elo needs every team's history. The P2 live feed stays strict-raise.
- stdlib `csv` (not pandas) for the P0 adapter; pandas enters in P1 when ratings need it.
- argparse (not Typer) for the CLI; processed output is JSONL (`data/processed/matches.jsonl`).
- No `configs/` yet — P0 has no tunable numbers; K-factors etc. arrive in P1.
- Plain commit messages — no Co-Authored-By trailer (Daniel's preference).
- 48-team list verified against post-playoff sources on 2026-07-01 (canonical names follow
  the martj42 dataset; FIFA-style variants are aliases).

## Next actions

1. Answer remaining SCOPE.md §7 open questions (repo name, football-data.org key, xG in/out,
   license).
2. P1: chronological Elo / attack-defence / form updaters (`src/engine/ratings/`) with
   config-driven K-factors under `configs/`; hand-computed-fixture tests; neutral-venue flag.
3. P1: Rung 0 rating-only logistic + calibration layer + walk-forward evaluation harness.

## Reopened decisions

None.

## Blocked / waiting

Nothing. Local `main` is 5 commits ahead of `origin/main` — push when ready.

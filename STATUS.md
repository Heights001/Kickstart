# STATUS.md

**Last updated:** 2026-07-02 (end of P2 session)
**Tournament context:** WC2026 live — R32 in progress, final July 19. **The MVP milestone
(P2) shipped 17 days before the final.**

## Current phase

**P2 — complete (MVP).** Both done-when criteria hold, all four gates green, 92 tests.

- `uv run engine simulate --pack packs/world_cup_2026` (as-of now, 100k runs, 4.4s):
  champion table with MC standard errors. Headline: **Argentina 28.0% ±0.14, France 22.4%,
  Spain 18.3%, Mexico 9.2%, Brazil 6.2%**. Facts corroborate reality — eliminated teams
  (South Korea, Saudi Arabia, …) at exactly 0.0; R32 losers (Germany, Netherlands, Japan,
  Ivory Coast) match the real shootout/knockout results.
- `--freeze 2026-06-10` (100k runs, 13.6s) reproduces a pre-tournament run from zero facts:
  **Spain 29.1%, Argentina 19.8%, France 10.3%** ante post.
- Every run appends per-team round-reach probabilities to
  `data/processed/prob_history.jsonl` (SCOPE §2.10).

Shipped this session:
- **Pack format data:** `format.yaml` (real draw, tiebreaker chains per SCOPE §2.8, hosts,
  full knockout template matches 73–104) and `bracket_allocation.yaml` — FIFA's complete
  Annex C (495 combinations), sourced from a machine-readable transcription independently
  verified against FIFA's regulations PDF (dw-football/wc2026-bracket). The live
  combination (BDEFGJKL) reproduces the actual R32 pairings (Germany–Paraguay,
  France–Sweden), confirming table + template end-to-end.
- **Shootout handling:** the backbone records 90-minute scores, so martj42's
  `shootouts.csv` is a second pack source; `Match.winner` (optional) is annotated at
  ingest (642 drawn matches across history). Drawn knockout facts without a winner raise.
- **Format interpreter** (`src/engine/simulation/`): group ranking
  (points → GD → GF → head-to-head → seeded random), best-thirds ranking, Annex C lookup,
  slot grammar (`1A`/`2B`/`T1E`/`W74`/`L101`), live-state builder (group facts = same-group
  first meetings; later meetings go to the knockout facts pool).
- **Match sampler:** Rung 0 + calibrator fitted strictly pre-as-of; venue-aware probs
  (host ⇒ home advantage, else neutral; rest-diff 0); conditional double-Poisson
  scorelines (λ clamped to config bounds); knockout draws renormalised.
- **Monte Carlo:** fully seeded, group fixtures vectorised across runs, completed
  stable groups ranked once, unused-fact integrity check, MC SE on every probability.
  ~23k runs/s live, ~7.4k runs/s full rewind.
- `engine ingest --refresh`: date-stamped raw downloads (raw/ stays append-only) for
  updating results through the final.

## Decisions made this session

- **Live state is CSV-only for now** (SCOPE §2.11 partial deviation): open question #2
  (football-data.org key) is still unanswered, and the backbone CSV already carries
  results through 2026-06-30 with regular updates. The API adapter can slot in behind
  the same interface later; nothing blocks on it.
- Static ratings within a simulation run (no in-run Elo updates).
- Venue approximation (hosts get home advantage; host-vs-host treated neutral) and
  rest-diff 0 for simulated matches — documented in format.yaml.
- λ clamped to [0.2, 3.5] because of the attack/defence scale drift (P1 caveat); this
  only affects group tiebreak resolution, not match winners.

## Honest caveats

- Backbone data lags ~1–2 days (currently through 6/30): matches played since are
  simulated, not facts. `engine ingest --refresh` + `engine ratings` before simulating
  picks up new results when martj42 updates.
- Fair-play tiebreaker approximated by seeded random (SCOPE-documented); shootout model
  is strength-proportional renormalisation.
- wc2014 calibration caveat from P1 still stands; attack/defence normalisation still
  worth revisiting (feeds scoreline sampler now).

## Next actions

1. P3: feature store + Rung 1 (feature LR) + Rung 2 (LightGBM+Optuna) with MLflow
   tracking and the promote-only-if-better gate.
2. Or pull forward P4's probability-over-time chart — prob_history already accumulates,
   and the tournament ends July 19 (the killer feature has a hard deadline).
3. Answer SCOPE §7 open questions (repo name, football-data.org key, xG, license).

## Reopened decisions

None.

## Blocked / waiting

Nothing.

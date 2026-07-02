# STATUS.md

**Last updated:** 2026-07-02 (end of P1 session)
**Tournament context:** WC2026 live — group stage complete, Round of 32 in progress. Final is
July 19. P2 (MVP milestone) before the final is the target that makes this tournament usable
end-to-end; earlier is better for the probability-over-time chart.

## Current phase

**P1 — complete.** Rung 0 beats the frequency baseline on held-out log loss in all three
backtest windows; ECE reported; all four gates green.

`uv run engine train --pack packs/world_cup_2026 --rung 0` (walk-forward, isotonic
calibration chosen on every window):

| window | n_test | log loss | baseline ll | Brier | ECE | baseline ECE |
|--------|-------:|---------:|------------:|------:|----:|-------------:|
| wc2014 | 64 | 0.9100 | 1.0594 | 0.5387 | 0.1541 | 0.0424 |
| wc2018 | 64 | 0.9828 | 1.0949 | 0.5815 | 0.0683 | 0.1032 |
| wc2022 | 64 | 1.0712 | 1.0744 | 0.6186 | 0.0833 | 0.0544 |

Shipped this session:
- Config layer: `configs/default.yaml` + typed loaders (`src/engine/core/config.py`); pack
  gains `competitions.yaml` (tournament → importance tier) and `backtests.yaml` (WC windows).
- Chronological ratings (`src/engine/ratings/`): Elo (K by tier, goal-margin multiplier,
  home advantage suppressed on neutral), multiplicative attack/defence, EW form, and the
  as-of feature store (pre-match snapshots; out-of-order matches raise). `engine ratings`
  writes `features.jsonl` + `ratings.json` (49,484 rows, 336 teams).
- Models + evaluation: OutcomeModel protocol, frequency baseline, Rung 0 multinomial LR on
  [elo_diff, neutral, rest_diff], isotonic/Platt/identity calibration picked by ECE on a
  walk-forward validation window, metrics (log loss / Brier / top-label ECE), and
  `engine train --rung 0` printing the gate table + writing `eval_rung0.json`.

## Decisions made this session

- Two pack files beyond SCOPE §3's listing: `competitions.yaml` and `backtests.yaml`
  (competition facts belong in pack YAML per rule 1).
- numpy + scikit-learn added; still no pandas (nothing needed it). MLflow deferred to P3 —
  P1's rung-vs-baseline comparison is recorded in the printed report + JSON artifact.
- Final Elo table sanity-checked (Argentina/Spain/France top-3 as of 2026-07-01).

## Honest caveats

- **wc2014 calibration:** model ECE 0.154 vs baseline 0.042. A constant-frequency predictor
  is trivially well-calibrated, so the gate (log loss) still passes cleanly, but Rung 0's
  calibration is only moderate on that window. Watch it when Rung 1 arrives.
- Attack/defence ratings drift off the 1.0-centred scale over 150 years (self-consistent,
  but revisit normalisation before the P2 scoreline sampler consumes them).

## Next actions

1. P2: `format.yaml` interpreter (12 groups of 4, full tiebreaker chain incl. best-thirds +
   FIFA allocation table) + `bracket_allocation.yaml` in the pack.
2. P2: conditional scoreline sampler + Monte Carlo simulator (`engine simulate --as-of now`).
3. P2: live-state ingestion (football-data.org + CSV fallback) and `prob_history` table.

## Reopened decisions

None.

## Blocked / waiting

Nothing.

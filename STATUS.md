# STATUS.md

**Last updated:** 2026-07-01 (pre-implementation)
**Tournament context:** WC2026 live — group stage complete, Round of 32 in progress. Final is
July 19. P2 (MVP milestone) before the final is the target that makes this tournament usable
end-to-end; earlier is better for the probability-over-time chart.

## Current phase

P0 — not started. No code exists yet.

## Locked this session (see SCOPE.md §2 for full detail)

- Engine/pack architecture; v1 ships only the `world_cup_2026` pack. EPL + data-poor-league
  packs deferred to post-v1.
- Model ladder (rating-logistic → feature LR → LightGBM+Optuna) with calibration on every rung
  and a promote-only-if-better gate.
- Conditional scoreline sampler for group tiebreakers; renormalised win probs for knockout draws.
- Data: martj42 historical results backbone + football-data.org live feed with CSV fallback;
  own in-engine Elo (no scraped ratings).
- Live-state simulation and rewind backtesting (freeze 2026-06-10) are core MVP.
- Ruff replaces Black (lint + format in one tool).

## Next actions

1. Answer SCOPE.md §7 open questions (name, football-data.org key, xG in/out, license).
2. P0: scaffold repo with uv + ruff + mypy + pytest + CI skeleton; commit SCOPE/CLAUDE/STATUS.
3. P0: implement `Match`/`Team` schema, team registry, CSV adapter; load the historical
   backbone end-to-end with alias-collision report.

## Reopened decisions

None.

## Blocked / waiting

Nothing — P0 has no external dependencies (backbone dataset is a public CSV).

# Architecture

## Engine / pack split

The engine (`src/engine/`) is competition-agnostic and **never imports pack code**.
A pack is a folder of YAML (plus, later, optional adapter code loaded by path)
declaring everything competition-specific:

```
packs/world_cup_2026/
    teams.yaml               # registry seed: canonical ids + aliases
    sources.yaml             # data sources, column mappings, name policy
    competitions.yaml        # tournament name -> importance tier
    backtests.yaml           # walk-forward evaluation windows
    format.yaml              # groups, tiebreakers, knockout template, hosts
    bracket_allocation.yaml  # FIFA Annex C: 495 third-place combinations
```

If engine code needs a competition fact, that fact moves into pack YAML — never
into engine code.

## Data flow

```
engine ingest    sources (CSV) ──▶ data/raw/ (append-only) ──▶ canonical Match
                 records (names resolved through the team registry)
                 ──▶ data/processed/matches.jsonl

engine ratings   chronological walk over matches ──▶ pre-match feature snapshots
                 (Elo, attack/defence, form, rest, experience)
                 ──▶ features.jsonl + ratings.json

engine train     walk-forward evaluation of a model rung vs the shipped rung,
                 calibration on a pre-test window, promotion gate, MLflow log
                 ──▶ model_registry.json (which rung simulation uses)

engine simulate  live state (facts) + promoted rung fitted strictly pre-as-of
                 ──▶ Monte Carlo over the format ──▶ champion/round-reach table
                 ──▶ prob_history.jsonl (append per run)

engine api / dashboard   serve the same artifacts + sampler
```

## Invariants (CLAUDE.md hard rules)

1. **Canonical schema only** downstream of ingestion: `Match`/`Team` Pydantic
   models; source quirks die inside adapters.
2. **Chronological integrity**: every rating/feature is computed strictly before
   the match it describes; walkers raise on out-of-order input; models and
   calibrators are fitted only on pre-`as_of` data and are therefore retrained per
   as-of rather than persisted.
3. **Registry resolution**: team names resolve through the alias registry;
   unresolved names raise with a suggested-alias report. One declared-authoritative
   source may register new canonical teams, explicitly and collision-checked.
4. **Determinism**: one seed from config threads through training, calibration,
   tiebreaks, and simulation; same config + data = same output.
5. **Promotion gate**: a rung ships only if it beats the currently shipped rung on
   mean held-out log loss with no-worse mean ECE, recorded in MLflow.

# CLAUDE.md — Working agreements for this repo

Read SCOPE.md first (design + locked decisions), then STATUS.md (current state + next actions).
Do not re-litigate locked decisions in SCOPE.md §2 unless STATUS.md flags one as reopened.

## Commands

```
uv sync                          # install/refresh environment
uv run pytest                    # full test suite
uv run pytest -k <expr> -x       # focused
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/
uv run engine <subcommand>       # CLI (see SCOPE.md §3)
```

All four gates (pytest, ruff check, ruff format --check, mypy) must pass before any commit.
CI runs the same gates — never rely on CI to catch what you didn't run locally.

## Architecture rules (hard)

1. **Engine/pack boundary:** nothing under `src/engine/` imports from `packs/`. Packs are data
   (YAML) plus optional adapter code loaded dynamically by path. If engine code needs to know a
   competition-specific fact, that fact belongs in the pack's YAML.
2. **Canonical schema only:** downstream of ingestion, code consumes `Match`/`Team` models only.
   Source-specific quirks die inside adapters.
3. **Chronological integrity:** any function computing a rating or feature takes an `as_of`
   date and must not read matches on/after it. Walk-forward splits only. A leaked feature is a
   P0 bug regardless of metric improvement.
4. **Team names resolve through the registry.** Never string-match team names ad hoc. Unresolved
   names raise with a suggested-alias report; do not auto-create teams.
5. **Determinism:** every stochastic path takes a seed from config. Same config + same data =
   same output, bit-for-bit where feasible.
6. **Model promotion gate:** a higher rung ships only if it beats the rung below on held-out
   log loss AND is at least as well calibrated (ECE). Record the comparison in MLflow.
7. **Probabilities are floats; no money in this codebase** — the Decimal convention from the
   fintech repos does not apply here.

## Style

- Python 3.12+, full type annotations, mypy-clean. Pydantic v2 for all boundary objects.
- Small modules, single responsibility, dependency injection over globals. No duplicated logic.
- Structured JSON logging via the shared logger in `utils/`; tqdm for long loops.
- Config in YAML under `configs/`; no magic numbers in code (Elo K-factors, sim run counts,
  calibration windows all live in config).
- `data/raw/` is immutable (append-only downloads); everything in `data/processed/` must be
  regenerable by `engine ingest` + `engine ratings`.
- Notebooks are for exploration only; production code paths never import from `notebooks/`.

## Testing expectations

- Unit tests alongside each module; golden-file tests for the format interpreter (tiny synthetic
  4-team tournaments with hand-computed tiebreaker outcomes).
- Simulation tests: fixed seed → exact expected output; probability sums ≈ 1; a team already
  eliminated in live state must show 0.0 champion probability.
- Rating tests: hand-computed Elo updates for known fixtures; neutral-venue flag respected.
- API tests via FastAPI TestClient.

## Session protocol

- End every session by updating STATUS.md: what changed, decisions made, next 1–3 actions,
  anything blocked.
- If a locked decision proves wrong in practice, stop, write the issue + proposed change in
  STATUS.md under "Reopened decisions", and confirm direction with Daniel before implementing.
- Prefer completing one phase gate over starting the next phase.

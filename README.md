# predictor-engine

Competition-agnostic football prediction engine. Competitions plug in as **packs**
(YAML config + data adapters); the engine trains calibrated match-outcome models and
Monte-Carlo-simulates the competition format to output championship probabilities.

v1 ships one pack: **FIFA World Cup 2026**.

See [SCOPE.md](SCOPE.md) for design and locked decisions, [STATUS.md](STATUS.md) for
current state, and [CLAUDE.md](CLAUDE.md) for working agreements.

## Quick start

```
uv sync
uv run engine ingest --pack packs/world_cup_2026
uv run pytest
```

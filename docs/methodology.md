# Methodology

## Ratings

### Elo

Ratings start at 1500. For a match with home rating `R_h`, away rating `R_a`:

```
W_e = 1 / (1 + 10^((R_a − R_h − H·[not neutral]) / 400))
R_h' = R_h + K · G · (W − W_e)          (zero-sum: away gets −Δ)
```

- `W` is the home result (1 / 0.5 / 0), `W_e` the expected score.
- `H` (config: 80) is home advantage, suppressed entirely on neutral venues.
- `K` scales with competition importance tier, mapped per-pack
  (`competitions.yaml`) with values in engine config: World Cup 60,
  continental 50, qualifiers 40, Nations League/other 30, friendlies 20.
- `G` is the eloratings.net goal-margin multiplier: 1.0 for a 1-goal margin,
  1.5 for 2, 1.75 for 3, +0.125 per further goal.

### Attack / defence

Multiplicative rates around a slowly adapting global goals-per-side mean `mu`:
expected home goals = `mu · attack_h · defence_a` (defence > 1 = leaky). Each
match nudges the four involved ratings toward the rates implied by the observed
score (EW step `alpha`), using pre-match values throughout. These feed the
scoreline sampler, not the outcome classifier.

### Form

Exponentially weighted result score on [0, 1]: `f' = d·f + (1−d)·result`
with decay `d = 0.85`; new teams start at 0.5.

All updaters run in one chronological walk that snapshots features **before**
seeing each match's result — a match can never leak its own outcome into its
features, and out-of-order input raises.

## Model ladder

Each rung is a drop-in 3-way classifier P(home/draw/away):

- **Rung 0** — multinomial logistic on [elo_diff, neutral, rest_diff]. The
  permanent baseline.
- **Rung 1** — scaled logistic on the full feature set (elo/form/attack/defence
  diffs, rest, neutral, log-scaled total and same-tier experience diffs).
- **Rung 2** — LightGBM multiclass; hyperparameters from a seeded Optuna TPE
  search validated on the calibration window.

### Calibration

For each rung, isotonic and Platt calibrators are fitted one-vs-rest per class
on a walk-forward validation window (the `calibration.window_years` before the
test window), renormalised row-wise, and the best of {identity, isotonic, Platt}
is picked by top-label ECE (Brier tie-break).

### Promotion gate

A rung ships only if, across all backtest windows (train < freeze; test on the
tournament's matches), it has **lower mean log loss and no-worse mean ECE** than
the currently shipped rung. Decisions are logged to MLflow and recorded in
`model_registry.json`. On the WC 2014/18/22 windows, rung 1 improved log loss
but worsened calibration (not promoted); rung 2 improved both (promoted).

## Simulation

- Completed matches are facts; only remaining matches are sampled. Group
  ranking follows points → goal difference → goals for → head-to-head (among the
  tied teams) → seeded random. Best thirds rank by points → GD → GF → random;
  their bracket slots come from FIFA's Annex C table (all 495 combinations,
  shipped in the pack).
- Match outcomes are sampled from the calibrated classifier; **scorelines** are
  then sampled from a double-Poisson grid (means from as-of attack/defence,
  clamped to config bounds) **conditioned on the sampled outcome** — so group
  tiebreakers see realistic goal counts without contradicting the classifier.
- Knockout draws resolve by renormalising the two win probabilities
  (ET/penalties ≈ strength-proportional).
- Default 100k runs, fully seeded. Every probability is reported with its Monte
  Carlo standard error `sqrt(p(1−p)/N)`.

## Approximations & caveats (documented, deliberate)

- Fair-play tiebreaker is not simulatable → final tiebreak is a seeded random
  draw. Immaterial at headline-probability resolution.
- Venue: a host playing a non-host gets home advantage; all other pairings are
  neutral; per-stadium assignment is not modelled. Rest-day diff for simulated
  matches is 0.
- MC standard error excludes model uncertainty; tournament outcomes are single
  draws from fat-tailed distributions. The walk-forward backtests are the only
  defensible quality claim.
- Optuna tunes on the same window the calibrator uses (mild optimism).
- Attack/defence ratings drift off the 1.0-centred scale over 150 years; the
  scoreline sampler clamps its Poisson means to compensate.

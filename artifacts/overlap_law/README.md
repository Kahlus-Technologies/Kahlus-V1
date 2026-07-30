# F9 evidence — the overlap inflation law

Tracked evidence for `docs/results/overlap_inflation_law_2026-07-28.md`.
`runs/` is gitignored, so cite these copies.

| file | contents |
| --- | --- |
| `overlap_law_results.json` | all 286 cells: W, H, chosen alpha, MSE_overlap / MSE_copy / MSE_future, observed inflation, the W/H bound, and the exact formula value |
| `overlap_law.png` | predicted bound vs observed inflation (log-log, identity line), and departure-from-bound vs the correction term |

Produced by `scripts/overlap_inflation_law.py` and `scripts/plot_overlap_law.py`.

**Law:** `inflation = MSE_future/MSE_overlap ≤ W/H`, with equality as
`MSE_copy/MSE_future → 0`. 286 configurations, 11 signals, **0 violations**,
median observed/bound 1.000.

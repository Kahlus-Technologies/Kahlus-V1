# F8 evidence — horizon-resolved forecast crossover

Tracked evidence for `docs/results/forecast_crossover_2026-07-27.md` (ledger row F8).

`runs/` is gitignored, so the working outputs there are not citable. These are the
committed copies. Small by design: JSON summaries and one figure, no raw tensors.

| file | produced by | contents |
| --- | --- | --- |
| `forecast_scaleup_results.json` | `scripts/forecast_scaleup_paired.py` | per-subject MSE for all 4 models × 9 horizons, the 39/12/27 subject split, chosen ridge alpha and GRU config per horizon, GRU seed spread, paired Wilcoxon + bootstrap CIs |
| `adversarial_validation.json` | `scripts/validate_forecast_claim.py` | the 8-test falsification suite (T1–T8) |
| `forecast_crossover.png` | `scripts/plot_forecast_scaleup.py` | horizon curve + per-subject paired deltas |

Regenerate:

```
PYTHONPATH=src python3 scripts/forecast_scaleup_paired.py \
    --data <sleep-cassette> --out runs/forecast_scaleup --subjects 78
PYTHONPATH=src python3 scripts/validate_forecast_claim.py \
    --data <sleep-cassette> --out runs/forecast_validation
python3 scripts/plot_forecast_scaleup.py
```

Headline: at h=16 (160 ms) the GRU scores 0.7268 vs a per-horizon-tuned ridge-AR
at 0.8298 — Δ +0.1031, Holm p=2.24e-06, 26/27 held-out subjects. The advantage
ends by ~320 ms. Read §2 and §4 of the finding doc before quoting any number:
the long-horizon claim is deliberately weaker than the primary run suggested.

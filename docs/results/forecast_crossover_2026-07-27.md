# F8 — Horizon-resolved crossover: the GRU beats a tuned linear AR below ~160 ms and not above

**Date:** 2026-07-27. **Status:** positive result, powered, pre-baseline-tuned.
**Artifacts:** `runs/forecast_scaleup/results.json`,
`runs/forecast_scaleup/forecast_crossover.png`.
**Reproduce:**

```
PYTHONPATH=src python3 scripts/forecast_scaleup_paired.py \
    --data <sleep-cassette> --out runs/forecast_scaleup --subjects 78
```

This replaces the invalidated `3.116 MSE / r=0.972` sidecar as the project's
forecasting result. That number was the 126/127 within-window overlap artifact
(see `docs/research/eval_leakage_audit_2026-07-21.md`); it is not cited here as
performance and is not comparable to anything below.

---

## Protocol

Real Sleep-EDF sleep-cassette, channel `EEG Fpz-Cz`, 100 Hz, z-scored per
recording. **All 78 subjects**, both nights per subject mapped to one subject id.
Shuffled subject-disjoint split (seed 0): **39 train / 12 validation / 27 test**,
31,200 train and 21,200 test windows.

The task is a **strictly-future single-sample forecast**: given `signal[i:i+127]`,
predict `signal[i+127+h-1]`. Input and target share **zero** samples at every
horizon — the property whose absence produced the historical artifact.

Both learned models are tuned on the **same held-out validation subjects**:

- ridge-AR: alpha swept over a 14-point grid to `2e7`, per horizon. The script
  warns when the selection lands on a grid edge; a first pass truncated at `500`
  did select the edge, so the grid was extended until every horizon selected an
  interior value. This matters — see §3.
- Kahlus GRU: `(hidden, steps)` swept over `{32,64} x {300,900}`, per horizon,
  then run at 3 seeds. Seed spread is reported and is small (≤0.05 MSE).

MSE is computed **per test subject**, so model differences are paired. Headline
test is Wilcoxon signed-rank over the 27 subjects; the CI is a 10,000-draw
bootstrap of the mean per-subject delta.

## 1. Result

Mean per-subject MSE (z-scored units). `Δ = ridge − GRU`; positive favours GRU.

| h | ms | persistence | ridge-AR | **GRU** | Δ | 95% CI | Wilcoxon p | subjects favouring GRU |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | :---: |
| 1 | 10 | 0.4322 | 0.2958 | **0.2754** | +0.0204 | [0.0125, 0.0288] | 2.6e-05 | 25/27 |
| 2 | 20 | 0.4449 | 0.3393 | **0.3190** | +0.0203 | [0.0127, 0.0275] | 1.6e-05 | 24/27 |
| 4 | 40 | 0.5691 | 0.4399 | **0.3950** | +0.0450 | [0.0381, 0.0522] | 1.5e-08 | **27/27** |
| 8 | 80 | 0.8568 | 0.6193 | **0.5566** | +0.0626 | [0.0477, 0.0775] | 1.5e-07 | 26/27 |
| 16 | 160 | 1.3814 | 0.8298 | **0.7268** | **+0.1031** | [0.0753, 0.1313] | 3.7e-07 | 26/27 |
| 32 | 320 | 1.7600 | **0.8700** | 0.8756 | −0.0056 | [−0.0250, 0.0150] | 0.43 | 10/27 |
| 64 | 640 | 2.0433 | **0.8771** | 0.9004 | −0.0234 | [−0.0320, −0.0144] | 4.1e-05 | 4/27 |
| 128 | 1280 | 2.4660 | **0.9198** | 0.9245 | −0.0047 | [−0.0124, 0.0030] | 0.28 | 10/27 |
| 256 | 2560 | 1.8746 | **0.9942** | 1.0099 | −0.0157 | [−0.0185, −0.0132] | 1.5e-08 | **0/27** |

**Best margin, h=16 (160 ms):** GRU 0.7268 vs tuned ridge 0.8298 — a **12.4%
relative reduction**, 26 of 27 held-out subjects, `p = 3.7e-07`. Against
persistence at the same lead: 1.3814 → 0.7268, a **47.4% reduction**.

## 2. What is actually claimed

**Claimed.** On single-channel sleep EEG, with a subject-disjoint split and a
per-horizon-tuned linear baseline, a small GRU gives a statistically robust
error reduction over that baseline for horizons from 10 ms to 160 ms, and the
margin grows monotonically with horizon across that range. Both learned models
beat persistence everywhere.

**Also claimed, and it cuts against us.** The advantage **inverts** past ~320 ms.
At 640 ms and 2560 ms the tuned linear model is better with high confidence
(`p = 4.1e-05` and `1.5e-08`; at 2560 ms **zero of 27** subjects favour the GRU).
The nonlinear architecture earns its complexity only in a bounded short-horizon
regime.

**Not claimed.** Nothing about clinical utility. 160 ms is roughly 400× shorter
than the ~1 min precursor timescale the HNPH line targets. Nothing multichannel —
this is one channel, which is the most likely reason the margin closes so early.
Nothing cross-dataset: one corpus.

## 3. The methodological finding, which may matter more than the numbers

A 24-subject pilot with the alpha grid truncated at `500` produced a *different
scientific conclusion*: it showed the GRU losing at h=16. The earlier 8-subject
run with alpha fixed at `5.0` showed the GRU winning at h=16 by 16%. Both were
artifacts of baseline tuning, in opposite directions.

The order of operations that produced a stable answer was: extend the grid until
no horizon selects an edge, give the deep model a matched sweep on the same
validation subjects, then read the result. Every step changed a number that a
paper would have reported.

This is the second inflation mechanism this project has now quantified on its own
work, alongside the 126× within-window overlap:

1. **within-example input/target overlap** — inflates by 126× and survives
   subject-held-out splitting and every remedy in the leakage literature, because
   it lives inside a single example rather than across the split;
2. **under-tuned linear baselines** — reverses the sign of the headline
   conclusion at 4 of 9 horizons here.

Literature grounding, novelty check and comparators:
`docs/research/baseline_fairness_literature_2026-07-27.md`. Briefly: the
within-example axis appears unaddressed in the leakage literature (Bernett 2024,
Del Pup 2025, Apicella 2024 all treat between-example sharing); the weak-baseline
pathology has a direct precedent in McGreivy & Hakim (2024, *Nat. Mach. Intell.*,
79% of ML-for-PDE papers) and Lin (2019) in IR; and Wu et al. (2025) name "few
linear baselines" as an explicit open gap across 149 EEG studies.

**Comparator to be careful about:** Pankka et al. (2024) already report WaveNet
beating an AR model on resting-state EEG at 150 ms — the same regime as our best
horizon. "A deep model beats AR at ~150 ms" is therefore *not* novel. What is
ours is the horizon-resolved crossover with per-subject paired statistics against
a baseline tuned to convergence, and the two inflation mechanisms.

## 4. Loose ends

- Persistence MSE is non-monotonic (2.4660 at 1280 ms, 1.8746 at 2560 ms). For
  z-scored signals an uncorrelated lag gives ~2.0, so exceeding it at 1280 ms
  implies anti-correlation at that lag — consistent with slow-wave periodicity in
  sleep EEG. Physically sensible, but unverified; do not present it as a finding.
- Single channel. Onojima et al. (2025) beat AR cross-subject using raw
  *multichannel* input; the obvious next test is whether the crossover moves right
  when the model gets spatial structure it can actually use.
- One corpus. `chbmit`, `siena-scalp-eeg`, `slpdb`, `eegmmidb` are on local disk;
  a claim of this shape needs the crossover replicated on at least two more.
- The sealed confirmation holdout (`scripts/seal_holdout.py`) has **not** been
  used here — this is the analysis cohort. A confirmatory run belongs on sealed
  data once the primary result is fixed in writing.

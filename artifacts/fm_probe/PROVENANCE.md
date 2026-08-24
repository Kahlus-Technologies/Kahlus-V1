# Provenance and claim boundaries — fm_probe artifacts

Applies to `scores.json`, `scores_full78.json`, `scores_full78_with_null.json`
(committed 2026-08-20, writer at `47b44bef`, before sharding/provenance fields
existed). The JSONs are preserved unmodified as historical evidence; this file
states what they do and do not support. Verified by the 2026-08-23 branch audit
(every mean/CI/win-count recomputes exactly from the per-fold data; `full78`
and `full78_with_null` are the same evidence, not divergent runs).

## Invalid: every 10 ms row

The task resamples Sleep-EDF 100→200 Hz with `scipy.signal.resample_poly`,
whose non-causal half-band FIR writes future original samples into the
interpolated context samples. A 127-lag ridge recovers the next ~2 originals
almost exactly. Null measurement (white noise, best possible MSE = variance,
exact production pipeline): ridge/variance = **0.066 at 10 ms** vs
0.935 / 0.971 / 0.973 at 80 / 160 / 320 ms. Cross-check: the native-100 Hz
audit (`../isolated_residual_evidence_20260815/tuned_ridge_audit_result.json`)
scores tuned ridge **0.293** at the same nominal 10 ms horizon where these
files claim **0.0106** — a 28× gap explained entirely by the filter.

Do not cite any `"10"` entry. The horizon was removed from the runner
(`scripts/fm_probe_local.py`), which now carries a white-noise selftest gate
that fails if any reported horizon scores <0.9× variance on noise.

## Caveats on the surviving rows

- **80 ms**: ≤~6% residual interpolation advantage (0.935 on the null),
  shared by all arms; cross-arm comparisons stand, absolute MSEs are mildly
  flattered. 160/320 ms: ≤3%.
- **Normalization**: these runs z-scored each recording with whole-clip
  (future-inclusive) scalar stats. Scale-only, applied identically to every
  arm, so arm-vs-arm comparisons are fair, but the runs predate the
  past-only normalization now in the runner.
- **Targets** land on interpolated (smoothed) samples, uniformly deflating
  all MSEs slightly relative to the native-rate task.

## What these files still support

At 160/320 ms under 78-subject LOSO: tuned ridge beats the frozen CBraMod
probe at 160 ms (+0.075, bootstrap CI excludes 0); at 320 ms the probe beats
ridge (−0.044, CI excludes 0) **but the random-init backbone beats the
pretrained one** (+0.0086, CI [+0.0046, +0.0126]) — the 320 ms gain is not
attributable to TUEG pretraining. `claim_scope`: single-channel Fpz-Cz
Sleep-EDF isolated forecasting only; no clinical claim; no foundation-model
claim. `stop_reason`: superseded — the 8×A100 factorial
(`scripts/slurm/run_everything_8xa100.sh`) regenerates all arms under the
fixed runner with recorded config + git sha.

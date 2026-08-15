# Isolated residual courtroom — first A100 evidence run, audited (2026-08-15)

## What ran

Krish executed the `kahlus_isolated_residual_handoff.zip` protocol (SHA256
`0a5ed2cf…` — matches the shipped zip byte-for-byte) on two A100s: BNCI2014_001
fixed split and Sleep-EDF cassette **78-subject LOSO**, GRU head only, one seed.
Stack: persistence → 16-lag ridge (α=20) → oscillator residual → GRU on the
leftover. Strictly-future isolated targets throughout.

## Results (their run, `runs/*/scores.json`)

- **BNCI:** GRU stack loses to ridge at all four horizons. Corpus stays negative.
- **Sleep-EDF (mean over 78 folds):** stack beats the 16-lag ridge at every
  horizon; per-fold sign test 77–78/78, p≈1e-22; win rule
  (residual-vs-zero < residual variance) true in 76–78/78 folds.

## The F26 audit (this package: `tuned_ridge_audit.py` / `_result.json`)

An undertuned linear baseline has manufactured a GRU "win" in this project
before, so the pack's weak ridge was re-scored against a **127-lag ridge with α
tuned per fold and horizon on held-out training subjects**, on identical
windows and identical train-fold-only z-scoring. Since rFFT bins are linear in
the window, the tuned full-window ridge also spans the oscillator features.

| h (ms) | weak ridge | tuned ridge | GRU stack | gap tuned−stack [95% CI] | stack wins |
|---|---|---|---|---|---|
| 1 (10)   | 0.2951 | 0.2932 | 0.2600 | +0.0332 [+0.0280, +0.0392] | 77/78 |
| 8 (80)   | 0.6603 | 0.6370 | 0.5418 | +0.0952 [+0.0799, +0.1119] | 78/78 |
| 16 (160) | 0.9004 | 0.8410 | 0.6929 | +0.1481 [+0.1219, +0.1774] | 75/78 |
| 32 (320) | 0.9872 | 0.8970 | 0.8025 | +0.0945 [+0.0720, +0.1239] | 72/78 |

Baseline tuning explains 5–49% of the raw gap depending on horizon — and a
subject-consistent nonlinear advantage of ~11–18% MSE survives at every
horizon, CIs excluding zero, majorities ≥72/78.

## Honest status

**Powered preliminary positive — not yet a publishable claim.** Open before any
paper-grade statement:

1. **One seed.** The F8 standard is 3 seeds; seed sensitivity unchecked.
2. **Provenance gap.** The run's script deviates from repo commit `e0ace5e`
   beyond the disclosed device fix: train-fold-only z-scoring (stricter than
   the shipped per-recording z-score — direction favors validity, but it is
   undisclosed in the pack README) and the modified `_forecast_common.py` it
   requires is not in the pack. Reproduce independently from repo code.
3. **One channel, one corpus.** Fpz-Cz, Sleep-EDF cassette. BNCI is negative
   under a weaker protocol (fixed 2-subject test); a BNCI LOSO with this stack
   has not run.
4. **Tension with F8.** F8 found the plain-GRU advantage gone by ~320 ms; this
   stack keeps +0.0945 (72/78) at 320 ms. Different architecture (residual
   stack vs plain GRU), split (LOSO vs fixed), and normalization — needs a
   controlled comparison, not a shrug.
5. **Baseline ladder not exhausted.** Gate demands best-baseline: GBM /
   stronger nonlinear-feature linear models next.

Device bug found by this run (heads never moved to GPU) is fixed upstream in
commit `369849af`. Do not cite the mean tables without the per-fold CIs; do not
put any of this in the IECBES submission — it is one-seed, night-of, and
co-author-unreviewed.

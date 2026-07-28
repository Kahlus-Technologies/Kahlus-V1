# Kahlus Findings Ledger

Durable record of every publishable finding, its data artifact, the code that produced it,
and the git tag that pins the tree at the point the finding was frozen.

Read alongside [AGENTS.md](../../AGENTS.md) (claim discipline) and
[a100-run-history.md](a100-run-history.md) (engineering-run provenance).

Rule: a finding is publishable when it clears the gate predicate
`split_disjoint AND finite AND beat_best_baseline AND controls_pass AND powered AND scoped_claim`,
**or** when it is an honest negative produced by machinery that has been validated on a
synthetic known-signal / known-null pair. Negatives count. A benchmark that only flatters
its own model is worthless.

## Findings

| ID | Finding | Verdict | Substrate / power | Headline number | Data artifact | Code | Tag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F0 | Neural-CASP gate suite (M0-M5) is deterministic, leakage-audited, and fails honestly | Arena works | Synthetic + public smoke | M0/M1/M5 pass; M3/M4 fail honestly (underpowered / external not committed) | `artifacts/forecastability_trial0_m0..m5/` | `src/neurotwin/forecastability/m0.py..m5.py` | `finding/neural-casp-gate-suite-v1` |
| F1 | Forecast "skill" of 3.116 MSE / 0.972 r is an overlap illusion, not forecasting | Invalidated | MOABB EEG v1 sidecar | Metric collapses once scored on strictly-future positions only | `docs/research/eval_leakage_audit_2026-07-21.md`, `a100-run-history.md` | `src/stf/*` overlap-scoring fix (commit `155a115f`) | `finding/forecast-overlap-illusion-v1` |
| F2 | Kahlus GRU does not beat trivial baselines under isolated (non-overlapping) forecasting | Negative | Sleep-EDF + BNCI2014_001, subject-held-out | GRU loses to persistence / ridge at h=1 | `artifacts/ridge_bnci_real/` | `scripts/bnci_isolated_forecast_check.py` | `finding/isolated-forecast-negative-v1` |
| F3 | Peripheral/autonomic channels add no residual forecastability for sleep-state transitions | Negative (scout) | Public Sleep-EDF smoke; synthetic known/null validated | Real-data RFS not distinguishable from null; synthetic known passes, null fails | `artifacts/interoception_rfs_scout/` | `src/neurotwin/forecastability/interoception_scout.py` | `finding/interoception-rfs-scout-v1` |
| F4 | Passive complexity (LZ / permutation / multiscale entropy) does not beat a spectral baseline for wake/NREM/REM discrimination | Powered negative | Full Sleep-EDF cassette: 78 subjects, 413,828 windows, claim bootstrap (n_boot=2000) | Wake RFS -0.330 [CI -0.349, -0.313]; NREM -0.284 [-0.297, -0.271]; REM -0.161 [-0.177, -0.146]; all CIs exclude 0 on the negative side | `artifacts/passive_pci_state/` | `src/neurotwin/forecastability/passive_pci.py`, `complexity_features.py` | `finding/passive-pci-negative-v1` |
| F5 | Propofol sedation: complexity beyond spectral for awake vs sedated | Powered negative (full cohort) | OpenNeuro ds005620: **21/21 subjects** (full cohort — supersedes the earlier 12/21 disk-limited partial run), 2527 windows, claim bootstrap (n_boot=2000) | Awake RFS −0.2072 [CI −0.3084, −0.1208]; sedated −0.2072 [CI −0.3077, −0.1198]; CIs exclude 0 on the negative side | `artifacts/propofol_pci_state/` | `propofol_pci.py` | `finding/propofol-pci-powered-negative-v1` |
| F6 | Autonomic block beyond cortical spectral for micro-arousal | Synthetic OK; NSRR loader fixed | Reformulated Y=arousal; claim bootstrap synthetic validated; real MESA/SHHS pending credentialed NSRR download | Known RFS 0.021; null −0.000; stub-EDF loader path reaches `mesa_status: evaluated` | `artifacts/autonomic_rfs_arousal/` | `autonomic_rfs.py`, `adapters/nsrr.py`, `_rfs_eval.py` | `finding/autonomic-rfs-pending-nsrr-v1` |
| F8 | Horizon-resolved crossover: GRU beats a per-horizon-tuned ridge-AR below ~160 ms; advantage ends by ~320 ms | **Powered positive (bounded regime), adversarially validated** | Sleep-EDF cassette, all 78 subjects, subject-disjoint 39/12/27, 31,200 train / 21,200 test windows, per-subject paired Wilcoxon + 10k bootstrap, 3 seeds; 8-test falsification suite | h=16 (160 ms): GRU 0.7268 vs tuned ridge 0.8298, Δ +0.1031 [0.0753, 0.1313], Holm p=2.2e-06, 26/27 subjects; vs persistence 1.3814 (−47.4%). Robust: 8/8 resampled splits positive (mean +0.0947, sd 0.0083); +0.0963 under non-overlapping windows. Controls: GRU **loses** to ridge on AR(2) where linear is optimal; shuffled-target and noise-input nulls both collapse to the variance floor | `artifacts/forecast_crossover/` (results, adversarial validation, figure) | `scripts/forecast_scaleup_paired.py`, `scripts/validate_forecast_claim.py`, `scripts/plot_forecast_scaleup.py` | `docs/results/forecast_crossover_2026-07-27.md` |

## What the findings jointly say

Every residual-forecastability probe (F3-F6) is a **negative**, produced by machinery that
is provably alive (synthetic known-signal fixtures pass, known-null fixtures fail, controls
collapse). F8 is the first powered **positive**, and it is deliberately narrow:

> Under strict subject-held-out evaluation against the best trivial baseline in the ladder,
> the residual forecastability claims that dominate the noninvasive-neural literature do not
> reproduce. The overlap illusion (F1) shows one mechanism by which they appear to; F8 shows
> a second — an under-tuned linear baseline — and, once both are removed, locates the
> bounded horizon regime (<=160 ms, single-channel sleep EEG) where a nonlinear model does
> hold a real, subject-consistent advantage, and the regime (>=320 ms) where it does not.

The arena (F0) is the asset. The negatives (F1-F6) are the evidence that the arena bites.
F8 is the evidence that it is not merely a machine for producing negatives.

**Open tension, F2 vs F8.** F2 recorded `FORECASTING_SKILL_DEAD` on BNCI2014_001 with **2**
test subjects and an untuned baseline. F8 finds the opposite on Sleep-EDF with 27 test
subjects and a converged baseline. These are different corpora, so F8 does not overturn F2 —
but F2's power and baseline tuning are both below the standard F8 now sets, and it should be
re-run under the F8 protocol before either is cited.

## Claim boundaries (do not cross)

- No clinical seizure/diagnosis claim; no consciousness claim; no "beat PCI" claim (F4 is a negative).
- The 3.116 MSE / 0.972 r number is retired as a forecasting result (F1).
- Passive PCI (F4) is a negative on Sleep-EDF cassette only; F5 tests propofol (ds005620).
- F5 claim scope must not imply TMS-PCI replacement or clinical anesthesia monitoring.

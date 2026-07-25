# Rao et al. 2025 (bioRxiv 2025.11.13.686572) — methods mining for F7

Read 2026-07-25 from the full preprint. Rao, Fredericks, Dresler, Rebollo, Zeitzer,
Schoch, Coleman — "Simultaneous stomach-brain electrophysiology reveals dynamic
coupling in human sleep." 60 participants, 3 nights (adaptation + 2 experimental),
64-ch EEG + 2-ch EOG + 3-ch EMG + 2-ch ECG + 4-ch EGG.

Purpose here: extract the analysis choices worth mirroring, and record one finding
that directly constrains what F7 should claim.

---

## The one predictive claim, in full

Section 2.5.2, n = 49 nights. Nested linear models predicting **subjective sleep
quality (SSQ, 1-5)**:

| model | result |
| --- | --- |
| Baseline PSG (%N1/%N2/%N3/%Wake, sleep-fragmentation indices, sex, condition, time-since-meal, coupling condition) | R² = 0.225, adj R² = 0.133, p = 0.017 |
| + EGG ISO amplitude variance | **ΔR² = 0.13**, partial R² = 0.16, p = 0.014 |
| + EEG sigma ISO | ΔR² = 0.11, p = 0.031 |
| + both EGG and EEG ISO | ΔR² = 0.23, p = 0.0047 |
| + EGG κ (phase coupling) | ΔR² = 0.0005, p = 0.84 (nothing) |
| + coupling MI (parietal) | ΔR² = 0.046, p = 0.17 (nothing) |

Their own framing is careful: the *intrinsic* infraslow rhythmicity of each organ
predicts SSQ, while the *coupling* between them does not.

## The finding that constrains F7

**They tested cardiac autonomic predictors and found nothing.** Same nested-model
setup, same outcome:

- HRV (RMSSD): partial r = 0.16, p = 0.29 with SSQ; **ΔR² < 0.03, p > 0.25**
- heart-rate infraslow power: same, ΔR² < 0.03
- Including HRV alongside EGG ISO did not improve fit (R² 0.37 vs 0.36)

So for *subjective sleep quality*, cardiac autonomic indices carry no incremental
information beyond PSG in this cohort. That is a published null on essentially the
Z-block F6/F7 uses.

**What this does and does not mean for us.** It does not sink F7: their outcome is
a night-level subjective rating; F6/F7's outcome is a within-night future arousal
or state transition at a fixed horizon, and Bonnet & Arand's ≥10-beat pre-arousal
HR acceleration is about exactly that within-night timescale. Different estimand,
different timescale. But it is a real prior against the broad "autonomic explains
sleep outcomes" story, and F7 should be framed narrowly around event-locked
prediction, not night-level quality. Worth citing in the writeup either way, since
it is the kind of null this arena exists to take seriously.

## Methods worth mirroring

1. **LMM with repeated measures within participant**, covariates sex, experimental
   condition, time since last meal. Our subject-level GroupKFold + nuisance block
   B is the cross-fitted analog; their covariates map onto B directly.
2. **Nested model comparison** (baseline → baseline + feature) reporting ΔR² and
   partial R². This is the same incremental-value question as RFS bits; ours is
   out-of-fold, theirs is in-sample with an adjusted-R² companion.
3. **Cross-validated ΔCVR² as a check.** They report ΔCVR² = −0.015 for the κ
   measure, i.e. they did look for CV shrinkage on at least one predictor. Good
   practice; our whole gate is that check.
4. **FDR correction** across the objective-measure family (reported as corrected
   q < 0.05 for SME / TST / WASO).
5. **Effect sizes with CIs throughout** — d_LMM = 1.1 (NREM vs REM gastric power),
   0.81 (ISO peak power), 0.33 (NREM vs wake ISO).
6. **Model-estimated marginal means ± 95% CI** rather than raw group means.

## Signal-processing details worth copying

- Gastric band 0.03–0.07 Hz (normogastric); total spectral power computed 0.01–0.15 Hz.
- ISO extracted as the **amplitude envelope of the band-limited signal**, then
  z-scored — peak at ~0.007 Hz (~2.5 min period).
- Cortical comparator: **sigma power 9–16 Hz**, its infraslow amplitude variance
  (<0.1 Hz). Sigma is the spindle/arousal-related marker they pair with gastric ISO.
- Sub-1 Hz structure is handled as amplitude/phase of a band-limited envelope, not
  as Welch band-power in short windows.

That last point matters for our slpdb work: BP and respiration are sub-1 Hz and the
neuroforecast slpdb loader already handles them as log power envelopes for the same
reason.

## Direct quotes worth keeping honest

Their conclusion is about *intrinsic rhythmicity*, not coupling: "consistency of
infraslow modulation within both the brain and stomach, rather than direct
stomach-brain coupling, best accounts for interindividual differences in subjective
sleep quality." Any Kahlus writeup that cites this paper as motivating a
*coupling* claim would be misreading it.

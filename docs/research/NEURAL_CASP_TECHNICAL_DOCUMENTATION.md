# Neural-CASP: a leakage-controlled incremental-value test for neurophysiological forecasting

Technical documentation for the Kahlus / Neural-CASP repository. Written for sleep
and EEG researchers and for ML practitioners working on physiological time series.
Assumes familiarity with polysomnography, cross-validation, and nested model
comparison.

**Status as of 2026-07-25.** Every real-data result in this repository is a
negative. That is stated first because it is the honest headline, not a caveat
buried at the end. The contribution is the evaluation apparatus and a set of
powered nulls produced by it; there is currently no positive finding.

Branch `fix/forecast-overlap-guard`. Repository `ptlnextdoor/Kahlus-V1`.

---

## 1. Estimand

For an outcome `Y` at horizon `h`, a nuisance/baseline block `B`, and a candidate
block `Z`, define the **residual forecastability score**:

```
RFS_bits = (NLL_B − NLL_{B+Z}) / ln 2
```

implemented in `src/neurotwin/forecastability/m1.py::_rfs_bits`, with `NLL` the
mean binary cross-entropy of a calibrated probability forecast
(`_nll`, probabilities clipped to [1e-5, 1−1e-5]).

RFS is the out-of-fold log-likelihood gain from adding `Z` to `B`, expressed in
bits per anchor. Under a correctly specified model it estimates the conditional
mutual information `I(Y; Z | B)`; for a fixed model class it is a lower bound.
This is not a new information-theoretic quantity. **Methodologically it is a
cross-fitted, out-of-sample analogue of nested ΔR² / ΔCVR² model comparison** — the
same incremental-value question, evaluated under subject-held-out cross-validation
rather than in-sample.

`RFS` is the one piece of project-specific vocabulary in this document. Everything
else uses standard terminology.

### Blocks

- **B (nuisance)** always contains the event base rate, recent event history, and
  cortical spectral features. A positive RFS therefore means `Z` carries
  information the cortex's own recent state does not already supply.
- **Z (candidate)** is the block under test — an autonomic block, a complexity
  block, a peripheral-channel block, depending on the finding.
- **Y** is a future event indicator at horizon `h` (arousal within the next `h`
  epochs; a stable stage transition; a macrostate label).

---

## 2. Evaluation protocol

### 2.1 Cross-fitting and clustering

Nuisance and residual models are fit out-of-fold with `GroupKFold` at the
**subject** level. Anchors within a recording are strongly dependent, so all
resampling is at the cluster (subject) level, never per-anchor. This mirrors the
repeated-measures structure handled by linear mixed models in the sleep
literature (e.g. Rao et al. 2025 fit LMMs with random effects per participant and
covariates for sex, condition, and time since last meal); the covariates that
enter their LMM enter our `B` block.

### 2.2 Baseline ladder

The residual is scored against the *best* of a candidate baseline set, selected by
NLL (`_best_baseline`): moving average, persistence, base rate, and a
nuisance-only cross-fitted model. Reporting a gain over a weak baseline is the
standard way incremental-value claims inflate; taking the minimum-NLL baseline
removes that degree of freedom.

### 2.3 Control battery

Every candidate result must survive:

| control | what it breaks |
| --- | --- |
| label shuffle | any association at all |
| time shift | temporal alignment |
| within-subject circular shift of `Z` | alignment while preserving `Z`'s autocorrelation |
| subject-identity probe | `Z` acting as a subject fingerprint (must stay ≤ chance + 0.2) |
| cluster permutation | provides the p-value |

The circular-shift surrogate is the same device used in the gastric-coupling
literature to preserve spectral structure while destroying phase alignment.

### 2.4 Certification rule

Implemented in `_rfs_eval.py::horizon_real_passes`. An edge is certified only if
**all** hold:

```
rfs_bits > 0.02
rfs_ci_low > 0                      # one-sided 95% subject-cluster bootstrap LCB
every control < 0.4 × rfs_bits
no nuisance-probe failures
(+ minimum subjects and minimum positive events)
```

The 0.02 bits/anchor floor is a detection threshold, not a claim of clinical
usefulness.

---

## 3. The leakage failure and its remediation

This section exists because the repository's original headline result was an
artifact. It is documented rather than deleted.

### 3.1 F1 — the overlap illusion

The historical forecasting task took a 128-sample window, used
`X = seq[0:127]` as input and `Y = seq[1:128]` as target, and scored MSE over the
**full 127-length target**. Input and target therefore shared **126 of 127
samples**. A model that copies the input shifted by one is exact on 126 positions;
the single genuinely-future sample is averaged away.

Measured on real Sleep-EDF with the identical ridge model:

| scoring rule | ridge MSE |
| --- | --- |
| old full-127-window (overlapping) | **0.0018** |
| isolated strictly-future sample | **0.227** |

A **126× gap from one model, one dataset, two scoring rules.** The historical
`6621642` sidecar (MSE 3.116 / Pearson 0.972) is marked INVALID in place — not
removed — in `docs/results/a100-run-history.md`,
`docs/audit/EXPERIMENT_VALIDITY_AUDIT.md`, and `docs/CLAIMS.md`.

This is the same failure mode Brookshire et al. (2024, *Frontiers in
Neuroscience*) document for translational EEG: segment-based rather than
subject-based holdout inflates apparent performance, and they find the majority of
surveyed DNN-EEG studies use the inflating split. Our case is the within-window
analogue — leakage between input and target rather than between train and test —
and it is arguably easier to miss because no split is violated at all. Del Pup et
al. (2025, *Computers in Biology and Medicine*) reach compatible conclusions on
partitioning strategy across >100,000 trained models.

### 3.2 The fix: strictly-future metric mask

Forecasting tasks now carry a boolean `metric_mask` that is true only on target
positions **absent from the input window**, applied to **both the training loss
and the evaluation metric** (`training/prepared_metrics.py`,
`benchmarks/baseline_suite.py`). Copyable positions can neither be optimised
against nor scored.

Regression test (`tests/data/test_forecast_no_overlap.py`): a predictor
constructed to exploit the overlap scores ≈0.059 unmasked and ≈0.470 masked — an
**8× jump**. That discriminating gap, not the agreement between two metrics, is
the evidence the mask works.

### 3.3 The guard

`data/forecast_contract.py::assert_forecast_scoring_is_leakage_safe` makes the
unsafe configuration unconstructible. A forecasting task is valid only if

```
forecast_horizon >= window_length          # windows disjoint by construction
  OR  a non-empty strictly-future metric_mask is attached
```

Called by both builders; a single home rather than a copy per call site. An
all-False mask is rejected as equivalent to no mask.

### 3.4 Person-level split audit

`data/audit.py::audit_person_level_split`. The pre-existing
`audit_split_manifest` verifies that declared `subject_id` values do not span
train/val/test — but it **trusts the labels**. When two nights of one person are
labelled as two subjects, that check passes while the person sits on both sides.
That is how a 26-person train/test overlap previously shipped.

Both dataset conventions in this repository collapse recordings to people:

```
Sleep-EDF   SC4nnN   ->  subject nn, night N     (SC4001, SC4002 = one person)
slpdb       slpNNx   ->  subject NN              (slp01a, slp01b = one person)
```

The lead test builds the exact historical manifest — SC4001 in train, SC4002 in
test — and asserts that `audit_split_manifest` **passes** while
`audit_person_level_split` **fails**. Where person identity cannot be established
the audit **fails closed** ("person identity unverified") rather than certifying
by default.

---

## 4. Findings

All numbers below are traceable to `docs/results/findings-ledger.md` and the
artifacts named in each row.

### F2 — isolated forecasting is a tie with ridge

Subject-held-out, strictly-future single-sample forecast, Sleep-EDF cassette and
BNCI2014-001. Every learned model trained **directly per horizon** (an earlier
version rolled a one-step GRU out autoregressively, which compounds error and is
not a like-for-like comparison; that version was discarded).

| horizon (samples @100 Hz) | random | mean-of-trace | persistence | ridge | GRU |
| --- | --- | --- | --- | --- | --- |
| 1 (10 ms) | 1.855 | 1.037 | 0.281 | 0.227 | **0.211** |
| 8 (80 ms) | 2.070 | 1.232 | 0.853 | 0.675 | **0.583** |
| 16 (160 ms) | 2.054 | 1.264 | 1.334 | 0.822 | **0.690** |
| 64 (640 ms) | 2.208 | 1.395 | 2.084 | **0.884** | 0.913 |

Both learned models separate clearly from the trivial baselines, and the margin
over persistence widens with horizon. But the GRU does not meaningfully beat a
127-tap ridge, and loses at 640 ms. At single-channel millisecond scale the signal
is close to linear and a regularised linear autoregression captures nearly all of
it. Consistent with the long-standing finding that well-regularised linear and
Riemannian methods remain competitive on EEG (Lotte et al. 2018), and that the
canonical deep baselines (Lawhern et al. 2018; Schirrmeister et al. 2017) earn
their keep on classification rather than short-horizon waveform forecasting.

### F3 — interoception scout, negative (scout-grade)

Peripheral channels (EOG, EMG, respiration) as `Z`; future sleep-stage transition
as `Y`; Sleep-EDF. Synthetic known-signal passes and known-null fails, so the
machinery is live; the real-data residual does not survive controls at scout
thresholds. Note the design limitation: Sleep-EDF cassette carries no clean ECG,
so the strongest autonomic channel (heart rate / HRV) is absent from `Z`.

### F4 — passive complexity does not beat a spectral baseline (powered)

Full Sleep-EDF cassette, **78 subjects, 413,828 windows**, claim bootstrap
(n_boot = 2000). `Z` = Lempel-Ziv complexity, permutation entropy, multiscale
entropy; `B` = spectral baseline; `Y` = wake/NREM/REM one-vs-rest.

| state | RFS bits | CI |
| --- | --- | --- |
| Wake | −0.330 | excludes 0, negative side |
| NREM | −0.284 | excludes 0, negative side |
| REM | −0.161 | excludes 0, negative side |

The complexity block actively *hurts* relative to spectral features.
`forecastability/passive_pci.py`, `complexity_features.py`.

### F5 — propofol sedation, powered negative (full cohort)

OpenNeuro **ds005620**, full **21/21 subjects**, 2527 windows, claim bootstrap.
Awake-vs-sedated discrimination, complexity `Z` beyond spectral `B`.

```
RFS = −0.2072 bits
awake   CI [−0.3084, −0.1208]
sedated CI [−0.3077, −0.1198]
```

Awake and sedated return the identical value **by construction**: they are two
mutually exclusive and exhaustive macrostates, so this is one binary
discrimination task scored from both label directions. Same conclusion as F4 under
a different etiology.

Scope: this is a passive-EEG complexity measure. It is **not** TMS-evoked PCI
(Casali-style) and carries no consciousness or clinical claim.

### F6 — autonomic → arousal (NSRR MESA/SHHS): built, not run

`Z` = autonomic block (HRV, respiration); `B` = cortical spectral + history;
`Y` = arousal within horizon. Loader written and unit-tested against synthetic
EDF stubs; synthetic known passes and known-null fails. **Real cohort unrun** —
NSRR credentialed access has been applied for, not obtained.

The prior evidence motivating this target has *temporal precedence*, which is what
makes it a forecasting rather than a correlational question: Bonnet & Arand (1997)
report heart-rate acceleration beginning **≥10 beats before** the scored EEG
arousal; Huo et al. (2023, *PLOS ONE*) characterise HRV/arousal coupling on MESA
specifically; Kuo et al. (2015) relate EEG beta power and HRV across cortical and
autonomic arousals.

**This is forecasting, not detection — the distinction is load-bearing.**
Arousal *detection* from ECG is close to solved: Basner et al. (2007) reach
AUC 0.91 from five post-onset beats, and Li et al. (2020) reach AUROC 0.93 on
MESA. Those models classify an arousal that is already underway, unconditioned.
Our estimand is strictly-future prediction at a fixed horizon, scored as
incremental value **beyond a cortical spectral + event-history baseline** — what
the autonomic block adds over what the cortex already tells you, before the event.
Any writeup that does not state this in its first paragraph will be read as a
worse version of Li 2020.

**Countervailing priors — there are three, and they are strong.**
1. Rao et al. (2025) tested cardiac autonomic predictors in a nested-model design
   and found nothing: HRV (RMSSD) partial r = 0.16, p = 0.29, **ΔR² < 0.03,
   p > 0.25** for subjective sleep quality. F6/F7 remain defensible only because
   their outcome is a *within-night, event-locked* arousal at a fixed horizon, not
   a night-level subjective rating — different estimand, different timescale.
   Recorded in `docs/research/rao_2025_methods_mining.md`.
2. **The autonomic-lead evidence is cohort-specific.** It is strong for
   *spontaneous* NREM arousals in healthy sleepers (Bonnet & Arand's ≥10 beats;
   Togo et al. 2006 at ~1 s before EEG desynchronisation; Attoh-Mensah et al. 2023
   finding cardiorespiratory shifts across the 30 s preceding micro-arousal) and
   **weak for respiratory-event arousals in apnea cohorts**, where airway
   reopening, hypoxia relief, cortical change and sympathetic discharge all land
   within seconds of each other. Bahr-Hamm et al. (2022) found no HRV-entropy
   difference for OSA respiratory cortical arousals and concluded they were
   uncoupled from autonomic activation.
3. Ribeiro et al. (2022, *PNAS*) find peripheral events usually precede cortical
   ones but **only about half recruit cortex** — which predicts a real-but-modest
   effect, not a large one.

Consequence for design: spontaneous and respiratory arousals must be **stratified,
not pooled**. Pooling averages a plausible effect against a documented null.

### F7 — autonomic → arousal on MIT-BIH slpdb: code complete, run pending

slpdb is the one fully open dataset carrying simultaneous cortical EEG and ECG
with scored arousals, so it can host the F6 estimand without credentialing.
Channels: ECG, BP, EEG (C4-A1), Resp; 250 Hz.

Two dataset facts, both verified empirically rather than assumed:

1. **Arousals are encoded as a suffix on the event token**, not as separate
   annotations. `LA` is a leg movement *with* arousal, `L` one without; likewise
   `HA`/`H` for hypopnea, `CAA`/`CA` for central apnea, `X` for obstructive apnea
   with arousal, `A` for a bare arousal. Sampling four records yielded 485
   arousal-marked events. Annotations also carry stray NUL bytes, stripped before
   parsing.
2. **18 records, 16 subjects.** `slp01a`/`slp01b` and `slp02a`/`slp02b` are the
   same people. Grouping by record would place one person on both sides of a
   subject-held-out split. The loader groups on the numeric prefix; the smoke run
   confirms the guard bites — given the first four records it reports 2 subjects,
   not 4, and refuses.

Signal handling follows the infraslow convention used in the gastric-coupling
literature: heart rate is derived by QRS detection (`wfdb` `xqrs`) on the raw
250 Hz ECG rather than fed as a raw waveform, and BP and respiration are carried
as log power envelopes on a 1 Hz grid rather than short-window Welch band power.
EEG is decimated to 32 Hz to keep the spindle/sigma range resolvable.

**Cohort caveat, recorded in `claim_scope`:** slpdb is a sleep-apnea cohort with a
high arousal base rate. It is not a healthy-sleeper sample and does not stand in
for MESA. Stronger than that: because its arousals are predominantly
respiratory-event arousals, it is the **sub-population where autonomic precedence
is least supported in the literature** (see F6). slpdb is therefore an adversarial
case for this hypothesis, not a small version of the real test — a null here is
close to uninformative about the MESA question, and must be reported as such.

---

## 5. Calibration evidence

Detection power is a rate over seeds, so a single run per condition cannot
estimate it. `scripts/autonomic_rfs_calibration_sweep.py` injects the
autonomic→arousal coupling at graded strengths into the synthetic fixture and runs
the full gate across seeds.

**24 seeds per point, synthetic ground truth:**

| injected coupling | detection power | median RFS |
| --- | --- | --- |
| 0.0 (null anchor) | **0.04** (1/24) | −0.0005 bits |
| 0.5 | 0.58 (14/24) | +0.0084 |
| 1.0 | **0.79** (19/24) | +0.0211 |
| 2.0 | 0.67 (16/24) | +0.0145 |

The null anchor fires on 1 of 24 seeds — **≈4%, essentially the nominal rate for a
one-sided 95% lower bound.** The single false positive (seed 2) has
`rfs_bits = 0.0022`, `ci_low = 0.00023`: barely clearing zero, which is what a
correctly calibrated borderline detection looks like. This is the quantitative
form of the claim that the gate does not rubber-stamp.

**Non-monotonicity at high coupling is real**, persisting across 24 seeds in both
panels (power 0.79 → 0.67; median bits 0.0211 → 0.0145). RFS is a *residual*: once
arousal becomes highly predictable from base rate and recent history alone, `B`
absorbs the signal and the residual attributable to `Z` shrinks. Detection power
is therefore the headline; the median-bits panel is reported alongside it and is
explicitly **not** a calibration curve.

Synthetic by construction. This validates the estimator before it is pointed at
real data; it is not evidence about any cohort.

---

## 6. Figure catalogue

### `runs/autonomic_calibration/autonomic_rfs_detection_power.png`
Two panels vs injected coupling strength: detection power (fraction of seeds whose
LCB clears zero) and median recovered RFS with a dashed zero reference. 24
seeds/point. **Shows:** the gate's operating characteristic on known ground truth.
**Does not show:** anything about real cohorts.
Regenerate: `PYTHONPATH=src python3 scripts/autonomic_rfs_calibration_sweep.py --out runs/autonomic_calibration --seeds 24 --strengths 0 0.5 1 2`
Data: `detection_power_sweep.csv`, `detection_power_per_draw.csv` (per-seed, for
auditing individual detections).
The suptitle is generated from the measured numbers; an earlier hardcoded version
asserted "no detections at the null anchor" while an 8-seed run was detecting 1
of 8, and would have contradicted its own data.

### `runs/amrith_isolated_check/isolated_forecast_sweep.png`
Isolated single-future-sample MSE vs forecast horizon (1–64 samples @100 Hz) for
random, mean-of-trace, persistence, ridge-AR, and the GRU, all trained directly
per horizon. **Shows:** the F2 tie. **Does not show:** any claim about
multi-channel or longer-timescale forecasting.
Regenerate: `PYTHONPATH=src python3 scripts/amrith_isolated_forecast_check.py --data <sleep-cassette> --out runs/amrith_isolated_check --subjects 8`
Data: `sweep.csv`, `headline.json`.

### `runs/amrith_waveform/amrith_waveform_diagram.png`
Two panels on one real Sleep-EDF window. Left: the old task, with the 126/127
shared region shaded so the input and target series visibly coincide. Right: the
isolated task — 127 known samples, then the true vs predicted strictly-future
sample. **Shows:** why the old metric read near-zero. The displayed window is not
cherry-picked and the model is visibly off on it, which is representative of
single-step EEG difficulty.
Regenerate: `PYTHONPATH=src python3 scripts/amrith_waveform_diagram.py --data <sleep-cassette> --out runs/amrith_waveform`

### `docs/paper/ieee_figure_source/figures/fig5_amrith_overlap.*`
The old-overlapping vs new-isolated MSE contrast (0.0018 vs 0.227). Source
`src/Figure5_amrith_overlap.py`, data `data/amrith_overlap_headline.json`, wired
into `render_all.py`.

### `docs/figures/hnph_protocol/`, `docs/analysis/ridge_eeg_figures/`, `docs/research/eeg_v1_figure_source/`
Protocol schematics, ridge diagnostics, and the eeg_v1 benchmark packet from
earlier commits. Provenance is in each directory's own source scripts.

---

## 7. Datasets

| dataset | access | role | note |
| --- | --- | --- | --- |
| Sleep-EDF Expanded (sleep-cassette) | open | F2, F3, F4 | no clean ECG; limits autonomic `Z` |
| BNCI2014-001 (MOABB) | open | F2 | motor imagery; benchmarking discipline per Jayaram & Barachant 2018 |
| OpenNeuro ds005620 | open | F5 | propofol sedation, 21 subjects, rest EEG only |
| MIT-BIH slpdb | open | F7 | EEG+ECG+BP+Resp, scored arousals, 16 subjects; apnea cohort |
| NSRR MESA / SHHS | credentialed | F6 | **applied for, not held** |

---

## 8. Related work and positioning

**On leakage.** Brookshire et al. (2024) and Del Pup et al. (2025) establish that
split methodology, not architecture, dominates apparent performance in
translational EEG. F1 extends that to within-window input/target overlap. The
guard and person-level audit are the enforcement mechanisms.

**On incremental value.** The nested-model design here is deliberately the same
shape as Rao et al. (2025) §2.5.2: a baseline model of macro-architectural and
covariate features, then the candidate block added and ΔR² / partial R² reported.
Our difference is cross-fitting and subject-held-out evaluation rather than
in-sample fit, plus a mandatory control battery. Rao et al. do report a
cross-validated ΔCVR² for one predictor (−0.015 for the phase-coupling measure),
so the CV-shrinkage concern is acknowledged in that literature.

**Read Rao correctly.** Their positive is *intrinsic* infraslow rhythmicity — EGG
ISO amplitude variance predicting subjective sleep quality (ΔR² = 0.13, p = 0.014,
n = 49 nights) — and their *coupling* measures added essentially nothing
(κ: ΔR² = 0.0005, p = 0.84; parietal MI: ΔR² = 0.046, p = 0.17). Their own
conclusion attributes the effect to consistency of infraslow modulation within
each organ rather than to stomach-brain coupling. Citing that paper as motivation
for a *coupling* claim misreads it.

**On directed information.** The companion `neuroforecast` line implements
causally-conditioned directed information graphs following Quinn, Kiyavash &
Coleman (2015), with transfer entropy (Schreiber 2000) as the pairwise special
case. That line addresses direction and direct-vs-mediated structure; the RFS gate
documented here addresses incremental predictive value under leakage control. They
are complementary, not competing.

**On sleep staging.** Chambon et al. (2017) and Eldele et al. (2021) are the
relevant deep baselines on the same data family. F4's negative concerns
*complexity features beyond a spectral baseline*, not staging accuracy; the
comparison is not to those systems' reported performance.

---

## 9. Limitations

1. **No positive real-data finding exists.** F2, F3, F4, F5 are negative; F6 is
   unrun; F7 is pending data transfer.
2. **The negatives are on targets where a null was the expected outcome.**
   Sleep-stage discrimination is well served by spectral features, so complexity
   adding nothing is unsurprising. This weakens the rhetorical force of the
   negatives even though it does not weaken their validity.
3. **F5 and F4 test one specific complexity block**, not "complexity" in general.
4. **Rao's HRV null is a real prior** against the broad autonomic story. The
   defence is timescale and outcome specificity, and it is a defence, not a
   refutation.
5. **slpdb is an apnea cohort.** Its high arousal base rate makes it a poor proxy
   for healthy-sleeper generalisation.
6. **Calibration evidence is synthetic.** It bounds the estimator's operating
   characteristic under a generative model we wrote; it is not external validation.
7. **The `B` block absorbs signal at high coupling**, so RFS is not monotone in
   effect size. Interpret detection power, not bits, as the calibration quantity.

---

## 10. Reproduction

```bash
# environment (CI-equivalent)
python3.11 -m venv .venv && .venv/bin/pip install -e '.[figures,moabb]' pytest ruff==0.16.0 edfio

# full suite
PYTHONPATH=src python -m pytest -q
uvx ruff@0.16.0 check .

# leakage regression + guard + person audit
PYTHONPATH=src python -m pytest -q \
  tests/data/test_forecast_no_overlap.py \
  tests/data/test_forecast_overlap_guard.py \
  tests/data/test_person_level_split_audit.py

# calibration
PYTHONPATH=src python3 scripts/autonomic_rfs_calibration_sweep.py \
  --out runs/autonomic_calibration --seeds 24 --strengths 0 0.5 1 2

# F5 propofol (requires ds005620 locally)
PYTHONPATH=src python3 scripts/run_propofol_pci_gate.py \
  --ds-root <ds005620> --bootstrap-mode claim --min-subjects 8

# F7 slpdb (requires local slpdb copy; streaming per-record over HTTP is impractical)
PYTHONPATH=src python3 scripts/run_slpdb_autonomic_gate.py \
  --out artifacts/slpdb_autonomic --bootstrap-mode claim --min-subjects 8
```

Every gate emits `claim_scope` and `stop_reason` into its JSON and Markdown
artifacts, so a run's permitted interpretation travels with its numbers.

---

## References

Bonnet, M. H., & Arand, D. L. (1997). Heart rate variability: sleep stage, time of
night, and arousal influences. *Electroencephalography and Clinical
Neurophysiology*, 102(5), 390–396.

Brookshire, G., et al. (2024). Data leakage in deep learning studies of
translational EEG. *Frontiers in Neuroscience*, 18, 1373515.

Chambon, S., et al. (2017). A deep learning architecture for temporal sleep stage
classification using multivariate and multimodal time series. *IEEE TNSRE*.

Del Pup, F., et al. (2025). The role of data partitioning on the performance of
EEG-based deep learning models in supervised cross-subject analysis. *Computers in
Biology and Medicine*.

Eldele, E., et al. (2021). An attention-based deep learning approach for sleep
stage classification with single-channel EEG. *IEEE TNSRE*.

Huo, J., et al. (2023). Coupling analysis of heart rate variability and cortical
arousal using a deep learning algorithm. *PLOS ONE*.

Jayaram, V., & Barachant, A. (2018). MOABB: trustworthy algorithm benchmarking for
BCIs. *Journal of Neural Engineering*, 15(6), 066011.

Kuo, T., et al. (2015). EEG beta power and heart rate variability describe the
association between cortical and autonomic arousals across sleep. *Autonomic
Neuroscience*.

Lawhern, V. J., et al. (2018). EEGNet: a compact convolutional neural network for
EEG-based brain-computer interfaces. *Journal of Neural Engineering*.

Lotte, F., et al. (2018). A review of classification algorithms for EEG-based
brain-computer interfaces: a 10 year update. *Journal of Neural Engineering*.

Quinn, C. J., Kiyavash, N., & Coleman, T. P. (2015). Directed information graphs.
*IEEE Transactions on Information Theory*, 61(12), 6887–6909.

Rao, A. A., Fredericks, M., Dresler, M., Rebollo, I., Zeitzer, J. M., Schoch,
S. F., & Coleman, T. P. (2025). Simultaneous stomach-brain electrophysiology
reveals dynamic coupling in human sleep. *bioRxiv* 2025.11.13.686572.

Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural
networks for EEG decoding and visualization. *Human Brain Mapping*.

Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*,
85(2), 461–464.

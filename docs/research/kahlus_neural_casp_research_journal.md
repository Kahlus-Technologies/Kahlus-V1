# Kahlus / Neural-CASP research journal

Working log covering the pivot from single-channel EEG forecasting to a
leakage-controlled forecastability arena, and everything that followed. Written
so it can be picked up cold: repo state, estimands, findings with numbers,
what's real, what's blocked, and what to do next.

Repos:
- `ptlnextdoor/Kahlus-V1` — the arena, gates, findings, adapters. `main` at
  `00979a3c` (merge of PR #72).
- `ptlnextdoor/neuroforecast` — certified directed-information graphs (separate
  line, Coleman-facing).
- `ptlnextdoor/kahlus-bench` — sealed synthetic benchmark that scores methods.

Local data: `/Users/aayu/datasets/kahlus_multidataset_public/{sleep-edfx,
ds005620}`. NSRR MESA/SHHS applied for, not yet in hand.

---

## What changed, in one paragraph

The original headline (Kahlus GRU beats ridge on next-sample EEG, MSE 3.116 / r
0.972) was an artifact. The task fed the model a 128-sample window as a
127-sample input and a 127-sample target shifted by one, so input and target
shared 126 of 127 samples. Scoring MSE over the full window rewards copying, not
forecasting. Once the metric is restricted to the strictly-future sample the GRU
only ties ridge. So the project stopped chasing a forecasting-skill headline and
turned into what it should have been: a leakage-controlled instrument for asking
whether a signal carries residual, subject-held-out information about a future
outcome, plus an honest set of results from running that instrument. Every
real-data result so far is a null, and every null is on a target where a null was
the expected outcome. The one target with real prior evidence for a positive
(autonomic activity predicting cortical arousal) needs a dataset that isn't on
disk yet.

---

## The estimand

For an outcome `Y` at horizon `h`, a nuisance/baseline block `B`, and a candidate
block `Z`, the residual forecastability score is

    RFS_bits = (NLL_B - NLL_{B+Z}) / ln 2  ≈  I(Y ; Z | B)

estimated by cross-fitted (out-of-fold) conditional log-likelihood, with the
split at the subject level (GroupKFold). `B` always contains base rate, recent
event history, and cortical spectral features, so a positive RFS means `Z` adds
information the cortex's own recent state does not already carry. Certification is
a subject-cluster bootstrap one-sided 95% lower bound; an edge is certified only
if LCB > 0. Controls that must collapse before any positive is allowed: label
shuffle, time shift, within-subject circular shift of `Z`, subject-identity probe
(≤ chance + 0.2), and cluster permutation for the p-value. This is the same
incremental-value question as a nested ΔCVR² model, cross-fitted and leakage-
audited, which is why it maps cleanly onto the Rao/Coleman nested-model design.

---

## The invalidation (F1)

The historical `6621642-ddpfix` `future_state_forecasting` sidecar reported MSE
3.116075 / Pearson 0.972108 against ridge 7.745 and persistence 8.204. That row
is now marked invalid in place (not deleted) in `docs/results/a100-run-history.md`,
`docs/audit/EXPERIMENT_VALIDITY_AUDIT.md`, and `docs/CLAIMS.md`, with a dated note
pointing at the audit. It is retained as evidence of the bug.

Mechanism: with input `X = seq[0:127]` and target `Y = seq[1:128]`, position
`Y[k] = seq[k+1]` is present in `X` for every `k` except the last. A model that
copies `X` shifted by one is exact on 126 of 127 positions; averaging MSE over the
window buries the single genuinely-future sample. Quantified on real Sleep-EDF
with the same ridge model:

| metric | ridge MSE |
| --- | --- |
| old full-127-window (overlap) | 0.0018 |
| isolated single future sample | 0.227 |

A 126x gap from one model, one dataset, two scoring rules. That 0.0018 is the
r≈0.97 illusion measured directly.

### The fix (PR #71, merged 2026-07-22)

Forecasting tasks now carry a strictly-future `metric_mask` that is true only on
target positions absent from the input window. The mask is applied to both the
training loss and the eval metric (`training/prepared_metrics.py`,
`benchmarks/baseline_suite.py`), so the copyable positions can neither be
optimized against nor scored. `forecastability/m0.py` refuses to score a
forecasting task that lacks the mask. `data/forecast_contract.py` adds a versioned
task contract with `target_overlaps_input` provenance so an overlapping,
claim-eligible task cannot be constructed silently.

Regression test (`tests/data/test_forecast_no_overlap.py`): a predictor built to
exploit the overlap scores ~0.059 unmasked and ~0.470 masked, an 8x jump. That
gap is the discriminating proof the mask works. (An earlier framing leaned on a
~1% agreement between the isolated-metric and masked-suite ridge numbers; that
agreement is supporting evidence only, since on near-stationary AR-like signals
per-position errors barely move under masking. The copy-trap jump is the real
test.)

---

## The arena and its results (PR #72, merged 2026-07-24, `00979a3c`)

Neural-CASP is the RFS gate plus a baseline ladder (must beat moving-average and
persistence), the control battery above, and the overlap audit, wrapped so every
run emits `claim_scope` and `stop_reason` in its artifacts. Shared horizon-shaped
primitives live in `forecastability/_rfs_eval.py`; state-shaped ones in
`forecastability/passive_pci.py`. The point of the arena is that it produces the
same kind of result whether the answer is yes or no, and it has been run on
targets across two etiologies.

### F2 — isolated EEG forecasting is a tie

Subject-held-out, strictly-future single-sample forecast on Sleep-EDF cassette
and BNCI2014-001. Every learned model trained directly per horizon (an earlier
version rolled a one-step GRU out autoregressively and it diverged, which was an
unfair setup and was corrected). Isolated MSE, lower is better:

| horizon (samples @100 Hz) | random | mean-of-trace | persistence | ridge | Kahlus GRU |
| --- | --- | --- | --- | --- | --- |
| 1 (10 ms) | 1.855 | 1.037 | 0.281 | 0.227 | 0.211 |
| 8 (80 ms) | 2.070 | 1.232 | 0.853 | 0.675 | 0.583 |
| 16 (160 ms) | 2.054 | 1.264 | 1.334 | 0.822 | 0.690 |
| 64 (640 ms) | 2.208 | 1.395 | 2.084 | 0.884 | 0.913 |

The GRU wins by a small margin at short leads and loses to ridge at 640 ms. Both
learned models clearly beat the trivial baselines, and the margin over
persistence grows with horizon. Read plainly: at single-channel millisecond scale
the signal is close to linear, so a 127-tap ridge captures nearly all of it and
the nonlinear model does not earn its complexity. Reproduce:
`scripts/amrith_isolated_forecast_check.py` (self-test gated).

### F3 — interoception scout, negative

Peripheral channels (EOG/EMG/respiration) as `Z`, EEG + history as `B`, future
sleep-stage transition as `Y`, on Sleep-EDF. Synthetic known-signal passes and
known-null fails, so the machinery is live; on real data the residual does not
survive controls at scout thresholds. Scout-grade (smoke), not powered. This is
the honest reason peripheral-on-Sleep-EDF is weak: the strong autonomic signal
lives in HRV/ECG, which Sleep-EDF cassette does not carry cleanly.

### F4 — passive complexity does not beat spectral, powered

Full Sleep-EDF cassette, 78 subjects, 413,828 windows, claim bootstrap
(n_boot=2000). Complexity block (Lempel-Ziv, permutation entropy, multiscale
entropy) as `Z`, spectral baseline as `B`, wake/NREM/REM one-vs-rest as `Y`. RFS
bits, all CIs excluding zero on the negative side: wake -0.330, NREM -0.284, REM
-0.161. The complexity block actively hurts relative to the spectral baseline.
`forecastability/passive_pci.py`, `complexity_features.py`.

### F5 — propofol sedation, powered negative (full cohort)

OpenNeuro ds005620, full 21/21-subject cohort (126 rest-EEG .vhdr files;
supersedes an earlier 12/21 disk-limited partial run), 2527 windows, claim
bootstrap. Awake-vs-sedated discrimination, complexity `Z` beyond spectral `B`.
RFS -0.2072 bits both ways: awake CI [-0.3084, -0.1208], sedated CI [-0.3077,
-0.1198]. Awake and sedated read the same value by construction, since they are
two mutually exclusive exhaustive macrostates and the discrimination is one
binary task scored from both label directions. Same conclusion as F4 under a
different etiology: complexity does not beat spectral. Reproduce:
`scripts/run_propofol_pci_gate.py --ds-root <ds005620> --bootstrap-mode claim
--min-subjects 8`. Fetch is per-subject fault-tolerant (`scripts/fetch_ds005620.py`);
subjects lacking a `task-sed_acq-rest` run are skipped rather than aborting the
pull.

### F6 — autonomic to arousal, built and synthetic-validated, real run pending

This is the reformulated target designed to have a plausible positive. `Y` = a
micro-arousal event in a future horizon, `Z` = an autonomic/peripheral block
(HRV, respiration), `B` = cortical spectral baseline plus recent-arousal history.
Estimand and gate identical to the above. Loader for NSRR PSG (MESA and SHHS as a
dataset-held-out second site) is written and unit-tested against a synthetic EDF
stub (`adapters/nsrr.py`, `tests/adapters/test_nsrr.py`); the channel-axis
stacking bug in the first draft is fixed (modalities stack on an explicit channel
axis, not concatenated along time). Synthetic known-signal passes and known-null
fails. It runs in one command once credentialed data lands.

The prior evidence for this target is real and includes temporal precedence,
which is what makes it forecastable rather than merely correlated: heart rate
accelerates at least 10 beats before the scored EEG arousal (Bonnet & Arand
1997); HRV-arousal coupling has been shown on MESA specifically (Huo et al.
2023); the coupling differs across NREM/REM (Kuo et al. 2015). Data is the only
blocker. The positive signal is HRV/ECG, which is not present in the public data
currently on disk, so there is no honest way to run this today on Sleep-EDF or
ds005620.

#### Calibration sweep, and why it is not yet presentable

`make_autonomic_fixture` gained a backward-compatible `strength` argument
(default 1.0, so existing tests are unaffected) that scales the injected
autonomic-to-arousal coupling. `scripts/autonomic_rfs_calibration_sweep.py`
sweeps strength and records recovered RFS with bootstrap CI. Single-seed result,
claim bootstrap:

| strength | RFS bits | LCB | certified (LCB>0) |
| --- | --- | --- | --- |
| 0.0 | -0.0008 | -0.0018 | no |
| 0.25 | 0.0009 | -0.0030 | no |
| 0.5 | 0.0059 | -0.0002 | no |
| 1.0 | 0.0211 | 0.0077 | yes |
| 2.0 | 0.0148 | 0.0022 | yes |
| 3.0 | 0.0081 | -0.0006 | no |

The null anchor reads ~0 and certification kicks in at real signal, which is the
good part. But RFS is non-monotonic: it peaks at strength 1.0 and rolls off. The
likely reason is that RFS is a residual (information beyond `B`), and at very high
coupling the arousal becomes predictable enough that the base-rate/recent-history
nuisance in `B` already captures it, shrinking the residual attributable to `Z`. I
have not confirmed that mechanism, and this is single-seed so the strength-3.0 dip
could be partly noise. The correct version is a multi-seed detection-power curve
(fraction of seeds certifying per strength). Until that exists this figure is not
something to put in front of anyone, and it is not used in any claim.

---

## Reproducibility notes worth keeping

CI on this repo runs `ruff check .` then `pytest`. Two real environment bugs
surfaced and are fixed:

1. Ruff version drift. CI installed ruff unpinned and resolved to 0.16.0, whose
   default rule set added import-sort checks that 0.15.9 (local) did not run, so
   CI reported 743 errors on code that passed locally. Fix: an explicit
   `[tool.ruff.lint] select = ["E4","E7","E9","F"]` in `pyproject.toml` so the
   lint contract is version-independent, plus `ruff==0.16.0` pinned in the CI
   workflow. Clean under both versions, zero code changes.
2. Missing test dependency. CI installed only `.[figures]`; the propofol and NSRR
   tests build synthetic BrainVision/EDF fixtures via `mne`, which lives in the
   `moabb` extra. Fix: install `.[figures,moabb]` in CI.

Both were caught by rebuilding CI's exact environment from scratch (fresh Python
3.11 venv, CI's literal install line) and again in a pristine clone, rather than
trusting the green check. Full suite in a clean clone: 875 passed, 5 skipped.
A `test_runner_bundle.py` ordering failure seen in the long-lived working
directory did not reproduce in the clone; it was local cruft from a day of manual
runs, not a repo bug.

---

## Code map (for a cold pickup)

- `forecastability/_rfs_eval.py` — shared horizon RFS primitives, cross-fit,
  controls, bootstrap, JSON/Markdown writers.
- `forecastability/passive_pci.py` — state-shaped RFS (F4), `evaluate_pci_fixture`.
- `forecastability/interoception_scout.py` — F3.
- `forecastability/autonomic_rfs.py` — F6 gate, `make_autonomic_fixture(strength=)`,
  `_evaluate_fixture`.
- `forecastability/propofol_pci.py` — F5, ds005620 loader.
- `forecastability/m0.py` — the trusted M0 harness; asserts the strictly-future
  mask.
- `forecastability/complexity_features.py` — LZ / permutation / multiscale entropy.
- `data/forecast_contract.py` — versioned forecast task contract, overlap
  provenance.
- `adapters/nsrr.py` — MESA/SHHS PSG loader (channel-axis correct, stub-tested).
- `scripts/` — `amrith_isolated_forecast_check.py` (F2), `run_propofol_pci_gate.py`
  (F5), `fetch_ds005620.py`, `autonomic_rfs_calibration_sweep.py`.
- Findings ledgers: `docs/results/findings-ledger.md`,
  `docs/paper/neural_casp_arena/findings.md`. Submission draft:
  `docs/paper/neurips_2026/submission.md`.

---

## Current status and next steps

The submission stands as an arena plus a set of powered honest negatives; it does
not need a positive to be a real contribution (NeurIPS Datasets & Benchmarks
shape). A positive would strengthen it, and the only target with real prior
support for one is F6, gated entirely on NSRR MESA/SHHS access (applied for). When
that lands, F6 is one command and returns either a certified positive (literature
says likely) or a powered null on a second modality.

Open, unblocked, not started: an overlap guard that fails task construction when
`forecast_horizon < window_length` without a strictly-future mask (Phase 2 of the
leakage remediation plan), and an adapter subject-id audit to confirm multi-night
recordings map to one subject and cannot leak across the split (Phase 3).

Open questions I actually want answered, some of them for Amrith:
- Is a physical/interventional experiment the more honest next step given the
  passive tie, and what is the minimal first one? Closed-loop acoustic
  stimulation or targeted reactivation timed to a detected state are the obvious
  candidates.
- Does threshold-finding methodology from implanted-electrode work transfer to
  non-invasive closed-loop paradigms at all.
- Why exactly is ridge so strong on single-channel EEG beyond "it's near-linear"
  — worth a raw-waveform diagnostic (input vs predicted, features labeled) that
  has not been made yet.

# Verified numbers for the BrainBodyFM paper — recomputed 2026-08-28

Every figure below was recomputed from the raw artifact
`docs/results/eeg_fm_benchmark_2026-08-24/paired_statistics_kahlus_reference.json`,
**not copied from SUMMARY.md**. All nine rows reproduce the summary exactly, so
the summary is trustworthy and can be quoted directly.

Reproduce: `scripts/verify_paper_numbers.py` (added alongside this file).

## The headline table — safe to put in the paper

Reference is `kahlus_gru`. Δ = best rival − Kahlus per test subject; positive Δ
means Kahlus has the lower error. n = 27 held-out subjects, percentile bootstrap
with 10,000 resamples, resampling **subjects** not windows. Holm-corrected
across a family of 8.

| h | Kahlus test MSE | best rival | Δ | Holm p | verdict |
|---:|---:|---|---:|---:|---|
| 1 | 0.3110 | cbramod full_ft | +0.0239 | — | not claim-eligible |
| 2 | 0.3624 | cbramod full_ft | +0.0152 | 0.0012 | **kahlus wins** |
| 4 | 0.4527 | cbramod full_ft | +0.0236 | 0.0001 | **kahlus wins** |
| 8 | 0.6357 | cbramod full_ft | +0.0506 | <0.0001 | **kahlus wins** |
| 16 | 0.8251 | cbramod full_ft | +0.0546 | <0.0001 | **kahlus wins** |
| 32 | 0.9720 | cbramod full_ft | −0.0015 | 0.9871 | tie |
| 64 | 1.0177 | labram_base full_ft | −0.0022 | 0.9713 | tie |
| 128 | 1.0467 | cbramod full_ft | +0.0085 | 0.0275 | **kahlus wins** |
| 256 | 1.1337 | labram_base full_ft | +0.0014 | 0.0871 | not significant |

**Claim that survives:** five wins of eight claim-eligible horizons after Holm
correction, three ties, **never a significant loss**.

**h = 1 is not claim-eligible** — it was dropped as leak-dominated
(`583cadf1`, "drop leak-dominated 10 ms horizon, gate it"). Do not quote its Δ
as a win. Quoting a horizon your own protocol excluded is the exact failure the
paper is about.

## The honest ceiling, which belongs in the paper

From h = 32 onward, **every entrant sits at test MSE ≈ 0.97–1.14** — at or worse
than predicting the mean. The "ties" there are *no model extracts signal at this
range*, not *the rival caught up*. Say so plainly; it costs nothing and pre-empts
the obvious reviewer question.

## Gate status — verified from `eeg_fm_benchmark_gate.json`

`scientific_claim_allowed = false`. Passed and failed:

| passed | failed |
|---|---|
| `baseline_table_present` | `beat_best_baseline` |
| `calibration_checked` | `controls_pass` (BIOT label-shuffle + time-shift) |
| `checkpoint_provenance_verified` | `no_grid_edge_selected` |
| `contamination_audit_passed` | random-twin kill criterion |
| `determinism_enabled` | `cluster_permutation_null` not implemented |
| `powered` (27 subjects, CIs within ratified MDE 0.10) | `subject_identity_probe` not implemented |

Claim scope: `external_pretrained_eeg_baseline_ranking_sleep_edf_forecast`.

**Lead with this.** A benchmark whose gate blocks its own headline is a stronger
paper than a benchmark that wins, especially at a workshop that convened a panel
on evaluation practice.

## The pretraining control

The random-init twin matched or beat the pretrained model in **25 of 72**
model × condition × horizon cells. EEGPT full-fine-tune: the twin is *strictly
better* at 7 of 9 horizons. LaBraM: at h = 2, 64, 128.

Two readings are consistent, and the run cannot separate them:

1. pretraining does not transfer to single-channel next-sample forecasting at
   ~1 s of context;
2. the adapters handicap the pretrained weights, so the twin control is
   uninformative rather than damning.

Four structural bottlenecks found in the PR #80 review push toward (2): EEGPT's
encoder is temporal-order-invariant as adapted (verified — permuting patches
leaves pre-pool tokens bit-identical); BIOT's STFT returns magnitudes only, so
phase is deleted; CBraMod's criss-cross attention sees a single token at one
channel (softmax weight ≡ 1.0, measured); and per-window FFT resampling puts
~13–21% of signal power into the last 4 samples, corrupting the most recent and
most predictive history.

**Write it as reading (2) unless Lane B says otherwise.** An earlier revision
said the evidence "leans toward (1)" and the review corrected it. Do not
un-correct it.

## F10 correlation law — cleared to cite

Merged via PR #81 (`e01e3d47`), verified an ancestor of `origin/main`.

- exact bound violated by 0 of 286 cells
- `W/H` shortcut violated by 223 of 286, always flattering
- white noise, W=32 H=1: reported r = 0.9843, true r_future = 0.0141
- mechanism to machine precision, median relative error 7.1e-16

## Numbers NOT to use

- **h = 1 Δ = +0.0239** — not claim-eligible, see above.
- **Anything from `paired_statistics.json`** (as opposed to
  `paired_statistics_kahlus_reference.json`) without checking the reference: it
  pairs against the *best incumbent*, which at h = 64 and 256 is `tuned_ridge_ar`,
  not Kahlus. An earlier revision mislabelled those rows and the review caught it.
- **`paired_statistics_incumbents.json`** — one unpushed commit (`617822ef`).
  Push before citing.
- **"4,000 µV" style raw HBN statistics** — that is drift, not signal.

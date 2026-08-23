# EEG foundation models vs Kahlus — result summary (2026-08-24)

Full detail: `EEG_FM_BENCHMARK_REPORT.md`. Machine-readable: `eeg_fm_benchmark.json`,
`paired_statistics.json`, `eeg_fm_benchmark_gate.json`. Audit trail: `attempts.log.gz`.

Scope: Lane A single-channel (Fpz-Cz) next-sample forecasting on Sleep-EDF, 39/12/27
subject-disjoint split, causal per-recording normalisation. This is one corpus, one task,
one lane. It says nothing about these models' published capabilities and nothing clinical.

## Headline

`kahlus_gru` ranks **first at six of nine horizons** (1, 2, 4, 8, 16, 128), second at one
(32), and third at two (64, 256) where the margins are 0.002 and 0.0005 — statistical ties.

On the paired subject-level test that actually decides ranking, Kahlus is **significantly
better than the best external entrant at five of eight claim-eligible horizons** (2, 4, 8,
16, 128 after Holm correction) and **statistically tied at the other three**. It never
significantly loses at any horizon.

| h | best rival | Δ (rival − kahlus) | 95% CI | Holm p | verdict |
|---:|---|---:|---|---:|---|
| 2 | cbramod full_ft | +0.0152 | [+0.0086, +0.0226] | 0.0012 | kahlus wins |
| 4 | cbramod full_ft | +0.0236 | [+0.0156, +0.0326] | 0.0001 | kahlus wins |
| 8 | cbramod full_ft | +0.0506 | [+0.0418, +0.0601] | <0.0001 | kahlus wins |
| 16 | cbramod full_ft | +0.0546 | [+0.0454, +0.0640] | <0.0001 | kahlus wins |
| 32 | cbramod full_ft | −0.0015 | [−0.0117, +0.0077] | 0.857 | tie |
| 64 | labram_base full_ft | −0.0002 | [−0.0077, +0.0059] | 0.330 | tie |
| 128 | cbramod full_ft | +0.0085 | [+0.0034, +0.0137] | 0.0221 | kahlus wins |
| 256 | labram_base full_ft | +0.0019 | [+0.0004, +0.0035] | 0.126 | not significant |

Horizons from 32 onward sit at test MSE ≈ 1.0–1.14 for *every* entrant, i.e. at or worse
than predicting the mean. The ties there are "no model extracts signal at this range",
not "the rival caught up".

## The negative control fired: pretraining bought nothing

Every external model ran twice — real pretrained weights, and a random-init twin of the
identical architecture. The twin matched or beat the pretrained model in **25 of 72**
model×condition×horizon cells, including EEGPT at 7 of 9 horizons under full fine-tuning
(its twin is *better* at h=1,2,4,8) and LaBraM at h=2, 64, 128.

Per plan section 5 this is a pre-registered kill criterion: it blocks the ranking claim
rather than annotating it. Two readings are consistent with the data and this experiment
cannot separate them:

1. Pretraining genuinely does not transfer to single-channel next-sample forecasting at
   ~1 s of context. These are masked-infilling classifiers evaluated off-task.
2. The adapters handicap the pretrained weights — documented handicaps include BIOT's
   single STFT token at this context length, EEGPT's monopolar `FPZ` standing in for the
   bipolar Fpz-Cz derivation, LaBraM's untrained `fc_norm` under the frozen probe, and
   context windows (1.0–1.25 s) far shorter than the 4 s+ these models were pretrained on.

Evidence leans toward (1) — LaBraM fine-tuned reaches 0.337 at h=1 versus Kahlus's 0.311,
so the architecture clearly learns the task, which is not how a broken harness behaves —
but "leans toward" is not "establishes". Resolving it needs Lane B (multi-channel, native
context), which is out of scope here.

BIOT is separately degenerate: 1.10–1.14 flat at every horizon, identical to its twin, and
it failed both corruption controls. Its rows are non-results, not weak results.

## Gate verdict: claim blocked (by design)

`scientific_claim_allowed = false`. What passed and what did not:

| passed | failed |
|---|---|
| `powered` (27 subjects, every CI within the ratified MDE 0.10) | `controls_pass` (BIOT label-shuffle + time-shift) |
| `protocol_ran_as_declared` | `beat_best_baseline` (ties at h=32, 64, 256 are not wins) |
| `single_test_look_per_family` | `no_grid_edge_selected` |
| `tuning_budget_symmetric` | random-twin kill criterion (above) |
| `determinism_enabled` | `cluster_permutation_null` not implemented |
| `checkpoint_provenance_verified` | `subject_identity_probe` not implemented |
| `index_audit_passed`, `normalization_causal` | |
| `harness_sanity_passed`, `incumbent_ladder_complete` | |

`beat_best_baseline` demands a win at *every* claim-eligible horizon and scores a tie as a
non-win. The gate refusing is therefore not "Kahlus lost" — it is the gate declining to
certify a ranking claim while a degenerate entrant sits in the table, two planned controls
are unimplemented, and three horizons are ties.

To convert this into a certifiable claim: implement the two missing controls, either drop
BIOT or give it a fair context, widen the grid axis that hit an edge, and restrict the
claim to the horizons where signal exists (2–16, plus 128).

## Provenance

- Shards ran at commit `0481891`, clean worktree, merged at `cd9bbb5`.
- 4 shards → 180 result rows; one test evaluation per (entrant, horizon) family, enforced
  by the append-only ledger.
- MDE 0.10 z-scored MSE, declared pre-data, ratified by Krish Garg 2026-08-21.
- 3 seeds, 14 configurations per entrant row, 10,000-draw subject-level bootstrap.
- All four checkpoints sha256-verified against the frozen manifest at load time.

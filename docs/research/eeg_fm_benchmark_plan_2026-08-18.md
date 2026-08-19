# External Pretrained EEG Models vs Kahlus Incumbents — Experiment Design

**Date:** 2026-08-18 · **Branch:** `kgarg2468/model-benchmark` · **Status:** prospective
design document. Nothing here has been run. No result in this file is evidence.

**Primary source:** `.context/eeg-fm-benchmark-research-2026-08-18.md` (condensed
two-worker literature and repo survey).

---

## 0. Purpose and claim boundary

### 0.1 What this plan is for

Place four external pretrained EEG models — CBraMod, EEGPT, LaBraM-Base, BIOT —
into the existing leakage-controlled forecasting harness alongside the Kahlus
incumbents (`kahlus_gru` from F8, the tuned ridge/AR ladder, persistence), under a
single protocol, a single normalisation convention, a symmetric tuning budget, and
a contamination audit that is machine-readable rather than prose.

This implements the P2 peer-review item already on the books:

> "Use tuned persistence, ridge, AR/VAR, TCN, GRU, Transformer, and real SSM
> baselines; **add LaBraM/BIOT only when task-compatible and contamination-audited**."
> — `docs/audit/REMEDIATION_ROADMAP.md:63`

and closes the contamination row that has been open since the leakage audit:

> "Pretraining contamination | not relevant to recovered GRU | **required before
> LaBraM/BIOT use** | P2" — `docs/audit/DATA_LEAKAGE_AUDIT.md:19`

### 0.2 The only claim shape this design can support

**Supported:**

> Under a leakage-controlled, subject-held-out Sleep-EDF strictly-future
> forecasting protocol (`kahlus.forecast.v2_nonoverlap`), with per-horizon tuning
> budgets matched across entrants and an identical bolt-on prediction head on every
> model, the ranking of {CBraMod, EEGPT, LaBraM-Base, BIOT, kahlus_gru, tuned
> ridge-AR, persistence} at horizons h ∈ {1…256} @ 100 Hz is *X*.

**Explicitly NOT supported, and never to be written, in any form:**

- "Kahlus is SOTA" / "Kahlus beats foundation models" / "Kahlus is a brain
  foundation model" — `AGENTS.md` TL;DR lists *brain foundation model* under
  "What Kahlus is NOT" and under "Never claim".
- Any statement about these external models' *published* capabilities. We are
  evaluating them **off-task**: they are bidirectional masked-infilling
  encoders/classifiers, not forecasters. A poor forecasting number is evidence
  about transfer to this task, **not** evidence that the model is weak.
  `docs/audit/LITERATURE_REPOSITORY_MATRIX.md:21` already records that their
  published classification metrics are not comparable to Kahlus forecasting MSE.
- Anything clinical, anything about seizure prediction, anything cross-dataset
  (this is one corpus), anything about scaling laws.

### 0.3 Gate predicate this design must satisfy

Per `AGENTS.md` TL;DR:

```text
GatePass = split_disjoint ∧ finite ∧ beat_best_baseline ∧ controls_pass ∧ powered ∧ scoped_claim
```

The `beat_best_baseline` term is what this whole exercise changes: after this run,
"best baseline" for Sleep-EDF forecasting includes four pretrained external models,
not just the internal ladder (persistence → ridge → GBM → standard EEG + B →
incumbent). That is the point. If the incumbents survive it, the F8 claim is
strictly stronger; if they don't, the ladder was incomplete and we say so.

---

## 1. Competitor slate

| # | Model | Params | Weights source | License | Why it is in |
|---|---|---:|---|---|---|
| 1 | **CBraMod** (ICLR 2025) | 4.9 M | HF `weighting666/CBraMod`; `braindecode/cbramod-pretrained` | MIT / Apache-2.0 | Strongest entrant under *independent* cross-subject benchmarks (OmniEEG-Bench full-FT rank 1). Released weights include a raw-patch reconstruction head (`proj_out`), so it can emit µV-scale waveform without us inventing a decoder. Positional channel handling means bipolar montages need no manual mapping. |
| 2 | **EEGPT** (NeurIPS 2024) | 25.3 M enc | figshare checkpoint (**must be mirrored**, see §8) + braindecode | Apache-2.0 | The only entrant whose pretraining corpus is disjoint from all three of Sleep-EDF / CHB-MIT / TUSZ (PhysioNet-MI, HGD, TSU, SEED, M3CV). Its 80 % channel-mask pretraining objective is the closest published analogue to our held-out-channel task. Upstream repo ships a Sleep-EDF harness. |
| 3 | **LaBraM-Base** (ICLR 2024) | 5.8 M | committed in upstream repo + braindecode | MIT | Field reference point; a reviewer will ask for it by name. Worst task fit of the four: VQ-token objective, **no raw-waveform decoder**, ≤256-patch cap (≈11 s at 23 ch), and known HP sensitivity — under-tuning it would be exactly the weak-baseline pathology this project exists to catch (§5.3). |
| 4 | **BIOT** `EEG-PREST-16-channels` | 3.4 M | upstream repo checkpoints | MIT | Its pretraining montage *is* CHB-MIT's bipolar layout, so it is the natural entrant if the arena ever extends past Sleep-EDF. **Hard rule: never the `six-datasets` checkpoint** — that one was pretrained on CHB-MIT + TUAB + TUEV directly. |

### 1.1 Explicitly excluded, with reasons recorded

| Candidate | Decision | Reason |
|---|---|---|
| **REVE** | Excluded for the Sleep-EDF arena | Documented collapse below 4 channels. Sleep-EDF (cassette) gives us 2 EEG channels. Excluding it here is a task-fit decision, **not** a quality judgement, and it may re-enter if a multi-channel corpus is added. |
| **LUNA** | Excluded pending legal review | CC-BY-ND. Fine-tuning and head-bolting plausibly produce a derivative work; this needs a human legal call before weights are downloaded, not after. Also TUEG-pretrained (contaminated for TUSZ). |
| **ZUNA 1.1** | Excluded, revisit | The only published model actually *designed* for masked-channel reconstruction — the closest conceptual match to our task — but the public port is encoder-only; the diffusion decoder is unreleased. Contacting the authors is the only path in. |
| **BrainOmni** | Deferred | Modern channel-flexible entrant, MIT. Reasonable fifth seat if the four above run cleanly; not in the v1 slate to keep the tuning budget affordable and symmetric. |

### 1.2 Integration path (design only — no code in this change)

- **braindecode as the unified adapter layer.** `Model.from_pretrained(...)`,
  loaders for `sleep_physionet` / `chb_mit` / `tuh`, and montage-interpolation
  wrappers. braindecode is already pinned in `external/upstreams.lock.json`.
- **Harness hook.** Each entrant becomes an `ExecutableBaselineRunner`
  (`src/neurotwin/benchmarks/baseline_runners.py:15-25`), registered in
  `EXECUTABLE_BASELINE_RUNNERS` (`src/neurotwin/benchmarks/baseline_suite.py:647`)
  **and** in `BASELINE_CATALOG` (`src/neurotwin/benchmarks/baseline_catalog.py`) —
  both, because a runner present in only one of the two is silently absent from a
  catalog-driven table. `gbm` was exactly that case until the sibling change on this
  branch registered it. Predictions must match `task.y_test.shape`.
- **Compute.** Every candidate is small (largest 25.3 M). On 1×A100-80 GB a frozen
  probe is minutes and a full Sleep-EDF fine-tune is roughly 1–3 h per model per
  lane. Compute is not the binding constraint; §8 is.

---

## 2. Contamination matrix

The rule that determines the arena: **a model may only be scored on a dataset its
pretraining corpus did not contain.** Pretraining contamination is not caught by
subject-disjoint splitting — the model saw the data before our split existed.

Statuses below are transcribed from `pretraining_contamination.py`; dataset ids are
that module's `SLEEP_EDF_EVAL_DATASET_ID` / `CHB_MIT_EVAL_DATASET_ID` /
`TUSZ_EVAL_DATASET_ID`.

| `model_id` | Sleep-EDF | CHB-MIT | TUSZ | Basis |
|---|---|---|---|---|
| `eegpt` | clean | clean | clean | Corpus is PhysioNet-MI, HGD, TSU-Benchmark, SEED, M3CV — disjoint from all three |
| `cbramod` | clean | clean | **contaminated** | TUEG-only pretraining; TUEG ⊇ TUSZ |
| `labram_base` | clean | *unverified* | **contaminated** | TUH corpora in the ~2,500 h mix cover TUSZ. CHB-MIT is not named, but the corpus list spans ~20 datasets plus unenumerated self-collected data, so overlap cannot be ruled out |
| `biot_prest16` | clean | clean | clean | `EEG-PREST-16-channels` is pretrained on MGH resting EEG only |
| `biot_six_datasets` | clean | **contaminated** | **contaminated** | Pretrained on TUAB, TUEV, CHB-MIT, IIIC, PREST, SHHS. **Banned from this benchmark entirely** — see below |
| LUNA (excluded from slate) | — | — | — | Not audited in the module because it is not an entrant; TUEG-pretrained, so TUSZ would be contaminated |
| `kahlus_gru`, ridge-AR, persistence | clean | clean | clean | No pretraining |

`biot_six_datasets` is `clean` on Sleep-EDF by the audit, and is still banned: the
same-family checkpoint `biot_prest16` is clean everywhere, so there is no scientific
reason to introduce a checkpoint that is unusable on two of three datasets, and
every reason not to leave one lying in the registry where a later run could pick it
up for CHB-MIT.

**Consequence:** Sleep-EDF is the only arena clean for every entrant, so the
**headline comparison lives there and only there.** CHB-MIT is a possible secondary
arena for the subset clean on it (EEGPT, CBraMod, BIOT `PREST-16`), gated on
resolving LaBraM-Base's *unverified* status with the authors. TUSZ is off the table
for a ranking claim regardless — two entrants are contaminated *and* no TUSZ
adapter exists in this repo (§8).

`unverified` is a distinct, load-bearing state: it is not "probably fine". Any
entrant still `unverified` on a dataset at run time is reported with its status in
the results table and excluded from the ranking claim on that dataset.

### 2.1 Machine-readable source of truth

This table is documentation, not the contract. The contract is
`src/neurotwin/benchmarks/pretraining_contamination.py` (sibling change on this
branch): a declared mapping of `model_id × dataset_id →
{clean, contaminated, unverified}` plus the evidence string for each cell, with
`contamination_status()` failing closed to `unverified` on unknown pairs and
`is_evaluation_permitted()` returning True only for explicitly audited `clean`
pairs. The runner must consult that module and refuse to score anything that is not
explicitly permitted; where
this document and that module disagree, **the module wins and this document is the
bug.** Its status column is copied verbatim into `baseline_table.csv` (§9).

---

## 3. Experiment design

### 3.1 Arena

| Item | Value |
|---|---|
| Dataset | PhysioNet Sleep-EDF Expanded, sleep-cassette |
| Protocol | `FORECAST_PROTOCOL_V2_NONOVERLAP` (`kahlus.forecast.v2_nonoverlap`, `src/neurotwin/data/forecast_contract.py`) |
| Sampling rate | 100 Hz |
| Task | strictly-future single-sample forecast: given `signal[i:i+127]`, predict `signal[i+127+h-1]`; **zero** shared samples at every horizon |
| Horizons | h ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} → 10 ms … 2560 ms |
| Split unit | subject (both nights mapped to one subject id), shuffled subject-disjoint, seed 0 |
| Split | **39 train / 12 validation / 27 test** of 78 subjects (≈50 / 15 / 35 %) — byte-identical to F8, `docs/results/forecast_crossover_2026-07-27.md:24-26` |
| Uncertainty | per-test-subject MSE → paired Wilcoxon signed-rank over 27 subjects + 10,000-draw bootstrap of the mean per-subject delta |

Holding the split fixed at F8's is deliberate: it makes every external model's
number directly comparable to an already-published, already-adversarially-validated
incumbent row rather than to a re-run of it.

### 3.2 Two lanes

**Lane A — single channel (`EEG Fpz-Cz`).** F8-comparable. The incumbent numbers in
`forecast_crossover_2026-07-27.md:48-58` drop straight in as a row. This lane is
*unfavourable* to the foundation models by construction — they were pretrained on
montages, and a 1-channel input discards the spatial structure they encode. Report
it as such; do not use Lane A alone to conclude anything about the FMs.

**Lane B — multi-channel.** All available Sleep-EDF cassette EEG channels, montage
handled per model (CBraMod positional, EEGPT name-keyed dict, LaBraM patch cap,
BIOT bipolar). This is the lane that is *fair to the FMs*, and it simultaneously
addresses F8's own stated limitation:

> "Nothing multichannel — this is one channel, which is the most likely reason the
> margin closes so early." — `forecast_crossover_2026-07-27.md:88-89`

Lane B therefore has independent scientific value even if every FM loses: it is the
multichannel replication that F8 §5 asks for
(`forecast_crossover_2026-07-27.md:191-193`), with Onojima et al. (2025) as the
comparator. The incumbents must be re-run in Lane B — a Lane-A incumbent number
must never be printed in a Lane-B table.

### 3.3 Three conditions per external model

| Condition | What it is | Label discipline |
|---|---|---|
| **(a) Acausal masked-infill** | Mask the future window; let the pretrained model infill it with full bidirectional context. Zero-shot (or few-train). | **Must be labelled "infilling", never "forecasting".** It sees samples after the target. It is an upper bound on the model's representational grasp of the signal, not a forecast, and it is not admissible in the ranking claim. Reported in a separate table with an explicit acausality warning. |
| **(b) Frozen encoder + trained head** | Encoder weights frozen; identical bolt-on head trained per horizon. | Causal, admissible. This is the cleanest measurement of "what did pretraining buy". |
| **(c) Full fine-tune** | Encoder + head trained end to end per horizon. | Causal, admissible. Most favourable to the FMs; the condition a reviewer will insist on. |

The ranking claim is computed over each model's **best admissible condition**
(max over b, c). Condition (a) exists to answer a different question — "is the
future window predictable at all from a pretrained EEG representation?" — and to
make it impossible for anyone to quietly reuse an acausal number as a forecast.

### 3.4 Symmetric tuning budget

Per `docs/research/baseline_fairness_literature_2026-07-27.md:73-79`, a
"tuned linear model matches the deep model" claim is only honest under a symmetric
tuning budget — and F8 demonstrated on its own data that an under-tuned baseline
*reverses the sign of the conclusion*
(`forecast_crossover_2026-07-27.md:93-96`).

Rules, per horizon, all selection on the 12 **validation** subjects only:

1. **Equal configuration count.** Every entrant, incumbent and external, receives
   the same number *N* of validation configurations per horizon per lane. *N* is
   fixed in writing before the first run and logged in the artifact.
2. **No edge selection.** If a selected hyperparameter lands on a grid edge, the
   grid is extended and the sweep re-run — for *every* entrant, not just the ones
   we like. Ridge-AR alpha runs the F8 14-point grid to `2e7`; the GRU gets its
   matched `(hidden, steps)` sweep; the FMs get learning rate, head width, and
   (condition c) unfreezing depth over an *N*-point grid.
3. **LaBraM gets a real sweep.** It is documented as HP-sensitive. Under-reporting
   it would be the McGreivy & Hakim weak-baseline pathology with the sign flipped
   in our favour, which is worse than losing.
4. **Seeds.** ≥3 seeds per selected configuration; seed spread reported alongside
   every mean, as in F8.
5. **Budget symmetry is reported, not asserted.** The artifact records
   configurations-tried, wall-clock, and gradient steps per entrant so a reader can
   check the symmetry claim instead of trusting it.

### 3.5 Incumbent ladder

| Entrant | Role |
|---|---|
| `persistence` | The floor. Beck et al. (2025): on volatile series no method reliably beats naive; it must always be reported. |
| `linear_ridge`, `autoregressive_ridge` | The tuned linear ladder, per-horizon alpha sweep to convergence |
| `gbm` | Required by the `AGENTS.md` baseline ladder; was runner-only until the sibling change registered it in `BASELINE_CATALOG`. Verify it appears in the table before trusting any ranking |
| `kahlus_gru` | The F8 incumbent: tiny GRU, context L=127, direct horizon regression |

`NeuralStateSpaceTranslator` is **not** an entrant. Its forecasting sidecar is
INVALIDATED (overlap artifact) and its masked reconstruction failed (r = −0.0125).
It must not appear in any table in this benchmark.

---

## 4. Controls

Every control below must be run for **every** entrant, external and incumbent, and
every one must collapse. Per `AGENTS.md` TL;DR: *label shuffle · time-shift ·
cluster-permutation null · subject probe ≤ chance + 0.2 · synthetic NULL must FAIL.*

| Control | Falsification condition |
|---|---|
| **Random-init architectural twin** | For each FM, the same architecture with randomly initialised weights, same head, same tuning budget. If the pretrained model does not beat its own random twin, its pretraining bought nothing on this task — and any ranking statement about "foundation models" collapses into a statement about architectures. **This is the most important control in the design.** |
| **Label shuffle** | A win that survives target permutation is not a win |
| **Time-shift** | A win that survives shifting targets off their true lag is not a win |
| **Subject-identity probe** | Any entrant whose representation predicts subject identity above chance + 0.20 is flagged. Motivated by the Identity Trap result (arXiv:2606.06647): identity leakage inflates FM scores by 13–89× in variance **even under subject-disjoint splits**. |
| **Synthetic NULL** | Must FAIL. If any entrant scores on a signal with no predictable structure, the harness is broken |
| **AR(2) sanity (F8 T5)** | On a stationary AR(2) process the minimum-MSE predictor is linear. Any entrant beating tuned ridge there indicates a harness artifact |
| **Index audit (F8 T1)** | Zero shared samples between context and target, verified at every horizon and every lane |

### 4.1 Metrics — and the normalisation blocker

- **Primary: z-scored MSE**, per test subject, one shared normalisation convention
  across all entrants and both lanes.
- **Secondary: Pearson r**, per test subject.
- **Tertiary: NLL / RFS-bits**, and *only* via an identical bolt-on Gaussian head
  fitted on every entrant's features.

Three hard conditions on the metrics:

1. **Fix or ablate the transductive z-scoring caveat first.** F8 currently z-scores
   each recording with its own full-recording mean and sd, which includes the test
   targets — standard practice, applied identically to every model, and explicitly
   listed as *not yet tested*
   (`docs/results/forecast_crossover_2026-07-27.md:174-179`). It was tolerable for
   a GRU-vs-ridge comparison where both models see the same normalisation. It is
   **not** tolerable across models with different pretraining-time normalisation
   assumptions, because a transductive statistic can interact differently with a
   pretrained encoder than with a ridge. Either switch to a causal /
   train-fitted normaliser, or run the benchmark under both and show the ranking is
   invariant. This is a **prerequisite**, not a robustness check (§8).
2. **One normalisation convention, stated once, applied everywhere.** No entrant
   may use its own pretraining-time normalisation for scoring; if a model needs its
   own input scaling, that happens inside its adapter and is documented, but the
   *metric* is computed in the shared convention.
3. **NLL and RFS-bits are our-head-on-their-features numbers.** None of these
   models has a native NLL or µV-scale probabilistic output; LaBraM cannot emit raw
   waveform at all. Every probabilistic number in this benchmark is a property of
   *our* Gaussian head reading *their* frozen features, and every table containing
   one carries that sentence. RFS bits are defined as
   `(NLL_baseline − NLL_model) / ln 2` per `AGENTS.md` §Layer 2.

---

## 5. Success and kill criteria

Decided now, before data.

### 5.1 What "Kahlus incumbents are top-2" requires

All of the following, jointly, in the Sleep-EDF arena, per lane:

1. `kahlus_gru` or the tuned ridge-AR ladder ranks in the top 2 by mean per-subject
   z-scored MSE at the horizon in question, against each FM's **best admissible
   condition** (b or c), not its worst.
2. The per-subject paired delta against the best FM has a bootstrap 95 % CI
   **excluding 0**.
3. Wilcoxon signed-rank across the 27 test subjects is significant after Holm
   correction over the 9 horizons.
4. **Every** control in §4 collapses, for every entrant — including the random-init
   twins.
5. Tuning-budget symmetry is demonstrable from the logged configuration counts.
6. Contamination status for every entrant on Sleep-EDF is `clean` in
   `pretraining_contamination.py`.

Fail any one and the claim is not made. There is no partial credit and no
"trending toward".

### 5.2 The expected outcome, stated in advance

The independent literature says the most likely result is a **tie or a modest FM
loss**, and we should not be surprised or disappointed by it:

- EEG-FM-Audit (arXiv:2605.26910): well-tuned supervised baselines match or beat
  EEG foundation models.
- arXiv:2605.11410: classical features recover ~79 % of the claimed FM advantage.
- arXiv:2607.24519: classical features beat FMs outright on CAUEEG.
- NeuralBench (arXiv:2605.08495): FMs "only marginally outperform task-specific
  models."
- Our own roadmap already plans for this shape:
  `docs/research/ROADMAP_TO_PUBLICATION.md:257-263` — three independent 2026
  results find richer architectures do not reliably help on EEG, and it instructs
  us to **"Plan the null as the deliverable."**

So both directions are publishable and neither is a failure:

- **Incumbents hold.** "A tiny GRU and a converged ridge match or beat four
  pretrained EEG models on leakage-controlled short-horizon Sleep-EDF forecasting,
  under matched tuning budgets" — a fourth independent confirmation of the
  weak-transfer finding, from the forecasting angle rather than classification.
- **FMs win.** Then the F8 claim is correctly narrowed in public, the ladder gets a
  new rung, and we have quantified what pretraining is worth on this task via the
  random-init twins. Also a result.

The one outcome that is *not* acceptable is a ranking produced by asymmetric
tuning, an acausal condition mislabelled as forecasting, or repeated test-set
looks.

### 5.3 Iteration policy (the "tweak until top-2" trap)

**Allowed:** iterate architecture and hyperparameters against the 39 train + 12
validation subjects, under symmetric budgets, for as long as we like.

**Allowed once:** a single held-out test evaluation per *configuration family*.

**Forbidden:** re-evaluating on the 27 test subjects and then changing anything
because the rank was disappointing. That is tuning on test — the exact leakage
class this repo exists to detect, and it would be self-refuting.

**Logged:** every attempt, every configuration family, every test evaluation, with
timestamps, in the run artifact. The count of test-set evaluations is reported in
the paper. A benchmark that hides its attempt count is not a benchmark.

The sealed confirmation holdout (`scripts/seal_holdout.py`) has still not been used
for F8; if it is used here, it is used **after** the ranking is fixed in writing,
not as an escape hatch from a bad result.

---

## 6. Proposed claim scope (NOT implemented — requires human sign-off)

`NARROW_CLAIM_SCOPES` (`src/neurotwin/gates/unified_gate.py:42-58`) is deliberately
constructed so that **no scope in it permits a SOTA, baseline-superiority, or
clinical claim** — that is stated in its own docstring. A cross-model ranking is a
baseline-superiority-adjacent claim, so it cannot be smuggled in under an existing
scope such as `eeg_future_forecasting_benchmark_ready`, which covers harness
readiness only.

**Proposed addition, for human governance review only:**

```python
"external_pretrained_eeg_baseline_ranking_sleep_edf_forecast"
```

Intended semantics, to be written into the docstring if approved:

> Permits reporting a *ranking* of named external pretrained EEG models against the
> internal baseline ladder, on the Sleep-EDF `kahlus.forecast.v2_nonoverlap`
> forecasting task only, at declared horizons, under a declared symmetric tuning
> budget, with every entrant `clean` in `pretraining_contamination.py` for that
> dataset. It does **not** permit: SOTA claims, claims about the external models'
> published tasks, claims on any other dataset, claims from acausal infilling
> conditions, or any clinical claim.

**Governance status: NOT IMPLEMENTED. Requires explicit human approval.** This
document proposes wording; it does not authorise the edit. Adding a scope to that
allowlist widens what the gate will let through, which is a policy change, not an
engineering change. An agent must not add it.

---

## 7. Sequencing

```text
P0  clear blockers (§8)                     ← currently blocking everything
P1  contamination module + adapters + registry rows (code, sibling changes)
P2  transductive-normalisation fix/ablation on the existing F8 lane
P3  Lane A dry run, CPU/small GPU, 2 subjects, all entrants, controls only
P4  Lane A full run (39/12/27), conditions a/b/c, 9 horizons
P5  Lane B full run (multi-channel), incumbents re-run in-lane
P6  gate → artifacts → write-up → (if approved) claim scope
```

No step may start before its predecessor's artifact exists and has been read.

---

## 8. Prerequisites and blockers

Nothing in §7 P4 onward may begin until every box below is checked. These are
blockers, not risks.

- [ ] **`p0_remediation_complete=false` freeze.** "No A100 work is allowed until R1
      through R6 are merged and their local protocol gates pass" —
      `docs/audit/REMEDIATION_ROADMAP.md:31-33`. R0/R1/R2 are recorded as merged;
      **R3–R6 status must be verified against the roadmap before any GPU
      allocation.** This design is written under the freeze and does not lift it.
- [ ] **`huggingface_hub` quarantine.**
      `tests/artifacts/test_docs_contracts.py:149-172` forbids `huggingface_hub`
      (among others) from appearing in `pyproject.toml`,
      `deploy/a100/environment-a100.yml`, and `requirements/cluster-a100.txt`. All
      external-model dependencies therefore live in a separate
      `requirements/eeg-fm-baselines.txt` (added in the sibling change), installed
      explicitly, never added to those three files. Do not "fix" the test.
- [ ] **Transductive z-score fix or ablation** (§4.1 condition 1) landed on the F8
      lane, with the F8 numbers re-confirmed or amended under the new convention,
      before any cross-model number is computed.
- [ ] **EEGPT figshare checkpoint mirrored.** It is the only contamination-clean
      entrant and it lives on figshare with no redundancy. Mirror it to controlled
      storage with a recorded SHA-256 **before** the run, not during. If it
      disappears un-mirrored, the cleanest arm of the benchmark is unreproducible.
      Checkpoint hashes go in the artifact.
- [ ] **`pretraining_contamination.py` merged** and consulted by the runner
      (sibling change on this branch).
- [ ] **`gbm` present in the produced `baseline_table.csv`**, not merely registered
      in `BASELINE_CATALOG`, so a ladder rung cannot silently vanish.
- [ ] **Weight licences reviewed** for redistribution and derivative-work terms;
      LUNA's CC-BY-ND question answered by a human before download.
- [ ] **No TUSZ adapter exists** (constants only, `m3.py`). TUSZ is out of scope for
      this benchmark; do not let a "quick TUSZ check" appear in the run plan.
- [ ] **Compute booked only after the above.** All four models fit on 1×A100-80 GB;
      the constraint is governance, not hardware.

---

## 9. Artifact contract for the eventual run

Follows the mandatory pattern from `AGENTS.md` (*runner → gate → JSON + Markdown*)
and the conventions in `src/neurotwin/forecastability/m0.py:28-79`.

Output root: `artifacts/eeg_fm_benchmark/`

| File | Content |
|---|---|
| `environment.json` | `_environment_payload()` convention: interpreter, platform, package versions |
| `git_state.json` | `_git_state()` convention: commit, dirty flag; a dirty worktree fails the gate unless explicitly waived |
| `baseline_table.csv` | one row per (`model_id`, lane, condition, horizon, seed); columns include mean per-subject MSE, Pearson r, NLL, RFS bits, **`contamination_status`**, `configs_tried`, `grid_edge_selected` |
| `controls.json` | every control in §4 per entrant, with pass/fail and the falsification threshold that was applied |
| `eeg_fm_benchmark_gate.json` | gate dossier: `split_disjoint`, `finite_metrics`, `beat_best_baseline`, `controls_pass`, `powered`, `claim_scope`, `contamination_audit_passed`, `tuning_budget_symmetric`, plus `stop_reason` |
| `EEG_FM_BENCHMARK_REPORT.md` | human-readable report generated from the JSON, never hand-edited |
| `checkpoints.json` | source URL, SHA-256, licence, and mirror location for every external checkpoint used |
| `attempts.log` | §5.3 iteration ledger: every configuration family and every test-set evaluation, timestamped |

Contract rules, inherited from the repo:

- Committed artifacts must match the code that produced them.
- Raw neural data never enters git; `runs/` is gitignored, so the write-up cites the
  `artifacts/` copies.
- Determinism: two rounds, hash-compared, as `run_m0_gate` does with
  `baseline_table_sha256_round_1/2`.
- The gate is conjunctive. A single failed check sets
  `scientific_claim_allowed=False` and records why.

---

## 10. Sources

**Benchmarks and audits of EEG foundation models**

- Channel Adaptation for EEG Foundation Models — arXiv:2604.23091 (CBraMod beats
  31×-larger models on 4 of 5)
- OmniEEG-Bench — arXiv:2606.00815
- NeuralBench — arXiv:2605.08495
- NeuroAtlas — arXiv:2605.14698
- EEG-FM-Audit — arXiv:2605.26910
- Identity Trap (subject-identity leakage under subject-disjoint splits) —
  arXiv:2606.06647
- Negative-control protocol / CAUEEG classical-feature result — arXiv:2607.24519
- Classical features recover ~79 % of FM advantage — arXiv:2605.11410
- Del Pup / Zanola, data partitioning — arXiv:2505.13021
- Beyond Accuracy (channel-dropout confound) — arXiv:2605.17562

**Forecasting comparators** (see
`docs/research/baseline_fairness_literature_2026-07-27.md` §3 for the full list)

- Pankka et al. (2024) — WaveNet beats AR at 150 ms on resting-state EEG. Same
  regime as our h=16; "a deep model beats AR at ~150 ms" is therefore *not* novel.
- Onojima et al. (2025) — complex-valued analytic-signal prediction from raw
  multichannel EEG; the direct comparator for Lane B.
- McGreivy & Hakim (2024, *Nat. Mach. Intell.*) — 79 % of ML-for-PDE papers beat
  only a weak baseline. The template for §3.4.
- Beck et al. (2025) — "Mind the naive forecast!" Persistence is the mandatory floor.

**Reference-hygiene flag (do not act on this in this change)**

`AGENTS.md` §13 cites a "Liu et al. 2026 EEG foundation-model survey". That
reference **could not be located on arXiv** during this research pass. The closest
matches are arXiv:2508.15716 and arXiv:2604.23525. This is flagged for a human to
correct in `AGENTS.md`; **this document does not edit `AGENTS.md`**, and no claim
here depends on that citation.

---

## 11. What this document is not

It is not a result, not an authorisation to spend A100 time, not an approved claim
scope, and not a commitment that the Kahlus incumbents will win. It is a
prospective protocol written before the data, so that whichever way the ranking
falls, the answer is one we are obliged to publish.

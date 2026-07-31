# Hypothesis register

Every hypothesis carries a **falsifier written before the experiment runs**. That
ordering is the whole point: a falsifier written afterwards is a rationalisation,
and an agent that can move its own goalposts will always find a positive.

Append-only. Never delete a row — move it to **Killed** with a date and reason.

---

## Rules

1. **Pre-register or don't run.** If the falsifier column is blank, the
   experiment is not allowed to start.
2. **A killed hypothesis is a good outcome.** It closes a decision and stops the
   program wasting cycles. Killing something counts as a successful cycle.
3. **Never re-open a killed line** without new evidence, and record what the new
   evidence is. The Killed table exists so the program stops re-litigating.
4. **Controls are mandatory.** Every experiment runs a null that must fail *and*
   a positive control that must pass. Without both you cannot distinguish "no
   signal" from "broken pipeline" — the most common way a research loop wastes a
   month.
5. **Inconclusive is a legitimate verdict.** Record it. Do not convert an
   inconclusive into a weak positive.

---

## Open

| ID | Hypothesis | Decision | Falsifier (pre-registered) | Status |
| --- | --- | --- | --- | --- |
| H1 | Prediction error keeps improving with model capacity/class — there is real headroom beyond the GRU | D1 | If ridge-127, a 4× larger GRU, a temporal CNN and a transformer all land within 5\% of each other at h=16, H1 is dead and we are at a practical ceiling | open |
| H2 | Cortical state transitions are predictable at ≥30 s lead from EEG alone | D2 | If a controlled attempt at 30 s lead fails to beat the base-rate/persistence baseline with CI excluding zero, on ≥40 subjects, H2 is dead for this signal | open |
| H3 | The F8 crossover moves to a longer horizon with multichannel input | D3 | If the crossover horizon with 3 channels is within one horizon step of the single-channel result, H3 is dead — spatial information does not extend forecastability | open |
| H4 | The `W/H` bound generalises beyond the shifted-window construction | D4 | If any other overlapping construction produces an inflation exceeding its derived copyable-position bound, H4 is dead | open |
| H5 | A material fraction of published EEG forecasting work uses `H < W` | D5 | If ≥60 papers are coded and the `INFLATED` rate is under 10\% with `UNRECOVERABLE` also low, H5 is dead and F9 stays a tool rather than a field finding | open |

## Killed

| ID | Hypothesis | Killed | Why | Evidence |
| --- | --- | --- | --- | --- |
| H6 | F9's `W/H` bound transfers from MSE to reported correlation, so one methods-section number audits both metrics | 2026-07-30 | **Falsified, 223/286 cells.** The exact bound is `1 + ((W−H)/H)·(s_pred·s_targ)^(−1/2)`; `W/H` is only the `s_pred = s_targ = 1` case. Regularised models shrink toward the mean exactly where the signal is unpredictable, so `s_pred < 1` (median 0.700) and the true deflation *exceeds* `W/H` — median 1.21×, up to 10.8×. `W/H` understates correlation inflation, and does so worst where there is least real signal. **Pre-registration caveat:** H6 was never in the Open table. Its falsifier was encoded in the harness (`gap_bound_W_over_H`, `wh_respected`) before the data run, so the test was not tuned to its outcome — but it does not carry the full guarantee the Open table is for, and is entered here rather than claimed as pre-registered. | `docs/results/correlation_inflation_law_2026-07-30.md`, `artifacts/correlation_law/` |

## Inconclusive — needs more power or better method

| ID | Hypothesis | Blocked on |
| --- | --- | --- |
| — | *(none yet)* | |

---

## Cycle log

One line per cycle. `advanced` / `killed` / `inconclusive` / `blocked`.

| Cycle | Date | Decision worked | Outcome | Note |
| --- | --- | --- | --- | --- |
| 0 | 2026-07-28 | — | setup | Register created; nothing run yet |
| 1 | 2026-07-29 | D1 | advanced | Built the capacity-ceiling harness (`scripts/capacity_ceiling.py`). Also merged the F8 branch into `research/auto-survey` — `forecast_scaleup_paired.py` was not reachable from this branch, which the loop prompt tells the agent to reuse. |
| 4 | 2026-07-30 | D3 | **retracted a claim of mine** | I asserted that the public `eeg-forecastability-limits` BNCI result (skill dying 64–256 ms) was a cross-dataset replication of F8's Sleep-EDF crossover (advantage gone by 320 ms). **Both halves were wrong.** (a) The BNCI 64–256 ms figure was an artifact of a fixed ridge `alpha=1.0` across 2,816 features; with alpha selected on a held-out training subject it survives to ~512 ms. (b) They are not the same quantity — F8 measures *GRU beats tuned ridge* (an architecture advantage), BNCI measures *AR beats the trace mean* (any skill at all). Two different comparisons agreeing in order of magnitude is not a replication. D3 stays open; a real cross-corpus test has to run the same estimand on both. |
| 3 | 2026-07-30 | D4 | **killed H6** | Derived the correlation analogue of F9 (F10), verified 0/286 violations on the exact bound — and falsified the convenient extrapolation that `W/H` carries over. It does not, 223/286, and it errs in the flattering direction. Recorded as H6 in Killed — see the caveat on that row about its pre-registration status. |
| 5 | 2026-07-31 | — | **corrected a number on the record** | The F12 commit message (`826c0ec`) and the primer both reported 5-class staging as "0.732 vs 0.734 balanced accuracy" for stage-only vs +EEG. **0.734 does not appear anywhere in the run log or the artifact.** The committed values at 60\,s are stage-only **0.732**, +EEG **0.713**, EEG-only **0.633** (`artifacts/stage_forecast/transition_detection_n1_stratified.json`, `run.log` line 395/1115). The direction of the finding is unchanged and in fact stronger: adding EEG to the current stage does not cost 0.002 balanced accuracy, it costs **0.019**. Caught by `docs/paper/primer/verify_primer.py`, which re-reads every primer number from its artifact. The commit message is immutable; this row is the correction of record. |
| 6 | 2026-07-31 | — | **F12 re-running at full scale** | Auditing F12's artifact for the primer surfaced two problems beyond the mis-transcribed number in cycle 5. (a) **It was run on 10 subjects, not 60** (`n_subjects: 10`, split 6 fit / 1 select / **3 test**) — the script's default is 60, so this was an explicit `--n-subjects 10`. Three test subjects is the same under-powering just fixed in the BNCI sweep, and the primer quoted F12 beside F11's 60-subject result without stating the difference. (b) **11 of 24 hyper-parameter selections sat at a grid edge** on a `C` grid of (1e-4 .. 1.0), including the stage-only baseline that the headline negative rests on. Fixes: `CVALS` extended to (1e-7 .. 1e2) and an `at_grid_edge()` helper now prints a warning the moment a selection pins, mirroring the BNCI fix. Re-running on the full 60 subjects (36 fit / 9 select / **15 test**, matching F11). Until it lands, F12's numbers in the primer are the 10-subject ones and are flagged as provisional. |
| 2 | 2026-07-29 | D1 | **harness rejected** | TESTMODE fixtures built (`scripts/testmode_fixtures.py`), harness scored **3/4**. Failed `small_win`: 4.5% planted nonlinear headroom, best model −2.6% vs ridge. On real data this would have fired H1's falsifier and killed the hypothesis for an optimisation reason, not a data reason. Root cause: one shared `lr=2e-2` across four families; the transformer diverged everywhere (−87%, −134%, −283%). Fixed with per-family LR ranking + early stopping. Re-validating. |

# The Kahlus Handbook

**Everything in this project, taught from zero.**

Last updated 2026-07-31. If a number here disagrees with a slide, an email, or an
older doc, **this file is right and the other thing is stale.** Every number was
read out of a run log or `results.json`, not from memory.

You need no background to read this. Terms are defined the first time they appear.
Read §1 and §3 and you understand the project. Read the rest when you need it.

---

<!--TOC-START-->
## Contents

1. [What Kahlus is, in one page](#1-what-kahlus-is-in-one-page)
2. [The vocabulary](#2-the-vocabulary)
3. [The core idea: how forecasting scores lie](#3-the-core-idea-how-forecasting-scores-lie)
4. [The experiments, one at a time](#4-the-experiments-one-at-a-time)
5. [The code](#5-the-code)
6. [The literature](#6-the-literature)
7. [The honest ledger](#7-the-honest-ledger)
8. [How to talk about this](#8-how-to-talk-about-this)

---

<!--TOC-END-->
## 1. What Kahlus is, in one page

**The original goal.** Predict what the brain is about to do, from EEG, far enough
ahead to matter — motivated by seizure prediction, where a minute of warning changes
someone's life.

**What actually happened.** The first big result was wrong. Not fraud, not a typo —
wrong in a way the whole field is exposed to. Chasing down *why* turned out to be
worth more than the original goal, and is now the centre of the project.

**One sentence:**

> Standard ways of scoring a forecast can make a model look far better than it is,
> by amounts you can compute exactly — and there are three separate mechanisms that
> do it.

| # | Mechanism | Effect |
| --- | --- | --- |
| 1 | The scored target window overlaps its own input | Error looks up to `W/H`× better than it is |
| 2 | The same bug in *correlation* instead of error | Worse, and the obvious formula **understates** it |
| 3 | The ground-truth labels are unreliable | Scores partly measure label noise |

**Why it matters.** All three are *measurement* problems, not modelling problems.
They apply to other people's work, on other datasets, with better models. A paper
about them is a paper about how the field measures itself.

**Where the biology stands.** We have real, statistically solid forecasting results.
They are modest, and other groups have published better ones (§6). The measurement
work is the contribution.

---

## 2. The vocabulary

### Signals

**EEG** — electrical voltage on the scalp from populations of neurons.
**Sampling rate (`fs`)** — measurements per second. Sleep-EDF 100 Hz, BNCI 250 Hz.
**Channel** — one electrode pair. `EEG Fpz-Cz` = voltage between forehead and top of head.
**EOG** — eye movement. **EMG** — muscle activity. Both recorded alongside EEG; both
contaminate it, and both are *required* to score sleep.
**PSG (polysomnography)** — the full overnight recording.
**Epoch** — a fixed chunk, conventionally 30 s in sleep work. One stage label per epoch.

### Sleep

**Stages** — **W** (wake), **N1** (drowsy/lightest), **N2** (light), **N3** (deep),
**REM** (dreaming). A night is a sequence of these.
**Hypnogram** — the timeline of stage labels, produced by a human expert. **This is
our ground truth, and §3.3 is about why that's a problem.**

### Forecasting

**Horizon (`h`/`H`)** — how far ahead you predict.
**Window / context (`W`/`L`)** — how much past data the model sees.
**Persistence baseline** — the dumbest forecast: *"what's happening now will still be
happening then."* **This is the baseline that matters.** For slow signals it is very
hard to beat, and a paper comparing only against something weaker is not telling you much.
**Majority / trace-mean baseline** — always guess the most common answer. Beating it
proves almost nothing.

### Scoring

**MSE** — mean of (prediction − truth)². Lower better. On normalised data, **1.0 = guessing the mean**, so below 1.0 beats guessing.
**Correlation `r`** — −1 to +1. `r²` ≈ fraction of variance explained; r = 0.3 means ~9%. Use `r²` to stay honest.
**Accuracy** — **dangerous with imbalanced classes.** If 80% of a night is N2, always saying "N2" scores 80% while knowing nothing.
**Balanced accuracy** — accuracy averaged per class. **Chance = 1/classes; for our 5 stages, 0.20.**
**AUC** — for yes/no questions. Probability the model ranks a true positive above a true negative. **0.5 = chance, 1.0 = perfect.** Immune to class imbalance.
**AP (average precision)** — like AUC but focused on the rare positive class. Always compare to the **base rate**.

### Models

**Ridge regression** — linear model with a penalty `alpha`. Too small → memorises noise; too large → predicts nothing. **Tuning it properly is not a detail — §4.6 is an entire finding about getting it wrong.**
**Logistic regression** — the classification version.
**GRU** — a neural net for sequences, with memory.
**Gradient boosting** — many small decision trees, each fixing the last one's errors.

### Discipline

**Subject-held-out** — train on some people, test on *different* people. Non-negotiable: two windows from one person aren't independent.
**Train / validation / test** — fit on train, choose settings on validation, report on test. **Choosing settings using test data is cheating** even by accident — the "winner's curse", which we shipped once (§7).
**Leakage** — the model sees what it shouldn't. §3 is a kind the usual guards miss.
**Causal vs non-causal filter** — a causal filter at time *t* uses only data before *t*. Non-causal ones see the future — standard offline, **but leakage in a forecasting experiment** (§4.5).
**p-value** — chance of seeing this if nothing real were happening. **Confidence interval** — plausible range; "excludes zero" means probably real.
**Wilcoxon signed-rank** — paired test: does the model beat the baseline *in the same people*, consistently? Stronger than a pooled number, which one odd subject can carry.
**Pre-registration** — writing down what would prove you wrong *before* running. `REGISTER.md` does this, because it is very easy to decide afterwards that whatever you found is what you wanted.

---

## 3. The core idea: how forecasting scores lie

This section is the project. Understand it and you can defend the work.

### 3.1 Mechanism one — the target contains the input

A very common way to set up EEG forecasting:

```
input  X = samples 0 .. W          (window of length W)
target Y = samples H .. H+W        (same length, shifted by H)
```

Looks reasonable — "predict the next window." **It is broken.**

Because the shift `H` is smaller than the window `W`, the target **overlaps** the
input. With `W = 127`, `H = 1`, the target shares **126 of its 127 samples** with the
input. One sample is genuinely in the future.

A model scores brilliantly by *copying*. It is not forecasting; it is transcribing,
and being graded on the transcription.

**Why the usual safeguards miss it.** Split by subject, cross-validate, avoid
overlapping train/test — none help. This leak is **inside a single example**. The
example is corrupt on its own. The leakage literature (Bernett et al., *Nature
Methods* 2024; Del Pup et al. 2025, five CV schemes over ~10⁵ trained EEG models;
Apicella et al. 2024) all treat information shared *between* examples.

**The formula.** Score one model's one set of predictions three ways — all positions,
only copyable, only genuinely-future:

```
MSE_overlap = [ (W−H)·MSE_copy + H·MSE_future ] / W
```

An **identity** — arithmetic, not a fit. Rearranged:

```
inflation = MSE_future / MSE_overlap  ≤  W / H
```

**In practice:** read `W` and `H` from any paper's methods section, divide. That is
how much its error may be flattered. No code, no data, no access to the authors.

**Verified:** 286 (W,H) combinations × 11 signals (7 real recordings from two public
datasets, 4 synthetic). **Zero violations.** Sharpest case: **white noise**, where
nothing is predictable, still inflates at the bound — proving the artifact belongs to
the **metric**, not the signal or model.

*(Finding F9 · `docs/results/overlap_inflation_law_2026-07-28.md`)*

![**F9 — the overlap inflation law.** *Left:* predicted bound `W/H`, computable from a
methods section alone, against observed MSE inflation. Points lie on the identity and
never above it; the region where a counterexample would appear is empty. The star is
this project's own invalidated result (W=127, H=1): predicted 127×, measured 126.1×.
*Right:* every departure from the bound is explained by one term, monotonically, across
five decades.](figures/f9_overlap_law.png)


### 3.2 Mechanism two — correlation is worse

Most papers report **correlation**, not MSE. Does `W/H` apply?

There is an exact bound, and it holds on all 286 cells. But the convenient shortcut
**fails**:

> `W/H` is violated on **223 of 286** cells — always in the direction that flatters
> the paper.

**Why.** The true factor is `1 + ((W−H)/H)·(s_pred·s_targ)^(−1/2)`, where the `s`
terms compare variability in the future part versus the copyable part. `W/H` is only
right when both are 1. Real models **shrink toward the average exactly where the
signal is unpredictable**, so `s_pred` ≈ 0.70 and true inflation *exceeds* `W/H` —
typically 1.2×, up to 10.8×. Verified to 7×10⁻¹⁶ — machine precision.

**Sharpest example:** white noise at W=32, H=1 reports **r = 0.984** when true
forecast correlation is **0.014**.

*(Finding F10)*

![**F10 — the correlation law, and a falsification.** *Left:* the exact bound is violated
by none of the 286 cells; the `W/H` shortcut carried over from F9 is violated by 223, all
in the shaded region where a reported number looks better than it is. *Right:* the
departure is not noise — rearranging the exact bound forces a relationship with no fitted
parameter, and every cell lies on that curve across three decades.](figures/f10_correlation_law.png)


### 3.3 Mechanism three — the labels are unreliable

For sleep, ground truth is a human expert's stage labels. Humans disagree. From a
meta-analysis (Lee et al., *J Clin Sleep Med* 2022):

| Stage | Inter-scorer agreement (Cohen's κ) |
| --- | --- |
| Overall | 0.76 |
| W | 0.70 |
| **N1** | **0.24** |
| N2 | 0.57 |
| N3 | 0.57 |
| REM | 0.69 |

κ = 1.0 is perfect, 0 is chance. **N1 at 0.24 means experts barely agree it is
happening.** Independently, Danker-Hopfe et al. (*J Sleep Res* 2009) report 82.0% raw
epoch agreement.

**You cannot predict a label more reliably than the label is defined.** A *transition*
is defined by **two** labels, so it is worse: at 82% agreement, both endpoints being
reproducible is ≈ 0.82² ≈ **0.67**. About a third of "transitions" may be two experts
disagreeing rather than a brain changing state.

**Measured in our data: 55% of scored transitions have N1 at an endpoint.** §4.7 shows
what happens when you remove them, and it is the most interesting result in the project.

---

## 4. The experiments, one at a time

Each: **what we asked, how, what we found, what it does *not* show.**

### 4.1 The original result, and its retraction

**Claimed:** MSE 3.116, r = 0.972.
**Reality:** W=127, H=1 — 126 of 127 target samples were in the input.

A reported correlation is uninformative unless `r ≥ 1 − H/W`. Here the threshold is
**0.992**. The reported **0.972 is below it** — so that number never constrained
forecast skill at all. It did not even reach what trivial copying achieves.

Kept in the repo as evidence of the bug, never as performance.

### 4.2 F8 — the crossover (a real, small positive)

**Question:** does a neural net beat a *properly tuned* linear model at short horizons?

**Method:** Sleep-EDF, all 78 subjects, subject-disjoint 39/12/27. Both models tuned
per horizon on the same validation subjects. Paired Wilcoxon, Holm-corrected across
nine horizons.

**Result:** at **160 ms** the GRU is **12.4%** below the linear baseline
(p = 2.2×10⁻⁶, better in **26 of 27** held-out subjects). Advantage **gone by 320 ms**.

**Eight falsification tests.** Two matter most:
- On synthetic AR(2) data, where linear is *provably* optimal, the same GRU with the
  same budget **loses** (p = 0.014) — the apparatus cannot manufacture wins.
- A linear model trained with the GRU's exact optimiser is *worse* than closed-form
  ridge, so the margin is architecture, not optimisation.

**One claim did not survive:** we reported ridge beating the GRU at 640 ms; under
non-overlapping windows it became a tie (p = 0.58). Narrowed and reported.

**Does not show:** clinical utility. 160 ms is ~400× shorter than the ~1-minute
timescale that matters.

![**F8 — the forecast crossover.** A small recurrent model beats a per-horizon-tuned
linear baseline below ~160 ms; the advantage is gone by 320 ms. Both models tuned on the
same validation subjects, paired Wilcoxon across 27 held-out
subjects.](figures/f8_forecast_crossover.png)


### 4.3 F9 — the overlap law
See §3.1. 286 cells, zero violations, machine precision.

### 4.4 F10 — the correlation law, and a falsification
See §3.2. Theorem holds 0/286; the convenient extrapolation is false 223/286.

**Worth internalising:** we *predicted* the shortcut would transfer, tested it, and it
did not. Recorded as killed hypothesis H6 — including a note that it was never
formally pre-registered, so it doesn't get to claim that credit.

### 4.5 The envelope discovery (BNCI)

**Question:** raw voltage stops being forecastable after ~0.5 s. Is that a fact about
the *brain*, or about *voltage*?

**Terminology:** the **band-power envelope** is how much energy the signal carries in
a frequency band (e.g. gamma, 30–100 Hz), tracked over time — a slow summary rather
than the fast raw wiggle.

**Result:** gamma envelope forecastable at **16 s** (r = 0.42, 3/3 held-out subjects)
while raw voltage was dead by 1 s. **A >30× horizon change from changing *what* you
predict, not *how*.**

**Three controls, because a number that size is usually a bug:**

1. **Non-causal filtering.** Standard tools (`sosfiltfilt`, `hilbert`) can see the
   future — §3.1's bug in another costume. Redone strictly causally: 0.437 → **0.447**.
   The non-causality contributed nothing.
2. **White noise through the same pipeline.** Gamma |r| ≤ 0.005 at every horizon. *This
   control also caught a real contamination:* causal **delta** scores **r = 0.998 on
   pure noise** at 64 ms — filter memory, not signal. Slow bands at short horizons are
   uninterpretable, and the README now says so.
3. **Trial-structure locking.** BNCI trials run ~7–8 s and 8192 ms is one period, so a
   model that learned the task rhythm would mimic a forecaster. Dense sweep found **no
   peak** at the trial period.

**Does not show:** dynamics. Real predictive skill decays with horizon; this is flat
from 2 s to 16 s. Long-memory structure, not prediction of change.

### 4.6 The regularisation correction (a lesson, in public)

The public repo claimed forecastability "dies between 64 and 256 ms." That was **our
own under-regularisation** — ridge `alpha` fixed at 1.0 across 2,816 features. Tuned
properly on a held-out training subject, skill survives to **~512 ms**, roughly double.

**Then the fix repeated the sin.** The first sweep capped alpha at 1e4 and the selector
picked that top value at four of five horizons — a grid edge. A sweep written to cure
under-regularisation was itself under-regularised. Grid now runs to 1e8 with an
explicit `!! GRID EDGE` warning.

**The general lesson:** *tuning the baseline properly changed the answer by 2×; tuning
the model changed it by 12%.* An under-tuned baseline is the most common way to
accidentally overstate a result.

### 4.7 F11 and F12 — Sleep-EDF, where the EMG question gets settled

**The worry:** scalp gamma is the band most contaminated by **muscle**. Slow jaw or
neck tension would look exactly like our result. BNCI has no muscle channel.

**Sleep-EDF has `EMG submental`** — a chin muscle channel. The question becomes a
measurement, not an argument.

**F11** (60 subjects, subject-disjoint 36/9/15, causal envelopes, strictly-future
targets):

| band | r at 60 s | after removing EMG **and** EOG | subjects beating persistence |
| --- | --- | --- | --- |
| low gamma | 0.294 | 0.181 | **15/15**, p = 6.1×10⁻⁵ |
| beta | 0.256 | 0.141 | **15/15**, p = 6.1×10⁻⁵ |

Every horizon, both bands: 15/15, CI excluding zero. (p = 6.1×10⁻⁵ is the *floor* for
n = 15 — all 15 in the same direction.)

**Reading it honestly:** it is **not** muscle (EMG correlation +0.24, removing it costs
little). But **eye movement carries a real share** — EOG correlation +0.63, and
residualising both costs about a third of the effect. During sleep that is physiology,
not contamination: REM is *scored by* eye movement and muscle atonia, so removing them
is conservative.

![**F11 — envelope horizon on Sleep-EDF.** *Left:* correlation with the true future value
decays but survives to 60 s in both bands; dashed lines are the same models after removing
eye and muscle artifact. *Middle:* MSE against the persistence baseline (dotted) and the
predict-the-mean line at 1.0 — the model beats both at every horizon. *Right:* the effect
is not carried by a few subjects; all 15 held-out subjects improve on persistence at 60 s.](figures/f11_envelope_horizon.png)


**F12 — predict the stage, not the amplitude.** Nobody cares about the gamma envelope
value in 60 seconds; they care about the state.

**5-class staging: a NEGATIVE.** At 60 s, balanced accuracy:

| input | balanced acc | vs persistence 0.732 |
| --- | --- | --- |
| stage_only | 0.732 | ties exactly — the harness validating itself |
| eeg_only | 0.633 | **worse** |
| eeg + stage | 0.713 | **worse** |

Given the current stage, EEG adds nothing. The model correctly learns to persist,
because transitions are 19% of windows and persistence minimises loss.

**Binary transition detection: a POSITIVE.** Asking "will the stage change within
60 s?" — and stratifying by label reliability (§3.3):

| input | AUC (all) | AUC excl-N1 | above chance, excl-N1 |
| --- | --- | --- | --- |
| stage_only (baseline) | 0.657 | 0.552 | +0.052 |
| eeg_only | 0.610 | **0.506** | +0.006 — **chance** |
| **eeg + stage** | **0.690** | **0.605** | **+0.105** |
| all + stage (full PSG) | 0.697 | 0.606 | +0.106 |

**N1 was 55% of transitions.** We expected removing noisy labels to *raise* AUC. It
lowers it — but it guts the **baseline** far more. Most of the baseline's apparent
skill was N1's instability, i.e. a lookup table of which stages are unstable.

> **On transitions experts actually agree on, EEG roughly doubles the above-chance
> signal (+0.105 vs +0.052).**

![**F12 — the label-reliability collapse.** *Left:* excluding N1 transitions lowers every
AUC — EEG-only falls all the way to chance. *Right:* the same numbers as margin above
chance, which is what the argument is about. Removing unreliable labels costs the baseline
two-thirds of its margin (0.157 → 0.052) but costs the model far less (0.190 → 0.105). If
the claim were false these two drops would be equal.](figures/f12_label_reliability.png)


**Reported alongside, because it matters:** `eeg_only` excluding N1 is **0.506 —
chance**. EEG *alone* predicts nothing here. It is the *combination* with the current
stage that works. And the AP gains are much smaller than the AUC gains, so AUC is the
favourable reading; both are in the results file.

---

## 5. The code

All in `scripts/`. Run from repo root with `PYTHONPATH=src`.

| File | What it does |
| --- | --- |
| `overlap_inflation_law.py` | F9. 286 (W,H) cells over 11 signals; asserts the decomposition per cell so a wrong mask fails loudly. |
| `correlation_inflation_law.py` | F10. Same for correlation. `--selftest` checks the theorem on adversarial inputs — it caught a sign condition I had wrong. |
| `plot_overlap_law.py`, `plot_correlation_law.py` | The figures. |
| `forecast_scaleup_paired.py` | F8. GRU vs tuned ridge, per horizon, paired stats. |
| `validate_forecast_claim.py` | The eight falsification tests for F8. |
| `envelope_horizon_sleepedf.py` | F11. Causal envelopes, EMG/EOG residualisation, paired per-subject tests. Also holds `sleep_span()` and `load_channels()` used by the others. |
| `envelope_stage_decomposition.py` | Splits F11's predictions by whether a transition occurred. |
| `stage_forecast_sleepedf.py` | F12. 5-class staging + binary transition detection + N1 stratification. |
| `envelope_null_control.py` | White noise through the whole envelope pipeline. Caught the delta-band filter artifact. |
| `capacity_ceiling.py` | Model-capacity sweep, four families. |
| `testmode_fixtures.py` | Planted-signal fixtures that test **the harness itself** before it touches real data. |

**Two files worth understanding as a pair.** `testmode_fixtures.py` builds data with a
*known* answer — a big effect, a small one, none. If the harness can't recover a
planted effect, the harness is broken. It caught three bugs before any conclusion
depended on them. `envelope_null_control.py` does the same from the other side: feed
it pure noise, and anything it reports is an artifact.

**Reproduce:**

```bash
PYTHONPATH=src python3 scripts/correlation_inflation_law.py --selftest
PYTHONPATH=src python3 scripts/correlation_inflation_law.py --out runs/correlation_law
PYTHONPATH=src python3 scripts/envelope_horizon_sleepedf.py --out runs/envelope_sleepedf
PYTHONPATH=src python3 scripts/stage_forecast_sleepedf.py --out runs/stage_forecast --context-s 60
```

Every run writes a `results.json` containing its own settings, so any number can be
traced back to the exact configuration that produced it.

---

## 6. The literature

**Know these.** You will be asked about them.

### Directly competing

**Kweon et al. 2026, "SSF-SET", IEEE J Biomed Health Inform.** Sleep stage
*forecasting* on **our exact dataset** (SleepEDF20/78), subject-wise CV. Transformer +
LSTM + quantised tokens + reinforcement learning. Accuracy 0.611, macro-F1 0.537 on
SleepEDF78. **State of the art on our data, and more sophisticated than anything we built.**

**Leguia et al. 2025, Brain Communications.** Predicts **NREM→REM transitions 2–4
minutes ahead at AUC 0.88**, using δ and σ band power in a simple model. Eight
subjects, ~30 consecutive nights, **sub-scalp implanted** electrodes.

**How we compare, honestly:** they are ahead. AUC 0.88 at 2–4 min beats our 0.690 at
1 min. Our defences are real but partial — implanted electrodes on 8 subjects, and
they predicted only NREM→REM, a transition between two of the *most reliably scored*
stages (REM κ = 0.69). We use scalp EEG on 45 subjects across all transitions,
including the ones nobody can score reliably.

**That contrast is the opening, not the weakness.** Nobody in this literature
stratifies by label reliability, and our N1 result says all-transition numbers are
partly measuring which stages are unstable.

### Supporting

**Hussain et al. 2022, Sensors** (99 citations) — sleep stage *classification* from
EEG biomarkers; establishes that delta power ratios track sleep depth. Explains why
band envelopes carry state information.

**Lee et al. 2022, J Clin Sleep Med** — the inter-scorer meta-analysis (κ = 0.76
overall, N1 = 0.24). **Our most important citation.** It makes the label-noise argument
quantitative instead of rhetorical.

**Danker-Hopfe et al. 2009, J Sleep Res** (427 citations) — independent confirmation,
82.0% raw epoch agreement under AASM.

### Leakage (for F9/F10)

**Bernett et al. 2024, Nature Methods**; **Del Pup et al. 2025**; **Apicella et al.
2024**. All address information shared **between** examples. **None addresses
information shared inside a single example** — that gap is the novelty claim, and it
should be stated carefully rather than overclaimed.

---

## 7. The honest ledger

### Solid
- **F9** — 286 cells, 0 violations, machine precision. Derived, not fitted.
- **F10** — theorem 0/286; `W/H` extrapolation falsified 223/286.
- **F8** — 78 subjects, 8 falsification tests, one self-retracted sub-claim.
- **F11** — 15/15 subjects every horizon, p = 6.1×10⁻⁵, CI excludes zero.
- **F12** — transition AUC 0.690 vs 0.657; excl-N1 0.605 vs 0.552.

### Retracted or corrected — all self-caught

| What | Why |
| --- | --- |
| MSE 3.116 / r 0.972 | Window overlap; fails the informativeness screen entirely |
| "BNCI replicates F8's crossover" | Different estimand **and** the BNCI number was a regularisation artifact |
| "Envelope r = 0.596 at 60 s" | Pilot had 2 test subjects; real number 0.294 |
| "Not muscle or eye artifact" | Not muscle. **Eye movement is real** (+0.63) |
| "Measured sleep" | Measured **wake** — recordings start hours before lights-out |
| "Forecastability dies at 256 ms" | Our own fixed `alpha=1.0`; properly tuned it is ~512 ms |
| Ridge beats GRU at 640 ms | Became a tie (p = 0.58) under non-overlapping windows |

**This list is an asset.** Seven self-caught errors, published rather than buried, is a
credential. Senior people read it as a signal about how you work.

### Open
- **The prevalence survey.** How often does published work use `H < W`? Until it's
  done, F9/F10 are tools for auditing one paper, not a finding about the field.
  **Highest-value undone thing, and it needs no GPU.**
- **Re-analysing published numbers** with the N1 stratification.

### Not claimed
- No clinical utility. Nothing here is near a device.
- The envelope result is about **representation**, not dynamics — flat r across an 8×
  horizon range is long memory, not prediction of change.
- The `s_pred < 1` shrinkage magnitudes are documented for ridge on these traces, not
  established as a general law.

---

## 8. How to talk about this

### 30 seconds
> Forecasting scores in this field can be inflated by measurement artifacts. We found
> three: the target window overlapping its input, the same bug behaving worse in
> correlation, and unreliable ground-truth labels. Each has an exact correction. We
> caught the first one in our own headline result and retracted it.

### For a sleep researcher
> Everyone reports transition prediction over all transitions. Over half involve N1,
> where inter-scorer κ is 0.24 — experts barely agree it happened. Excluding them, the
> standard "current stage" baseline drops to near-chance, which means a lot of
> published performance is measuring *which stages are unstable* rather than prediction.

### For a methods reviewer
> `MSE_overlap = [(W−H)·MSE_copy + H·MSE_future]/W` is an identity, so
> `inflation ≤ W/H` from the methods section alone. The correlation analogue has an
> exact bound too, but `W/H` understates it on 223 of 286 cells, because regularised
> models shrink toward the mean where the signal is unpredictable.

### Questions you will get

**"Isn't it obvious band envelopes are smoother and easier to predict?"**
Partly — which is why persistence is the baseline, not the mean. Persistence gets
smoother too. The claim is the quantitative horizon comparison under matched protocol
with artifact controls, plus the fact that the forecasting literature works on voltage.

**"Leguia et al. get AUC 0.88 at 4 minutes. You get 0.690 at 1."**
Correct, and they're ahead. Implanted electrodes, 8 subjects, and only NREM→REM —
between two of the most reliably scored stages. Our contribution isn't a better
forecaster; it's that all-transition numbers include transitions nobody can score.

**"Why should I trust these numbers?"**
Every run writes its settings beside its results; the harness is tested on planted
signals before touching real data; white noise goes through the same pipeline as a
null; and seven errors are documented in §7 rather than removed.

**"What's the actual contribution?"**
Three measurement corrections that apply to other people's work, each derived and
verified — one of which we found by breaking our own result.

---

## Figures

Every figure regenerates from a committed artifact:

```bash
python3 scripts/plot_sleepedf_figures.py --layout ieee --out docs/figures/ieee   # paper
python3 scripts/plot_sleepedf_figures.py --layout wide --out docs/figures        # slides
```

**Two layouts, and the difference matters.** `docs/figures/ieee/` is 3.5 in wide with
panels stacked vertically, for a two-column IEEE paper. Its fonts are set in absolute
points *at that width* — 8 pt in the file is 8 pt on the page. The common failure is
drawing a 13-inch-wide figure and letting the journal shrink it into a column, which
divides every font by ~3.7 and makes the axes unreadable. Place them with
`\includegraphics[width=\columnwidth]{...}` and do not rescale. Both render at 301 dpi
at final size, above the 300 dpi most venues require.

`docs/figures/` holds the wide side-by-side versions, for slides and this handbook.

---

## Where to go next

- Hypotheses and their pre-written falsifiers: `docs/research/program/REGISTER.md`
- Per-finding write-ups: `docs/results/`
- Evidence, one directory per finding: `artifacts/`
- Two-page brief for external readers:
  `docs/paper/overlap_law_brief/kahlus_metric_inflation_brief.pdf`

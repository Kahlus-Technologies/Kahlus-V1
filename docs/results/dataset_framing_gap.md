# The dataset framing gap

**Status:** open, with the closing experiment running.
**Last updated:** 2026-07-31.

## The gap, stated plainly

The paper is framed on **BNCI2014_001**, a motor-imagery EEG corpus. The
powered results it leans on were measured on **Sleep-EDF**, a sleep corpus.

Specifically:

| Finding | What it says | Measured on |
| --- | --- | --- |
| F8 | GRU beats tuned ridge at short horizons; the advantage is gone by 320 ms | Sleep-EDF |
| F9 | Overlap inflation law, exact bound | Sleep-EDF |
| F10 | The correlation analogue holds; the `W/H` shortcut does **not** | Sleep-EDF |
| F11 | Band envelopes forecastable to 60 s, 15/15 held-out subjects | Sleep-EDF |
| F12 | Stage forecasting: negative as posed, weak when re-posed | Sleep-EDF |

Every figure in `docs/paper/results_figures/` draws on Sleep-EDF. A reader who
takes the framing at face value would expect otherwise.

## Why this is a real problem and not a labelling nit

The two corpora differ in the way that matters most for a forecasting claim:
they are different **brain states**. Sleep-EDF is spontaneous sleep, where the
signal has strong slow structure and long autocorrelation. BNCI2014_001 is
awake, cued motor imagery, where the informative component is a brief
task-locked response. A short-horizon nonlinear advantage is exactly the kind of
result that could depend on which of those regimes you are in.

So "it will transfer" is a hypothesis, not a formality.

## The retraction that made the gap explicit

An earlier version of this project claimed the gap was already closed: that the
public `eeg-forecastability-limits` BNCI result (skill dying at 64--256 ms) was a
cross-dataset replication of F8's Sleep-EDF crossover (advantage gone by 320 ms).
**That claim was retracted.** It was wrong twice over, for independent reasons.

1. **The BNCI number came from an under-tuned baseline.** It used a fixed ridge
   `alpha=1.0` across 2,816 features. With alpha selected on a held-out training
   subject, skill survives to roughly 512 ms. The "limit" was a property of the
   hyper-parameter, not of the brain.

2. **The subtler one: the two numbers were never the same estimand.** F8 measures
   *GRU beats tuned ridge* --- an architecture question. The BNCI figure measured
   *AR beats the trace mean* --- an existence question, whether there is any
   forecastable structure at all. Two different comparisons landing in the same
   order of magnitude is a coincidence, not a replication.

The second reason is the one worth carrying forward, because it survives any
amount of extra tuning. Before calling anything a replication, ask whether the
quantity being estimated is the same quantity. Same dataset, same metric, and
same direction are not sufficient.

Recorded in `docs/research/program/REGISTER.md`, cycle 4.

## How the gap is being closed

By running the experiment. `scripts/bnci_crossover_sweep.py` is
`forecast_scaleup_paired.py` with the corpus swapped and nothing else changed
that matters:

* subject-disjoint splits, same splitter, same seed policy
* ridge alpha tuned per horizon on validation subjects
* GRU (hidden, steps) tuned on the **same** validation subjects, several seeds
* MSE scored per test subject, so the comparison is paired
* strictly-future single-sample targets; the target is never inside the input

Two design faults were found and fixed in that script before it was trusted:

* **The ridge grid chose the regularisation, not the data.** Validation selected
  `alpha = 0.1`, the bottom of the inherited grid, at every horizon. The grid is
  now extended down to `1e-6`, and any selection landing on an edge prints a
  warning. A baseline pinned to its grid edge is under-tuned --- which is
  precisely the failure this project has already been caught by twice.
* **The test was too small to detect anything.** A 35/15/50 split left 3 test
  subjects, and a 3-pair Wilcoxon cannot return p < 0.25 at any effect size.
  Replaced with **9 leave-one-subject-out folds**, tuning kept strictly inside
  each fold.

BNCI is 250 Hz against Sleep-EDF's 100 Hz, so horizons are specified in
milliseconds and converted per dataset. Comparing `h=1` across the two would
compare 4 ms against 10 ms and call it replication.

## Both outcomes are publishable, and neither is spun

**If the GRU advantage replicates on BNCI**, the claim is a genuine cross-corpus
finding and the paper's framing is earned.

**If ridge wins on BNCI under a symmetric budget**, the claim narrows to
brain-state dependence: a nonlinear short-horizon advantage exists in
spontaneous sleep and not in cued motor imagery. That is *more* interesting than
a clean replication, because it says something about when the advantage exists
rather than merely that it does. It is only embarrassing if the framing is left
unchanged after the result comes in.

The one outcome that would be a failure of process is quietly reporting whichever
corpus happened to agree.

## What must change when the result lands

- [ ] Update this file with the verdict and the artifact path.
- [ ] Update the framing in the paper to match what was actually measured.
- [ ] Update the F12/F8 discussion in `docs/paper/primer/kahlus_primer.tex`
      (Part VI) and re-run `docs/paper/primer/verify_primer.py`.
- [ ] Add a cycle row to `docs/research/program/REGISTER.md`.

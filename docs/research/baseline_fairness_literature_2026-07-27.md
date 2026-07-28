# Literature grounding: what makes an EEG forecasting claim credible

**Date:** 2026-07-27. **Purpose:** establish, from current literature, (a) whether
the within-window overlap finding is novel, (b) what a defensible baseline
comparison requires, (c) who the direct comparators are. Searched via Consensus.

This exists because the previous headline (`3.116` MSE / `r=0.972`) was an
overlap artifact, and the replacement claim has to survive the exact scrutiny
that killed it.

---

## 1. Is the within-window overlap finding novel? — Yes, on a distinct axis

The leakage literature is large and good, but it addresses a **different overlap
axis**. Two axes must be kept apart:

| axis | what shares samples | fixed by | in literature? |
| --- | --- | --- | --- |
| **between-example** | example *i* and example *j* | subject-wise CV, non-overlapping stride | extensively |
| **within-example** | `X` and `Y` of the *same* example | `horizon >= window_length` | **not found** |

Evidence that the field covers only the first axis:

- Bernett et al. (2024, *Nature Methods*) give seven questions for preventing
  leakage in biological ML. All concern train/test information sharing; none ask
  whether a regression target overlaps its own input window.
- Del Pup et al. (2025) compare five CV settings over 100,000 trained EEG models
  and conclude subject-based and nested CV are required. Our artifact **survives
  both** — it is not a partitioning problem.
- Lee et al. (2023) study overlapped subject-wise vs trial-wise CV under cropping
  augmentation; again, overlap *between* augmented samples.
- Dehghani et al. (2019) quantify overlapping vs non-overlapping sliding windows
  in HAR and find the apparent benefit of overlap is an artifact of
  subject-dependent CV — the same lesson, still on the between-example axis.
- Wosiak et al. (2025) argue temporal segmentation should be a "first-class
  design choice" and call for transparent reporting of window length and
  shift/overlap — and still do not mention horizon versus window length.
- Apicella et al. (2024) taxonomize leakage across the ML workflow; the
  within-example forecasting case is absent.

**Conclusion:** the 126/127 result is a leakage *mode* that the current remedy
set does not catch, because every remedy operates on how examples are assigned to
splits, and this one lives inside a single example. That is the contribution.
No scoop found.

## 2. What a "we beat the baseline" claim now requires

The strongest precedent for the paper's shape is outside neuroscience:

- **McGreivy & Hakim (2024, *Nature Machine Intelligence*)** — systematic review
  of ML-for-PDE solving: **79% (60/76)** of papers claiming to beat a numerical
  method compared against a *weak baseline*. They attribute it to researcher
  degrees of freedom and positive-result bias. This is the template: quantify the
  weak-baseline rate, then show what a strong baseline does.
- **Lin (2019, *SIGIR Forum*)** and **Yang et al. (2019)** — the "neural hype"
  papers in IR. Yang et al. re-ran five neural models against strong baselines and
  found most reported gains illusory, one additive. Directly analogous method.
- **Beck et al. (2025, *Applied Intelligence*)** — "Mind the naive forecast!" On
  highly volatile series, *no* method (ARIMA, ETS, BVAR, LSTM, DeepVAR, TFT,
  XGBoost) consistently beats the naive no-change forecast; ML degrades faster
  than statistical models as volatility rises. Establishes that the persistence
  baseline is the right floor and must be reported.
- **Wu et al. (2025, *IEEE Access*)** — review of 149 EEG seizure-prediction
  studies (2018–May 2025). Names **"few linear baselines"** as one of four
  persistent field-wide failures, and calls explicitly for "unified benchmarks
  that pit CNN, Transformer, GNN, and related models against **well-tuned linear
  baselines**." Our corrected benchmark is a direct response to a stated gap.
- **Puffay et al. (2023, *J. Neural Eng.*)** — 29 EEG-to-speech deep learning
  studies; recurrent pitfalls are "biased cross-validations, data leakage leading
  to over-fitted models," plus model/data-size mismatch.

**Operational consequence for our run:** the linear baseline gets a per-horizon
regularisation sweep on held-out validation subjects, and the grid must not be
truncated at its edge. A 24-subject pilot selected the top of a grid that stopped
at `alpha=500`, i.e. the baseline was *still* under-tuned; the grid now runs to
`5e4` and the script warns on edge selection. The GRU receives a matched
(hidden, steps) sweep on the same validation subjects — a "tuned linear model
matches the deep model" claim is only honest under a symmetric tuning budget.

## 3. Direct comparators for the forecasting result

These are the papers a reviewer will ask about:

- **Pankka et al. (2024)** — WaveNet forecasts resting-state EEG **150 ms ahead**
  in theta and alpha, MAE 1.0 ± 1.1 µV / 0.9 ± 1.1 µV, outperforming an AR model
  on amplitude and phase; probabilistic output allows discarding uncertain
  predictions. **This is the same horizon regime as our h=16 (160 ms).** So
  "a deep model beats AR at ~150 ms" is already published — our contribution
  cannot be that claim, and must be the methodology plus the horizon-resolved
  crossover.
- **Onojima et al. (2025)** — temporal-convolutional encoder predicting the
  complex-valued analytic signal directly from **raw multichannel** EEG, beating
  an AR baseline subject-wise *and* cross-subject over 25 participants, with a
  missing-signal-imitation mechanism for electrode dropout. Relevant twice: it is
  a comparator, and it confirms multichannel is where the nonlinear model should
  earn its keep. Our current task is single-channel (Fpz-Cz), which is the most
  likely reason our GRU only ties a tuned ridge.

## 4. What this implies for the paper

The contribution is **not** "our GRU forecasts EEG well" — Pankka already has the
150 ms result. It is a two-mechanism audit of how EEG forecasting claims inflate,
each quantified on the same data and model:

1. **Within-example input/target overlap** — 126× on real Sleep-EDF; survives
   subject-held-out splitting and every remedy in §1; novel axis.
2. **Under-tuned linear baselines** — a known pathology (§2) not previously
   quantified for EEG forecasting; our own pilot reproduced it accidentally.

Plus a corrected, horizon-resolved benchmark with per-subject paired statistics
on all 78 Sleep-EDF subjects, reporting honestly where the deep model wins, ties,
and loses.

## References

1. Bernett J. et al. (2024). Guiding questions to avoid data leakage in biological machine learning applications. *Nature Methods*.
2. Joeres R. et al. (2025). Data splitting to avoid information leakage with DataSAIL. *Nature Communications*.
3. Apicella A. et al. (2024). Don't push the button! Exploring data leakage risks in machine learning and transfer learning. *Artificial Intelligence Review*.
4. Del Pup F. et al. (2025). The role of data partitioning on the performance of EEG-based deep learning models in supervised cross-subject analysis. *Computers in Biology and Medicine*.
5. Lee H.-T. et al. (2023). Risk of data leakage in estimating the diagnostic performance of a deep-learning-based computer-aided system for psychiatric disorders. *Scientific Reports*.
6. Dehghani A. et al. (2019). A quantitative comparison of overlapping and non-overlapping sliding windows for human activity recognition using inertial sensors. *Sensors*.
7. Wosiak A. et al. (2025). Impact of temporal window shift on EEG-based machine learning models for cognitive fatigue detection. *Algorithms*.
8. McGreivy N. & Hakim A. (2024). Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations. *Nature Machine Intelligence*.
9. Lin J. (2019). The neural hype and comparisons against weak baselines. *ACM SIGIR Forum*.
10. Yang W. et al. (2019). Critically examining the "neural hype": weak baselines and the additivity of effectiveness gains from neural ranking models. *SIGIR*.
11. Beck N. et al. (2025). Mind the naive forecast! A rigorous evaluation of forecasting models for time series with low predictability. *Applied Intelligence*.
12. Wu Y. et al. (2025). A review of machine learning and deep learning trends in EEG-based epileptic seizure prediction. *IEEE Access*.
13. Puffay C. et al. (2023). Relating EEG to continuous speech using deep neural networks: a review. *Journal of Neural Engineering*.
14. Pankka H. et al. (2024). Enhanced EEG forecasting: a probabilistic deep learning approach. *bioRxiv*.
15. Onojima T. et al. (2025). Direct prediction of the complex-valued analytic signal of EEG from raw multichannel data. *bioRxiv*.

# Roadmap: from here to a paper the field actually uses

Written 2026-07-25. Companion to `NEURAL_CASP_TECHNICAL_DOCUMENTATION.md` and
`kahlus_neural_casp_research_journal.md`.

> **Revised 2026-07-25 after the P1–P7 literature searches.** See
> `literature_grounding_2026-07-25.md` for what changed and why. Six material
> revisions: A2's novelty repositions (the empirical overlap sweep is partly done;
> the *analytic law* is the named gap), A3 stops competing with NeuralBench, B1
> must disclaim any mutual-information interpretation (McAllester–Stratos
> impossibility), B2's prior shifts toward a negative, **slpdb is demoted from
> smoke test to adversarial case**, and Track D's clinical hook is weaker than
> assumed in apnea cohorts.

---

## 0. The honest starting position

What exists and is real:

- A leakage-controlled incremental-value gate (RFS) with cross-fitting, a
  baseline ladder, a five-control battery, and subject-cluster bootstrap
  certification.
- Calibration evidence: **4% false-positive rate at the null anchor** (1/24
  seeds), essentially nominal for a one-sided 95% LCB.
- A concrete, quantified leakage finding: **126× inflation** from input/target
  overlap, with the historical claim marked invalid in place.
- Structural prevention: a guard that makes the unsafe task unconstructible, and
  a person-level split audit that catches nights-as-identities.
- Four powered or scout-grade negatives (F2–F5) across two etiologies.

What does not exist:

- Any positive real-data finding.
- MESA/SHHS access (applied for, not held).
- An external replication of anything.

**The citation question, answered honestly.** Nobody can plan 30k citations, and
any plan that claims to is selling something. But the *shape* of work that gets
cited that heavily is knowable, and it is not "a result." It is one of:

1. a **tool or benchmark** people run every week (CASP, MOABB, EEGNet, Braindecode),
2. a **dataset** people build on (PDB, Sleep-EDF, MESA),
3. a **methodological correction** the field must cite to justify its own design
   choices (leakage papers, reporting standards).

Kahlus's realistic path is (1) + (3). Not (2) — we generate no data. And a result
paper, even a positive one, is the *least* likely route to that citation profile.
Plan accordingly: **build the thing people must use and must cite when they
justify their evaluation design.**

---

## Track A — Ship the correction paper (highest certainty, do first)

**Thesis:** input/target overlap and label-level identity confusion silently
inflate neurophysiological forecasting results, and here is a gate, a guard, and
an audit that make the inflated version unconstructible.

This is submittable *now*. It needs no positive.

### A1. Consolidate and merge (days)
- Merge the current branch. Full-suite green gate; ruff clean.
- Split into separate PRs if you want the clean history the plan originally
  specified; otherwise land as one with atomic per-finding commits.

### A2. Strengthen the leakage claim to publication grade (1–2 weeks)

**Read `Albelali & Ahmed 2025` (arXiv 2512.06932) first** — nearest neighbour, and
credited with "split before sequence construction — prevent future leakage from
prebuilt input-output windows." If it already quantifies input/target overlap, A2
narrows to the analytic law plus enforcement.

**Position carefully.** The empirical overlap sweep is partly done: Oran & Yıldırım
(2024) swept 0/25/50/75 % overlap in EEG; Lee et al. (2023) ran 75 % overlap
against four CV schemes; Dehghani et al. (2019) showed overlap buys 8–13 % F1 under
subject-dependent CV and nothing under subject-independent CV. **Our mechanism is
different and must be said so in the abstract:** those study *train/test
dependence*; ours is **input/target overlap inside a single window**, which
survives a perfectly correct subject-held-out split. That is why ours is 126× and
theirs are 8–13 %. Without this sentence we get filed under Brookshire as
incremental.

The 126× number is currently one model on one dataset. Make it a *phenomenon*:
- Sweep the overlap fraction (126/127, 96/127, 64/127, 0/127) and show inflation
  as a smooth function of overlap. That converts an anecdote into a curve.
- Repeat across ≥3 public datasets (Sleep-EDF, BNCI2014-001, ds005620) and ≥4
  model classes (ridge, GRU, TCN, EEGNet). A 4×4 grid where *every cell* inflates
  is what makes it undeniable.
- Add the *analytic* prediction: for an AR(p) process with known autocorrelation,
  derive expected MSE under overlap vs isolated. **P1 names this as an open gap** —
  "the field still lacks a widely adopted analytic law linking overlap fraction or
  stride to expected optimism under realistic AR or Gaussian physiological signal
  models." Do not derive from scratch: Medrano et al. (2024) give Gaussian-ACF →
  effective sample size validated on electrophysiological signals; Afyouni et al.
  (2019) give autocorrelation-corrected effective DoF; Gahtan & Bronstein (2025)
  give β-mixing → effective blocks. **Paskaramoorthy et al. (2025) is the
  structural template** — analytic bias bounds for IID-resampled Sharpe ratios
  derived from first-lag autocorrelation. Same shape of theorem, different field.
- State Bergmeir et al. (2018) precisely: K-fold CV *is* valid for purely
  autoregressive models with uncorrelated candidate errors. Overclaiming that "CV
  is invalid on time series" is wrong and citable against us.

**This is the single highest-leverage piece of work available right now.** It is
fully doable with data already on disk, needs no credentialing, and produces the
kind of result other authors are *obligated* to cite.

### A3. Release the artifact properly (revised — do not build a benchmark)
The benchmark slot filled up in the last twelve months: NeuralBench (Meta, 36 EEG
tasks / 94 datasets), EEG-FM-Bench, OmniEEG-Bench. Competing loses.

- Ship `leakage-check` as a **small composable dependency**, designed to be called
  *inside* MOABB / NeuralBench / Braindecode pipelines: takes a task spec, reports
  overlap fraction and whether scoring is copy-safe. MOABB won by building on MNE
  and scikit-learn to cut switching costs; do the same.
- `pip install neural-casp`, one-command reproduction, sealed test data.
- **Adoption comes from tools people run, not papers people read.** A one-function
  check other benchmarks adopt beats a benchmark nobody switches to.

**Venue:** NeurIPS Datasets & Benchmarks, or *Journal of Neural Engineering* /
*Imaging Neuroscience* for the methods framing. Brookshire 2024 landed in
*Frontiers in Neuroscience* and is well-cited; this is the same genre with a
sharper mechanism and shipped enforcement.

---

## Track B — The genuinely novel methods contribution

**The open problem, stated precisely:** neural directed-information and
mutual-information estimators (DINE, transformer-DINE, MINE/InfoNCE) give
*asymptotic* consistency under ergodicity assumptions. There is no
**distribution-free, finite-sample** guarantee on the false-certify rate of a
learned information estimate. Our bootstrap floor is empirical, not guaranteed.

**The contribution:** replace the bootstrap detection floor with a
**split-conformal** floor computed on held-out *subject clusters*, giving
`P(certify | true null) ≤ α` at finite n under cluster exchangeability.

**Mandatory disclaimer (McAllester & Stratos 2018).** No distribution-free
high-confidence lower bound on mutual information from N samples can exceed
O(log N). So B1 must **not** claim a distribution-free CI on an MI. The escape,
stated up front: RFS is a difference of out-of-fold mean log-losses between nested
models with clipped probabilities — a **bounded functional of held-out
predictions**, not an MI estimate. Conformal/concentration applies to *that*.
Reading RFS as a bound on `I(Y; Z | B)` inherits the impossibility and is not
certified. (This is also why the RFS coinage earns its keep: calling it an MI
estimate walks straight into the theorem.)

**Prior art that narrows the claim.** Oliveira et al. (2022, JMLR) already prove
split conformal stays valid on dependent processes with an explicit coverage
penalty. So the contribution is *not* "cluster conformal" — it is applying a
finite-sample floor to a **one-sided certification decision** on subject-clustered
physiological data, with the exchangeability assumption stated and its violation
bounded.

Why this is real and not a reskin:
- Conformal prediction is distribution-free and finite-sample by construction.
- The hard part — and the actual research — is that anchors within a subject are
  dependent, so iid conformal does not apply. Cluster/block conformal with a
  stated exchangeability assumption and a sensitivity bound on its violation is a
  genuine methods result.
- It is testable against the calibration harness that already exists: the 24-seed
  detection-power sweep becomes the empirical check on the theoretical bound.

### Staging
- **B1 (CPU, now):** conformal floor on the *existing* linear estimator. Prove the
  finite-sample floor is tighter or better-calibrated than the bootstrap floor on
  the synthetic harness. This isolates the statistical contribution from any
  architecture question.
- **B2 (needs GPU, only if B1 holds):** swap in an S4/Mamba encoder; show the
  detection floor drops on nonlinear scenarios. **Gate: if the floor does not
  drop, do not ship the backbone.** A transformer that certifies nothing new is
  not a contribution, and saying so is a publishable finding in itself.

**Venue:** NeurIPS/AISTATS main track. This is the piece with actual theoretical
novelty.

---

## Track C — The result (gated on data, lowest certainty)

F6/F7: autonomic block → future arousal, beyond cortical spectral baseline.

**Honest odds assessment.** Rao et al. tested cardiac autonomic predictors in a
nested design and found **ΔR² < 0.03, p > 0.25** for subjective sleep quality.
That is a published null on essentially this Z-block. Our defence is timescale and
outcome specificity — within-night event-locked arousal, where Bonnet & Arand's
≥10-beat pre-arousal HR acceleration lives — and that is a *defence*, not a
refutation. **Treat a null as the modal outcome and design the writeup so a null
is publishable.** That is exactly what the detection floor is for: a null above
the floor is a real negative; below it, it is silence.

- **C1 (revised):** slpdb is an **adversarial case, not a smoke test.** P3 rates
  autonomic-lead evidence for respiratory-event arousals in apnea cohorts at
  weak (3/10), versus strong (8/10) for spontaneous arousals in healthy sleepers —
  and slpdb's arousals are predominantly respiratory. A null there says little
  about MESA. Still worth running (it is the only open dataset with simultaneous
  EEG+ECG+scored arousals), but **stratify spontaneous (`A`) vs respiratory
  (`HA`/`CAA`/`X`) as a primary axis**, since pooling averages a plausible effect
  against a documented null (Bahr-Hamm 2022).
- **C2:** MESA when credentialed. Thousands of subjects, this is the real run.
  **Pre-register the spontaneous-arousal subgroup as primary** — that is where the
  ≥10-beat (Bonnet & Arand) and 30-second (Attoh-Mensah 2023) precursor evidence
  actually lives.
- **Framing gate for all of Track C:** ECG-only arousal *detection* already reaches
  AUROC 0.93 on MESA (Li 2020) and AUC 0.91 from five post-onset beats (Basner
  2007). Every artifact must open by distinguishing **strictly-future forecasting,
  conditioned on a cortical baseline** from concurrent detection. Skip this and a
  reviewer files us as a worse Li 2020.
- **C3:** SHHS as dataset-held-out replication. **The dataset-held-out axis is
  worth more than the primary result** — almost nobody does it, and it is what
  separates "we found something" from "it transfers."

---

## Track D — Clinical translation (long horizon, name it, don't chase it yet)

Real clinical use requires prospective validation, IRB, and a regulatory pathway.
None of that is reachable from retrospective public data. What *is* reachable and
worth doing now:

- Keep every `claim_scope` and `stop_reason` honest, because a device program
  later will need exactly this evidence discipline.
- The clinical hook is **weaker than assumed, and cohort-dependent.** In community
  cohorts arousal burden holds up (Shahrbabaki 2021, n=8001: CV-mortality HR 2.17
  in SOF women, 1.60 in SHHS women). In **apnea** cohorts it largely fails once
  desaturation is modelled — Labarca et al. (2023) found hypoxic and ventilatory
  burden predicted incident CVD in MESA/MrOS while **arousal burden did not**, and
  Azarbarzin et al. (2023) found arousal-only hypopneas unassociated with outcomes
  across three cohorts. Cite both sides.
- **Better target if we want clinical meaning: arousal *intensity*, not counts.**
  Gao et al. (2025) found quantitative arousal intensity predicted cognition and
  sleepiness where counts do not, and P6's gap table shows intensity/timing thin
  across every outcome column. Less crowded question, more clinically defensible
  outcome.
- **Do not put a clinical claim in a methods paper.**

---

## Sequencing

```
Now        A1 (merge) ─┬─ A2 (overlap-inflation study)  ← highest leverage
                       └─ B1 (conformal floor, CPU)     ← the real novelty
Then       A3 (release artifact + CLI)  →  submit Track A
Parallel   C1 slpdb smoke  →  C2 MESA when access lands  →  C3 SHHS transfer
Later      B2 (backbone, only if B1 holds)
Never yet  D (clinical claims)
```

## Kill criteria (decide these now, not after seeing data)

- **A2** fails if inflation does *not* scale with overlap fraction. Then F1 is a
  quirk of one setup, not a phenomenon — report it smaller and move on.
- **B1** fails if the conformal floor is not better-calibrated than the bootstrap
  floor. Then there is no methods paper; say so and keep the bootstrap.
- **B2** fails if the floor does not drop with a richer encoder. **Expect this.**
  Three independent 2026 results (Kuruppu/Wagh/Varatharajah's critical review;
  Yang & Van Hulle's ICLR "Are EEG Foundation Models Worth It?"; Meta's
  NeuralBench) all find richer architectures do not reliably help on EEG and show
  no clear scaling law. Plan the null as the deliverable — a fourth confirmation
  from a new angle (certification rather than accuracy) is worth more than a
  marginal positive.
- **C2** returns a null above the detection floor → publish the null with the
  power analysis. Below the floor → report as underpowered, not as a null.

## The one-sentence positioning

> A leakage-controlled, finite-sample-certified test for whether a physiological
> signal carries genuine incremental information about a future neural event —
> plus the evidence that widely-used evaluation designs inflate exactly this
> quantity.

---

# Research prompts

Copy-paste. Each is self-contained.

### P1 — Overlap inflation (Consensus / Semantic Scholar)
> Find work quantifying how input/target overlap, window striding, or
> autocorrelation inflates reported performance in time-series forecasting
> evaluation, especially physiological signals (EEG, ECG, PPG). I want: (a) papers
> that measure inflation as a function of overlap fraction, (b) any analytic
> treatment for AR/Gaussian processes, (c) proposed reporting standards. Also
> search adjacent fields — finance, climate, remote sensing — where lookahead bias
> is well studied and may already have the analytic result.

### P2 — Finite-sample guarantees for learned information estimates (algorithmic)
> Survey finite-sample and distribution-free guarantees for neural mutual-
> information and directed-information estimators (MINE, InfoNCE/CPC, DV bounds,
> DINE, transformer-DINE). Specifically: where does the guarantee break from
> asymptotic to finite-sample? Has anyone applied conformal prediction to certify
> an information-theoretic estimate? What is the state of the art for conformal
> prediction under *clustered / non-exchangeable* data (block conformal, subject-
> level exchangeability, hierarchical conformal)? Cite theorems and their exact
> assumptions.

### P3 — Autonomic precursors of cortical arousal (biomedical)
> What is the evidence that autonomic signals (heart rate, HRV, respiration,
> pulse-wave amplitude, peripheral arterial tone) *precede* cortical EEG arousal
> during sleep? I want temporal-precedence evidence specifically — lead times,
> event-locked analyses, and any predictive (not merely correlational) modelling.
> Include negative results. Distinguish spontaneous arousals from respiratory-event
> arousals, and note which findings are apnea-cohort-specific vs healthy sleepers.

### P4 — Arousal detection state of the art (biomedical / benchmarking)
> What are the current best-performing automated arousal detection and prediction
> methods on NSRR datasets (MESA, SHHS, MrOS) and the PhysioNet 2018 Challenge?
> Report architectures, input modalities, evaluation protocol (subject-held-out?
> cross-dataset?), and headline metrics. I especially want to know whether anyone
> evaluates *dataset-held-out* transfer, and what the reported drop is.

### P5 — Benchmark adoption (meta / strategy)
> How did widely-adopted ML benchmarks in biosignal and neuroscience domains
> achieve adoption — MOABB, Braindecode, BIDS, OpenNeuro, the PhysioNet
> Challenges? What made the field standardise on them: tooling, a challenge event,
> institutional backing, a leaderboard, a paper? What did failed benchmarks lack?

### P6 — Clinical significance of arousal burden (biomedical, for Track D framing)
> Summarise evidence linking sleep arousal burden / arousal index to clinical
> outcomes — cardiovascular events, mortality, cognition, daytime function. Which
> measures are prognostic, in which cohorts, with what effect sizes? Note where
> associations are confounded by comorbid apnea severity.

### P7 — Grok / X, field-pulse
> Who is currently working on evaluation rigour, data leakage, or reproducibility
> in EEG and physiological machine learning? Names, labs, recent threads and
> preprints from the past 12 months. Also: who is publicly critical of EEG
> foundation models and their benchmarks?

---

## What I would do first, if forced to pick one thing

**A2 — the overlap-inflation study.** It uses data already on disk, needs no
credentialing, converts the existing 126× anecdote into a curve plus an analytic
prediction, and produces the artifact other authors have to cite when they defend
their own evaluation design. It is the shortest path from what exists today to
something the field must reckon with.

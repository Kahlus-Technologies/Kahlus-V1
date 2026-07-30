# Literature grounding — what the seven searches changed

Date: 2026-07-25. Sources: six Consensus deep searches (P1–P6) and one X/field-pulse
(P7), run against the prompts in `ROADMAP_TO_PUBLICATION.md`.

This file records only what **changes the plan**. Confirmations are noted briefly;
the substance is the six places where the literature contradicts, narrows, or
reframes what we intended to do.

---

## 1. The analytic law is a real, named gap — and it is our opening

P1's closing line is the single most useful sentence returned by any of the seven
searches:

> the field ... still lacks a widely adopted analytic law linking **overlap
> fraction or stride** to **expected optimism** under realistic AR or Gaussian
> physiological signal models.

That is the A2 deliverable, identified as an open gap by an independent survey.
Take it.

**But the empirical half is partly done, and we must not claim it.** Oran & Yıldırım
(2024) already swept **0/25/50/75 % overlap × 0.5/1/2/4 s windows in 36 subjects**
on EEG mental arithmetic. Lee et al. (2023) ran 75 % overlap against four CV
schemes in EEG. Dehghani et al. (2019) showed the cleanest diagnosis in HAR:
overlap buys 8–13 % F1 under subject-dependent CV and **exactly nothing** under
subject-independent CV.

**Revision to A2.** The contribution is *not* "we swept overlap fraction first."
It is:

1. **A different mechanism.** Every paper above studies *train/test dependence* —
   related windows crossing folds. Ours is **input/target overlap inside a single
   window**, which survives a perfectly correct subject-held-out split. That is why
   our effect is 126× and theirs are 8–13 % F1 or 28 % accuracy: different failure,
   different order of magnitude. This distinction must be stated in the abstract or
   a reviewer will file us under Brookshire and reject as incremental.
2. **The analytic law**, derived for AR(p)/Gaussian, matched against the empirical
   curve.
3. **Shipped enforcement** — the guard, not just the warning.

**Scoop risk to check before writing.** Albelali & Ahmed (2025), *Hidden Leaks in
Time Series Forecasting: How Data Leakage Affects LSTM Evaluation Across
Configurations and Validation Strategies* (arXiv 2512.06932), is the nearest
neighbour, and P1's standards table credits it with "split before sequence
construction — prevent future leakage from prebuilt input-output windows." **Read
this paper first.** If it already quantifies input/target overlap, A2 narrows to
the analytic law plus enforcement.

**Machinery to reuse rather than reinvent.** The derivation should not start from
scratch:
- Medrano et al. (2024) — parametric Gaussian ACF → effective sample size, validated
  on electrophysiological power signals. Closest physiological instrument.
- Afyouni et al. (2019) — autocorrelation-corrected effective DoF for Pearson r.
- Gahtan & Bronstein (2025) — β-mixing → effective blocks → generalization bounds.
- **Paskaramoorthy et al. (2025) is the structural template**: they derive analytic
  bias bounds for IID-resampled rolling-window Sharpe ratios **from first-lag
  autocorrelation**. That is the shape of theorem we want, in finance rather than
  physiology.
- Bergmeir et al. (2018) — the narrow positive: K-fold CV *is* valid for purely
  autoregressive models when candidate errors are uncorrelated. State this
  precisely; overclaiming that "CV is invalid on time series" is wrong and citable
  against us.

---

## 2. Track B has a hard impossibility result in its path

P2 confirms the gap — the coverage table returns **GAP in all four columns** for
"conformalized information estimation." Nobody has conformalized an MI/DI estimate.

**But McAllester & Stratos (2018) prove a distribution-free impossibility:** any
high-confidence lower bound on mutual information from N samples cannot exceed
**O(log N)**. This directly constrains DV/MINE/InfoNCE-style lower bounds read as
universal certificates.

**Revision to B1 — this is the important one.** We must **not** claim a
distribution-free finite-sample confidence interval on a mutual information. The
escape is precise and must be stated in the first paragraph of any B1 writeup:

> RFS is not an MI estimate. It is a **difference of out-of-fold mean
> log-losses between two nested models**, with probabilities clipped to
> [1e-5, 1−1e-5]. It is therefore a **bounded** functional of held-out
> predictions, and a conformal/concentration argument over subject clusters
> applies to *that* quantity. Any interpretation of RFS as a bound on
> `I(Y; Z | B)` inherits the McAllester–Stratos limitation and is not
> certified by our procedure.

That is a genuinely defensible position — and it is also *why* the RFS coinage
earns its keep. If we called it an MI estimate we would walk straight into the
impossibility result.

**Prior art that narrows the novelty claim.** Oliveira et al. (2022), *Split
Conformal Prediction and Non-Exchangeable Data* (JMLR), already proves split
conformal stays valid on many dependent processes with an explicit **coverage
penalty** from concentration + decoupling — aimed at exactly our setting.
Barber et al. (2022) covers weighted/non-exchangeable. Gibbs et al. (2023) gives
exact finite-sample coverage over **finite-dimensional** shift classes, and
impossibility otherwise.

So B1's contribution is **not** "cluster conformal." It is: applying a
conformal/concentration floor to a *certification decision* — a one-sided
gate — on subject-clustered physiological data, with the exchangeability
assumption stated and its violation bounded. Narrower than the roadmap implied,
still real, and now correctly positioned.

---

## 3. slpdb is the weakest possible cohort for our hypothesis

This is the finding that most changes Track C, and it is not good news.

P3 splits the evidence cleanly:

| | spontaneous NREM arousals, healthy sleepers | respiratory-event arousals, apnea cohorts |
|---|---|---|
| autonomic **leads** EEG | **yes**, beat-level: Bonnet & Arand ≥10 beats; Togo ~1 s before desync; Attoh-Mensah shifts across the **30 s preceding** micro-arousal | **usually not shown** — evidence strength 3/10 |
| design | event-locked to EEG arousal onset | event-locked to **respiratory recovery** |
| what it supports | temporal precedence | predictive **surrogacy** |

**slpdb is a sleep-apnea cohort.** Its arousals are predominantly
respiratory-event arousals — the exact class where P3 rates "respiratory-event
papers prove autonomic lead over EEG onset" at **weak (3/10)**. In apnea,
airway reopening, hypoxia relief, cortical change and sympathetic discharge all
land within a few seconds, so precedence is nearly unidentifiable by construction.

Worse, there is a direct negative in that cohort: Bahr-Hamm et al. (2022) found
**no significant HRV-entropy difference** across sleep stages for OSA respiratory
cortical arousals and concluded those arousals were uncoupled from autonomic
activation. And Crescimanno et al. (2018) found only **31 % / 36 %** agreement
between cortical and autonomic arousals in ventilated ALS patients.

**Revisions to Track C:**
- **Demote slpdb from "smoke test" to "adversarial case."** It is not a small
  version of MESA. It is the sub-population where the hypothesis is least likely
  to hold. A null there is close to uninformative about the real question, and
  must be reported that way in `claim_scope` — the current scope string
  ("apnea_cohort_not_healthy_sleepers") is correct but understates it.
- **Stratify spontaneous vs respiratory arousals as a primary analysis axis, not a
  covariate.** The adapter already parses event tokens, so `A` (bare arousal) is
  separable from `HA`/`CAA`/`X` (respiratory). Pooling them averages a plausible
  effect with a documented null. This is a concrete code change to
  `adapters/slpdb.py` and the fixture builder.
- **On MESA, pre-register the spontaneous-arousal subgroup** as the primary
  analysis. That is where the ≥10-beat and 30-s precursor evidence actually lives.

**Encouraging, and directly ours:** Attoh-Mensah et al. (2023) found tachogram,
pulse-wave amplitude, pulse transit time and respiratory amplitude all shifting
during the **30 seconds preceding** EEG micro-arousal, and explicitly note the
result "needs finer continuous validation." Our horizon is in that window and our
method is that finer validation. Ribeiro et al. (2022, *PNAS*) adds the mechanistic
frame: peripheral events usually precede but **only about half recruit cortex** —
which predicts a real-but-modest RFS rather than a large one, and gives us a
principled reason to expect a small effect size instead of explaining one away
after the fact.

---

## 4. Arousal *detection* from ECG is close to solved — our framing must not collide with it

P4 returns numbers that would sink a carelessly framed Track C:

- Basner et al. (2007): ECG-only, five post-onset beats → **AUC 0.91**.
- Li et al. (2020): ECG model on **MESA** → **AUROC 0.93**.
- Olsen et al. (2018), Wisconsin Sleep Cohort: precision 0.72, sensitivity 0.63.

A reviewer who reads "autonomic signals predict cortical arousal" will reply that
this was done on MESA in 2020 at AUROC 0.93.

**Revision:** every Track C artifact must state the distinction in its first
sentence. Those are **detection** results — concurrent or post-onset beats,
classifying an arousal that is happening. Ours is **strictly-future forecasting at
a fixed horizon, scored as incremental value over a cortical spectral + history
baseline.** The nuisance block B is what makes it a different question: we are
asking what the autonomic signal adds *beyond what the cortex already tells you*,
before the event. Basner's five beats are post-onset and unconditioned.

**Bonus: C3 now has a citable benchmark.** Ehrlich et al. (2024) give real
dataset-held-out numbers — models score 0.82–0.83 F1 on their own cohort and
0.50–0.61 off it, a **0.12–0.32 absolute F1 drop**. If our MESA→SHHS transfer is
run, that is the number to compare against, and P4 confirms almost nobody does
this: "true dataset-held-out transfer has been evaluated only sparsely."

---

## 5. Do not build a benchmark. Build the check that plugs into theirs.

P5 confirms the A3 thesis — adoption follows tooling, and MOABB won by building on
MNE and scikit-learn to cut switching costs — but P7 adds the fact that changes the
plan: **the benchmark slot is now crowded.** NeuralBench (Meta, ~May 2026; 36 EEG
tasks, 94 datasets), EEG-FM-Bench, and OmniEEG-Bench all launched inside twelve
months.

**Revision to A3:** competing is the wrong move and we would lose. Ship
`leakage-check` as a **small, composable dependency** — takes a task spec, reports
overlap fraction and whether scoring is copy-safe — designed to be called *inside*
MOABB/NeuralBench/Braindecode pipelines. P5's failure-mode table is explicit that
narrow tools stall when they lack reusable code and external validation, and that
what wins is making good practice easier than bad practice. A one-function check
other benchmarks adopt beats a benchmark nobody switches to.

P5 also names an unfilled gap we already satisfy: SpeechBrain-MOABB criticises
MOABB and Braindecode for lacking **full-pipeline hyperparameter search and
multi-seed evaluation**. Our 24-seed detection-power sweep is exactly that
discipline.

---

## 6. Track D's clinical hook is weaker than the roadmap assumed

P6 is the most sobering return, and it corrects a claim currently in
`NEURAL_CASP_TECHNICAL_DOCUMENTATION.md`.

- **In OSA, hypoxic burden beats arousal burden.** Labarca et al. (2023): in MESA
  and MrOS, hypoxic and ventilatory burden predicted incident CVD, **arousal burden
  did not**. Azarbarzin et al. (2023), three cohorts: **arousal-only** hypopneas
  were unassociated with outcomes after adjusting for desaturation-defined events.
- **In community cohorts it holds up.** Shahrbabaki et al. (2021), 8001 people:
  arousal burden → CV mortality HR **2.17** (SOF women), **1.60** (SHHS women);
  weaker and mixed in men.

**Revisions:**
- The technical documentation currently motivates the arousal target via
  "arousal burden associates with cardiovascular outcomes — Huo et al. 2023." That
  is defensible only for **community cohorts**; in the apnea cohorts where our only
  open dataset lives, the association largely fails. Add the countervailing
  citation rather than leaving the one-sided version standing.
- **Gao et al. (2025) points at a better target.** Quantitative arousal *intensity*
  predicted cognition and sleepiness where arousal *counts* do not, and P6's gap
  table shows "arousal intensity / timing" thin across every outcome column. If we
  want a clinically meaningful `Y`, **graded arousal intensity is a better outcome
  than a binary event indicator** — and it is a less crowded question.

---

## 7. Field pulse — who reviews us, and who is already here

P7 names the community our Track A lands in:

- **Del Pup, Atzori et al. (Padova)** — the most active leakage/partitioning group;
  >100k models trained, recommends nested leave-n-subjects-out. Likely reviewers.
- **Brookshire et al.** — the anchor citation, still central.
- **Varatharajah, Wagh, Kuruppu (UMN/UIUC/Mayo)** — *EEG Foundation Models: A
  Critical Review* (arXiv 2507.11783 → JNE 2026).
- **Yang, Van Hulle et al. (KU Leuven)** — ICLR 2026, *Are EEG Foundation Models
  Worth It?*: fine-tuned FMs do not consistently beat compact nets in low-data or
  hard-transfer regimes; **no clear scaling law**.
- **Banville, d'Ascoli, King (Meta)** — NeuralBench; foundation models only
  *marginally* beat task-specific models, many cognitive decoding tasks near chance.

**Implication for B2.** The roadmap already gated the backbone on "does the
detection floor drop." Three independent 2026 results now say richer architectures
do *not* reliably help on EEG. **Raise the prior that B2 returns a negative, and
plan the writeup so that negative is the publishable outcome** — it would be a
fourth independent confirmation from a new angle (certification rather than
accuracy), which is worth more than a marginal positive.

---

## Net effect on the plan

| Track | Change |
|---|---|
| **A2** | Still the top priority — the analytic law is a *named* open gap. But novelty repositions to within-window mechanism + analytic law + enforcement. **Read Albelali 2025 first** for scoop risk. Reuse Medrano/Afyouni/Gahtan machinery; Paskaramoorthy is the template theorem. |
| **A3** | Do **not** build a competing benchmark (NeuralBench/EEG-FM-Bench/OmniEEG-Bench all <12 months old). Ship a composable `leakage-check` that plugs into them. |
| **B1** | Keep — the gap is confirmed across all four coverage columns. **Must disclaim MI interpretation** (McAllester–Stratos O(log N) impossibility). Position against Oliveira 2022 as prior art; contribution is the certification decision, not cluster conformal itself. |
| **B2** | Prior of a negative raised substantially. Plan the null as the deliverable. |
| **C** | **slpdb demoted to adversarial case** — apnea/respiratory arousals are where autonomic lead is *least* supported (3/10). **Stratify spontaneous vs respiratory as a primary axis.** Pre-register the spontaneous subgroup on MESA. Frame forecasting-vs-detection explicitly or collide with AUROC 0.93. |
| **C3** | Ehrlich 2024 gives the transfer benchmark: 0.12–0.32 absolute F1 drop cross-cohort. |
| **D** | Arousal burden loses to hypoxic burden in apnea cohorts; holds in community cohorts. Fix the one-sided citation in the technical doc. **Arousal intensity is a better target than binary events** (Gao 2025) and less crowded. |

## Immediate code consequences

1. `adapters/slpdb.py` — separate spontaneous (`A`) from respiratory
   (`HA`/`CAA`/`X`) arousal tokens; carry the distinction into the fixture so the
   gate can be run per-class.
2. `scripts/run_slpdb_autonomic_gate.py` — `claim_scope` should say the cohort is
   adversarial for the hypothesis, not merely unrepresentative.
3. `NEURAL_CASP_TECHNICAL_DOCUMENTATION.md` §F6 and §Related work — add the
   Labarca/Azarbarzin countervailing evidence beside Huo 2023; add the
   detection-vs-forecasting distinction with Basner/Li numbers.

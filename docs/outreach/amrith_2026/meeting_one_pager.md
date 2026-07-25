# Kahlus — where it stands, the live hypothesis, and what I'd ask you

**For:** Amrith Lotlikar (in-person, ~30 min) · **From:** Aayushya · 2026-07-24
**One-line:** the passive forecasting result is an honest tie with ridge; the real
opportunity is a literature-backed autonomic→arousal target I've built the gate for
but can't run until credentialed data lands.

---

## 1. Honest status (no dressing up)

- **Your metric point was right.** On the isolated task (predict only the
  strictly-future 128th sample, subject-held-out), the Kahlus GRU **ties** ridge —
  it does not meaningfully beat it. The earlier "big improvement" was the 126/127
  input–target overlap you flagged, not forecasting skill.
- **Why ridge is strong (your June question):** at millisecond single-channel
  scale, EEG is near-linear and autocorrelation-dominated — a 127-tap linear filter
  already captures almost all of it. There's little nonlinear structure left for a
  GRU to win on. The implication: the architecture isn't the bottleneck; the
  **target and the signal** are.
- **What I built instead of chasing that tie:** a leakage-controlled evaluation
  arena (Neural-CASP) — cross-fitted residual-forecastability score, baseline
  ladder (must beat persistence/moving-average), mandatory negative controls, and
  an overlap/copy-trap audit. It has produced honest **powered negatives** (sleep
  complexity, propofol sedation 21/21 subjects) — proof the gate *bites* rather
  than rubber-stamps.

## 2. The live hypothesis (where a real positive plausibly exists)

**Estimand:** does an autonomic/peripheral block **Z** (HRV, respiration) carry
residual forecastability for a **future cortical arousal Y**, beyond the cortex's
own spectral baseline **B**, under subject-held-out cross-fitting?

**Why this one, not the others** — it has genuine prior evidence with *temporal
precedence* (i.e. it's forecastable, not just correlational):
- Heart rate accelerates **≥10 beats before** the EEG arousal; sympathetic
  activation *precedes* arousals [3] (Bonnet 1997, 429 cites).
- The autonomic↔cortical arousal coupling is established across NREM/REM [1].
- It's been shown **on the MESA dataset specifically** [2] — the data I've applied
  for.

This is the opposite of the negatives so far: those targets (sleep-stage,
complexity) were ones where a null was *expected*. This one the literature says
should be positive.

## 3. The apparatus (already built, gate-ready)

- RFS gate + cross-fitting + controls (shuffle / time-shift / circular-shift
  surrogate / subject-probe) + cluster-bootstrap CIs + cluster-permutation p.
- NSRR **MESA/SHHS loader** written and unit-tested (synthetic known-signal
  passes, known-null fails — machinery validated).
- Runs in **one command** the moment credentialed data lands. Pre-registered so a
  null is a real, reportable negative and a positive can't be p-hacked into being.

## 4. Data status (the one real blocker)

- The positive signal is **HRV/ECG** — not present in the public data on disk
  (Sleep-EDF has no clean ECG; that's why the peripheral scout there was a
  negative). So there is no honest runnable-today positive on current data.
- **MESA + SHHS credentialed access: applied for (NSRR).** On arrival: F6 runs,
  yielding either a certified positive (literature-likely) or an honest powered
  null.

## 5. What I would not do

Bring you a manufactured positive. Every result in the arena is real; the positive
above is a **target + apparatus + prior evidence**, with the verdict deliberately
pending the data. No clinical, seizure, or consciousness claims.

## 6. What I'd most value your take on (in person)

1. Given the passive tie, is a **physical/interventional experiment** the more
   honest next step — and what's the *minimal* first one? (Your stimulation-
   threshold work is why I want to ask you specifically — e.g. closed-loop acoustic
   or targeted-reactivation cueing timed to a detected state.)
2. Does your threshold-finding methodology transfer at all to non-invasive /
   closed-loop paradigms, or is implanted-electrode the only regime where it's
   meaningful?
3. Is the certified directed-information / multi-organ direction a real thesis, or
   a detour from what you'd consider a fundable question?
4. Where am I misunderstanding the problem?
5. What would a strong SIMR→lab transition actually require from me?

---

### References
[1] [EEG beta power and heart rate variability describe the association between cortical and autonomic arousals across sleep](https://consensus.app/papers/details/7a7f06c77b0a57eaa6c101fe86cb4b3e/?utm_source=claude_code) (Kuo et al., 2015, *Autonomic Neuroscience*)
[2] [Coupling analysis of heart rate variability and cortical arousal using a deep learning algorithm](https://consensus.app/papers/details/fb49dac1da7d5200a934e6355a6a1464/?utm_source=claude_code) (Huo et al., 2023, *PLOS ONE* — MESA dataset)
[3] [Heart rate variability: sleep stage, time of night, and arousal influences](https://consensus.app/papers/details/590e7395ba1d565f96212a47adb52bb7/?utm_source=claude_code) (Bonnet & Arand, 1997, *Electroenceph. Clin. Neurophysiol.* — HR accel ≥10 beats pre-arousal)

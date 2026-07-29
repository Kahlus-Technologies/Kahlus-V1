# North star

Read this first, every cycle. If a cycle's work does not move one of the
decisions below, the cycle picked the wrong work.

---

## The question

**Can a model see a change in brain state before it happens, with enough lead
time to act?**

Everything else is instrumental.

## Where we actually are

Two results stand up:

- **F9** — a window-scored forecast metric inflates by at most `W/H`. Derived,
  286 configurations, zero violations. This is a *methods* result.
- **F8** — with honest scoring, a GRU beats a tuned ridge out to 160 ms on 78
  subjects; the advantage is gone by 320 ms.

Seven other findings are negative or incomplete. That is the real shape.

## The uncomfortable fact this program exists to resolve

**160 ms is roughly 400× short of the timescale that would matter clinically, and
the margin over a linear baseline is small (0.83 → 0.73 MSE).** Raw next-sample
EEG forecasting may be close to its ceiling — not because our models are weak, but
because the signal is substantially stochastic.

If that is true, optimising forecasters is wasted effort and the program must
redirect. **We do not currently know whether it is true.** That is the first
decision below, and it gates everything else.

## Open decisions, in priority order

Each cycle must move exactly one. A cycle that *closes* a decision by killing a
line is a successful cycle — arguably the most valuable kind.

### D1. Is there headroom? *(gates everything)*
Does prediction error keep falling as model capacity and class increase, or does
it plateau? If ridge-127, a larger GRU, and a temporal CNN all land within a few
percent of each other, we are at a practical ceiling for this task and D2 becomes
the whole program.

**Closes when:** a capacity-saturation curve exists across ≥4 model classes at
h ∈ {1, 16, 64}, with the plateau (or absence of one) visible.

**Caveat to state in any writeup:** this measures an *empirical* ceiling for the
classes tried, not an information-theoretic bound. Do not call it the latter.

### D2. Is the transition task reachable?
Predicting a *state change* (arousal, stage transition, seizure onset) minutes
ahead is the clinically meaningful task and has external benchmarks. Is there
detectable signal at 30–60 s lead?

**Closes when:** an honest, controlled attempt at ≥30 s lead reports either a
result that beats its baseline with controls passing, or a powered null.

### D3. Does the F8 crossover move with multichannel input?
Single channel is the likeliest reason the advantage dies at 320 ms.
Sleep-EDF has Fpz-Cz, Pz-Oz and EOG on disk.

**Closes when:** the crossover horizon is measured with multichannel input under
the F8 protocol, and compared to the single-channel result.

### D4. Is the F9 bound the general case?
Does it hold for overlapping constructions other than the shifted window
(strided multi-step targets, autoregressive rollout scoring)?

**Closes when:** the copyable-position count is derived for ≥2 other
constructions and tested.

### D5. Prevalence — how often does published work do this?
Turns F9 from a tool into a finding about the field. Mechanical; lowest risk.

**Closes when:** ≥60 papers coded per `../prevalence_survey/PROTOCOL.md`.

## What would make us stop

- If **D1** shows a hard plateau and **D2** returns a powered null at useful lead
  times, then the honest conclusion is that this signal does not support the
  original ambition, and the project's contribution is the methods work (F1, F8,
  F9). **That is a real outcome, not a failure**, and the program should say so
  rather than grinding.

## What "best in the field" would require

Not a feeling. An external scoreboard we do not control — a public benchmark or
held-out contest someone else scores. Until a target like that is named, treat
"best" as undefined and optimise the decisions above instead.

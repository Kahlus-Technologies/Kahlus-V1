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
| — | *(none yet)* | | | |

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

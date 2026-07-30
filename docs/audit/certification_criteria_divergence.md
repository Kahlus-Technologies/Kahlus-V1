# Certification criteria divergence

Audit finding, 2026-07-26. Status: **latent, not live** — nothing is currently
miscertified. Recorded before any consolidation so the reconciliation is
reviewable on its own.

## What was found

Ten predicates across `src/neurotwin/forecastability/` ask the same six questions
and disagree on the constants.

### Effect-size floor — 30x spread, two modules have none

| Module | Floor on `rfs_bits` | Anchor |
| --- | ---: | --- |
| `m5.py` | 0.001 | `m5.py:345` |
| `_rfs_eval.py`, `passive_pci.py` | 0.02 | `_rfs_eval.py:147,186`; `passive_pci.py:367,402` |
| `m1.py`, `m4.py` | 0.03 | `m1.py:444`; `m4.py:408` |
| `m2.py` | **none** | `m2.py:565,583` |
| `m3.py` | **none** | `m3.py:392-417` |

`m3` carries the seizure claim and certifies on `rfs_ci_low > 0` plus controls
and power, with no effect-size floor at all. That is the `ci_only` protocol
`scripts/false_certification_benchmark.py` exists to argue against.

### Control ratio — 0.4 vs 0.5, and m2 has it backwards

- **0.40**: `_rfs_eval.py:148,149,188`; `m1.py:446,447`; `m2.py:574`;
  `passive_pci.py:368,369,404`
- **0.50**: `m2.py:595`; `m3.py:406,409`; `m5.py:347,349`

`m2.py:574` holds the **synthetic** fixture — ground truth known, no claim at
stake — to 0.40, while `m2.py:595` holds **real Sleep-EDF** to 0.50. Real data
gets 25% more leakage headroom than the machinery check.

### Sign inversion in the failure-list form

`m2.py:595`, `m3.py:406,409`, `m5.py:347,349` evaluate
`control >= full * ratio` with no guard that `full > 0`. With `full = -0.10` the
bar becomes `-0.05`, so a control at `-0.20` records no failure — the worse the
primary performs, the easier its own control check becomes.

Currently masked because `rfs_ci_low <= 0` fires separately in the same failure
list. That masking is incidental: it depends on the CI check staying in the list
and on the bootstrap CI agreeing with the point estimate.

## Why nothing is miscertified

- `m3_gate_report.json` is `gate_passed: false`.
- `m2`'s real Sleep-EDF pass was **back-adjudicated** under the stricter 0.40:
  real effect `+0.06008` bits, time-shift control `+0.01337` = **22.3%** of the
  effect, against a 40% limit. It passes either way.

So consolidating to the stricter constants is a fix, not a rewrite of history —
but that is a checked fact, not an assumption. `git blame` shows the 0.50s all
come from one commit (`8d727de7`, 2026-07-02) and the 0.40s from later
consolidation work (2026-07-22/23) that never went back. Drift, not design.

## What was done

`criteria.py` extracts one `GateCriteria` + `evaluate()` and declares each
module's constants **verbatim**, with file:line anchors. Verified
behaviour-preserving: 8 comparisons against `horizon_real_passes` on real
archived payloads, 0 mismatches. `docs/audit/verdict_oracle.json` freezes all 13
archived verdicts; `scripts/ops/freeze_verdicts.py --check` fails on any flip.

The sign guard is written once, correctly, in `evaluate()`.

## What is deliberately NOT done

Reconciling the constants. That is a scientific decision and belongs in its own
commit naming the justification and the results expected to change. Folding a
threshold move into a refactor is how a published result's meaning changes
without anyone noticing.

Open question for that commit: **should `m3` have an effect-size floor, and what
should it be?** It is the only gate with a clinical claim attached and the only
one with no floor.

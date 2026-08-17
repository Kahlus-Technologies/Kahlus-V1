# F10 — the correlation inflation law, and why F9's `W/H` does not extend to it

**Date:** 2026-07-30
**Harness:** `scripts/correlation_inflation_law.py`, figure `scripts/plot_correlation_law.py`
**Artifact:** `runs/correlation_law/results.json` (286 cells), `runs/correlation_law/correlation_law.png`
**Status:** theorem proved and verified; one natural extrapolation from F9 **falsified**

---

## Why this exists

F9 bounded how much window-overlap inflates **MSE**. But MSE is not what most
biosignal forecasting papers lead with — they lead with **r** or **R²**. If the
`W/H` bound transferred, one number off a methods section would audit both
metrics and F9 would roughly double in reach.

It does not transfer. That is the main result here, and it points the wrong way
from the convenient one: **`W/H` *understates* correlation inflation.** A reader
who took F9's bound and applied it to a reported correlation would conclude the
paper is less inflated than it actually is.

---

## The theorem

Under the flawed protocol — `X = seq[0:W]`, `Y = seq[H:H+W]` — the `W` scored
positions split into `W−H` **copyable** (set `C`) and `H` **strictly future**
(set `F`). Pool all prediction/target pairs and take deviations about the global
means, which is exactly what a reported `r` is built from:

```
A_C = Σ_C (p−μ_p)²    A_F = Σ_F (p−μ_p)²
B_C = Σ_C (t−μ_t)²    B_F = Σ_F (t−μ_t)²
S_C = Σ_C (p−μ_p)(t−μ_t)              S_F = Σ_F (p−μ_p)(t−μ_t)
```

By AM-GM, `√((A_C+A_F)(B_C+B_F)) ≥ √(A_C B_C) + √(A_F B_F)`. Since `r_overlap`
and the mixture below share a numerator and differ only in denominator:

> **Theorem (exact, no distributional assumption).**
> `|r_overlap| ≤ | μ·r_copy + (1−μ)·r_future |`, where
> `μ = √(A_C B_C) / (√(A_C B_C) + √(A_F B_F))`.
> When `r_overlap ≥ 0` — the only case in which a paper is claiming skill — the
> absolute values drop.

**The sign condition is load-bearing.** With a negative pooled covariance the
larger denominator pulls `r_overlap` *toward* zero and the inequality reverses.
The first version of this derivation asserted the signed form unconditionally;
`--selftest` on adversarial draws failed it, which is how the condition ended up
in the theorem instead of in a referee report.

Because `r_copy ≤ 1`, the corollary an auditor would want is:

> `(1 − r_future) / (1 − r_overlap) ≤ 1/(1−μ)`

The **correlation gap** deflates by at most `1/(1−μ)`. Setting `μ = (W−H)/W`
— which is what per-position variance homogeneity would give — recovers `W/H`,
the same constant as F9.

---

## Verification

Same grid as F9: `W ∈ {32,64,128,256} × H ∈ {1,…,128}`, `H < W`, on 11 traces
(7 recorded channels from Sleep-EDF and MIT-BIH polysomnographic, 4 synthetic
processes). 286 cells.

| Claim | Violations |
| --- | --- |
| Pooled identity for `r_overlap` | 0 / 286 (asserted per cell) |
| **Mixture theorem** | **0 / 286** |
| **Exact gap bound `1/(1−μ)`** | **0 / 286** |
| `W/H` shortcut carried over from F9 | **223 / 286** |

The first three are what a correct derivation and correct code look like. The
fourth is the finding.

---

## Why `W/H` fails, exactly

`1/(1−μ) = 1 + g/h`, and expanding `g/h` in per-position variances gives, with
no approximation:

> `gap bound = 1 + ((W−H)/H) · (s_pred · s_targ)^(−1/2)`

where `s_pred`, `s_targ` are the future/copy per-position variance ratios of the
**prediction** and the **target**. `W/H` is exactly the special case
`s_pred = s_targ = 1`.

This is verified, not asserted — the shrinkage-predicted bound reproduces the
measured exact bound to **median relative error 7.1×10⁻¹⁶** (max 4.5×10⁻¹⁴)
across all 286 cells. Rearranged, it forces a parameter-free collapse:

```
(exact − 1) / (W/H − 1) = (s_pred · s_targ)^(−1/2)
```

Every cell lies on that one curve over three decades (figure, right panel).

**Which term does the damage.** `s_targ` has median **0.996** — the targets are
variance-stationary within the window, as expected. `s_pred` has median
**0.700**, range `[0.008, 1.382]`. So prediction shrinkage is the entire
mechanism: a regularised model shrinks toward the mean *precisely where the
signal is unpredictable*, deflating `A_F`, raising `μ` above `(W−H)/W`, and
pushing the true bound above `W/H`.

Median `exact / (W/H)` is **1.209** — so `W/H` typically understates correlation
inflation by about 20%. The tail is much worse.

---

## The part that should worry a reader

The failure is not random across signals. It is **monotone in how forecastable
the signal is**:

| Trace | `W/H` holds | median `s_pred` | median excess |
| --- | --- | --- | --- |
| `slpdb_resp` (nasal) | 18/26 | 1.026 | 0.98 |
| `slpdb_bp` | 20/26 | 0.984 | 1.02 |
| `sleepedf_eog` | 6/26 | 0.874 | 1.06 |
| `sleepedf_eeg_fpz_cz` | 4/26 | 0.716 | 1.15 |
| `random_walk` | 2/26 | 0.710 | 1.25 |
| `slpdb_ecg` | 3/26 | 0.497 | 1.41 |
| `ar1_0.5` | 0/26 | 0.098 | 2.89 |
| `white_noise` | 0/26 | 0.038 | **4.79** |

**`W/H` is least reliable exactly where the inflation matters most.** On a
strongly autocorrelated signal the model barely shrinks and `W/H` is nearly
right. On a signal with little forecastable structure the model shrinks hard,
and `W/H` can understate by an order of magnitude.

The sharpest single cell — **white noise, `W`=32, `H`=1**:

> reported **`r = 0.9843`**, true **`r_future = 0.0141`**, `W/H` understates the
> deflation by **10.8×**.

Nothing whatsoever is forecastable in white noise. The protocol still reports
`r = 0.98`. As in F9, the artifact belongs to the **metric**, not to the signal
or the model — but here it is worse than F9's version of the same statement,
because the correlation case has no safe data-free bound to fall back on.

---

## What an auditor can and cannot do

**Can:** given `(W, H, r_reported)`, the *necessary* condition
`r_reported ≥ 1 − H/W` is still checkable with no data. Below it, the reported
number is weaker than trivial copy-and-persist reaches and constrains forecast
skill not at all. For the historical Kahlus bug (`W`=127, `H`=1) the threshold is
`r ≥ 0.99213`; the reported value was **0.972**, i.e. below it — so that number
never bounded forecast skill in the first place.

**Cannot:** convert a reported `r` into a *lower bound* on `r_future` using
`W/H`. That step is unsound, by 223/286 here. It needs `s_pred`, which needs the
predictions, which needs the authors. The honest ask is for `r` recomputed on
the strictly-future positions — one line of code for anyone holding the model.

---

## Reproduce

```bash
PYTHONPATH=src python3 scripts/correlation_inflation_law.py --selftest
PYTHONPATH=src python3 scripts/correlation_inflation_law.py --out runs/correlation_law
python3 scripts/plot_correlation_law.py --results runs/correlation_law/results.json
```

## Not claimed

- Nothing about model quality. As with F9, a good model and a useless one are
  inflated by the same mechanism.
- The mixture theorem is proved; the *prevalence* of `H < W` in the published
  literature is still unmeasured (register H5, decision D5). Until that is done,
  F9 and F10 are tools for auditing a paper, not a claim about the field.
- `s_pred < 1` is documented here for ridge on these traces. It is expected of
  any shrinkage estimator, but the specific magnitudes are not a general law.

# F9 — The overlap inflation law: `inflation ≤ W/H`, from the methods section alone

**Date:** 2026-07-28. **Status:** exact bound, derived and verified.
**Artifacts (tracked):** `artifacts/overlap_law/`.
**Reproduce:**

```
PYTHONPATH=src python3 scripts/overlap_inflation_law.py --out runs/overlap_law
python3 scripts/plot_overlap_law.py
```

F1 recorded that the invalidated `3.116 MSE / r=0.972` result was a 126× overlap
artifact. This asks the next question: **is that inflation predictable in closed
form?** It is. The consequence is that a reader can compute how inflated a
published forecasting number is from the window length and horizon printed in its
methods section — no code, no data, no rerun.

---

## 1. The construction and the identity

From a sequence of length `W+H`:

```
X = seq[0 : W]        the input window
Y = seq[H : H+W]      the target window, shifted by H
```

`Y[k] = X[k+H]` whenever `k < W-H`. So of the `W` scored target positions,
`W-H` are **copyable** — literally present in the input at a shifted index — and
only `H` are **strictly future**.

Both scores below come from **one model and one set of predictions**. Only the
scoring mask differs, so no argument about model capacity, training, or tuning
can account for the gap:

| quantity | mask |
| --- | --- |
| `MSE_overlap` | all `W` positions — what the flawed protocol reports |
| `MSE_future` | the last `H` positions — the honest forecast error |
| `MSE_copy` | the first `W-H` positions |

These satisfy an identity, not a fit:

```
MSE_overlap = [ (W-H)·MSE_copy + H·MSE_future ] / W
```

which rearranges to

```
inflation ≡ MSE_future / MSE_overlap
          = (W/H) / ( 1 + ((W-H)/H)·(MSE_copy/MSE_future) )
```

Two claims follow:

- **(a) Bound.** `inflation ≤ W/H`, always. Equality iff `MSE_copy = 0`.
- **(b) Asymptote.** `inflation → W/H` as `MSE_copy/MSE_future → 0`.

`W/H` requires no data. For the historical bug (`W=127, H=1`) it predicts **127×**;
the repo measured **126.1×**.

## 2. Verification

**286 configurations. 11 signals. Zero bound violations.**

`W ∈ {32, 64, 128, 256}` × `H ∈ {1,2,4,8,16,32,64,128}` with `H < W`, over
signals chosen to span autocorrelation structure as widely as possible: three
Sleep-EDF channels (Fpz-Cz, Pz-Oz, EOG), four slpdb channels (ECG, BP, O2-A1 EEG,
nasal respiration), and four synthetic processes (white noise, AR(1) φ=0.5,
AR(1) φ=0.95, random walk).

| statistic | value |
| --- | --- |
| bound violations | **0 / 286** |
| observed/bound, median | **1.000** |
| observed/bound, mean | 0.981 |
| observed/bound, min | 0.292 |
| exact formula vs observed, max relative error | **1.3e-15** |

The exact-formula agreement is machine precision because the decomposition is an
identity; the script asserts it per cell, so a broken mask fails loudly rather
than producing a plausible number.

Every departure from the bound is explained by the single correction term, and it
is monotone across five decades:

| `MSE_copy/MSE_future` | cells | mean observed/bound |
| --- | ---: | ---: |
| `< 1e-5` | 126 | **1.000** |
| `1e-5 – 1e-3` | 96 | 0.998 |
| `1e-3 – 1e-2` | 38 | 0.986 |
| `1e-2 – 1e-1` | 22 | 0.887 |
| `> 1e-1` | 4 | 0.428 |

**The sharpest test is white noise.** A signal with no predictable structure at
all still shows inflation at the bound (median 1.000, min 0.997), because the
copyable positions are present in the input regardless of whether anything is
forecastable. This is what establishes that the artifact is a property of the
**metric**, not of the signal or the model.

The largest deviations are slpdb blood pressure (`MSE_copy/MSE_future` up to
0.16), where the copy is reproduced poorly, so the bound is loose exactly as the
formula says it should be. The bound is never violated.

## 3. What this is for

Given `W` and `H` from a methods section, `W/H` is an upper bound on the factor
by which a window-scored forecast metric overstates true forecast skill. Two
practical uses:

1. **Triage a published number without rerunning anything.** A paper reporting a
   128-sample target window at horizon 1 is reporting a number that can be up to
   128× better than its forecast error.
2. **A one-line reviewer check.** `H ≥ W` is sufficient for the metric to be
   clean. Anything else needs a strictly-future mask.

## 4. Scope and limits

- The bound governs the **metric**. It says nothing about whether a model is good;
  a model with genuine skill and a model with none are inflated by the same factor.
- It assumes the shifted-window construction above (target = input window shifted
  by `H`). Other overlapping constructions need their own copyable-position count,
  though the same identity applies once that count is known.
- Tightness depends on how exactly the copyable positions are reproduced. A weak
  or heavily regularised model leaves the bound loose. **The bound is one-sided:
  a real result can never exceed it, so quoting `W/H` is always safe as a
  worst case.**
- Verified with ridge. The identity is model-independent by construction, but only
  ridge was swept here; the decomposition assertion would catch a mask error under
  any model.
- **Not yet done:** the prevalence survey. Establishing how often published EEG
  forecasting work uses `H < W` is what turns this from a tool into a finding
  about the field, and it is the obvious next step
  (cf. McGreivy & Hakim 2024, *Nat. Mach. Intell.*: 79% of ML-for-PDE papers
  compared against a weak baseline).

# F10 — correlation inflation law

Evidence for `docs/results/correlation_inflation_law_2026-07-30.md`.

| File | What it is |
| --- | --- |
| `results.json` | 286 cells: `W ∈ {32,64,128,256} × H ∈ {1..128}`, `H < W`, on 11 traces. Per cell: the six pooled deviation moments, `r_overlap` / `r_copy` / `r_future`, both candidate bounds, the shrinkage diagnostics `s_pred` / `s_targ`, and the copy-and-persist floor. |
| `correlation_law.png` | Figure. Left: the exact bound holds (0/286), the `W/H` shortcut does not (223/286). Right: the parameter-free collapse `(exact−1)/(W/H−1) = (s_pred·s_targ)^(−1/2)`. |

## Headline

The mixture theorem and the exact gap bound are violated by **no cell**. The
`W/H` shortcut carried over from F9 is violated by **223 of 286**, always in the
direction that makes a paper look *better* than it is.

Mechanism verified to machine precision (median relative error 7.1×10⁻¹⁶):
prediction shrinkage, `s_pred` median 0.700 against `s_targ` median 0.996.

Sharpest cell — white noise, `W`=32, `H`=1: reported `r = 0.9843`, true
`r_future = 0.0141`.

## Regenerate

```bash
PYTHONPATH=src python3 scripts/correlation_inflation_law.py --selftest
PYTHONPATH=src python3 scripts/correlation_inflation_law.py --out runs/correlation_law
python3 scripts/plot_correlation_law.py --results runs/correlation_law/results.json
```

Deterministic: `SEED = 0`, shared with F9 via `scripts/overlap_inflation_law.py`.

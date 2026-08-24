"""F10: the correlation analogue of the overlap inflation law.

F9 bounded how much window-overlap inflates MSE. That bound is easy because MSE
is a per-position average, so it decomposes additively and the law is an
identity. Correlation is not an average -- it is a ratio of a covariance to a
product of standard deviations -- so the same argument does not apply, and the
question of whether a bound exists at all is open.

It does, it is exact, and it needs no distributional assumption.

SETUP. Under the flawed protocol, from a sequence of length W+H:

    X = seq[0 : W]      Y = seq[H : H+W]      Y[k] = X[k+H] for k < W-H

so of the W scored positions, W-H are *copyable* (set C) and H are strictly
future (set F). Pool all (n_score x W) prediction/target pairs and take
deviations about the global means mu_pred, mu_targ -- exactly the quantities a
paper's reported r is built from. Write

    A_C = sum_C (p - mu_p)^2      A_F = sum_F (p - mu_p)^2
    B_C = sum_C (t - mu_t)^2      B_F = sum_F (t - mu_t)^2
    S_C = sum_C (p - mu_p)(t - mu_t)                S_F = sum_F ...

Then, by definition and with no approximation,

    r_overlap = (S_C + S_F) / sqrt( (A_C + A_F)(B_C + B_F) )
    r_copy    = S_C / sqrt(A_C B_C)        r_future = S_F / sqrt(A_F B_F)

THEOREM (mixture bound; exact, assumption-free).

    sqrt((A_C + A_F)(B_C + B_F)) >= sqrt(A_C B_C) + sqrt(A_F B_F)

  -- squaring reduces this to A_C B_F + A_F B_C >= 2 sqrt(A_C B_F A_F B_C),
  which is AM-GM. Write mu = sqrt(A_C B_C) / (sqrt(A_C B_C) + sqrt(A_F B_F)),
  so that S_C + S_F = (g + h)(mu*r_copy + (1-mu)*r_future) with g + h the
  smaller denominator. Both r_overlap and the mixture therefore share a
  numerator and differ only in denominator, giving

    |r_overlap| <= | mu * r_copy + (1 - mu) * r_future |,

  with equality iff A_C B_F = A_F B_C. When r_overlap >= 0 -- the only case in
  which a paper is claiming forecast skill -- the signs drop out:

    r_overlap <= mu * r_copy + (1 - mu) * r_future.

  A reported correlation is at most a variance-weighted blend of how well the
  model copies and how well it forecasts.

  The sign condition is not cosmetic: with a negative pooled covariance the
  larger denominator pulls r_overlap *toward* zero, so the inequality reverses.
  A `--selftest` run on adversarial inputs caught exactly that case.

COROLLARY (the auditable form; requires r_overlap >= 0). Because r_copy <= 1,

    1 - r_overlap >= mu(1 - r_copy) + (1 - mu)(1 - r_future) >= (1 - mu)(1 - r_future)

so the *correlation gap* deflates by at most 1/(1 - mu). Expanding mu in
per-position variances gives, still with no approximation,

    (1 - r_future) / (1 - r_overlap)  <=  1 + ((W-H)/H) * (s_pred * s_targ)^-0.5

with s_pred, s_targ the future/copy per-position variance ratios of the
prediction and of the target. W/H -- F9's constant -- is exactly the special
case s_pred = s_targ = 1.

THE CONVENIENT EXTRAPOLATION IS FALSE. It would have been very useful if W/H
carried over, because then one number off a methods section would audit both
metrics. This script tested that and it fails on 223 of 286 cells, always in
the flattering direction: s_targ is ~1 (targets are variance-stationary within
the window, median 0.996) but s_pred has median 0.700, because a regularised
model shrinks toward the mean precisely where the signal is unpredictable. So
the true deflation EXCEEDS W/H -- median 1.21x, up to 10.8x -- and W/H
UNDERSTATES correlation inflation, worst where there is least real signal.
See docs/results/correlation_inflation_law_2026-07-30.md.

WHAT SURVIVES WITH NO DATA, ONLY (W, H, r_reported):

  Not an implied floor on r_future -- that step needs the W/H bound and is
  therefore unsound. What survives is a one-directional screen. When
  s_pred * s_targ <= 1 the exact bound is at least W/H, so the threshold for a
  non-vacuous floor is at least 1 - H/W and

      r_reported <  1 - H/W   =>   the reported number bounds nothing.

  Caveat: 39 of the 286 measured cells have s_pred * s_targ > 1 (the exact
  bound sits BELOW W/H there, min 0.85x), so the screen is only sound under
  that condition; check s_pred * s_targ, or use the exact bound, before
  applying it to a cell whose prediction variance exceeds the target's.

  The converse does NOT hold: clearing the screen does not make a number
  informative, it only fails to prove it uninformative. For the historical
  Kahlus bug (W=127, H=1) the screen sits at r >= 0.99213 and the reported
  value was 0.972 -- below the level trivial copy-and-persist reaches, so that
  number never constrained forecast skill at all.

WHAT THIS SCRIPT CHECKS.
  (a) the pooled identity for r_overlap holds to machine precision;
  (b) the mixture bound is never violated (this is the theorem, so a violation
      means the code is wrong, not the maths);
  (c) whether the *stationarity-dependent* W/H form -- the only one an auditor
      can apply to a paper they cannot re-run -- survives on real signals;
  (d) the empirical floor: what r a pure copy-and-persist predictor, which does
      no forecasting whatsoever, scores under the flawed protocol.

  PYTHONPATH=src python3 scripts/correlation_inflation_law.py --out runs/correlation_law
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from overlap_inflation_law import (  # noqa: E402
    RIDGE_ALPHAS,
    SEED,
    SHIFTS,
    WINDOWS,
    load_real_signals,
    multi_ridge,
    predict,
    synth,
    windows_from_trace,
)

N_WINDOWS = 6000


def _moments(pred: np.ndarray, targ: np.ndarray, split: int) -> dict:
    """Deviation moments about the GLOBAL pooled means, split copy / future.

    The global mean is the one a paper's reported r is built from, so the
    decomposition has to use it -- using per-set means would answer a different
    question than the one being audited.
    """
    p = pred - pred.mean()
    t = targ - targ.mean()
    pc, pf = p[:, :split], p[:, split:]
    tc, tf = t[:, :split], t[:, split:]
    return {
        "A_C": float((pc ** 2).sum()), "A_F": float((pf ** 2).sum()),
        "B_C": float((tc ** 2).sum()), "B_F": float((tf ** 2).sum()),
        "S_C": float((pc * tc).sum()), "S_F": float((pf * tf).sum()),
    }


def _r(s: float, a: float, b: float) -> float:
    d = np.sqrt(a * b)
    return float(s / d) if d > 0 else float("nan")


def copy_persist(x: np.ndarray, w: int, h: int) -> np.ndarray:
    """The dumbest predictor that exploits the flaw: copy what is copyable,
    hold the last input sample across the H strictly-future positions.

    It performs zero forecasting by construction, so whatever r it scores under
    the flawed protocol is a floor that any reported number must clear before it
    can mean anything.
    """
    out = np.empty((x.shape[0], w), dtype=np.float64)
    out[:, : w - h] = x[:, h:w]                  # exactly the copyable positions
    out[:, w - h:] = x[:, w - 1:w]               # persistence on the rest
    return out


def one_cell(seqs: np.ndarray, w: int, h: int) -> dict:
    x, y = seqs[:, :w], seqs[:, h:h + w]
    n_fit = int(0.6 * x.shape[0])
    xf, yf, xs, ys = x[:n_fit], y[:n_fit], x[n_fit:], y[n_fit:]

    # Tune on the audited protocol's own objective, as in F9: the fairest
    # reading of a flawed metric is the alpha that metric would itself pick.
    best = (np.inf, None, None)
    for alpha in RIDGE_ALPHAS:
        b = multi_ridge(xf, yf, alpha)
        m = float(np.mean((ys - predict(b, xs)) ** 2))
        if m < best[0]:
            best = (m, alpha, b)
    _, alpha, b = best

    pred = predict(b, xs)
    targ = ys.astype(np.float64)
    split = w - h
    m = _moments(pred, targ, split)

    r_overlap = _r(m["S_C"] + m["S_F"], m["A_C"] + m["A_F"], m["B_C"] + m["B_F"])
    r_copy = _r(m["S_C"], m["A_C"], m["B_C"])
    r_future = _r(m["S_F"], m["A_F"], m["B_F"])

    # (a) identity: the pooled r really is built from these six sums.
    direct = float(np.corrcoef(pred.ravel(), targ.ravel())[0, 1])
    assert abs(direct - r_overlap) < 1e-9, f"pooled identity broken: {direct} vs {r_overlap}"

    # (b) the exact mixture bound.
    g, hh = np.sqrt(m["A_C"] * m["B_C"]), np.sqrt(m["A_F"] * m["B_F"])
    mu = float(g / (g + hh)) if (g + hh) > 0 else float("nan")
    mixture = mu * r_copy + (1.0 - mu) * r_future
    # Signed form is only valid for r_overlap >= 0; the magnitude form always is.
    mixture_ok = (abs(r_overlap) <= abs(mixture) + 1e-9)

    # (c) exact gap-deflation bound vs the methods-section-only W/H form.
    gap_obs = (1.0 - r_future) / (1.0 - r_overlap) if r_overlap < 1.0 else float("inf")
    bound_exact = 1.0 / (1.0 - mu) if mu < 1.0 else float("inf")
    bound_wh = w / h

    # (d) the floor a non-forecasting predictor reaches under the same protocol.
    cp = copy_persist(xs, w, h)
    r_floor = float(np.corrcoef(cp.ravel(), targ.ravel())[0, 1])

    # Why the W/H form fails, exactly. 1/(1-mu) = 1 + g/h, and
    #     g/h = sqrt( (A_C/A_F)(B_C/B_F) ) = ((W-H)/H) * (s_pred * s_targ)^-0.5
    # with s_pred, s_targ the per-position future/copy variance ratios of the
    # prediction and the target. So
    #     gap bound = 1 + ((W-H)/H) * (s_pred * s_targ)^-0.5,
    # and W/H is exactly the special case s_pred = s_targ = 1. Two distinct
    # causes push it away from that: a regularised model shrinks toward the mean
    # precisely where the signal is unpredictable (s_pred < 1), and the signal
    # itself need not be variance-stationary within the window (s_targ != 1).
    # Recording both makes the mechanism checkable rather than asserted.
    s_pred = ((m["A_F"] / h) / (m["A_C"] / split)) if m["A_C"] > 0 else float("nan")
    s_targ = ((m["B_F"] / h) / (m["B_C"] / split)) if m["B_C"] > 0 else float("nan")
    bound_from_shrinkage = (1.0 + (split / h) / np.sqrt(s_pred * s_targ)
                            if s_pred > 0 and s_targ > 0 else float("nan"))

    # What an auditor with only (W, H, r_overlap) would conclude.
    implied = 1.0 - bound_wh * (1.0 - r_overlap)
    return {
        "W": w, "H": h, "alpha": alpha,
        "A_C": m["A_C"], "A_F": m["A_F"], "B_C": m["B_C"], "B_F": m["B_F"],
        "pred_var_ratio_future_over_copy": s_pred,
        "targ_var_ratio_future_over_copy": s_targ,
        "gap_bound_from_shrinkage": float(bound_from_shrinkage),
        "r_overlap": r_overlap, "r_copy": r_copy, "r_future": r_future,
        "mu": mu, "mixture_bound": mixture,
        "mixture_respected": bool(mixture_ok),
        "gap_deflation_observed": gap_obs,
        "gap_bound_exact": bound_exact,
        "gap_bound_W_over_H": bound_wh,
        "exact_respected": bool(gap_obs <= bound_exact * (1 + 1e-9)),
        "wh_respected": bool(gap_obs <= bound_wh * (1 + 1e-9)),
        "r_copy_persist_floor": r_floor,
        "vacuity_threshold": 1.0 - h / w,
        # One-directional: failing this proves the number bounds nothing;
        # passing it does NOT prove the number is informative.
        "clears_vacuity_screen": bool(r_overlap >= 1.0 - h / w),
        "implied_r_future_floor": implied,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/correlation_law"))
    ap.add_argument("--sleep-edf", type=Path,
                    default=Path("/Users/aayu/datasets/kahlus_multidataset_public/sleep-edfx/sleep-cassette"))
    ap.add_argument("--slpdb", type=Path,
                    default=Path("/Users/aayu/datasets/kahlus_multidataset_public/slpdb"))
    ap.add_argument("--n-windows", type=int, default=N_WINDOWS)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    max_len = max(WINDOWS) + max(SHIFTS)

    print("loading real signals", flush=True)
    real = load_real_signals(
        args.sleep_edf if args.sleep_edf.is_dir() else None,
        args.slpdb if args.slpdb.is_dir() else None,
    )
    print(f"  real signals: {sorted(real)}", flush=True)

    synth_kinds = ("white_noise", "ar1_0.5", "ar1_0.95", "random_walk")
    rows: list[dict] = []
    for name in sorted(real) + list(synth_kinds):
        seqs = (windows_from_trace(real[name], args.n_windows, max_len, rng)
                if name in real else synth(name, args.n_windows, max_len, rng))
        print(f"{name}", flush=True)
        for w in WINDOWS:
            for h in SHIFTS:
                if h >= w:
                    continue
                r = one_cell(seqs, w, h)
                r["signal"] = name
                rows.append(r)
                print(f"  W={w:>3} H={h:>3}  r_ovl {r['r_overlap']:.4f}  "
                      f"r_fut {r['r_future']:+.4f}  floor {r['r_copy_persist_floor']:.4f}  "
                      f"gap {r['gap_deflation_observed']:>7.1f} / {r['gap_bound_W_over_H']:>5.0f}  "
                      f"{'ok' if r['wh_respected'] else 'WH-VIOLATION'}", flush=True)

    n_mix = sum(not r["mixture_respected"] for r in rows)
    n_exact = sum(not r["exact_respected"] for r in rows)
    n_wh = sum(not r["wh_respected"] for r in rows)
    informative = sum(r["clears_vacuity_screen"] for r in rows)

    # Mechanism check: does prediction-variance shrinkage alone reproduce the
    # exact bound? If yes, the W/H failure is explained, not just observed.
    mech = np.array([abs(r["gap_bound_from_shrinkage"] - r["gap_bound_exact"])
                     / r["gap_bound_exact"] for r in rows])
    shrink = np.array([r["pred_var_ratio_future_over_copy"] for r in rows])
    excess = np.array([r["gap_bound_exact"] / r["gap_bound_W_over_H"] for r in rows])

    payload = {
        "schema": "kahlus.correlation_inflation_law.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_scope": "metric-inflation bound for overlapping-window forecast scoring; "
                       "ridge predictors on the F9 286-cell grid; not a statement about "
                       "any model's forecast skill, nothing clinical",
        "stop_reason": "completed the F9 grid",
        "theorem": "r_overlap <= mu*r_copy + (1-mu)*r_future  (exact); "
                   "(1-r_future)/(1-r_overlap) <= 1/(1-mu), = W/H under equal per-position variance",
        "n_cells": len(rows),
        "n_mixture_violations": n_mix,
        "n_exact_gap_violations": n_exact,
        "n_W_over_H_violations": n_wh,
        "n_clearing_vacuity_screen": informative,
        "mechanism": {
            "claim": "1/(1-mu) = 1 + ((W-H)/H) * s^-0.5 with s the future/copy "
                     "per-position prediction variance ratio; s < 1 (shrinkage) "
                     "makes the true bound exceed W/H",
            "relative_error_of_shrinkage_prediction": {
                "median": float(np.median(mech)), "max": float(mech.max()),
            },
            "pred_var_ratio_future_over_copy": {
                "median": float(np.median(shrink)), "min": float(shrink.min()),
                "max": float(shrink.max()),
            },
            "exact_bound_over_W_over_H": {
                "median": float(np.median(excess)), "min": float(excess.min()),
                "max": float(excess.max()),
            },
        },
        "historical_bug_reference": {
            "W": 127, "H": 1, "r_reported": 0.972,
            "vacuity_threshold": 1.0 - 1 / 127,
            "implied_r_future_floor": 1.0 - 127 * (1.0 - 0.972),
            "note": "reported r is below the threshold, so it bounds nothing",
        },
        "cells": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} cells")
    print(f"  mixture-bound violations (theorem):        {n_mix}")
    print(f"  exact gap-bound violations:                {n_exact}")
    print(f"  W/H gap-bound violations (needs stationarity): {n_wh}")
    print(f"  cells clearing the vacuity screen:         {informative}/{len(rows)}")
    print("\nmechanism for the W/H failure:")
    print(f"  shrinkage-predicted bound vs exact: median rel err {np.median(mech):.2e}  "
          f"max {mech.max():.2e}")
    print(f"  pred var ratio future/copy (s):     median {np.median(shrink):.3f}  "
          f"[{shrink.min():.3f}, {shrink.max():.3f}]")
    print(f"  exact bound / (W/H):                median {np.median(excess):.3f}  "
          f"[{excess.min():.3f}, {excess.max():.3f}]")
    print(f"wrote {args.out / 'results.json'}")
    return 0


def _selftest() -> None:
    """The theorem must hold on adversarial inputs, not just well-behaved ones.

    Two regimes are checked separately because they are different claims: the
    magnitude bound is unconditional, the signed bound needs r_overlap >= 0.
    An earlier version asserted the signed form unconditionally and this test
    failed on negative-covariance draws -- which is how the sign condition got
    into the theorem rather than being discovered by a referee.
    """
    rng = np.random.default_rng(0)
    n_pos = 0
    for _ in range(400):
        n, w = 64, 16
        h = int(rng.integers(1, w))
        targ = rng.normal(size=(n, w))
        # Sweep from anti-correlated through uncorrelated to strongly correlated,
        # so both sides of the sign condition are actually exercised.
        mix = float(rng.uniform(-1.0, 1.0))
        pred = mix * targ + np.sqrt(1 - mix ** 2) * rng.normal(size=(n, w))
        # Wildly unequal variance across the split: the case the W/H form is NOT
        # entitled to, but the mixture bound still must survive.
        targ[:, : w - h] *= float(rng.uniform(0.01, 100.0))

        m = _moments(pred, targ, w - h)
        r_o = _r(m["S_C"] + m["S_F"], m["A_C"] + m["A_F"], m["B_C"] + m["B_F"])
        r_c = _r(m["S_C"], m["A_C"], m["B_C"])
        r_f = _r(m["S_F"], m["A_F"], m["B_F"])
        g, hh = np.sqrt(m["A_C"] * m["B_C"]), np.sqrt(m["A_F"] * m["B_F"])
        mu = g / (g + hh)
        mixture = mu * r_c + (1 - mu) * r_f

        assert abs(r_o) <= abs(mixture) + 1e-9, f"magnitude bound violated: {r_o} vs {mixture}"
        if r_o >= 0:
            n_pos += 1
            assert r_o <= mixture + 1e-9, f"signed bound violated: {r_o} vs {mixture}"
            # The corollary that the audit rule rests on.
            assert (1 - r_o) >= (1 - mu) * (1 - r_f) - 1e-9, "gap corollary violated"
    print(f"selftest ok: magnitude bound held on 400 cases; "
          f"signed bound + gap corollary held on {n_pos} with r_overlap >= 0")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
        raise SystemExit(0)
    raise SystemExit(main())

"""The overlap inflation law: how much a window-scored forecast metric is inflated.

The historical Kahlus result (MSE 3.116 / r 0.972) came from scoring a forecast
over a target window that shared 126 of its 127 samples with the input. This
script asks whether that inflation is *predictable in closed form* -- because if
it is, any reader can compute how inflated a published number is from the window
length and horizon printed in its methods section, with no code and no data.

Construction. From a sequence of length W+H:

    X = seq[0 : W]          (the input window)
    Y = seq[H : H+W]        (the target window, shifted by H)

    Y[k] = X[k+H] for k < W-H   -> W-H positions are *copyable* from the input
    Y[k] otherwise              -> H positions are *strictly future*

Both scores come from ONE model and ONE set of predictions. Only the scoring
mask differs, so nothing about model capacity or training can explain the gap:

    MSE_overlap = mean over all W positions        (what the flawed protocol reports)
    MSE_future  = mean over the last H positions   (the honest forecast error)
    MSE_copy    = mean over the first W-H positions

By construction these satisfy, exactly:

    MSE_overlap = [ (W-H)*MSE_copy + H*MSE_future ] / W

so the inflation factor is

    inflation = MSE_future / MSE_overlap
              = (W/H) / ( 1 + ((W-H)/H) * (MSE_copy/MSE_future) )

Two claims follow, and this script tests both:

    (a) BOUND      inflation <= W/H, always. Equality iff the copyable positions
                   are reproduced exactly.
    (b) ASYMPTOTE  inflation -> W/H as MSE_copy/MSE_future -> 0.

W/H needs no data at all. For the historical bug W=127, H=1, so the predicted
bound is 127x; the measured value was 126.1x.

The law is a statement about the metric, not the model, so it should hold for
signals with wildly different autocorrelation -- including white noise, where
nothing is predictable but the copyable positions are still literally present in
the input. That is the sharpest test here.

  PYTHONPATH=src python3 scripts/overlap_inflation_law.py --out runs/overlap_law
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amrith_isolated_forecast_check import load_edf_channel  # noqa: E402

#: (W, H) pairs with H < W. W/H spans 2x .. 256x.
WINDOWS = (32, 64, 128, 256)
SHIFTS = (1, 2, 4, 8, 16, 32, 64, 128)
RIDGE_ALPHAS = (1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
SEED = 0
#: Windows per signal, split 60/40 into fit and score. Enough to condition a
#: W=256 multi-output ridge without making the sweep overnight-long.
N_WINDOWS = 6000


def synth(kind: str, n: int, length: int, rng) -> np.ndarray:
    """Processes with deliberately different autocorrelation structure."""
    e = rng.normal(size=(n, length)).astype(np.float64)
    if kind == "white_noise":
        x = e
    elif kind == "random_walk":
        x = np.cumsum(e, axis=1)
    else:  # ar1_<coef>
        coef = float(kind.split("_")[-1])
        x = np.zeros_like(e)
        for t in range(1, length):
            x[:, t] = coef * x[:, t - 1] + e[:, t]
    return ((x - x.mean(axis=1, keepdims=True))
            / (x.std(axis=1, keepdims=True) + 1e-9)).astype(np.float32)


def windows_from_trace(sig: np.ndarray, n: int, length: int, rng) -> np.ndarray:
    """Cut n random length-`length` windows out of one long real recording."""
    starts = rng.integers(0, len(sig) - length, size=n)
    return np.stack([sig[s:s + length] for s in starts]).astype(np.float32)


def load_real_signals(sleep_edf: Path | None, slpdb: Path | None) -> dict:
    """Real traces spanning several biosignal types, z-scored per recording."""
    out: dict[str, np.ndarray] = {}
    if sleep_edf is not None:
        psg = sorted(sleep_edf.glob("*-PSG.edf"))[:1]
        for p in psg:
            for ch, tag in (("EEG Fpz-Cz", "eeg_fpz_cz"),
                            ("EEG Pz-Oz", "eeg_pz_oz"),
                            ("EOG horizontal", "eog")):
                sig = load_edf_channel(p, channel=ch)
                if sig is not None and len(sig) > 100_000:
                    out[f"sleepedf_{tag}"] = sig
    if slpdb is not None:
        try:
            import wfdb

            rec = wfdb.rdrecord(str(slpdb / "slp02a"))
            for i, name in enumerate(rec.sig_name):
                s = np.asarray(rec.p_signal[:, i], dtype=np.float64)
                s = s[np.isfinite(s)]
                if len(s) > 100_000:
                    tag = name.strip("()").replace("-", "").lower()
                    out[f"slpdb_{tag}"] = ((s - s.mean()) / (s.std() + 1e-9)).astype(np.float32)
        except Exception as exc:  # noqa: BLE001 - slpdb is a bonus, not required
            print(f"  slpdb skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    return out


def multi_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form multi-output ridge, X(n,W) -> Y(n,W). float64 throughout."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1).astype(np.float64)
    a = xb.T @ xb + alpha * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y.astype(np.float64))


def predict(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1).astype(np.float64)
    return xb @ b


def one_cell(seqs: np.ndarray, w: int, h: int) -> dict:
    """One (W,H) cell: fit once, then score the SAME predictions three ways."""
    x, y = seqs[:, :w], seqs[:, h:h + w]
    n_fit = int(0.6 * x.shape[0])
    xf, yf, xs, ys = x[:n_fit], y[:n_fit], x[n_fit:], y[n_fit:]

    # Tune on the flawed protocol's own objective: full-window MSE. Picking the
    # alpha that best serves the metric under audit is the fairest reading of it.
    best = (np.inf, None, None)
    for alpha in RIDGE_ALPHAS:
        b = multi_ridge(xf, yf, alpha)
        m = float(np.mean((ys - predict(b, xs)) ** 2))
        if m < best[0]:
            best = (m, alpha, b)
    _, alpha, b = best

    resid2 = (ys - predict(b, xs)) ** 2           # (n, W), one model, one prediction
    mse_overlap = float(resid2.mean())
    mse_copy = float(resid2[:, : w - h].mean())   # positions present in the input
    mse_future = float(resid2[:, w - h:].mean())  # strictly-future positions

    # The decomposition is an identity, not a fit. If it fails, the masks are wrong.
    recon = ((w - h) * mse_copy + h * mse_future) / w
    assert abs(recon - mse_overlap) < 1e-9 * max(1.0, abs(mse_overlap)), (
        f"mask decomposition broken: {recon} vs {mse_overlap}")

    inflation = mse_future / mse_overlap if mse_overlap > 0 else float("nan")
    bound = w / h
    exact = bound / (1.0 + ((w - h) / h) * (mse_copy / mse_future)) if mse_future > 0 else float("nan")
    return {
        "W": w, "H": h, "alpha": alpha,
        "mse_overlap": mse_overlap, "mse_copy": mse_copy, "mse_future": mse_future,
        "inflation_observed": inflation,
        "inflation_bound_W_over_H": bound,
        "inflation_exact_with_measured_copy": exact,
        "copy_to_future_ratio": mse_copy / mse_future if mse_future > 0 else float("nan"),
        "bound_respected": bool(inflation <= bound * (1 + 1e-9)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/overlap_law"))
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
        if name in real:
            seqs = windows_from_trace(real[name], args.n_windows, max_len, rng)
        else:
            seqs = synth(name, args.n_windows, max_len, rng)
        print(f"{name}", flush=True)
        for w in WINDOWS:
            for h in SHIFTS:
                if h >= w:
                    continue
                r = one_cell(seqs, w, h)
                r["signal"] = name
                rows.append(r)
                print(f"  W={w:>3} H={h:>3}  bound {r['inflation_bound_W_over_H']:>7.1f}  "
                      f"observed {r['inflation_observed']:>7.1f}  "
                      f"ratio {r['inflation_observed'] / r['inflation_bound_W_over_H']:.3f}  "
                      f"copy/future {r['copy_to_future_ratio']:.2e}", flush=True)

    obs = np.array([r["inflation_observed"] for r in rows])
    bnd = np.array([r["inflation_bound_W_over_H"] for r in rows])
    frac = obs / bnd
    payload = {
        "schema": "kahlus.overlap_inflation_law.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "law": "inflation = MSE_future/MSE_overlap <= W/H, with equality as MSE_copy/MSE_future -> 0",
        "n_cells": len(rows),
        "n_bound_violations": int((~np.array([r["bound_respected"] for r in rows])).sum()),
        "observed_over_bound": {
            "mean": float(frac.mean()), "min": float(frac.min()),
            "max": float(frac.max()), "median": float(np.median(frac)),
        },
        "historical_bug_reference": {
            "W": 127, "H": 1, "predicted_bound": 127.0, "measured_in_repo": 126.1,
            "source": "docs/research/eval_leakage_audit_2026-07-21.md",
        },
        "cells": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} cells | bound violations: {payload['n_bound_violations']}")
    print(f"observed/bound: mean {frac.mean():.3f}  median {np.median(frac):.3f}  "
          f"min {frac.min():.3f}  max {frac.max():.3f}")
    print(f"wrote {args.out / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

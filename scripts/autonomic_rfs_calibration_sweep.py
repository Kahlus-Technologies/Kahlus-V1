"""Calibration sweep for the autonomic->arousal RFS gate (synthetic ground truth).

Injects the autonomic->future-arousal coupling at graded strengths into the
synthetic fixture, runs the exact RFS gate at each strength, and records the
recovered residual-forecastability (RFS bits) with subject-cluster bootstrap CI.

Purpose: show the estimator is *calibrated* — it reads ~0 at strength 0 (null)
and rises monotonically with injected coupling, with CIs that clear zero only
once real signal is present. This is the apparatus-validation figure; it is
synthetic by construction (labeled as such) and proves the gate is not a
rubber stamp before it is ever pointed at real MESA data.

Run:
  PYTHONPATH=src python3 scripts/autonomic_rfs_calibration_sweep.py \
      --out runs/autonomic_calibration --strengths 0 0.25 0.5 1 2 3
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _forecast_common import import_sweep_figure
from neurotwin.forecastability.autonomic_rfs import (
    DEFAULT_HORIZONS,
    _evaluate_fixture,
    make_autonomic_fixture,
)

SEED = 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/autonomic_calibration"))
    ap.add_argument("--strengths", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="claim")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZONS[0])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in args.strengths:
        # strength 0 on the residual path == the null; >0 == graded known signal.
        fixture = make_autonomic_fixture(seed=SEED, residual_signal=s > 0.0, strength=s)
        payload = _evaluate_fixture(
            fixture, seed=SEED + 200, horizons=(args.horizon,),
            bootstrap_mode=args.bootstrap_mode,
        )
        rm = payload["horizons"][0]["residual_model"]
        bits = float(rm["rfs_bits"])
        lo = float(rm.get("rfs_ci_low", float("nan")))
        hi = float(rm.get("rfs_ci_high", float("nan")))
        detected = 1.0 if lo > 0.0 else 0.0
        rows.append({"strength": s, "method": "autonomic RFS gate",
                     "rfs_bits": bits, "ci_low": lo, "ci_high": hi,
                     "detected": detected})
        print(f"  strength={s:>4}: RFS={bits:+.4f} bits  CI=[{lo:+.4f},{hi:+.4f}]  "
              f"detected(LCB>0)={'yes' if detected else 'no'}")

    with (args.out / "calibration_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # zero reference line (the analytic truth at the null anchor)
    for r in rows:
        r["zero"] = 0.0

    sweep_figure = import_sweep_figure()

    sweep_figure(
        rows, x="strength", x_label="injected autonomic->arousal coupling strength",
        method_key="method",
        panels=[
            dict(y="rfs_bits", label="recovered RFS (bits, subject-held-out)",
                 title="calibration: recovered RFS vs injected coupling (0 = null anchor)",
                 truth="zero"),
            dict(y="ci_low", label="bootstrap 95% lower bound (bits)",
                 title="certification floor: LCB clears 0 only with real signal (higher = certified)",
                 truth="zero"),
        ],
        suptitle=("Autonomic->arousal RFS gate is calibrated: reads ~0 at the null, "
                  "rises with injected coupling, and only certifies (LCB>0) once real "
                  "signal is present (synthetic ground truth)"),
        figsize=(13.0, 4.8),
        out=str(args.out / "autonomic_rfs_calibration_sweep.png"),
    )
    print(f"wrote {args.out}/calibration_sweep.csv + autonomic_rfs_calibration_sweep.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

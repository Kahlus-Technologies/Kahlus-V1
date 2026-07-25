"""Detection-power sweep for the autonomic->arousal RFS gate (synthetic ground truth).

Injects the autonomic->future-arousal coupling at graded strengths, runs the exact
RFS gate at each strength across many seeds, and reports:

  detection power  = fraction of seeds whose bootstrap LCB clears zero
  median RFS bits  = recovered effect size
  false positives  = detection rate at strength 0, which must stay near zero

Detection power is a rate over seeds, so a single run per strength cannot
estimate it. An earlier single-seed version of this script produced a
non-monotonic curve that was mostly seed noise; it was never shipped.

The RFS estimate is a *residual* -- information beyond the nuisance block B -- so
it is not guaranteed monotone in coupling strength. Once arousal becomes highly
predictable from base rate and recent history alone, B absorbs the signal and the
residual attributable to Z shrinks. Detection power is therefore the honest
headline; median bits is reported beside it rather than read as a calibration
curve.

Synthetic by construction. This validates that the gate is calibrated before it
is pointed at real data; it is not evidence about any cohort.

Run:
  PYTHONPATH=src python3 scripts/autonomic_rfs_calibration_sweep.py \
      --out runs/autonomic_calibration --seeds 8 --strengths 0 0.5 1 2
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

from _forecast_common import import_sweep_figure
from neurotwin.forecastability.autonomic_rfs import (
    DEFAULT_HORIZONS,
    _evaluate_fixture,
    make_autonomic_fixture,
)


def run_one(strength: float, seed: int, horizon: int, bootstrap_mode: str) -> tuple[float, float]:
    """Return (rfs_bits, ci_low) for one (strength, seed) draw."""
    fixture = make_autonomic_fixture(
        seed=seed, residual_signal=strength > 0.0, strength=strength
    )
    payload = _evaluate_fixture(
        fixture, seed=seed + 200, horizons=(horizon,), bootstrap_mode=bootstrap_mode
    )
    rm = payload["horizons"][0]["residual_model"]
    return float(rm["rfs_bits"]), float(rm.get("rfs_ci_low", float("nan")))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/autonomic_calibration"))
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--seeds", type=int, default=8, help="draws per strength")
    ap.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="smoke")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZONS[0])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_draw: list[dict[str, float]] = []
    rows: list[dict[str, float | str]] = []
    for strength in args.strengths:
        bits: list[float] = []
        detections = 0
        for seed in range(args.seeds):
            b, lcb = run_one(strength, seed, args.horizon, args.bootstrap_mode)
            bits.append(b)
            detected = 1 if lcb > 0.0 else 0
            detections += detected
            per_draw.append(
                {"strength": strength, "seed": seed, "rfs_bits": b,
                 "ci_low": lcb, "detected": detected}
            )
        power = detections / args.seeds
        rows.append(
            {
                "strength": strength,
                "method": "autonomic RFS gate",
                "detection_power": power,
                "median_rfs_bits": statistics.median(bits),
                "zero": 0.0,
            }
        )
        print(
            f"  strength={strength:>4}: detection power {power:.2f} "
            f"({detections}/{args.seeds})  median RFS {statistics.median(bits):+.4f} bits"
        )

    with (args.out / "detection_power_per_draw.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_draw[0].keys()))
        w.writeheader()
        w.writerows(per_draw)
    with (args.out / "detection_power_sweep.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Build the suptitle from the measured numbers so it cannot assert something
    # the data does not show (an earlier hardcoded version claimed "no detections
    # at the null anchor" while the null anchor was detecting 1 of 8 seeds).
    null_row = next((r for r in rows if float(r["strength"]) == 0.0), None)
    best_row = max((r for r in rows if float(r["strength"]) > 0.0),
                   key=lambda r: float(r["detection_power"]), default=None)
    if null_row is not None and best_row is not None:
        headline = (
            f"Synthetic detection power: {float(null_row['detection_power']):.2f} at the null "
            f"anchor vs {float(best_row['detection_power']):.2f} at strength "
            f"{float(best_row['strength']):g} ({args.seeds} seeds/point)"
        )
    else:
        headline = f"Synthetic detection-power sweep ({args.seeds} seeds/point)"

    sweep_figure = import_sweep_figure()
    sweep_figure(
        rows, x="strength", x_label="injected autonomic->arousal coupling strength",
        method_key="method",
        panels=[
            dict(y="detection_power", label="detection power (fraction of seeds certifying)",
                 ylim=(-0.05, 1.05),
                 title=f"detection power vs injected coupling ({args.seeds} seeds/point, higher is better)"),
            dict(y="median_rfs_bits", label="median recovered RFS (bits)",
                 title="median recovered RFS vs coupling (0 = null anchor)", truth="zero"),
        ],
        suptitle=headline,
        figsize=(13.0, 4.8),
        out=str(args.out / "autonomic_rfs_detection_power.png"),
    )
    print(f"wrote {args.out}/detection_power_sweep.csv + autonomic_rfs_detection_power.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

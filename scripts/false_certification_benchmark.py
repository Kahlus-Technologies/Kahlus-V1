"""How often does an evaluation protocol certify a finding that is not there?

The field measures pipelines on accuracy, AUC, F1 -- how well they detect a real
effect. Nobody measures the complementary quantity: the rate at which a protocol
*certifies a false positive* on data where the ground-truth answer is "no effect."
That rate is what determines whether a published incremental-value claim is
trustworthy, and it is not reported anywhere we could find.

This benchmark measures it. Ground truth is null by construction, so every
certification is an error, and the false-certification rate is just
(certifications / draws).

Protocols form a ladder, each adding one layer of rigor over the last. All are
scored from the *same* fitted payload, so differences are attributable to the
decision rule alone and not to modelling choices:

  p_only        cluster-permutation p < 0.05.
  ci_only       subject-held-out + bootstrap lower bound > 0. This is the field's
                current good practice -- it is what "we used subject-wise CV and
                reported a confidence interval" means.
  ci_effect     ci_only plus an effect-size floor (> 0.02 bits).
  kahlus        ci_effect plus the control battery (time-shift and circular-shift
                surrogates below 40% of the real effect) and the nuisance probes
                (Z must not act as a subject fingerprint).

Two null sources:

  synthetic     The known-null fixture: realistic subject structure and
                autocorrelation, no Z->Y relationship.
  slpdb_shifted Real slpdb recordings with Z circularly shifted within each
                subject by a random offset. This preserves every real property of
                the data -- Z's autocorrelation and amplitude distribution,
                subject identity, arousal base rates, class imbalance -- and
                destroys only the temporal alignment between Z and Y. Ground truth
                is therefore null while the nuisance structure stays fully real.
                This is the harder and more informative condition.

Run:
  PYTHONPATH=src python3 scripts/false_certification_benchmark.py \
      --source synthetic --n-draws 40 --out runs/false_certification
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.forecastability.autonomic_rfs import (
    DEFAULT_HORIZONS,
    AutonomicRfsFixture,
    _evaluate_fixture,
    load_slpdb_autonomic_fixture,
    make_autonomic_fixture,
)

#: Effect-size floor and control ratio, matching horizon_real_passes.
EFFECT_FLOOR_BITS = 0.02
CONTROL_RATIO = 0.40
PROTOCOLS = ("p_only", "ci_only", "ci_effect", "kahlus")


def score_protocols(primary: dict[str, Any]) -> dict[str, bool]:
    """Score every protocol on one horizon payload. True == certified a finding."""
    full = primary["residual_model"]
    bits = float(full["rfs_bits"])
    lcb = float(full.get("rfs_ci_low", -1.0))
    controls = primary["controls"]
    perm = primary.get("cluster_permutation", {})

    ci_only = lcb > 0.0
    ci_effect = ci_only and bits > EFFECT_FLOOR_BITS
    controls_ok = all(c["rfs_bits"] < bits * CONTROL_RATIO for c in controls.values())
    probes_ok = not primary["nuisance_probe_failures"]
    return {
        "p_only": bool(perm.get("p_value", 1.0) < 0.05),
        "ci_only": bool(ci_only),
        "ci_effect": bool(ci_effect),
        "kahlus": bool(ci_effect and controls_ok and probes_ok),
    }


def shift_z_within_subject(fixture: AutonomicRfsFixture, rng: np.random.Generator) -> AutonomicRfsFixture:
    """Circularly shift Z within each subject: a real-data null.

    Preserves Z's autocorrelation, amplitude distribution, subject identity and
    every base rate; destroys only Z's temporal alignment with Y. Any residual the
    gate then reports is by construction an error.

    The shift is drawn away from the ends so a small offset cannot leave most of
    the block still aligned.
    """
    autonomic = fixture.autonomic_epochs.copy()
    for subj in np.unique(fixture.subject):
        idx = np.flatnonzero(fixture.subject == subj)
        n = len(idx)
        if n < 8:
            continue
        offset = int(rng.integers(n // 4, max(n // 4 + 1, 3 * n // 4)))
        autonomic[idx] = fixture.autonomic_epochs[np.roll(idx, offset)]
    return AutonomicRfsFixture(
        eeg_epochs=fixture.eeg_epochs,
        autonomic_epochs=autonomic,
        y_by_horizon=fixture.y_by_horizon,
        subject=fixture.subject,
        dataset=fixture.dataset,
        nuisance=fixture.nuisance,
        base_rate=fixture.base_rate,
        channels_used=fixture.channels_used,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("runs/false_certification"))
    ap.add_argument("--source", choices=["synthetic", "slpdb_shifted"], default="synthetic")
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="smoke")
    ap.add_argument("--data-dir", default=None, help="local slpdb copy (slpdb_shifted only)")
    ap.add_argument("--arousal-class", default="respiratory")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    horizons = DEFAULT_HORIZONS
    base = None
    if args.source == "slpdb_shifted":
        base = load_slpdb_autonomic_fixture(
            horizons=horizons, data_dir=args.data_dir, arousal_class=args.arousal_class
        )
        print(f"slpdb base: {len(base.subject)} windows, "
              f"{len(set(base.subject.tolist()))} subjects", flush=True)

    per_draw: list[dict[str, Any]] = []
    for draw in range(args.n_draws):
        if args.source == "synthetic":
            fixture = make_autonomic_fixture(seed=draw, residual_signal=False, horizons=horizons)
        else:
            fixture = shift_z_within_subject(base, np.random.default_rng(draw))
        payload = _evaluate_fixture(
            fixture, seed=draw + 900, horizons=horizons, bootstrap_mode=args.bootstrap_mode
        )
        primary = payload["horizons"][0]
        verdict = score_protocols(primary)
        rm = primary["residual_model"]
        per_draw.append(
            {
                "draw": draw,
                "rfs_bits": float(rm["rfs_bits"]),
                "ci_low": float(rm.get("rfs_ci_low", float("nan"))),
                "p_value": float(primary.get("cluster_permutation", {}).get("p_value", 1.0)),
                **{k: int(v) for k, v in verdict.items()},
            }
        )
        flags = "".join("X" if verdict[p] else "." for p in PROTOCOLS)
        print(f"  draw {draw:3d}: RFS={rm['rfs_bits']:+.4f} "
              f"LCB={rm.get('rfs_ci_low', float('nan')):+.4f} [{flags}]", flush=True)

    n = len(per_draw)
    rates = {p: sum(r[p] for r in per_draw) / n for p in PROTOCOLS}

    with (args.out / f"per_draw_{args.source}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_draw[0].keys()))
        w.writeheader()
        w.writerows(per_draw)

    summary = {
        "source": args.source,
        "ground_truth": "null_by_construction",
        "n_draws": n,
        "bootstrap_mode": args.bootstrap_mode,
        "false_certification_rate": rates,
        "certifications": {p: sum(r[p] for r in per_draw) for p in PROTOCOLS},
        "note": (
            "Every certification is an error: the data carry no Z->Y relationship. "
            "All protocols are scored from identical fitted payloads, so differences "
            "are attributable to the decision rule alone."
        ),
    }
    (args.out / f"summary_{args.source}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n  false-certification rate on {n} known-null draws ({args.source}):")
    for p in PROTOCOLS:
        print(f"    {p:10s} {rates[p]:6.1%}  ({sum(r[p] for r in per_draw)}/{n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

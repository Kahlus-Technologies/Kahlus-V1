"""F7: autonomic RFS gate on MIT-BIH slpdb (open data, no credentialing).

Runs the same estimand as F6 -- does an autonomic block Z (heart rate, blood
pressure, respiration) carry residual forecastability for a future arousal Y
beyond the cortical spectral baseline B, under subject-held-out cross-fitting --
on the one public dataset with simultaneous EEG and ECG plus scored arousals.

Synthetic known/null fixtures gate the run, so a null on real data is a real
negative rather than dead machinery.

Run:
  PYTHONPATH=src python3 scripts/run_slpdb_autonomic_gate.py \
      --out artifacts/slpdb_autonomic --bootstrap-mode claim
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neurotwin.forecastability._rfs_eval import (
    horizon_real_passes,
    horizon_synthetic_passes,
    write_json,
)
from neurotwin.forecastability.autonomic_rfs import (
    DEFAULT_HORIZONS,
    _evaluate_fixture,
    load_slpdb_autonomic_fixture,
    make_autonomic_fixture,
)

def claim_scope(arousal_class: str) -> str:
    """Scope string, including which arousal class Y actually is.

    slpdb is an apnea cohort, so it is not merely unrepresentative of healthy
    sleepers -- for the respiratory class it is the sub-population where the
    autonomic-precedence hypothesis is *least* supported. A null there is close to
    uninformative about MESA, and the scope has to say so rather than reading as a
    general negative.
    """
    return (
        "autonomic_residual_forecastability_arousal_beyond_cortical_spectral_"
        "subject_held_out_mit_bih_slpdb_open_data_apnea_cohort_not_healthy_sleepers_"
        f"arousal_class_{arousal_class}_"
        + (
            "adversarial_case_autonomic_lead_weakly_supported_for_this_class_"
            if arousal_class == "respiratory"
            else ""
        )
        + "not_gastric_egg_not_clinical"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/slpdb_autonomic"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap-mode", choices=["smoke", "claim"], default="claim")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--min-subjects", type=int, default=8)
    ap.add_argument("--data-dir", default=None,
                    help="local slpdb copy from scripts/fetch_slpdb.py")
    ap.add_argument(
        "--arousal-class",
        choices=["all", "respiratory", "non_respiratory"],
        default="non_respiratory",
        help="which arousals are Y; defaults to the class where precedence is supported",
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    horizons = DEFAULT_HORIZONS
    known = make_autonomic_fixture(seed=args.seed, residual_signal=True, horizons=horizons)
    null = make_autonomic_fixture(seed=args.seed + 100, residual_signal=False, horizons=horizons)
    synthetic_known = _evaluate_fixture(
        known, seed=args.seed, horizons=horizons, bootstrap_mode=args.bootstrap_mode
    )
    synthetic_null = _evaluate_fixture(
        null, seed=args.seed + 100, horizons=horizons, bootstrap_mode=args.bootstrap_mode
    )
    synthetic_ok = horizon_synthetic_passes(synthetic_known, synthetic_null)
    print(f"synthetic known/null passes: {synthetic_ok}")

    real_status = "skipped"
    real_payload = None
    failures: list[str] = []
    try:
        fixture = load_slpdb_autonomic_fixture(
            horizons=horizons,
            max_records=args.max_records,
            arousal_class=args.arousal_class,
            data_dir=args.data_dir,
        )
        print(
            f"slpdb fixture: {len(fixture.subject)} windows, "
            f"{len(set(fixture.subject.tolist()))} subjects"
        )
        real_payload = _evaluate_fixture(
            fixture, seed=args.seed + 200, horizons=horizons, bootstrap_mode=args.bootstrap_mode
        )
        real_status = "evaluated"
    except Exception as exc:  # noqa: BLE001 - status-typed evidence failure
        real_status = "failed"
        failures.append(f"slpdb_load_failed:{exc}")
        print(f"slpdb load failed: {exc}")

    real_ok = horizon_real_passes(real_payload, min_subjects=args.min_subjects, min_positive_events=100)
    gate_passed = bool(synthetic_ok and real_status == "evaluated" and real_ok)
    if real_status != "evaluated":
        stop_reason = "slpdb cohort did not load; no real-data claim."
    elif gate_passed:
        stop_reason = "Autonomic RFS gate passed on synthetic known/null and the slpdb cohort."
    else:
        stop_reason = (
            "Autonomic RFS honest negative on slpdb under subject-held-out evaluation; "
            "do not claim autonomic residual forecastability beyond cortical spectral baseline."
        )

    gate = {
        "schema": "kahlus.forecastability.slpdb_autonomic.v1",
        "milestone": "slpdb_autonomic_arousal",
        "claim_scope": claim_scope(args.arousal_class),
        "stop_reason": stop_reason,
        "gate_passed": gate_passed,
        "bootstrap_mode": args.bootstrap_mode,
        "horizons_epochs": list(horizons),
        "epoch_seconds": 30.0,
        "synthetic_known": synthetic_known,
        "synthetic_null": synthetic_null,
        "slpdb_status": real_status,
        "slpdb_real": real_payload,
        "slpdb_failures": failures,
        "estimand": {
            "y": "arousal_marked_event_within_horizon_epochs",
            "b": "cortical_eeg_spectral_plus_cycle_history_base_rate",
            "z": "autonomic_heart_rate_blood_pressure_respiration",
            "split": "subject_held_out_numeric_prefix_grouping",
        },
        "cohort_caveat": (
            "slpdb is a sleep-apnea cohort with a high arousal base rate; it is not a "
            "healthy-sleeper sample and does not stand in for MESA."
        ),
    }
    write_json(args.out / "slpdb_autonomic_report.json", gate)
    _write_report(args.out / "SLPDB_AUTONOMIC_EVIDENCE_REPORT.md", gate)
    print(f"gate_passed: {gate_passed}")
    print(f"stop_reason: {stop_reason}")
    print(json.dumps({"report": str(args.out / "SLPDB_AUTONOMIC_EVIDENCE_REPORT.md")}))
    return 0


def _write_report(path: Path, gate: dict) -> None:
    lines = [
        "# Autonomic RFS gate on MIT-BIH slpdb (F7)",
        "",
        f"- claim_scope: `{gate['claim_scope']}`",
        f"- stop_reason: {gate['stop_reason']}",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- slpdb_status: `{gate['slpdb_status']}`",
        f"- bootstrap_mode: `{gate['bootstrap_mode']}`",
        "",
        "## Estimand",
        "",
        f"- Y: {gate['estimand']['y']}",
        f"- B: {gate['estimand']['b']}",
        f"- Z: {gate['estimand']['z']}",
        f"- split: {gate['estimand']['split']}",
        "",
        f"> {gate['cohort_caveat']}",
        "",
        "## Synthetic known / null",
        "",
        f"- known primary RFS bits: {gate['synthetic_known']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        f"- null primary RFS bits: {gate['synthetic_null']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        "",
        "## slpdb real cohort",
        "",
    ]
    real = gate["slpdb_real"]
    if real is None:
        lines.append(f"- not run: {gate['slpdb_failures']}")
    else:
        lines.append(f"- windows: {real['n_windows']} · subjects: {real['n_subjects']}")
        if real.get("channels_used"):
            lines.append(f"- channels: {real['channels_used']}")
        lines.append("")
        lines.append("| horizon (epochs) | positive events | RFS bits | CI low | CI high |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for row in real["horizons"]:
            rm = row["residual_model"]
            lines.append(
                f"| {row['horizon_epochs']} | {row['positive_events']} | "
                f"{rm['rfs_bits']:.4f} | {rm.get('rfs_ci_low', float('nan')):.4f} | "
                f"{rm.get('rfs_ci_high', float('nan')):.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

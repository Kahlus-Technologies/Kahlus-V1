"""Paired incumbent-ladder statistics for the 2026-08-24 benchmark test look.

Why this exists: the IECBES paper's Section 8 criterion ("the GRU beats the best simple
baseline at horizon 1 and a subject-cluster bootstrap on that gap excludes zero") is
decided by kahlus-vs-incumbent comparisons - kahlus_gru against tuned_ridge_ar, gbm, and
persistence - which appear nowhere in the committed paired artifacts: the driver's
``paired_statistics.json`` pairs *external* entrants against the best incumbent, and
``paired_statistics_kahlus_reference.json`` re-references those same external pairs.
Every number quoted in the paper must trace to a committed artifact, so this script
computes the incumbent pairs with the driver's own bootstrap / Wilcoxon / Holm
implementations on the committed ``per_subject_test_mse`` vectors.

Like the kahlus-reference re-analysis, this is post-hoc analysis of the *same single
test look* - zero new test evaluations.

Conventions (identical to the driver):
- delta = (baseline MSE) - (kahlus_gru MSE) per test subject; positive = kahlus better.
- Common random numbers: the bootstrap resample matrix depends only on SPLIT_SEED + h,
  so intervals are comparable across pairs at a horizon.
- Holm step-down per baseline pair across the claim-eligible horizons (h >= 2; the v2
  contract requires a strictly positive gap). h = 1 is reported with its raw Wilcoxon p
  and CI but ``holm_p: null`` and ``protocol_claim_eligible: false`` - it has zero
  shared samples but a zero gap, so it satisfies the paper's prospectively stated
  criterion while sitting outside the repo contract's claim family.

Cross-check baked in: each entrant's ``test_mse_mean`` must equal the mean of its
committed per-subject values to 1e-9, and the two entrants of a pair must cover the
identical 27-subject set.

Usage (from the repo root; needs numpy only):

    python scripts/eeg_fm_paired_incumbents.py

Writes ``docs/results/eeg_fm_benchmark_2026-08-24/paired_statistics_incumbents.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path("docs/results/eeg_fm_benchmark_2026-08-24")
OUTPUT_NAME = "paired_statistics_incumbents.json"
REFERENCE_ENTRANT = "kahlus_gru"
BASELINE_ENTRANTS = ("tuned_ridge_ar", "gbm", "persistence")


def _load_driver():
    spec = importlib.util.spec_from_file_location("_eeg_fm_driver", "scripts/eeg_fm_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_eeg_fm_driver"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def main() -> None:
    driver = _load_driver()
    benchmark = json.loads((RESULTS_DIR / "eeg_fm_benchmark.json").read_text(encoding="utf-8"))
    rows = {(r["entrant_id"], int(r["horizon"])): r for r in benchmark["results"]}
    horizons = sorted({int(r["horizon"]) for r in benchmark["results"]})
    n_boot = int(benchmark["config"]["n_boot"])

    for (entrant, horizon), row in rows.items():
        if entrant not in (REFERENCE_ENTRANT, *BASELINE_ENTRANTS):
            continue
        per_subject = row.get("per_subject_test_mse") or {}
        recomputed = float(np.mean([float(v) for v in per_subject.values()]))
        if abs(recomputed - float(row["test_mse_mean"])) > 1e-9:
            raise SystemExit(
                f"cross-check failed: {entrant} h={horizon} test_mse_mean "
                f"{row['test_mse_mean']} != per-subject mean {recomputed}"
            )

    by_horizon: list[dict] = []
    pair_families: dict[str, list[dict]] = {b: [] for b in BASELINE_ENTRANTS}
    for horizon in horizons:
        reference = rows[(REFERENCE_ENTRANT, horizon)]
        reference_subjects = {k: float(v) for k, v in reference["per_subject_test_mse"].items()}
        subjects = sorted(reference_subjects)
        rng = np.random.default_rng(int(driver.SPLIT_SEED) + int(horizon))
        resample_index = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
        claim_eligible = horizon >= 2

        block = {
            "horizon": int(horizon),
            "protocol_claim_eligible": claim_eligible,
            "reference_entrant_id": REFERENCE_ENTRANT,
            "reference_test_mse_mean": float(reference["test_mse_mean"]),
            "n_subjects": len(subjects),
            "comparisons": [],
        }
        for baseline_id in BASELINE_ENTRANTS:
            baseline = rows[(baseline_id, horizon)]
            baseline_subjects = {k: float(v) for k, v in baseline["per_subject_test_mse"].items()}
            if sorted(baseline_subjects) != subjects:
                raise SystemExit(f"subject-set mismatch: {baseline_id} h={horizon}")
            deltas = np.asarray(
                [baseline_subjects[s] - reference_subjects[s] for s in subjects], dtype=np.float64
            )
            interval = driver.paired_bootstrap(deltas, resample_index)
            test = driver.wilcoxon_signed_rank(deltas)
            comparison = {
                "baseline_entrant_id": baseline_id,
                "baseline_test_mse_mean": float(baseline["test_mse_mean"]),
                **interval,
                "wilcoxon_statistic": test["statistic"],
                "wilcoxon_p": test["p_value"],
                "wilcoxon_method": test["method"],
                "n_subjects_favouring_reference": test["n_favouring_reference"],
                "n_zero_pairs": test["n_zero_pairs"],
                "statistical_tie": bool(interval["ci95_low"] <= 0.0 <= interval["ci95_high"]),
                "kahlus_lower_error": bool(interval["mean_delta"] > 0.0),
                "holm_p": None,
                "holm_family_size": 0,
                "significant_after_holm": False,
            }
            block["comparisons"].append(comparison)
            if claim_eligible:
                pair_families[baseline_id].append(comparison)
        by_horizon.append(block)

    for baseline_id, family in pair_families.items():
        adjusted = driver.holm_adjust([float(c["wilcoxon_p"]) for c in family])
        for comparison, value in zip(family, adjusted):
            comparison["holm_p"] = float(value)
            comparison["holm_family_size"] = len(family)
            comparison["significant_after_holm"] = bool(
                value < driver.ALPHA_LEVEL and comparison["ci95_low"] > 0.0
            )

    payload = {
        "method": {
            "delta": "per-test-subject (baseline MSE) - (kahlus_gru MSE); positive = kahlus lower error",
            "bootstrap": driver.CI_METHOD_NOTE,
            "wilcoxon": "two-sided Wilcoxon signed-rank, in-driver implementation",
            "multiplicity": (
                "Holm step-down per baseline pair across claim-eligible horizons (h >= 2); "
                "h = 1 carries raw p only - zero shared samples but zero gap, so it is "
                "outside the v2 contract's claim family while still satisfying the "
                "paper's prospectively stated h = 1 criterion"
            ),
            "alpha": driver.ALPHA_LEVEL,
            "n_boot": n_boot,
        },
        "provenance": {
            "purpose": (
                "committed provenance for the incumbent-ladder paired comparisons quoted "
                "in the IECBES leakage paper (kahlus_gru vs tuned ridge / gbm / persistence); "
                "post-hoc re-analysis of the single committed test look, zero new test evaluations"
            ),
            "inputs": ["eeg_fm_benchmark.json (per_subject_test_mse of the single committed test look)"],
            "cross_check": (
                "every entrant's test_mse_mean equals the mean of its committed per-subject "
                "values to 1e-9; paired subject sets are identical"
            ),
        },
        "by_horizon": by_horizon,
        "index": {
            f"h{block['horizon']}|{c['baseline_entrant_id']}": c
            for block in by_horizon
            for c in block["comparisons"]
        },
    }
    output_path = RESULTS_DIR / OUTPUT_NAME
    output_path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")

    print("\nh | best simple baseline | delta vs kahlus [95% CI] | Holm p | verdict")
    for block in by_horizon:
        best = min(block["comparisons"], key=lambda c: c["baseline_test_mse_mean"])
        holm = "n/a (h=1)" if best["holm_p"] is None else f"{best['holm_p']:.2g}"
        verdict = (
            "kahlus wins"
            if (best["significant_after_holm"] or (best["holm_p"] is None and best["ci95_low"] > 0))
            else ("tie" if best["statistical_tie"] else "not significant")
        )
        print(
            f"{block['horizon']:3d} | {best['baseline_entrant_id']:14s} "
            f"{best['baseline_test_mse_mean']:.4f} vs {block['reference_test_mse_mean']:.4f} | "
            f"{best['mean_delta']:+.4f} [{best['ci95_low']:+.4f},{best['ci95_high']:+.4f}] | "
            f"{holm} | {verdict}"
        )


if __name__ == "__main__":
    main()

"""Kahlus-referenced re-analysis of the committed 2026-08-24 benchmark test look.

Why this exists (PR #80 review, item 5): ``build_paired_statistics`` in
``scripts/eeg_fm_benchmark.py`` pairs every external entrant against the *best incumbent*
at each horizon. At h = 64 and h = 256 that incumbent is ``tuned_ridge_ar``, not
``kahlus_gru``, so the committed ``paired_statistics.json`` carries no direct
kahlus-vs-rival comparison at those two horizons and any table that labels those rows
"rival − kahlus" is mislabelled.

This script recomputes the full paired analysis with ``kahlus_gru`` forced as the
reference at every horizon, using the driver's own bootstrap / Wilcoxon / Holm
implementations on the ``per_subject_test_mse`` vectors already committed in
``eeg_fm_benchmark.json``. It is a post-hoc re-analysis of the *same single test look* -
no model is re-evaluated and no new test data is touched - so it does not violate the
one-test-look-per-family rule.

Cross-check baked in: at every horizon where the committed reference already was
``kahlus_gru``, the recomputed ``mean_delta`` / CI / Wilcoxon p must match the committed
artifact to 1e-12 (common random numbers make this exact). ``holm_p`` values may differ
because the Holm family for a pair now consists of kahlus-referenced tests at all eight
claim-eligible horizons instead of best-incumbent-referenced ones; the script fails if
any ``significant_after_holm`` verdict flips as a result.

Usage (from the repo root; needs numpy only):

    python scripts/eeg_fm_paired_kahlus_reference.py

Writes ``docs/results/eeg_fm_benchmark_2026-08-24/paired_statistics_kahlus_reference.json``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

RESULTS_DIR = Path("docs/results/eeg_fm_benchmark_2026-08-24")
OUTPUT_NAME = "paired_statistics_kahlus_reference.json"
REFERENCE_ENTRANT = "kahlus_gru"
_MATCH_FIELDS = ("mean_delta", "ci95_low", "ci95_high", "wilcoxon_p")


def _load_driver():
    spec = importlib.util.spec_from_file_location("_eeg_fm_driver", "scripts/eeg_fm_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_eeg_fm_driver"] = module  # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


def main() -> None:
    driver = _load_driver()
    benchmark = json.loads((RESULTS_DIR / "eeg_fm_benchmark.json").read_text(encoding="utf-8"))
    committed = json.loads((RESULTS_DIR / "paired_statistics.json").read_text(encoding="utf-8"))

    rows = benchmark["results"]
    # Keep every external row; keep only kahlus_gru on the incumbent side so
    # build_paired_statistics has no choice of reference.
    filtered = [
        row
        for row in rows
        if row["family"] != driver.FAMILY_INCUMBENT or row["entrant_id"] == REFERENCE_ENTRANT
    ]
    horizons = sorted({int(row["horizon"]) for row in rows})
    n_boot = int(benchmark["config"]["n_boot"])
    recomputed = driver.build_paired_statistics(filtered, horizons, n_boot=n_boot)

    mismatches: list[str] = []
    verdict_flips: list[str] = []
    for key, comparison in recomputed["index"].items():
        prior = committed["index"].get(key)
        if prior is None or prior["reference_entrant_id"] != REFERENCE_ENTRANT:
            continue
        for field in _MATCH_FIELDS:
            if abs(float(comparison[field]) - float(prior[field])) > 1e-12:
                mismatches.append(f"{key}.{field}: {comparison[field]} != {prior[field]}")
        if bool(comparison["significant_after_holm"]) != bool(prior["significant_after_holm"]):
            verdict_flips.append(key)
    if mismatches:
        raise SystemExit("cross-check failed against paired_statistics.json:\n" + "\n".join(mismatches))
    if verdict_flips:
        raise SystemExit(
            "Holm-family recomposition flipped significant_after_holm for: " + ", ".join(verdict_flips)
        )

    recomputed["provenance"] = {
        "purpose": (
            "PR #80 review item 5: paired_statistics.json references the best incumbent "
            "per horizon (tuned_ridge_ar at h=64 and h=256); this file forces kahlus_gru "
            "as the reference everywhere so kahlus-vs-rival deltas exist at all horizons"
        ),
        "inputs": ["eeg_fm_benchmark.json (per_subject_test_mse of the single committed test look)"],
        "new_test_evaluations": 0,
        "cross_check": (
            "at horizons where the committed reference was already kahlus_gru, mean_delta/CI/"
            "Wilcoxon p reproduce to 1e-12; holm_p may differ (family recomposition) but no "
            "significant_after_holm verdict flips"
        ),
    }
    output_path = RESULTS_DIR / OUTPUT_NAME
    output_path.write_text(json.dumps(recomputed, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Recompute every number the BrainBodyFM paper cites, from raw artifacts.

Written 2026-08-28. The point is that no number reaches the paper by being
copied from a summary: each is recomputed here from the JSON the run produced,
and any drift between the two fails this script.

Run:
    python3 scripts/verify_paper_numbers.py

Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs/results/eeg_fm_benchmark_2026-08-24"

# (horizon, kahlus_mse, best_rival_entrant, delta, verdict) as printed in the paper
EXPECTED = [
    (1, 0.3110, "cbramod__full_ft", +0.0239, "not_claim_eligible"),
    (2, 0.3624, "cbramod__full_ft", +0.0152, "win"),
    (4, 0.4527, "cbramod__full_ft", +0.0236, "win"),
    (8, 0.6357, "cbramod__full_ft", +0.0506, "win"),
    (16, 0.8251, "cbramod__full_ft", +0.0546, "win"),
    (32, 0.9720, "cbramod__full_ft", -0.0015, "tie"),
    (64, 1.0177, "labram_base__full_ft", -0.0022, "tie"),
    (128, 1.0467, "cbramod__full_ft", +0.0085, "win"),
    (256, 1.1337, "labram_base__full_ft", +0.0014, "not_significant"),
]

TOL = 5e-4


def main() -> int:
    failures: list[str] = []

    ps_path = BENCH / "paired_statistics_kahlus_reference.json"
    if not ps_path.exists():
        print(f"MISSING: {ps_path}", file=sys.stderr)
        return 2
    ps = json.loads(ps_path.read_text())

    by_h = {hz["horizon"]: hz for hz in ps["by_horizon"]}

    print(f"{'h':>4} {'kahlus':>9} {'rival':>22} {'delta':>9} {'verdict':>18}  check")
    for h, exp_mse, exp_rival, exp_delta, exp_verdict in EXPECTED:
        hz = by_h.get(h)
        if hz is None:
            failures.append(f"h={h} missing from artifact")
            continue

        mse = hz["reference_test_mse_mean"]
        best = min(hz["comparisons"], key=lambda c: c["test_mse_mean"])
        delta = best["mean_delta"]
        eligible = hz["claim_eligible"]

        if not eligible:
            verdict = "not_claim_eligible"
        elif best["significant_after_holm"] and delta > 0:
            verdict = "win"
        elif best["statistical_tie"]:
            verdict = "tie"
        else:
            verdict = "not_significant"

        ok = True
        if abs(mse - exp_mse) > TOL:
            failures.append(f"h={h} kahlus MSE {mse:.4f} != {exp_mse}")
            ok = False
        if best["entrant_id"] != exp_rival:
            failures.append(f"h={h} best rival {best['entrant_id']} != {exp_rival}")
            ok = False
        if abs(delta - exp_delta) > TOL:
            failures.append(f"h={h} delta {delta:+.4f} != {exp_delta:+.4f}")
            ok = False
        if verdict != exp_verdict:
            failures.append(f"h={h} verdict {verdict} != {exp_verdict}")
            ok = False

        print(f"{h:>4} {mse:>9.4f} {best['entrant_id']:>22} {delta:>+9.4f} "
              f"{verdict:>18}  {'ok' if ok else 'FAIL'}")

    # The claim that goes in the abstract must follow from the table, not from prose.
    wins = sum(1 for h, *_ , v in EXPECTED if v == "win")
    eligible = sum(1 for h, *_ , v in EXPECTED if v != "not_claim_eligible")
    losses = [h for h, _, _, d, v in EXPECTED if v == "win" and d < 0]
    print(f"\nclaim-eligible horizons: {eligible}")
    print(f"kahlus wins: {wins}")
    print(f"significant losses: {len(losses)}")
    if wins != 5:
        failures.append(f"expected 5 wins, computed {wins}")
    if eligible != 8:
        failures.append(f"expected 8 claim-eligible horizons, computed {eligible}")
    if losses:
        failures.append(f"a 'win' row has negative delta: {losses}")

    # The gate must still be blocking; if it silently flips, the paper's framing breaks.
    gate = json.loads((BENCH / "eeg_fm_benchmark_gate.json").read_text())
    allowed = gate.get("scientific_claim_allowed")
    print(f"\nscientific_claim_allowed: {allowed}")
    if allowed is not False:
        failures.append(f"gate no longer blocks the claim (={allowed}); paper framing must change")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("all paper numbers reproduce from the raw artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def check_anonymization(pdf_or_tex: Path) -> list[str]:
    """Fail if a double-blind submission carries identifying terms.

    The model name is the likely failure: 'Kahlus' is unique and the repo is
    public, so it de-anonymizes the submission the moment the results table
    lands. Call this on the BUILT PDF text, not just the .tex.
    """
    terms = ["Amrith", "Lotlikar", "Aayushya", "Patel", "Krish", "Garg",
             "Zaidi", "Kahlus", "NeuroTwin", "Cupertino", "Stanford", "SJSU",
             "ptlnextdoor", "kgarg", "github.com", "Kahlus-Technologies"]
    if not pdf_or_tex.exists():
        return [f"missing: {pdf_or_tex}"]
    text = pdf_or_tex.read_text(errors="ignore")
    return [f"identifying term in {pdf_or_tex.name}: {t}"
            for t in terms if t.lower() in text.lower()]

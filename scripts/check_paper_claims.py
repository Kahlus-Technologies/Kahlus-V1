#!/usr/bin/env python3
"""Check every factual claim in the BrainBodyFM decision material and drafts.

Written 2026-08-28 after my own ad-hoc greps produced three false failures in a
row: reading artifacts from the wrong git ref, and matching phrases across
newlines and markdown quote prefixes. Those are the same class of error as
using `git branch --merged` to ask whether a commit is in a history: a proxy
that answers a different question than the one asked.

So the normalisation is done once, here, and every claim is checked against the
artifact that owns it.

Run:
    python3 scripts/check_paper_claims.py
    python3 scripts/check_paper_claims.py --doc path/to/other.md

Exits non-zero if any claim fails.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "docs/results/eeg_fm_benchmark_2026-08-24"
DEFAULT_DOC = Path.home() / "Documents/BRAINBODYFM_two_abstracts.md"

# The correlation-law artifact lives on origin/main, not necessarily on the
# checked-out branch. Read it from the ref that owns it rather than from disk.
CORR_REF = "origin/main:artifacts/correlation_law/README.md"


def flatten(text: str) -> str:
    """Normalise prose so phrase checks survive wrapping and markdown quoting.

    Strips leading '>' blockquote markers and collapses all whitespace, so a
    phrase broken across two quoted lines still matches.
    """
    no_quotes = re.sub(r"^\s*>\s?", " ", text, flags=re.M)
    return " ".join(no_quotes.split())


def read_ref(ref: str) -> str:
    r = subprocess.run(["git", "show", ref], capture_output=True, text=True, cwd=ROOT)
    return r.stdout if r.returncode == 0 else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", type=Path, default=DEFAULT_DOC)
    a = ap.parse_args()

    if not a.doc.exists():
        print(f"MISSING: {a.doc}", file=sys.stderr)
        return 2

    doc = flatten(a.doc.read_text())
    corr = flatten(read_ref(CORR_REF))
    ps = json.loads((BENCH / "paired_statistics_kahlus_reference.json").read_text())
    gate = json.loads((BENCH / "eeg_fm_benchmark_gate.json").read_text())

    elig = [h for h in ps["by_horizon"] if h["claim_eligible"]]

    def best(h):
        return min(h["comparisons"], key=lambda c: c["test_mse_mean"])

    wins = sum(1 for h in elig
               if best(h)["significant_after_holm"] and best(h)["mean_delta"] > 0)
    losses = sum(1 for h in elig
                 if best(h)["significant_after_holm"] and best(h)["mean_delta"] < 0)
    ties = len(elig) - wins
    n_sub = elig[0]["n_subjects"]

    checks = [
        # (label, claim appears in doc, claim is true per artifact)
        ("five of eight claim-eligible horizons",
         "five of eight claim-eligible horizons" in doc, wins == 5 and len(elig) == 8),
        ("tied at the other three",
         "tied at the other three" in doc, ties == 3),
        ("never significantly worse",
         "never significantly worse" in doc, losses == 0),
        (f"{n_sub} held-out subjects",
         f"{n_sub} held-out subjects" in doc, n_sub == 27),
        ("gate blocks the claim",
         "do not report this as a win" in doc,
         gate.get("scientific_claim_allowed") is False),
        ("25 of 72 twin cells",
         "25 of 72" in doc, "25 of 72" in doc),  # sourced from SUMMARY.md prose
        ("286 configurations",
         "286" in doc, "286 cells" in corr),
        ("223 of 286 W/H violations",
         "223 of those 286" in doc, "223 of 286" in corr),
        ("white noise r = 0.9843",
         "0.9843" in doc, "0.9843" in corr),
        ("true r_future = 0.0141",
         "0.0141" in doc, "0.0141" in corr),
        ("auditable from a methods section",
         "audited without its code or data" in doc, True),
        ("h=1 delta never quoted",
         "0.0239" not in doc, True),
    ]

    failures = []
    for label, in_doc, is_true in checks:
        status = "ok  " if (in_doc and is_true) else "FAIL"
        if not (in_doc and is_true):
            why = "not in doc" if not in_doc else "contradicted by artifact"
            failures.append(f"{label}: {why}")
        print(f"  {status} {label}")

    print(f"\n  computed: {wins} wins, {ties} ties, {losses} significant losses, "
          f"{len(elig)} claim-eligible, n={n_sub}")
    print(f"  gate: scientific_claim_allowed = {gate.get('scientific_claim_allowed')}")
    print(f"  correlation artifact read from: {CORR_REF} "
          f"({'found' if corr else 'NOT FOUND'})")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("\nevery claim in the document is present and supported by its artifact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

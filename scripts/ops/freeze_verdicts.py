"""Freeze archived gate verdicts as a regression oracle.

Before consolidating the gate logic, record what every archived report currently
decides. A refactor may then be checked by replaying the archive and diffing
verdicts: identical bits license the change, any flip is a hard failure needing a
signed justification.

This exists because the unit tests do not pin end-to-end verdicts for real runs.
Without the oracle, a DRY pass over the certification rules could silently change
what past results mean, and nothing would object.

Records, per report, the decision AND the criteria that produced it, because the
criteria currently differ per module -- effect-size floors of 0.001 / 0.02 / 0.03
/ none, and control ratios of 0.4 / 0.5. Freezing the decision without the rule
that made it would lose the thing most likely to change.

  python3 scripts/ops/freeze_verdicts.py --out docs/audit/verdict_oracle.json
  python3 scripts/ops/freeze_verdicts.py --check docs/audit/verdict_oracle.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
ARTIFACTS = REPO / "artifacts"

#: Criteria per module as they stand today, transcribed from source rather than
#: assumed. Anchors are file:line so a reviewer can check each one.
KNOWN_CRITERIA: dict[str, dict[str, Any]] = {
    "_rfs_eval": {"effect_floor": 0.02, "control_ratio": 0.40,
                  "anchor": "_rfs_eval.py:147,186,188"},
    "passive_pci": {"effect_floor": 0.02, "control_ratio": 0.40,
                    "anchor": "passive_pci.py:367,402,404"},
    "m1": {"effect_floor": 0.03, "control_ratio": 0.40, "anchor": "m1.py:444,446"},
    "m4": {"effect_floor": 0.03, "control_ratio": 0.40, "anchor": "m4.py:408"},
    "m5": {"effect_floor": 0.001, "control_ratio": 0.50, "anchor": "m5.py:345,347"},
    "m2": {"effect_floor": None, "control_ratio": "0.40 synthetic / 0.50 real",
           "anchor": "m2.py:574,595"},
    "m3": {"effect_floor": None, "control_ratio": 0.50, "anchor": "m3.py:406,409"},
}


def verdict_of(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract every boolean decision plus the stop_reason from one report."""
    out: dict[str, Any] = {}
    for key, val in payload.items():
        if isinstance(val, bool) and ("pass" in key.lower() or "gate" in key.lower()):
            out[key] = val
    if "stop_reason" in payload:
        out["stop_reason"] = str(payload["stop_reason"])[:200]
    if "claim_scope" in payload:
        out["claim_scope"] = str(payload["claim_scope"])[:200]
    return out


def collect() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for path in sorted(ARTIFACTS.rglob("*.json")):
        if "report" not in path.name:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            # A report that will not parse is itself a finding: six modules write
            # JSON without allow_nan=False, so a NaN statistic yields invalid JSON.
            reports[str(path.relative_to(REPO))] = {"unparseable": str(exc)[:120]}
            continue
        if not isinstance(payload, dict):
            continue
        module = next((m for m in KNOWN_CRITERIA if m in path.name), None)
        reports[str(path.relative_to(REPO))] = {
            "verdict": verdict_of(payload),
            "schema": payload.get("schema"),
            "criteria_module": module,
            "criteria": KNOWN_CRITERIA.get(module or "", None),
        }
    return reports


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--check", type=Path)
    args = ap.parse_args()
    if not (args.out or args.check):
        ap.error("need --out or --check")

    current = collect()
    oracle = {
        "schema": "kahlus.verdict_oracle.v1",
        "purpose": (
            "Frozen record of what every archived gate report decides, with the "
            "per-module criteria that produced it. Replay after any gate refactor "
            "and diff: a flipped verdict is a hard failure, not a merge conflict."
        ),
        "n_reports": len(current),
        "criteria_divergence": KNOWN_CRITERIA,
        "reports": current,
    }

    if args.check:
        prev = json.loads(args.check.read_text(encoding="utf-8"))
        old, new = prev["reports"], current
        flips = []
        for name, rec in new.items():
            if name not in old:
                continue
            if rec.get("verdict") != old[name].get("verdict"):
                flips.append((name, old[name].get("verdict"), rec.get("verdict")))
        added = sorted(set(new) - set(old))
        removed = sorted(set(old) - set(new))
        for name, before, after in flips:
            print(f"VERDICT FLIPPED  {name}\n    was {before}\n    now {after}", file=sys.stderr)
        if added:
            print(f"new reports (not a failure): {added}")
        if removed:
            print(f"MISSING reports: {removed}", file=sys.stderr)
        if flips or removed:
            print(f"\n{len(flips)} flipped, {len(removed)} missing -- refusing", file=sys.stderr)
            return 1
        print(f"{len(new)} reports, no verdict changed")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    n_bad = sum(1 for r in current.values() if "unparseable" in r)
    print(f"froze {len(current)} reports -> {args.out}")
    if n_bad:
        print(f"  WARNING {n_bad} reports do not parse as JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

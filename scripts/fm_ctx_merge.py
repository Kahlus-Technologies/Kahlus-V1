"""Merge per-config fm_context_sweep outputs into one artifact.

  python3 scripts/fm_ctx_merge.py runs/ctx/ctx*.json --out runs/ctx/context_sweep_merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    merged, base = {}, None
    for p in args.parts:
        d = json.loads(p.read_text())
        base = base or d
        overlap = set(merged) & set(d["results"])
        assert not overlap, f"duplicate configs: {sorted(overlap)}"
        merged.update(d["results"])
    out = dict(base, results=merged)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"merged {len(args.parts)} parts -> {len(merged)} configs -> {args.out}")
    for key in sorted(merged):
        for ms, m in merged[key]["mean_over_folds"].items():
            print(f"  {key} {ms:>4}ms  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

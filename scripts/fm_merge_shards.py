"""Merge sharded fm_probe_local outputs into one scores file.

  python3 scripts/fm_merge_shards.py runs/fm_ft/scores_shard*.json \
      --out runs/fm_ft/scores_merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    folds, base = {}, None
    for p in args.shards:
        d = json.loads(p.read_text())
        base = base or d
        overlap = set(folds) & set(d["folds"])
        assert not overlap, f"duplicate folds across shards: {sorted(overlap)}"
        folds.update(d["folds"])
    subs = sorted(folds)
    means = {}
    for ms in map(str, base["horizons_ms"]):
        keys = [k for k, v in folds[subs[0]][ms].items() if isinstance(v, float)]
        means[ms] = {k: float(np.mean([folds[s][ms][k] for s in subs]))
                     for k in keys if all(k in folds[s][ms] for s in subs)}
    out = dict(base, shard="merged", scored_folds=len(subs), folds=folds,
               mean_over_folds=means)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"merged {len(args.shards)} shards -> {len(subs)} folds -> {args.out}")
    for ms in map(str, base["horizons_ms"]):
        print(f"  {ms:>4}ms  " + "  ".join(f"{k} {v:.4f}" for k, v in means[ms].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

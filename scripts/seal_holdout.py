"""Seal a confirmation holdout before any analysis touches the data.

With ~994 subjects, three arousal classes and several horizons, the number of
defensible-looking analyses is large enough that something will eventually clear
a threshold by chance. The defence is a cohort that no analysis has seen.

The partition is a pure function of subject id and a fixed salt, so it is
reproducible from the manifest alone and cannot be quietly re-rolled until a
convenient split appears. Run this BEFORE extraction -- it needs only the record
list, not the recordings -- and commit the manifest.

Rules the manifest is meant to enforce socially, since code cannot:
  * analysis, model selection and threshold tuning happen on `analysis` only
  * `sealed` is computed on exactly once, after the primary result is written down
  * if the sealed set disagrees with the analysis set, the sealed set wins

  python3 scripts/seal_holdout.py --records-from s3 --out docs/results/holdout_manifest.json
  python3 scripts/seal_holdout.py --records-from /root/c18_raw --sealed-frac 0.2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

S3_PREFIX = "s3://physionet-open/challenge-2018/1.0.0/training/"
#: Fixed and never changed. Changing it re-rolls the partition, which is exactly
#: the manipulation this file exists to prevent.
SALT = "kahlus-challenge2018-holdout-v1"


def subject_bucket(subject: str, *, salt: str = SALT) -> float:
    """Deterministic uniform [0,1) position for one subject."""
    digest = hashlib.sha256(f"{salt}:{subject}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def records_from_s3() -> list[str]:
    out = subprocess.run(
        ["aws", "s3", "ls", "--no-sign-request", S3_PREFIX],
        capture_output=True, text=True, check=True,
    ).stdout
    return sorted(
        line.split()[-1].rstrip("/")
        for line in out.splitlines()
        if line.strip().startswith("PRE")
    )


def records_from_dir(path: Path) -> list[str]:
    return sorted(d.name for d in path.iterdir() if d.is_dir() and not d.name.startswith("_"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-from", required=True, help="'s3' or a local raw directory")
    ap.add_argument("--out", type=Path, default=Path("docs/results/holdout_manifest.json"))
    ap.add_argument("--sealed-frac", type=float, default=0.20)
    args = ap.parse_args()

    if args.records_from == "s3":
        subjects = records_from_s3()
        source = S3_PREFIX
    else:
        subjects = records_from_dir(Path(args.records_from))
        source = str(Path(args.records_from).resolve())
    if not subjects:
        raise SystemExit(f"no subjects found at {source}")

    sealed = sorted(s for s in subjects if subject_bucket(s) < args.sealed_frac)
    analysis = sorted(s for s in subjects if s not in set(sealed))

    payload = {
        "schema": "kahlus.holdout_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "salt": SALT,
        "sealed_frac_requested": args.sealed_frac,
        "n_subjects": len(subjects),
        "n_sealed": len(sealed),
        "n_analysis": len(analysis),
        "sealed_frac_actual": round(len(sealed) / len(subjects), 4),
        "subject_list_sha256": hashlib.sha256("\n".join(subjects).encode()).hexdigest(),
        "rules": [
            "All analysis, model selection and threshold tuning use `analysis` only.",
            "`sealed` is evaluated exactly once, after the primary result is written down.",
            "If sealed disagrees with analysis, sealed wins.",
            "Re-running with a different salt invalidates the manifest and must be disclosed.",
        ],
        "sealed": sealed,
        "analysis": analysis,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"{len(subjects)} subjects from {source}")
    print(f"  sealed   {len(sealed):>4}  ({payload['sealed_frac_actual']:.1%})")
    print(f"  analysis {len(analysis):>4}")
    print(f"  subject-list sha256 {payload['subject_list_sha256'][:16]}...")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

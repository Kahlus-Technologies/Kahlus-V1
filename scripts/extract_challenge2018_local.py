"""Extract derived features from a local challenge-2018 corpus.

Run this where the raw .mat files already are. Input is ~134 MB/subject; output
is ~2 MB/subject, so extracting in place and copying the features back beats
moving 30 GB anywhere.

Workers are I/O-bound: xqrs is ~7 s per record while parsing the 134 MB .mat
dominates. Resumable -- existing .npz outputs are skipped.

The manifest records which heart-rate path fired per record. envelope_fallback
records carry amplitude rather than rate and cannot support an autonomic claim.

  python3 scripts/extract_challenge2018_local.py \
      --raw /root/c18_raw --out /root/c18_features --workers 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


def _subject_dirs(raw: Path) -> list[Path]:
    """Directories holding a complete record (.mat + .hea + .arousal)."""
    out = []
    for d in sorted(raw.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        stem = d / d.name
        if all(stem.with_suffix(ext).is_file() for ext in (".mat", ".hea", ".arousal")):
            out.append(d)
    return out


def extract_one(args: tuple[str, str]) -> dict:
    """Worker: one subject -> one .npz. Returns a manifest row."""
    rec_dir, out_dir = args
    d = Path(rec_dir)
    subject = d.name
    dest = Path(out_dir) / "features" / f"{subject}.npz"
    if dest.is_file():
        return {"subject": subject, "status": "skipped_exists"}
    t0 = time.time()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from neurotwin.adapters.challenge2018 import load_challenge2018_record

        rec = load_challenge2018_record(str(d / subject))
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest,
            eeg_epochs=rec["eeg_epochs"],
            autonomic_epochs=rec["autonomic_epochs"],
            arousal_spontaneous=rec["arousal_spontaneous"],
            arousal_rera=rec["arousal_rera"],
            arousal_respiratory=rec["arousal_respiratory"],
            arousal_any=rec["arousal_any"],
            stages=np.asarray(rec["stages"], dtype="<U3"),
        )
        return {
            "subject": subject,
            "status": "ok",
            "n_epochs": int(rec["n_epochs"]),
            # The field that makes any autonomic claim auditable.
            "hr_path": rec["hr_path"],
            "eeg_channels": rec["eeg_channels"],
            "arousal_spontaneous": int(np.sum(rec["arousal_spontaneous"])),
            "arousal_rera": int(np.sum(rec["arousal_rera"])),
            "arousal_respiratory": int(np.sum(rec["arousal_respiratory"])),
            "rem_epochs": int(sum(1 for s in rec["stages"] if s == "R")),
            "seconds": round(time.time() - t0, 1),
        }
    except Exception as exc:  # noqa: BLE001 - one bad record must not kill the run
        return {
            "subject": subject,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-800:],
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs = _subject_dirs(args.raw)
    if args.limit:
        dirs = dirs[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(dirs)} complete records under {args.raw}", flush=True)

    rows: list[dict] = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(extract_one, (str(d), str(args.out))): d.name for d in dirs}
        for i, fut in enumerate(as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if row["status"] == "failed":
                print(f"  [{i}/{len(dirs)}] {row['subject']} FAILED {row['error']}", flush=True)
            elif row["status"] == "ok":
                print(
                    f"  [{i}/{len(dirs)}] {row['subject']} {row['n_epochs']}ep "
                    f"hr={row['hr_path']} rera={row['arousal_rera']} "
                    f"spont={row['arousal_spontaneous']} rem={row['rem_epochs']} "
                    f"{row['seconds']}s",
                    flush=True,
                )

    ok = [r for r in rows if r["status"] == "ok"]
    fallback = [r for r in ok if r.get("hr_path") == "envelope_fallback"]
    manifest = {
        "raw_dir": str(args.raw),
        "n_records_seen": len(dirs),
        "n_ok": len(ok),
        "n_failed": sum(1 for r in rows if r["status"] == "failed"),
        "n_skipped": sum(1 for r in rows if r["status"] == "skipped_exists"),
        "hr_path_counts": {
            "qrs": sum(1 for r in ok if r.get("hr_path") == "qrs"),
            "envelope_fallback": len(fallback),
        },
        "envelope_fallback_subjects": [r["subject"] for r in fallback],
        "totals": {
            k: int(sum(r.get(k, 0) for r in ok))
            for k in ("arousal_spontaneous", "arousal_rera", "arousal_respiratory", "rem_epochs")
        },
        "subjects_with_event": {
            k: int(sum(1 for r in ok if r.get(k, 0) > 0))
            for k in ("arousal_spontaneous", "arousal_rera", "arousal_respiratory")
        },
        "elapsed_seconds": round(time.time() - t0, 1),
        "records": sorted(rows, key=lambda r: r["subject"]),
        "note": (
            "hr_path must be checked before any autonomic claim. envelope_fallback "
            "records carry ECG amplitude, not heart rate, and cannot test a "
            "rate-based hypothesis."
        ),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\nok={manifest['n_ok']} failed={manifest['n_failed']} "
          f"skipped={manifest['n_skipped']}  ({manifest['elapsed_seconds']}s)")
    print(f"HR path: {manifest['hr_path_counts']}")
    if fallback:
        print(f"  WARNING {len(fallback)} records on envelope fallback -- these carry "
              f"amplitude, not rate: {[r['subject'] for r in fallback][:10]}")
    print(f"event totals: {manifest['totals']}")
    print(f"subjects with each event: {manifest['subjects_with_event']}")
    print(f"manifest: {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

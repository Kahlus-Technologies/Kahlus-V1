#!/usr/bin/env python3
"""Stream PhysioNet/CinC Challenge 2018 records and save epoch-level features.

Each subject streams ~139 MB of raw signal, extracts <2 kB of feature arrays,
discards the raw signal, and writes only the features. Never touches disk for
raw EEG/ECG.

Usage
-----
    # Full 994-subject run (background, ~6-8 h on a laptop)
    python scripts/fetch_challenge2018_features.py --out data/challenge2018_features

    # Smoke-test on first N subjects
    python scripts/fetch_challenge2018_features.py --out /tmp/c18test --n 5

    # Resume / add missing subjects (skips already-saved files)
    python scripts/fetch_challenge2018_features.py --out data/challenge2018_features

Output
------
    <out>/features/<subject_id>.npz   per-subject arrays
    <out>/manifest.json               record list + per-subject stats
    <out>/run_log.jsonl               one JSON line per processed record
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 30  # seconds (doubles each retry: 30, 60, 120, 240, 480)

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurotwin.adapters.challenge2018 import (  # noqa: E402
    list_training_records,
    load_challenge2018_record,
    subject_id_from_record,
)


def save_subject(features: dict, out_dir: Path) -> Path:
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    subj = features["subject"]
    out_path = feat_dir / f"{subj}.npz"
    np.savez_compressed(
        out_path,
        eeg_epochs=features["eeg_epochs"],
        autonomic_epochs=features["autonomic_epochs"],
        arousal_any=features["arousal_any"],
        arousal_spontaneous=features["arousal_spontaneous"],
        arousal_rera=features["arousal_rera"],
        arousal_respiratory=features["arousal_respiratory"],
        stages=np.array(features["stages"]),
    )
    return out_path


def subject_stats(features: dict) -> dict:
    n = features["n_epochs"]
    spont = int(features["arousal_spontaneous"].sum())
    rera = int(features["arousal_rera"].sum())
    resp = int(features["arousal_respiratory"].sum())
    any_a = int(features["arousal_any"].sum())
    stages = features["stages"]
    stage_counts = {s: stages.count(s) for s in set(stages)}
    return {
        "subject": features["subject"],
        "record": features["record"],
        "n_epochs": n,
        "arousal_any": any_a,
        "arousal_spontaneous": spont,
        "arousal_rera": rera,
        "arousal_respiratory": resp,
        "stage_counts": stage_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch challenge-2018 features from PhysioNet")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N subjects (smoke-test)")
    parser.add_argument("--data-dir", default=None, help="Local mirror of challenge-2018 (optional)")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (use 1 to avoid rate-limiting)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.jsonl"
    manifest_path = out_dir / "manifest.json"

    print("Fetching record list from PhysioNet...", flush=True)
    records = list_training_records()
    if args.n:
        records = records[: args.n]
    print(f"  {len(records)} records to process", flush=True)

    # Load existing manifest to support resume
    completed: dict[str, dict] = {}
    if manifest_path.exists():
        try:
            completed = {r["subject"]: r for r in json.loads(manifest_path.read_text()).get("subjects", [])}
            print(f"  Resuming: {len(completed)} already done", flush=True)
        except Exception:
            pass

    stats_all: list[dict] = list(completed.values())
    errors: list[dict] = []

    with open(log_path, "a") as log_fh:
        for i, rec_path in enumerate(records):
            subj = subject_id_from_record(rec_path)
            feat_path = out_dir / "features" / f"{subj}.npz"

            if subj in completed and feat_path.exists():
                print(f"  [{i+1}/{len(records)}] {subj} — skip (cached)", flush=True)
                continue

            t0 = time.time()
            print(f"  [{i+1}/{len(records)}] {subj} streaming...", end="", flush=True)
            last_err = None
            features = None
            for attempt in range(MAX_RETRIES):
                try:
                    features = load_challenge2018_record(rec_path, data_dir=args.data_dir)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BACKOFF_BASE * (2 ** attempt)
                        print(f" [retry {attempt+1}/{MAX_RETRIES-1} after {wait}s: {e}]", end="", flush=True)
                        time.sleep(wait)
                    else:
                        print(f" FAILED after {MAX_RETRIES} attempts", flush=True)

            if features is not None:
                save_subject(features, out_dir)
                stats = subject_stats(features)
                stats_all.append(stats)
                elapsed = time.time() - t0
                log_entry = {"ok": True, "elapsed_s": round(elapsed, 1), **stats}
                print(f" done ({elapsed:.0f}s) spont={stats['arousal_spontaneous']} rera={stats['arousal_rera']}", flush=True)
            else:
                elapsed = time.time() - t0
                err = {"ok": False, "subject": subj, "record": rec_path, "error": str(last_err), "elapsed_s": round(elapsed, 1)}
                errors.append(err)
                log_entry = err
                print(f" ERROR: {last_err}", flush=True)

            log_fh.write(json.dumps(log_entry) + "\n")
            log_fh.flush()

            # Write updated manifest after every subject (safe for crash/resume)
            manifest = {
                "dataset": "challenge-2018",
                "n_total": len(records),
                "n_done": len(stats_all),
                "n_errors": len(errors),
                "subjects": stats_all,
                "errors": errors,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    print("\n=== Summary ===")
    print(f"  Processed: {len(stats_all)}")
    print(f"  Errors:    {len(errors)}")
    if stats_all:
        total_spont = sum(s["arousal_spontaneous"] for s in stats_all)
        total_rera = sum(s["arousal_rera"] for s in stats_all)
        total_resp = sum(s["arousal_respiratory"] for s in stats_all)
        total_epochs = sum(s["n_epochs"] for s in stats_all)
        print(f"  Total epochs:       {total_epochs:,}")
        print(f"  Spontaneous arousals: {total_spont:,}  ({100*total_spont/total_epochs:.1f}%)")
        print(f"  RERA arousals:       {total_rera:,}  ({100*total_rera/total_epochs:.1f}%)")
        print(f"  Respiratory events:  {total_resp:,}  ({100*total_resp/total_epochs:.1f}%)")
        subjects_with_spont = sum(1 for s in stats_all if s["arousal_spontaneous"] > 0)
        print(f"  Subjects with ≥1 spontaneous arousal epoch: {subjects_with_spont}/{len(stats_all)}")

    if errors:
        print("\n  Failed records:")
        for e in errors[:10]:
            print(f"    {e['record']}: {e['error']}")
        if len(errors) > 10:
            print(f"    ... and {len(errors)-10} more (see {log_path})")


if __name__ == "__main__":
    main()

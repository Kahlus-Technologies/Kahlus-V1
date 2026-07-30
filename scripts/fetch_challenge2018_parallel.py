#!/usr/bin/env python3
"""Parallel challenge-2018 feature extractor using multiple workers.

Splits the 994 training records across N worker processes. Each worker
streams records from PhysioNet and saves only the feature arrays.
Uses file-based locking so workers don't double-process the same subject.

Usage:
    python scripts/fetch_challenge2018_parallel.py --out data/challenge2018 --workers 4 --n 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

MAX_RETRIES = 3
RETRY_BACKOFF = 20  # seconds

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neurotwin.adapters.challenge2018 import (  # noqa: E402
    list_training_records,
    load_challenge2018_record,
    subject_id_from_record,
)


def save_subject(features: dict, feat_dir: Path) -> None:
    subj = features["subject"]
    np.savez_compressed(
        feat_dir / f"{subj}.npz",
        eeg_epochs=features["eeg_epochs"],
        autonomic_epochs=features["autonomic_epochs"],
        arousal_any=features["arousal_any"],
        arousal_spontaneous=features["arousal_spontaneous"],
        arousal_rera=features["arousal_rera"],
        arousal_respiratory=features["arousal_respiratory"],
        stages=np.array(features["stages"]),
    )


def try_claim(lock_dir: Path, subj: str) -> bool:
    """Atomically claim a subject. Returns True if claimed by this worker."""
    claim = lock_dir / f"{subj}.claimed"
    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def worker(records: list[str], out_dir: Path, worker_id: int) -> None:
    feat_dir = out_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lock_dir = out_dir / ".locks"
    lock_dir.mkdir(exist_ok=True)
    log_path = out_dir / f"worker_{worker_id}.jsonl"

    with open(log_path, "a") as log_fh:
        for rec_path in records:
            subj = subject_id_from_record(rec_path)

            # Skip if already saved
            if (feat_dir / f"{subj}.npz").exists():
                continue

            # Claim this subject
            if not try_claim(lock_dir, subj):
                continue

            t0 = time.time()
            print(f"[W{worker_id}] {subj} streaming...", flush=True)

            features = None
            last_err = None
            for attempt in range(MAX_RETRIES):
                try:
                    features = load_challenge2018_record(rec_path)
                    break
                except Exception as e:
                    last_err = e
                    if attempt < MAX_RETRIES - 1:
                        wait = RETRY_BACKOFF * (2 ** attempt)
                        print(f"[W{worker_id}] {subj} retry {attempt+1} ({e}), sleeping {wait}s", flush=True)
                        time.sleep(wait)

            elapsed = time.time() - t0
            if features is not None:
                save_subject(features, feat_dir)
                spont = int(features["arousal_spontaneous"].sum())
                rera = int(features["arousal_rera"].sum())
                n = features["n_epochs"]
                entry = {"ok": True, "subject": subj, "n_epochs": n,
                         "arousal_spontaneous": spont, "arousal_rera": rera,
                         "elapsed_s": round(elapsed, 1), "worker": worker_id}
                print(f"[W{worker_id}] {subj} done ({elapsed:.0f}s) spont={spont} rera={rera}", flush=True)
            else:
                entry = {"ok": False, "subject": subj, "error": str(last_err),
                         "elapsed_s": round(elapsed, 1), "worker": worker_id}
                print(f"[W{worker_id}] {subj} FAILED: {last_err}", flush=True)

            log_fh.write(json.dumps(entry) + "\n")
            log_fh.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--n", type=int, default=None, help="Limit total subjects")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching record list...", flush=True)
    records = list_training_records()
    if args.n:
        records = records[: args.n]
    print(f"  {len(records)} records, {args.workers} workers", flush=True)

    import multiprocessing
    # Split records across workers
    chunks = [records[i::args.workers] for i in range(args.workers)]

    procs = []
    for wid, chunk in enumerate(chunks):
        p = multiprocessing.Process(target=worker, args=(chunk, out_dir, wid))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Final summary
    feat_dir = out_dir / "features"
    done = list(feat_dir.glob("*.npz"))
    print(f"\nDone: {len(done)} subjects saved to {feat_dir}")

    # Aggregate stats from worker logs
    stats = []
    for wid in range(args.workers):
        log = out_dir / f"worker_{wid}.jsonl"
        if log.exists():
            for line in log.read_text().splitlines():
                if line.strip():
                    stats.append(json.loads(line))

    ok = [s for s in stats if s.get("ok")]
    if ok:
        total_spont = sum(s["arousal_spontaneous"] for s in ok)
        total_epochs = sum(s["n_epochs"] for s in ok)
        print(f"Total epochs:              {total_epochs:,}")
        print(f"Spontaneous arousal epochs: {total_spont:,} ({100*total_spont/total_epochs:.1f}%)")
        print(f"Subjects with ≥1 spont:    {sum(1 for s in ok if s['arousal_spontaneous']>0)}/{len(ok)}")


if __name__ == "__main__":
    main()

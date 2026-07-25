"""Fetch MIT-BIH slpdb record-by-record, skipping what is already complete.

`wfdb.dl_database` is all-or-nothing: an interrupted run leaves no usable
checkpoint and the next attempt restarts the whole database. PhysioNet is slow
enough for slpdb (18 two-hour 250 Hz recordings, ~260 MB) that this reliably
fails to finish. This fetches one record at a time, treats a record as complete
only when .hea, .dat and .st are all present, and skips those, so progress is
monotonic across interruptions.

Also emits a fingerprint (record list + per-file sizes + sha256 of the header
set) so a downstream run can assert it read the same cohort, matching the
pattern used for ds005620.

Run:
  python3 scripts/fetch_slpdb.py --dest /path/to/slpdb
  python3 scripts/fetch_slpdb.py --dest /path/to/slpdb --max-subjects 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

BASE_URL = "https://physionet.org/files/slpdb/1.0.0"
REQUIRED_EXT = (".hea", ".dat", ".st")
_SUBJECT = re.compile(r"^slp(\d+)")


def subject_of(record: str) -> str:
    m = _SUBJECT.match(record)
    if m is None:
        raise ValueError(f"unrecognized slpdb record: {record!r}")
    return m.group(1)


def is_complete(dest: Path, record: str) -> bool:
    return all((dest / f"{record}{ext}").is_file() for ext in REQUIRED_EXT)


def remote_size(record: str, ext: str) -> int | None:
    """Content-Length for one remote file, or None if the server won't say."""
    proc = subprocess.run(
        ["curl", "-sS", "--fail", "-I", f"{BASE_URL}/{record}{ext}"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        if line.lower().startswith("content-length:"):
            return int(line.split(":", 1)[1].strip())
    return None


def verify_sizes(dest: Path, records: list[str]) -> list[str]:
    """Return records whose local files are shorter than the remote original.

    Existence alone is not integrity: an interrupted transfer leaves a short file
    that `is_complete` accepts. A truncated .dat would be read as a shorter
    recording and silently corrupt any downstream gate, so check the byte counts
    before certifying the cohort.
    """
    truncated = []
    for record in records:
        for ext in REQUIRED_EXT:
            path = dest / f"{record}{ext}"
            expected = remote_size(record, ext)
            if expected is not None and path.is_file() and path.stat().st_size < expected:
                print(f"  TRUNCATED {record}{ext}: "
                      f"{path.stat().st_size} < {expected} bytes", flush=True)
                truncated.append(record)
                break
    return truncated


def fetch_record(record: str, dest: Path, *, attempts: int = 3) -> bool:
    """Download one record's files via curl.

    wfdb's downloader reads the whole body in one stream; PhysioNet resets the
    connection partway through slpdb's ~32 MB .dat files, and a reset discards
    everything already transferred. curl with --retry-all-errors and -C - resumes
    from the partial file instead, so a reset costs only the unfetched tail.
    """
    for ext in REQUIRED_EXT:
        target = dest / f"{record}{ext}"
        url = f"{BASE_URL}/{record}{ext}"
        for attempt in range(attempts):
            proc = subprocess.run(
                ["curl", "-sS", "--fail", "--retry", "5", "--retry-delay", "3",
                 "--retry-all-errors", "-C", "-", "-o", str(target), url],
                capture_output=True, text=True, timeout=1800,
            )
            # curl exits 33 when the server rejects a resume on an already-complete
            # file; treat an existing non-empty file as done rather than retrying.
            if proc.returncode == 0 or (proc.returncode == 33 and target.is_file()):
                break
            print(f"    {record}{ext} attempt {attempt + 1} failed "
                  f"(exit {proc.returncode}): {proc.stderr.strip()[:120]}", flush=True)
            time.sleep(5)
    return is_complete(dest, record)


def write_fingerprint(dest: Path, records: list[str]) -> Path:
    entries = {}
    digest = hashlib.sha256()
    for record in sorted(records):
        sizes = {}
        for ext in REQUIRED_EXT:
            path = dest / f"{record}{ext}"
            sizes[ext] = path.stat().st_size if path.is_file() else None
        entries[record] = {"subject": subject_of(record), "sizes": sizes}
        header = dest / f"{record}.hea"
        if header.is_file():
            digest.update(header.read_bytes())
    payload = {
        "dataset_id": "slpdb",
        "source": "https://physionet.org/content/slpdb/1.0.0/",
        "n_records": len(records),
        "n_subjects": len({subject_of(r) for r in records}),
        "note": (
            "slpdb has more records than subjects: slp01a/slp01b and slp02a/slp02b "
            "are the same people. Group by the numeric prefix for subject-held-out splits."
        ),
        "header_set_sha256": digest.hexdigest(),
        "records": entries,
    }
    out = dest / "dataset_fingerprint.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--max-subjects", type=int, default=None,
                    help="stop once this many unique subjects are complete")
    args = ap.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    import wfdb

    records = list(wfdb.get_record_list("slpdb"))
    complete = [r for r in records if is_complete(args.dest, r)]
    print(f"{len(complete)}/{len(records)} records already complete "
          f"({len({subject_of(r) for r in complete})} subjects)", flush=True)

    for record in records:
        subjects_done = {subject_of(r) for r in complete}
        if args.max_subjects is not None and len(subjects_done) >= args.max_subjects:
            print(f"reached {len(subjects_done)} subjects; stopping", flush=True)
            break
        if is_complete(args.dest, record):
            continue
        print(f"  fetching {record} ...", flush=True)
        if fetch_record(record, args.dest):
            complete.append(record)
            print(f"    ok ({len(complete)}/{len(records)} records, "
                  f"{len({subject_of(r) for r in complete})} subjects)", flush=True)
        else:
            print(f"    giving up on {record}", flush=True)

    print("verifying byte counts against the server ...", flush=True)
    truncated = verify_sizes(args.dest, complete)
    for record in truncated:
        print(f"  re-fetching truncated {record} ...", flush=True)
        fetch_record(record, args.dest)
    still_bad = verify_sizes(args.dest, truncated) if truncated else []
    if still_bad:
        print(f"STILL TRUNCATED after re-fetch: {still_bad} -- excluded", flush=True)
        complete = [r for r in complete if r not in still_bad]

    fingerprint = write_fingerprint(args.dest, complete)
    print(f"complete: {len(complete)} records, "
          f"{len({subject_of(r) for r in complete})} subjects", flush=True)
    print(f"fingerprint: {fingerprint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

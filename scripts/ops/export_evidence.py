"""Export the public evidence bundle from this private repo.

The public repo is a LOSSY ONE-WAY MIRROR produced by this script. It is never
edited directly. Hand-curating a sibling public repo means public and private
diverge silently and every release becomes a reconciliation exercise; generating
it means the private repo stays the single source of truth.

Two tiers, because retention is a promise and the promise differs:

  custody      manifests, hashes, claim_scope, gate results, withdrawal registry.
               Tiny. Permanent. Never rewritten. This is what a citation points
               at and what must still resolve in twenty years.
  convenience  figures and derived arrays. Re-derivable from custody + public
               raw data. Explicitly disposable, garbage-collectable.

What is deliberately NOT exported: source code, adapters, private paths,
anything under src/. The export is allowlist-driven -- a file is published only
if it matches a rule below, so adding a new artifact type is a deliberate act.

  python3 scripts/ops/export_evidence.py --out ../kahlus-evidence
  python3 scripts/ops/export_evidence.py --out ../kahlus-evidence --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: (glob, tier). Allowlist: anything not matched here is never published.
EXPORT_RULES: list[tuple[str, str]] = [
    # --- custody: small, permanent, citation-bearing ------------------------
    ("artifacts/**/*_report.json", "custody"),
    ("artifacts/**/*EVIDENCE_REPORT.md", "custody"),
    ("docs/results/findings-ledger.md", "custody"),
    ("docs/results/holdout_manifest.json", "custody"),
    ("docs/audit/*.json", "custody"),
    ("docs/audit/*.md", "custody"),
    ("runs/**/*.csv", "custody"),
    ("**/dataset_fingerprint.json", "custody"),
    ("ops/jobs.yaml", "custody"),
    # --- convenience: re-derivable ------------------------------------------
    ("runs/**/*.png", "convenience"),
    ("docs/paper/**/figures/*.png", "convenience"),
    ("docs/paper/**/figures/*.pdf", "convenience"),
]

#: Never exported even if a rule would otherwise match.
DENY_SUBSTRINGS = ("/.venv/", "/node_modules/", "/.git/", "pw.txt", ".env", ".pem", ".key")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect() -> list[tuple[Path, str]]:
    seen: dict[Path, str] = {}
    for pattern, tier in EXPORT_RULES:
        for p in REPO.glob(pattern):
            if not p.is_file():
                continue
            s = str(p)
            if any(d in s for d in DENY_SUBSTRINGS):
                continue
            # custody wins if a file matches both tiers
            if p not in seen or tier == "custody":
                seen[p] = tier
    return sorted(seen.items())


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", action="store_true",
                    help="report what would be exported and exit non-zero if the "
                         "destination is stale")
    args = ap.parse_args()

    files = collect()
    if not files:
        print("nothing matched the export rules", file=sys.stderr)
        return 1

    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain"))
    entries = []
    for src, tier in files:
        rel = src.relative_to(REPO)
        entries.append({"path": str(rel), "tier": tier,
                        "sha256": sha256(src), "bytes": src.stat().st_size})

    manifest = {
        "schema": "kahlus.evidence_bundle.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_dirty": dirty,
        "n_files": len(entries),
        "bytes_by_tier": {
            t: sum(e["bytes"] for e in entries if e["tier"] == t)
            for t in ("custody", "convenience")
        },
        "tiers": {
            "custody": "Permanent. Never rewritten. Citations resolve here.",
            "convenience": "Re-derivable from custody plus public raw data. Disposable.",
        },
        "generated_by": "scripts/ops/export_evidence.py",
        "note": (
            "This bundle is a one-way export from a private repository. Do not edit "
            "it directly -- changes will be overwritten by the next export."
        ),
        "files": entries,
    }

    if args.check:
        prev = args.out / "evidence_manifest.json"
        print(f"{len(entries)} files  "
              f"custody={manifest['bytes_by_tier']['custody']/1e6:.1f}MB  "
              f"convenience={manifest['bytes_by_tier']['convenience']/1e6:.1f}MB")
        if dirty:
            print("WARNING: source tree is dirty; export would not be reproducible")
        if not prev.is_file():
            print("destination has no manifest -- export needed")
            return 1
        old = json.loads(prev.read_text(encoding="utf-8"))
        stale = {e["path"] for e in entries} != {e["path"] for e in old.get("files", [])} or any(
            e["sha256"] != o["sha256"]
            for e in entries
            for o in old.get("files", [])
            if e["path"] == o["path"]
        )
        print("STALE - export needed" if stale else "up to date")
        return 1 if stale else 0

    if dirty:
        print("refusing to export from a dirty tree: commit first so the bundle "
              "names a real source commit", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    for tier in ("custody", "convenience"):
        shutil.rmtree(args.out / tier, ignore_errors=True)
    for src, tier in files:
        dest = args.out / tier / src.relative_to(REPO)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    (args.out / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"exported {len(entries)} files from {commit[:12]} -> {args.out}")
    print(f"  custody     {manifest['bytes_by_tier']['custody']/1e6:8.1f} MB")
    print(f"  convenience {manifest['bytes_by_tier']['convenience']/1e6:8.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Execute one registered job. Refuses anything not in ops/jobs.yaml.

This exists so the cluster operator never has to run arbitrary shell supplied by
someone else. The requester names a job and supplies typed parameters; this
validates both against the registry and builds the command line itself. A request
for a job that is not registered, or a parameter that fails its constraint, is
rejected before anything executes.

What it deliberately does NOT do:
  * accept a command, shell string, or extra flags from the request
  * let the request choose the working directory (operator sets KAHLUS_WORK)
  * interpolate anything into a shell -- subprocess is called without shell=True

Run:
  python3 scripts/ops/run_job.py --job smoke --params '{"raw_dir":"...","out_dir":"..."}'
  python3 scripts/ops/run_job.py --request job.json --evidence-out evidence/
  python3 scripts/ops/run_job.py --list
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "ops" / "jobs.yaml"
_VAR = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def load_registry() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))


def coerce(name: str, raw: Any, spec: dict[str, Any]) -> Any:
    """Validate one parameter against its declared type and constraints."""
    kind = spec.get("type", "str")
    if kind == "int":
        val = int(raw)
    elif kind == "float":
        val = float(raw)
    elif kind in {"str", "path"}:
        val = str(raw)
    else:
        raise ValueError(f"{name}: unknown param type {kind!r}")

    if kind in {"int", "float"}:
        if "min" in spec and val < spec["min"]:
            raise ValueError(f"{name}={val} below min {spec['min']}")
        if "max" in spec and val > spec["max"]:
            raise ValueError(f"{name}={val} above max {spec['max']}")
    if "choices" in spec and val not in spec["choices"]:
        raise ValueError(f"{name}={val!r} not in {spec['choices']}")
    if kind == "path":
        # Reject traversal and shell metacharacters outright. Paths are still
        # passed as argv elements, never through a shell, but a request has no
        # business containing these.
        if ".." in val or any(c in val for c in ";|&$`\n"):
            raise ValueError(f"{name}: unsafe path {val!r}")
    return val


def resolve(job: dict[str, Any], params: dict[str, Any]) -> list[str]:
    """Build argv from the registry template. The request never supplies argv."""
    specs = job.get("params") or {}
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")

    resolved: dict[str, Any] = {}
    for name, spec in specs.items():
        if name in params:
            resolved[name] = coerce(name, params[name], spec)
        elif "default" in spec:
            resolved[name] = spec["default"]
        elif spec.get("required"):
            raise ValueError(f"missing required parameter: {name}")

    def sub(token: str) -> str:
        def repl(m: re.Match[str]) -> str:
            key = m.group(1)
            if key not in resolved:
                raise ValueError(f"template references unset parameter {key!r}")
            return str(resolved[key])

        return _VAR.sub(repl, token)

    argv = [str(t) for t in job["command"]]
    for group in job.get("args") or []:
        argv.extend(sub(str(t)) for t in group)
    return argv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job")
    ap.add_argument("--params", default="{}", help="JSON object")
    ap.add_argument("--request", type=Path, help="JSON file with {job, params}")
    ap.add_argument("--evidence-out", type=Path, default=Path("evidence"))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    jobs = reg["jobs"]

    if args.list:
        for name, j in jobs.items():
            print(f"{name:16s} {j['description'].strip().splitlines()[0]}")
            for p, s in (j.get("params") or {}).items():
                bits = [s.get("type", "str")]
                if s.get("required"):
                    bits.append("required")
                if "default" in s:
                    bits.append(f"default={s['default']}")
                print(f"    {p:14s} ({', '.join(bits)})")
        return 0

    if args.request:
        req = json.loads(args.request.read_text(encoding="utf-8"))
        job_name, params = req["job"], req.get("params", {})
    else:
        job_name, params = args.job, json.loads(args.params)
    if not job_name:
        ap.error("need --job or --request")

    if job_name not in jobs:
        print(f"REFUSED: {job_name!r} is not a registered job.", file=sys.stderr)
        print(f"  registered: {sorted(jobs)}", file=sys.stderr)
        return 2

    job = jobs[job_name]
    try:
        argv = resolve(job, params)
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    timeout = int(job.get("timeout_seconds") or reg["defaults"]["timeout_seconds"])
    print(f"job      : {job_name}")
    print(f"argv     : {argv}")
    print(f"timeout  : {timeout}s")
    print(f"cwd      : {REPO}")
    if args.dry_run:
        return 0

    args.evidence_out.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    t0 = time.time()
    log_path = args.evidence_out / "run.log"
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(  # noqa: S603 - argv built from registry, never shell
            argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
            timeout=timeout, check=False,
        )
    elapsed = round(time.time() - t0, 1)

    # Collect declared evidence files from any parameter that looks like an output dir.
    collected = []
    for fname in job.get("evidence") or []:
        for key in ("out_dir", "out"):
            if key in params:
                src = Path(str(params[key]))
                cand = src / fname if src.is_dir() else src
                if cand.is_file():
                    shutil.copy2(cand, args.evidence_out / cand.name)
                    collected.append(cand.name)
                    break

    result = {
        "schema": "kahlus.ops.job_result.v1",
        "job": job_name,
        "params": params,
        "argv": argv,
        "exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "started_utc": started.isoformat(),
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "host": os.uname().nodename,
        "evidence_files": sorted(collected + ["run.log"]),
    }
    (args.evidence_out / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"exit {proc.returncode} in {elapsed}s -> {args.evidence_out}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

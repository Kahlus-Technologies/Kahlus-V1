#!/usr/bin/env bash
# Chapman A100 bootstrap for the challenge-2018 cohort.
#
# The node is self-sufficient: it clones the repo, pulls raw data straight from
# the AWS Open Data mirror, and extracts features locally. No Hetzner, no rsync,
# no laptop staging.
#
# Note this workload has NO GPU component -- it is .mat parsing plus xqrs plus
# numpy. What the node contributes is cores, RAM, disk and bandwidth. Do not
# request a GPU for it.
#
#   scripts/cluster/chapman_bootstrap.sh /shared/persistent/neurotwin [n_workers]
#
# Stages are independent and resumable; re-running skips completed work.
set -euo pipefail

ROOT="${1:?usage: chapman_bootstrap.sh <work-dir> [n_workers]}"
WORKERS="${2:-$(nproc)}"
S3="s3://physionet-open/challenge-2018/1.0.0/training/"
RAW="$ROOT/c18_raw"
FEATURES="$ROOT/c18_features"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== chapman bootstrap ==="
echo "  work dir : $ROOT"
echo "  workers  : $WORKERS"
echo "  repo     : $REPO_DIR"
mkdir -p "$RAW" "$FEATURES"

# --- 0. preflight -----------------------------------------------------------
# Check the two things that silently invalidate everything downstream.
echo
echo "--- preflight ---"
command -v aws >/dev/null || { echo "FATAL: aws cli not found"; exit 1; }
python3 -c "import wfdb, numpy, scipy" 2>/dev/null || {
  echo "FATAL: wfdb/numpy/scipy missing. environment-a100.yml does not include wfdb --"
  echo "       install it before running:  pip install wfdb"
  exit 1
}
aws s3 ls --no-sign-request "$S3" >/dev/null 2>&1 || {
  echo "FATAL: cannot reach S3 from this node. Compute nodes on many clusters have"
  echo "       no outbound internet; check whether a proxy is required."
  exit 1
}
AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
echo "  free space: ${AVAIL_GB}G (need ~140G for raw + derived)"
[ "$AVAIL_GB" -ge 140 ] || { echo "FATAL: insufficient disk"; exit 1; }
echo "  preflight OK"

# --- 1. seal the holdout BEFORE any data lands ------------------------------
# Needs only the record list, so it runs first and cannot be influenced by
# anything seen in the recordings.
MANIFEST="$ROOT/holdout_manifest.json"
if [ ! -f "$MANIFEST" ]; then
  echo
  echo "--- sealing confirmation holdout ---"
  python3 "$REPO_DIR/scripts/seal_holdout.py" --records-from s3 --out "$MANIFEST"
else
  echo "  holdout manifest already sealed: $MANIFEST"
fi

# --- 2. smoke test on 5 subjects --------------------------------------------
# Catches pipeline bugs in minutes rather than after a full extraction.
SMOKE_DONE="$ROOT/.smoke_ok"
if [ ! -f "$SMOKE_DONE" ]; then
  echo
  echo "--- smoke: 5 subjects ---"
  aws s3 cp --no-sign-request --recursive "$S3" "$RAW/" \
    --exclude "*" $(aws s3 ls --no-sign-request "$S3" | head -5 \
      | awk '{print "--include \""$NF"*\""}' | tr -d '\n' | xargs echo) 2>/dev/null \
    || aws s3 sync --no-sign-request "$S3" "$RAW/" --exclude "*" --include "tr03-000*"
  python3 "$REPO_DIR/scripts/extract_challenge2018_local.py" \
    --raw "$RAW" --out "$FEATURES" --workers "$WORKERS" --limit 5
  echo "  smoke passed -- inspect $FEATURES/manifest.json before continuing"
  touch "$SMOKE_DONE"
fi

# --- 3. full pull ------------------------------------------------------------
echo
echo "--- full pull (133 GB, ~90 min at 25 MB/s) ---"
aws s3 sync --no-sign-request "$S3" "$RAW/"
echo "  records on disk: $(find "$RAW" -maxdepth 1 -mindepth 1 -type d | wc -l)"

# --- 4. extract --------------------------------------------------------------
echo
echo "--- extract ($WORKERS workers) ---"
python3 "$REPO_DIR/scripts/extract_challenge2018_local.py" \
  --raw "$RAW" --out "$FEATURES" --workers "$WORKERS"

echo
echo "=== done ==="
echo "  features : $FEATURES"
echo "  manifest : $FEATURES/manifest.json"
echo "  holdout  : $MANIFEST"
echo
echo "Check hr_path counts in the manifest before any autonomic claim:"
echo "  python3 -c \"import json;m=json.load(open('$FEATURES/manifest.json'));print(m['hr_path_counts'])\""
echo
echo "Next: REM positive control on the ANALYSIS split only. The sealed subjects"
echo "stay untouched until a primary result has been written down."

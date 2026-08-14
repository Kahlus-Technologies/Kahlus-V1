#!/usr/bin/env bash
# Isolated residual-field job, Sleep-EDF LOSO. Not the leaky overlapping trainer.
#SBATCH --job-name=kahlus-isolated-residual-sleep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs runs/isolated_residual_sleep
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${PYTHON_BIN:=python3}"
: "${OUT:=$ROOT/runs/isolated_residual_sleep}"
: "${STEPS:=8000}"
: "${HIDDEN:=192}"
: "${MODELS:=gru,ssm,nfc}"
: "${MAX_SUBJECTS:=0}"
: "${DATA_ROOT:=}"

"$PYTHON_BIN" scripts/isolated_residual_a100.py \
  --corpus sleep-edf \
  --loso \
  --device auto \
  --steps "$STEPS" \
  --hidden "$HIDDEN" \
  --models "$MODELS" \
  --stride 128 \
  --max-windows 4000 \
  --max-subjects "$MAX_SUBJECTS" \
  ${DATA_ROOT:+--data-root "$DATA_ROOT"} \
  --out "$OUT"

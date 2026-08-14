#!/usr/bin/env bash
# Isolated residual-field job. Not the leaky overlapping trainer.
#SBATCH --job-name=kahlus-isolated-residual
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
mkdir -p logs runs/isolated_residual
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
: "${PYTHON_BIN:=python3}"
: "${OUT:=$ROOT/runs/isolated_residual}"
: "${STEPS:=8000}"
: "${HIDDEN:=192}"
: "${MODELS:=gru,ssm,nfc}"

"$PYTHON_BIN" scripts/isolated_residual_a100.py \
  --corpus bnci \
  --device auto \
  --steps "$STEPS" \
  --hidden "$HIDDEN" \
  --models "$MODELS" \
  --stride 128 \
  --max-windows 8000 \
  --out "$OUT"

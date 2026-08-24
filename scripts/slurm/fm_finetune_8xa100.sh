#!/usr/bin/env bash
# Fine-tune arm of the FM forecasting courtroom, sharded across 8 A100s.
# One independent worker per GPU; folds are split by index modulo 8.
# Not slurm-array because the box runs bare Docker: plain background workers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs runs/fm_ft
: "${PYTHON_BIN:=python3}"
: "${MODELS:=gru-stack,cbramod,cbramod-rand,cbramod-ft,cbramod-ft-rand}"
: "${SEEDS:=5}"   # matches run_everything_8xa100.sh; artifact records the value
: "${STEPS:=2000}"
: "${NW:=8}"
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
# offline weights: unpack of the handoff puts the HF cache here
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
# Sleep-EDF cassette PSG dir on this box (same data as the isolated-residual run)
export KAHLUS_SLEEP_ROOT="${KAHLUS_SLEEP_ROOT:?set KAHLUS_SLEEP_ROOT to the sleep-cassette dir}"

pids=()
for i in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PYTHON_BIN" scripts/fm_probe_local.py \
    --device cuda \
    --max-subjects 0 --max-windows 1200 \
    --models "$MODELS" --seeds "$SEEDS" --gru-steps "$STEPS" \
    --shard "$i/$NW" \
    --out "runs/fm_ft/scores_shard$i.json" \
    > "logs/fm_ft_shard$i.log" 2>&1 &
  pids+=($!)
done
echo "launched $NW workers: ${pids[*]}"
for p in "${pids[@]}"; do wait "$p"; done

"$PYTHON_BIN" scripts/fm_merge_shards.py runs/fm_ft/scores_shard*.json \
  --out runs/fm_ft/scores_merged.json
echo "DONE -> runs/fm_ft/scores_merged.json"

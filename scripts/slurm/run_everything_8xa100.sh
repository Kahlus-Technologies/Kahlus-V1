#!/usr/bin/env bash
# THE one-command cluster job: every open experiment, all GPUs, self-merging.
#
#   Job A  fine-tune factorial (fold-sharded, all 8 GPUs):
#          persist | tuned ridge | GRU stack | CBraMod frozen / frozen-random /
#          fine-tuned / fine-tuned-from-random, 78-subject Sleep-EDF LOSO.
#   Job B  context x channel capability sweep (config-per-GPU):
#          {1 s, 15 s, 30 s} x {1, 3} channels at 80-2560 ms, FULL 78 subjects.
#
# Not the leaky trainer. Do not run scripts/slurm/train_a100.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${PYTHON_BIN:=python3}"
: "${NW:=8}"
: "${SEEDS:=5}"
: "${STEPS:=2000}"
: "${CTX_SUBJECTS:=0}"          # 0 = all 78
: "${CTX_SEEDS:=3}"
: "${CTX_STEPS:=1500}"
export PYTHONPATH="$ROOT/scripts:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/hf_cache}"
export HF_HUB_OFFLINE=1
export KAHLUS_SLEEP_ROOT="${KAHLUS_SLEEP_ROOT:?set KAHLUS_SLEEP_ROOT to the sleep-cassette dir}"

echo "== preflight =="
[ -d "$KAHLUS_SLEEP_ROOT" ] || { echo "FATAL: no data at $KAHLUS_SLEEP_ROOT"; exit 1; }
n_psg=$(ls "$KAHLUS_SLEEP_ROOT"/*-PSG.edf 2>/dev/null | wc -l)
echo "PSG files: $n_psg (need ~153)"; [ "$n_psg" -ge 100 ] || { echo "FATAL: too few PSGs"; exit 1; }
[ -d "$HF_HOME/hub" ] || { echo "FATAL: no bundled weights at $HF_HOME"; exit 1; }
"$PYTHON_BIN" - <<'PY' || { echo "FATAL: python deps missing (pip install braindecode transformers huggingface_hub scipy mne)"; exit 1; }
import braindecode, mne, scipy, torch
assert torch.cuda.is_available(), "no CUDA visible"
print(f"deps OK, {torch.cuda.device_count()} GPUs visible")
PY
"$PYTHON_BIN" scripts/fm_probe_local.py --selftest >/dev/null && echo "probe selftest OK"
"$PYTHON_BIN" scripts/fm_context_sweep.py --selftest >/dev/null && echo "sweep selftest OK"

mkdir -p logs runs/fm_ft runs/ctx evidence_out

echo "== job A: fine-tune factorial, $NW fold-shards =="
pids=()
for i in $(seq 0 $((NW - 1))); do
  CUDA_VISIBLE_DEVICES="$i" "$PYTHON_BIN" -u scripts/fm_probe_local.py \
    --device cuda --max-subjects 0 --max-windows 1200 \
    --models gru-stack,cbramod,cbramod-rand,cbramod-ft,cbramod-ft-rand \
    --seeds "$SEEDS" --gru-steps "$STEPS" --shard "$i/$NW" \
    --out "runs/fm_ft/scores_shard$i.json" \
    > "logs/fm_ft_shard$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
"$PYTHON_BIN" scripts/fm_merge_shards.py runs/fm_ft/scores_shard*.json \
  --out runs/fm_ft/scores_merged.json

echo "== job B: context x channel sweep, config-per-GPU =="
cfgs=("200 1" "3000 1" "6000 1" "200 3" "3000 3" "6000 3")
pids=()
for i in "${!cfgs[@]}"; do
  read -r ctx ch <<< "${cfgs[$i]}"
  CUDA_VISIBLE_DEVICES=$((i % NW)) "$PYTHON_BIN" -u scripts/fm_context_sweep.py \
    --device cuda --max-subjects "$CTX_SUBJECTS" --max-windows 500 --stride 400 \
    --gru-steps "$CTX_STEPS" --seeds "$CTX_SEEDS" \
    --contexts "$ctx" --channel-sets "$ch" \
    --out "runs/ctx/ctx${ctx}_ch${ch}.json" \
    > "logs/ctx${ctx}_ch${ch}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
"$PYTHON_BIN" scripts/fm_ctx_merge.py runs/ctx/ctx*_ch*.json \
  --out runs/ctx/context_sweep_merged.json

echo "== package send-back =="
cp runs/fm_ft/scores_merged.json runs/ctx/context_sweep_merged.json evidence_out/
cp logs/fm_ft_shard0.log evidence_out/ 2>/dev/null || true
( cd evidence_out && shasum -a 256 * > SHA256SUMS 2>/dev/null || sha256sum * > SHA256SUMS )
tar czf kahlus_evidence_$(date +%Y%m%d).tar.gz evidence_out
echo "DONE. Send back: kahlus_evidence_$(date +%Y%m%d).tar.gz (plus logs/ if anything failed)"

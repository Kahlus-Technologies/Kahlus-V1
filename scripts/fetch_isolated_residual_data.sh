#!/usr/bin/env bash
# Pull the two public corpora the isolated residual job needs.
# Run this on the login node BEFORE sbatch. Training jobs must not download.
set -euo pipefail

SLEEP_DEST="${SLEEP_DEST:-$HOME/datasets/sleep-edfx/sleep-cassette}"
BNCI_DEST="${BNCI_DEST:-$HOME/.cache/kahlus/moabb/MNE-bnci-data/~bci/database/001-2014}"
SLEEP_URL="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

echo "Sleep-EDF cassette -> $SLEEP_DEST  (~7-8 GB, PhysioNet)"
mkdir -p "$SLEEP_DEST"
if ! ls "$SLEEP_DEST"/*-PSG.edf >/dev/null 2>&1; then
  wget -r -N -c -np -nH --cut-dirs=3 -P "$(dirname "$SLEEP_DEST")" "$SLEEP_URL"
fi
echo "PSG files: $(ls -1 "$SLEEP_DEST"/*-PSG.edf 2>/dev/null | wc -l)"

echo "BNCI2014_001 -> $BNCI_DEST  (~0.8 GB, MOABB)"
mkdir -p "$BNCI_DEST"
if ! ls "$BNCI_DEST"/A0*T.mat >/dev/null 2>&1; then
  python3 - <<'PY'
from pathlib import Path
import os
dest = Path(os.environ.get("BNCI_DEST", Path.home() / ".cache/kahlus/moabb/MNE-bnci-data/~bci/database/001-2014"))
# moabb writes to $MNE_DATA/MNE-bnci-data/..., so MNE_DATA is the moabb dir
os.environ.setdefault("MNE_DATA", str(dest.parents[3]))
from moabb.datasets import BNCI2014_001
BNCI2014_001().download()
print("moabb download done")
PY
fi
echo "done. DATA_ROOT=$SLEEP_DEST"
echo "Then: DATA_ROOT=$SLEEP_DEST sbatch scripts/slurm/isolated_residual_sleep_a100.sh"

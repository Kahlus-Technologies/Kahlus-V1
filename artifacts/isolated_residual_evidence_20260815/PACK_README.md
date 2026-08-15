# Kahlus Isolated-Residual Evidence

## Scope

This package records a completed leakage-controlled EEG forecasting run from the
`kahlus_isolated_residual_handoff.zip` handoff. It is not the legacy
overlapping-window trainer. The task is: observe samples `0:126` and predict
only one strictly future sample at horizons 1, 8, 16, and 32.

The prediction stack is persistence, last-16-sample ridge, an oscillator
residual, and then a GRU residual head. The claim rule is encoded by
`gru_win`: the GRU must improve on ridge and beat predicting zero residual.

## Completed Runs

| Corpus | Split | Result |
| --- | --- | --- |
| BNCI2014_001 | Fixed subject-disjoint split: train `1,3,4,5,7,9`, val `2`, test `6,8` | Completed. The GRU did not beat ridge at any horizon. |
| Sleep-EDF cassette | 78 subject-held-out LOSO folds | Completed. Mean GRU MSE was lower than mean ridge MSE at every horizon. |

### BNCI MSE

| Horizon | Ridge | GRU stack | `gru_win` |
| --- | ---: | ---: | --- |
| 1 | 0.29645 | 0.41052 | false |
| 8 | 1.03545 | 1.38399 | false |
| 16 | 1.10781 | 1.56651 | false |
| 32 | 1.21500 | 1.68235 | false |

### Sleep-EDF Mean-Over-Folds MSE

| Horizon | Ridge | GRU stack |
| --- | ---: | ---: |
| 1 | 0.29514 | 0.26003 |
| 8 | 0.66034 | 0.54176 |
| 16 | 0.90041 | 0.69289 |
| 32 | 0.98715 | 0.80246 |

Read the full score structures in `runs/*/scores.json`; they include per-fold
Sleep-EDF results and all baselines. The package does not add confidence
intervals or significance tests, so these values support an experimental result
but are not a final publication claim on their own.

## Runtime Provenance

- Input handoff SHA256: `0a5ed2cf5ab1c016239a5e98619a66eb1786a640b1f523572f178bcb5651a39c`
- Base Kahlus-V1 commit: `4041a686d89ecf6a279fd84f5323e253d485fd5b`
- Container: `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`
- Hardware: one NVIDIA A100 80 GB per independent evaluation.
- BNCI ran on host GPU 0; Sleep-EDF ran on host GPU 5.
- Both preflight files confirm exactly one visible CUDA device.

## Required Runtime Fix

The supplied runner created GRU and SSM modules on CPU while it moved input
tensors to CUDA. This package includes the original runner backup and the
patched runner. The patch adds `.to(device)` to those learned modules only; it
does not change model architecture, training steps, splits, horizons, data, or
metrics. `logs/cuda_gru_probe.log` shows the one-step CUDA validation.

## Package Contents

- `runs/`: final BNCI and Sleep-EDF `scores.json` artifacts.
- `preflight/`: one-A100 visibility checks for both containers.
- `logs/`: CUDA probe and execution logs.
- `scripts/`: patched runner plus the supplied pre-patch runner backup.
- `launch_*.sh`, `data_prep_status.txt`, `launch_status.txt`: data and Docker
  launch provenance.
- `SHA256SUMS`: integrity hashes for all included files.

Excluded by design: raw EEG/EDF/MAT files, cached data, Docker layers,
credentials, tokens, keys, checkpoints, environment files, and nested archives.

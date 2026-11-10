# KAHLUS-V1 / NeuroTwin

> [!NOTE]
> **Active work moved to [Kahlus-V2](https://github.com/Kahlus-Technologies/Kahlus-V2)** (arousal forecasting and a safety veto for closed-loop sleep stimulation, on NSRR). This repository holds the v1 NeuroTwin EEG work and is kept for reference.

> A leakage-controlled benchmark and model scaffold for neural translation:
> can noninvasive biosignals forecast future brain and body state under
> honest, held-out splits?

Kahlus is the leakage-controlled engine behind NeuroTwin. It asks one falsifiable
question — **do noninvasive recordings (fMRI, EEG, behavior, stimulus) carry
residual information about *future* neural state that survives strict
subject/site/dataset holdout** — and refuses to let a model answer it by leaking.
The current architecture is NeuroTwin NFC (a Neural Field Compiler) that treats
every modality as a partial observation of one latent neural field.

The claim is deliberately **not** "first brain foundation model", "first
multimodal brain model", or a clinical digital twin. The defensible v1 target is
stricter: cross-modal neural translation, missing-modality reconstruction,
future-state forecasting, and few-shot subject adaptation, all measured under
held-out subject/site/dataset splits with a pre-declared leakage contract.

The full research constitution, claim boundaries, and gate discipline live in
**[AGENTS.md](AGENTS.md)**.

---

## The headline result

On recovered EEG v1 sidecars the honest finding is a **split verdict** — and an
overlap audit later showed the forecasting number was inflated by input–target
overlap. Under isolated (strictly future-sample) evaluation on Sleep-EDF and
BNCI2014_001, the Kahlus GRU does **not** beat the best trivial baseline. Report
the loss: a leakage-controlled benchmark that only flattered its own model would
be worthless.

| task / audit | honest status | note |
| --- | --- | --- |
| **Masked reconstruction** | **lose to ridge** | aggregate reconstruction failed (see A100 ledger) |
| **Isolated future-sample forecast** | **lose to persistence / ridge** | Sleep-EDF + BNCI subject-held-out |
| **Overlapping forecast sidecar (legacy)** | not a whole-model claim | 3.116 MSE / 0.972 r is a narrow, overlap-contaminated sidecar only |

Do **not** headline 3.116 MSE / 0.972 Pearson as whole-model performance.

![Recovered Kahlus v1 versus standard EEG baselines](docs/research/eeg_v1_figure_source/figures/Figure3_eeg_v1_baseline_ranking.png)

Publishing the reconstruction loss and the isolated-forecast loss is the point.
Neural-CASP (gates, copy-trap, overlap audit, residual forecastability) is the
real product surface — not a forecasting-skill claim.

**Findings ledger (F0–F6):** [`docs/results/findings-ledger.md`](docs/results/findings-ledger.md)  
**NeurIPS submission packet:** [`docs/paper/neurips_2026/`](docs/paper/neurips_2026/)  
**Arena paper packet:** [`docs/paper/neural_casp_arena/`](docs/paper/neural_casp_arena/)

Powered Passive PCI on full Sleep-EDF cassette (F4): complexity block **hurts**
prediction vs spectral baseline — wake RFS **−0.330** bits (CI excludes 0).
Tag: `finding/passive-pci-negative-v1`.

Every figure is rendered only from cached CSV/JSON evidence artifacts (no raw
tensors, no waveform overlays without provenance); the figure-source packet and
its provenance rule are in
[`docs/research/eeg_v1_figure_source/`](docs/research/eeg_v1_figure_source/).

## Why this exists

The neural foundation-model lane is crowded (TRIBE, BrainVista, Brain-OF,
BrainOmni, Brain Harmony), and most benchmarks are built by the same group that
builds the model. Leakage — reusing records, overlapping held-out groups,
repeated windows, forbidden target metadata — is the field's quiet failure mode,
and it inflates exactly the future-state and cross-subject claims that matter
most. Kahlus is built the other way around: **the split manifest is frozen at
recording-manifest time, before preprocessing, windowing, or augmentation**, and
every competitor is a first-class baseline rather than a strawman.

## The tasks (v1 neural translation)

Five task families, each scored under held-out subject/site/dataset splits:

1. **Future-state forecasting** — predict future neural state from past.
2. **Masked reconstruction** — recover masked channels of the present.
3. **Cross-modal translation** — map one modality to another.
4. **Subject adaptation** — few-shot to a held-out subject.
5. **Dataset/site generalization** — transfer across recording sites.

Baseline failures are explicit and excluded from rankings rather than silently
imputed. The competitor registry names the crowded lanes instead of pretending
they do not exist.

## Honest scope

- The headline numbers are from the **recovered EEG v1 benchmark**, rendered from
  cached evidence artifacts. Synthetic smoke results elsewhere in the repo are
  plumbing checks, not science, and are labeled as such.
- Kahlus **loses** masked reconstruction to a linear ridge. That is reported, not
  buried.
- Directed information / neural translation is not new; the contribution here is
  the **leakage contract + held-out-split discipline + explicit competitor
  baselines**, validated end-to-end on a benchmark that can fail its own model.
- Clinical, physiological, and "digital twin" claims are out of scope for v1 by
  the constitution in [AGENTS.md](AGENTS.md).

## Independent coupling benchmark

The directed-information estimator lane is checked separately by
[kahlus-bench](https://github.com/ptlnextdoor/kahlus-bench), a leakage-sealed
benchmark that certifies neural-coupling estimators against synthetic systems
whose true directed information is analytically known. That benchmark reads
ground truth; the estimators never do, so the pass is an independent check.

---

## Current Scaffold

- Shared `NeuralEventBatch` schema for fMRI, EEG, MEG, spikes, behavior, stimulus, anatomy, and clinical metadata.
- `SplitManifest` built at recording-manifest time, before preprocessing, windowing, or augmentation.
- Leakage checks and audits for record reuse, held-out group overlap, repeated windows, and forbidden target metadata.
- Required v1 Neural Translation task registry.
- Competitor registry that names crowded lanes instead of pretending they do not exist.
- Synthetic CPU smoke data for testable benchmark plumbing. These results are not science.
- Optional MOABB EEG adapter and BIDS/OpenNeuro manifest scanner. BIDS can emit prepared events from precomputed `_timeseries.npy/.npz/.tsv/.csv` derivatives without raw NIfTI preprocessing.
- Prepared `event_manifest.json` artifacts for offline train/eval jobs. Synthetic and MOABB preparation emit `NeuralEventBatch` `.npz` files; BIDS is manifest-only in this pass.
- Prepared-manifest NeuroTwin training path with single-task or `neural_translation_v1` multi-task training, DDP under `torchrun`, checkpoint save/resume, best checkpoints, objective-level JSONL metrics, gradient accumulation, bf16 autocast hooks, CSV/JSON metrics, config snapshots, environment info, split hashes, and synthetic/demo labels.
- Prepared training uses train for optimization, val for periodic evaluation and best-checkpoint selection, and test for final held-out reporting.
- Prepared-manifest benchmark suite covers the five v1 task families: forecasting, masked reconstruction, cross-modal translation, subject adaptation, and dataset/site generalization. Baseline failures are explicit and excluded from rankings.
- A100 is the canonical cluster target through `configs/train/*_a100.yaml`, `scripts/slurm/*_a100.sh`, and `docs/deploy/A100_RUNBOOK.md`; H100 remains a compatible high-memory variant. Docker/handoff assets live under `deploy/a100/`.
- Chapman A100 first launch has a guarded one-command path in `scripts/cluster/chapman_a100_first_run.sh`, with explicit preflight against placeholder paths, zero-window data, missing CUDA, and non-persistent run roots.
- Modular NeuroTwin model internals now expose modality encoders, `transformer`/`ssm_fallback` backbone selection, geometry/metadata encoders, projection heads, and leakage-safe subject-adapter controls. `mamba` remains an upstream baseline target, not a wired NeuroTwin backbone selector.
- Experimental NFC internals expose `NeuralFieldCompiler`, `LatentNeuralField`, causal field updates, low-rank pair kernels, observation operators, stimulus conditioning, uncertainty maps, and a synthetic latent-field suite. Synthetic NFC outputs are plumbing checks, not science.
- Reproducibility helpers for deterministic seeds, config snapshots, environment capture, git commit capture, and stable manifest hashes.
- CLI surface:
  - `nt doctor`
- `nt data prepare --dataset synthetic --split subject`
- `nt data smoke --dataset moabb --split subject --out-dir /tmp/neurotwin_moabb_smoke`
- `nt data audit --dataset synthetic`
  - `nt split audit --dataset synthetic --split subject`
  - `nt estimate --config <experiment>`
  - `nt train --dry-run --config <experiment>`
  - `nt train --config <experiment>`
  - `nt train --config <experiment> --resume runs/<run_id>/checkpoint.pt`
  - `nt eval --suite translation_smoke`
  - `nt eval --suite neural_translation_v1 --event-manifest <prepared>/event_manifest.json --split-manifest <prepared>/split_manifest.json`
  - `nt eval audit --suite neural_translation_v1 --event-manifest <prepared>/event_manifest.json --split-manifest <prepared>/split_manifest.json`
  - `nt report --suite translation_smoke`
  - `nt report --run-dir runs/<run_id>`
  - `nt report --compare runs/<run_a> runs/<run_b> --out-dir reports/compare`

## Install

Local CPU/dev:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Optional real-data adapters:

```bash
python -m pip install -e '.[moabb]'
```

BIDS derivative scanning is included in the base install; it does not install
heavy BIDS preprocessing libraries.

A100/H100 jobs should use prepared local manifests and data roots. Do not depend on internet downloads inside training jobs.

For the first Chapman A100 run:

```bash
scripts/cluster/chapman_a100_first_run.sh /path/to/shared/persistent/neurotwin
```

## Local Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m neurotwin.cli doctor
PYTHONPATH=src python3 -m neurotwin.cli data prepare --dataset synthetic --split subject --out-dir /tmp/neurotwin_prepared
PYTHONPATH=src python3 -m neurotwin.cli data smoke --dataset moabb --split subject --out-dir /tmp/neurotwin_moabb_smoke
scripts/prepare_moabb_smoke.sh /tmp/neurotwin_moabb_smoke
scripts/prepare_moabb_benchmark.sh /tmp/neurotwin_moabb_benchmark
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/neurotwin_v1_a100.yaml
PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/synthetic_debug.yaml
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/prepared_synthetic_debug.yaml --run-root /tmp/neurotwin_runs
PYTHONPATH=src python3 -m neurotwin.cli train --config configs/train/moabb_smoke_locked.yaml --run-root /tmp/neurotwin_moabb_runs
PYTHONPATH=src python3 -m neurotwin.cli eval --suite translation_smoke
PYTHONPATH=src python3 -m neurotwin.cli eval audit --suite neural_translation_v1 --event-manifest /tmp/neurotwin_prepared/event_manifest.json --split-manifest /tmp/neurotwin_prepared/split_manifest.json
PYTHONPATH=src python3 -m neurotwin.cli eval --suite neural_translation_v1 --event-manifest /tmp/neurotwin_prepared/event_manifest.json --split-manifest /tmp/neurotwin_prepared/split_manifest.json --train-steps 1
PYTHONPATH=src python3 -m neurotwin.cli eval --suite nfc_synthetic --out-dir /tmp/neurotwin_nfc_synthetic --train-steps 1 --seed 0
PYTHONPATH=src python3 -m neurotwin.cli report --suite translation_smoke
bash -n scripts/slurm/*.sh
```

## Repo Rule

Raw public neural data is never committed. The repo stores manifests, configs, adapters, reproducible download/prep scripts, benchmark definitions, and license notes.

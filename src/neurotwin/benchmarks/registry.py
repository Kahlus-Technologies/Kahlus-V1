from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompetitorSpec:
    competitor_id: str
    display_name: str
    family: str
    role: str
    upstream_url: str
    license_status: str
    notes: str
    implementation_status: str = "unavailable"


def competitor_registry() -> tuple[CompetitorSpec, ...]:
    return (
        CompetitorSpec(
            competitor_id="tribe_v2",
            display_name="TRIBE v2",
            family="stimulus-to-fMRI",
            role="Primary baseline for video/audio/language -> fMRI prediction and in-silico neuroscience claims.",
            upstream_url="https://arxiv.org/abs/2605.04326",
            license_status="CC BY-NC reference; no code or weights vendored",
            notes="Local tribe_style is a clean-room stimulus-to-fMRI approximation, not exact TRIBE v2.",
            implementation_status="clean_room_approximation",
        ),
        CompetitorSpec(
            competitor_id="brainvista",
            display_name="BrainVista",
            family="naturalistic fMRI next-token prediction",
            role="Primary baseline for multimodal stimulus-conditioned fMRI rollout.",
            upstream_url="https://arxiv.org/abs/2602.04512",
            license_status="paper-first; verify code/weights before reuse",
            notes="Crowds future fMRI from stimulus and long-horizon rollout claims.",
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="brain_of",
            display_name="Brain-OF",
            family="fMRI/EEG/MEG foundation model",
            role="Boss-fight baseline for generic multimodal neural foundation modeling.",
            upstream_url="https://arxiv.org/abs/2602.23410",
            license_status="paper-first; verify code/weights before reuse",
            notes="Overlaps with universal tokenizer/translator; NeuroTwin must be stricter on translation benchmarks and held-out splits.",
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="brainomni",
            display_name="BrainOmni",
            family="EEG/MEG foundation model",
            role="Baseline for electrophysiology tokenization and sensor-layout generalization.",
            upstream_url="https://arxiv.org/abs/2505.18185",
            license_status="paper-first; verify code/weights before reuse",
            notes="Covers unified EEG/MEG tokenizer and unseen-device generalization.",
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="brain_harmony",
            display_name="Brain Harmony",
            family="structural MRI + fMRI",
            role="Baseline for compact anatomy/function brain tokens.",
            upstream_url="https://arxiv.org/abs/2509.24693",
            license_status="paper-first; verify code/weights before reuse",
            notes="Crowds structure + function tokenization.",
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="transformer",
            display_name="Transformer",
            family="general sequence model",
            role="Strong shared-data baseline.",
            upstream_url="https://arxiv.org/abs/1706.03762",
            license_status="implementation local/permissive",
            notes="Must use the exact same splits and inputs as NeuroTwin.",
            implementation_status="local_baseline",
        ),
        CompetitorSpec(
            competitor_id="mamba_ssm",
            display_name="Mamba/SSM",
            family="selective state-space model",
            role="Strong long-sequence dynamics baseline.",
            upstream_url="https://github.com/state-spaces/mamba",
            license_status="permissive upstream; pin commit before vendoring",
            notes="Baseline and possible NeuroTwin core, but not by itself a novelty claim.",
            implementation_status="approximation",
        ),
        CompetitorSpec(
            competitor_id="eegpt",
            display_name="EEGPT",
            family="pretrained EEG foundation model",
            role="External pretrained EEG baseline; the only frontier-scale transformer entrant whose published pretraining corpus is disjoint from our Sleep-EDF/CHB-MIT/TUSZ eval sets (BIOT PREST-16 is likewise disjoint).",
            upstream_url="https://github.com/BINE022/EEGPT",
            license_status="Apache-2.0 code (NeurIPS 2024); weights hosted off-repo on figshare, verify terms before reuse",
            notes=(
                "Pretrained by dual self-supervised spatio-temporal masking (80% channel mask) for representation "
                "learning, not forecasting; downstream use is classification/masked infill. Pretraining corpus is "
                "PhysioNet-MI, HGD, TSU-Benchmark, SEED, M3CV. Checkpoint is distributed only via a figshare share "
                "link, which is fragile and must be mirrored and hashed before any benchmark run. No weights vendored."
            ),
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="labram",
            display_name="LaBraM",
            family="pretrained EEG foundation model",
            role="External pretrained EEG baseline; field reference point for neural-tokenizer EEG pretraining.",
            upstream_url="https://github.com/935963004/LaBraM",
            license_status="MIT code (ICLR 2024); base checkpoint committed in-repo, Large/Huge weights withheld by authors",
            notes=(
                "Pretrained with vector-quantized neural-token masked prediction for classification; there is no "
                "raw-waveform decoder, so it is not a forecaster and any forecasting comparison needs a bolted-on "
                "head. Pretraining is ~2,500h across ~20 corpora including TUH corpora, so TUSZ evaluation is "
                "contaminated. Only the Base checkpoint is publicly available (committed in the upstream repo); "
                "Large and Huge are withheld. No weights vendored."
            ),
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="cbramod",
            display_name="CBraMod",
            family="pretrained EEG foundation model",
            role="External pretrained EEG baseline; strongest reported cross-subject EEG foundation model at small parameter count.",
            upstream_url="https://github.com/wjq-learning/CBraMod",
            license_status="MIT code (ICLR 2025); weights hosted off-repo on Hugging Face (weighting666/CBraMod), verify model-card terms before reuse",
            notes=(
                "Criss-cross transformer pretrained by masked patch reconstruction/infill, not forecasting; released "
                "weights do include a raw-patch reconstruction head. Pretraining corpus is TUEG (>9,000h), which is a "
                "superset of TUSZ, so TUSZ evaluation is contaminated. No weights vendored."
            ),
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="biot",
            display_name="BIOT",
            family="pretrained biosignal transformer",
            role="External pretrained EEG baseline; native bipolar montage matches CHB-MIT without channel remapping.",
            upstream_url="https://github.com/ycq091044/BIOT",
            license_status="MIT code (NeurIPS 2023); checkpoints committed in-repo, no external weight host",
            notes=(
                "Biosignal transformer pretrained by masked/supervised classification objectives, not forecasting. "
                "Checkpoints ship inside the upstream repo. The EEG-PREST-16-channels checkpoint is MGH resting EEG "
                "only and is clean for our eval sets; the six-datasets checkpoint was pretrained on CHB-MIT (plus "
                "TUAB/TUEV/IIIC) and is FORBIDDEN for CHB-MIT evaluation and unsafe for TUSZ. Checkpoint identity "
                "must be recorded per run. No weights vendored."
            ),
            implementation_status="unavailable",
        ),
        CompetitorSpec(
            competitor_id="modality_specialist",
            display_name="Modality-specialist baselines",
            family="specialist models",
            role="NDT3/Neuroformer for spikes, Braindecode/EEGNet-style EEG, fMRI-specific baselines.",
            upstream_url="https://github.com/joel99/ndt3",
            license_status="mixed; quarantine restricted adapters",
            notes="Prevents weak generalist comparisons.",
            implementation_status="unavailable",
        ),
    )

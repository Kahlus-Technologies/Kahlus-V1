from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


CatalogStatus = Callable[[set[str], set[str]], str]


@dataclass(frozen=True)
class BaselineCatalogEntry:
    model_id: str
    display_name: str
    status: str | CatalogStatus
    notes: str
    upstream_reference: str | None = None
    exact_reproduction: bool = False
    uses_upstream_code: bool = False
    uses_upstream_weights: bool = False

    def catalog_row(self, task_ids: set[str], modalities: set[str]) -> dict[str, object]:
        status = self.status(task_ids, modalities) if callable(self.status) else self.status
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "status": status,
            "notes": self.notes,
            "upstream_reference": self.upstream_reference,
            "exact_reproduction": self.exact_reproduction,
            "uses_upstream_code": self.uses_upstream_code,
            "uses_upstream_weights": self.uses_upstream_weights,
        }


def baseline_catalog_rows(task_ids: set[str], modalities: set[str]) -> list[dict[str, object]]:
    return [spec.catalog_row(task_ids, modalities) for spec in BASELINE_CATALOG]


def _tribe_style_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "clean_room_approximation" if "stimulus_to_fmri_response" in task_ids and "fmri" in modalities else "unavailable"


def _brainvista_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "local_approximation" if "future_state_forecasting" in task_ids and "fmri" in modalities else "unavailable"


def _brain_of_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if "masked_neural_reconstruction" in task_ids and len(modalities) >= 2 else "unavailable"


def _brainomni_catalog_status(task_ids: set[str], modalities: set[str]) -> str:
    return "approximation" if modalities & {"eeg", "meg"} else "unavailable"


BASELINE_CATALOG: tuple[BaselineCatalogEntry, ...] = (
    BaselineCatalogEntry(
        model_id="persistence",
        display_name="Persistence",
        status="local_baseline",
        notes="Last-observation or identity-style forecast with shape-safe fallback.",
    ),
    BaselineCatalogEntry(
        model_id="train_mean",
        display_name="Train Mean",
        status="local_baseline",
        notes="Broadcast train-target mean negative baseline.",
    ),
    BaselineCatalogEntry(
        model_id="random_permutation",
        display_name="Random Permutation",
        status="negative_control",
        notes="Seeded permutation of training targets with target-shaped output.",
    ),
    BaselineCatalogEntry(
        model_id="linear_ridge",
        display_name="Linear Ridge",
        status="local_baseline",
        notes="Closed-form sanity baseline on identical prepared windows.",
    ),
    BaselineCatalogEntry(
        model_id="gbm",
        display_name="Gradient Boosting",
        status="local_baseline",
        notes="Local multi-output histogram gradient-boosting regressor on flattened prepared windows; scikit-learn implementation with shared splits.",
    ),
    BaselineCatalogEntry(
        model_id="autoregressive_ridge",
        display_name="Autoregressive Ridge",
        status="local_baseline",
        notes="Ridge from previous source timepoint to next target timepoint where sequence shapes allow.",
    ),
    BaselineCatalogEntry(
        model_id="mlp",
        display_name="MLP",
        status="local_baseline",
        notes="Per-timepoint neural-window baseline.",
    ),
    BaselineCatalogEntry(
        model_id="tcn",
        display_name="TCN",
        status="local_baseline",
        notes="Local temporal convolution baseline.",
    ),
    BaselineCatalogEntry(
        model_id="transformer",
        display_name="Transformer",
        status="local_baseline",
        notes="Small local Transformer with shared splits.",
    ),
    BaselineCatalogEntry(
        model_id="ssm_fallback",
        display_name="GRU (Legacy ID)",
        status="local_baseline",
        notes="GRU retained under the historical ssm_fallback identifier; it is not reported as a state-space model.",
    ),
    BaselineCatalogEntry(
        model_id="tiny_ssm",
        display_name="Tiny Diagonal SSM",
        status="local_baseline",
        notes="Stable diagonal recurrent state-space baseline with causal sequence processing.",
    ),
    BaselineCatalogEntry(
        model_id="neurotwin",
        display_name="NeuroTwin",
        status="local_baseline",
        notes="Current NeuroTwin implementation under the same task API.",
    ),
    BaselineCatalogEntry(
        model_id="pair_operator",
        display_name="NeuroTwin Pair-Operator",
        status="local_baseline",
        notes="fMRI-first pairwise neural operator ablation; clean-room NeuroTwin architecture, not an upstream reproduction.",
    ),
    BaselineCatalogEntry(
        model_id="pair_operator_no_pair_state",
        display_name="NeuroTwin Pair-Operator No Pair State",
        status="local_baseline",
        notes="Pair-Operator ablation with node/temporal path intact and pair-state update disabled.",
    ),
    BaselineCatalogEntry(
        model_id="tribe_style",
        display_name="TRIBE-Style",
        status=_tribe_style_catalog_status,
        notes="NeuroTwin-native stimulus-to-fMRI approximation; not an exact TRIBE v2 reproduction.",
        upstream_reference="TRIBE v2",
        exact_reproduction=False,
        uses_upstream_code=False,
        uses_upstream_weights=False,
    ),
    BaselineCatalogEntry(
        model_id="tribe_style_clean_room",
        display_name="TRIBE-Style Clean Room",
        status=_tribe_style_catalog_status,
        notes="NeuroTwin-native clean-room stimulus-to-fMRI approximation; not exact TRIBE v2 and uses no TRIBE code or weights.",
        upstream_reference="TRIBE v2",
        exact_reproduction=False,
        uses_upstream_code=False,
        uses_upstream_weights=False,
    ),
    BaselineCatalogEntry(
        model_id="brainvista_style",
        display_name="BrainVista-Style",
        status=_brainvista_catalog_status,
        notes="Local approximation: causal history-only fMRI rollout plus optional lagged stimulus features; not an exact BrainVista reproduction.",
        upstream_reference="BrainVista",
    ),
    BaselineCatalogEntry(
        model_id="brain_of_style",
        display_name="Brain-OF-Style",
        status=_brain_of_catalog_status,
        notes="Approximate multimodal masked reconstruction lane; not an exact Brain-OF reproduction.",
        upstream_reference="Brain-OF",
    ),
    BaselineCatalogEntry(
        model_id="brainomni_style",
        display_name="BrainOmni-Style",
        status=_brainomni_catalog_status,
        notes="Approximate EEG/MEG tokenizer lane; not an exact BrainOmni reproduction.",
        upstream_reference="BrainOmni",
    ),
    BaselineCatalogEntry(
        model_id="braindecode_wrapper",
        display_name="Braindecode Wrapper",
        status="unavailable",
        notes="Optional EEG wrapper slot; exact use requires installed Braindecode and compatible task protocols.",
        upstream_reference="Braindecode",
    ),
    BaselineCatalogEntry(
        model_id="cebra_wrapper",
        display_name="CEBRA Wrapper",
        status="unavailable",
        notes="Optional neural-behavior embedding wrapper slot; exact use requires installed CEBRA and aligned behavior data.",
        upstream_reference="CEBRA",
    ),
    BaselineCatalogEntry(
        model_id="eegpt",
        display_name="EEGPT (External Pretrained)",
        status="unavailable",
        notes=(
            "External pretrained weights not vendored; integration pending contamination audit + task-compatibility "
            "adapter. Pretrained for masked representation learning, not forecasting; checkpoint is figshare-hosted "
            "and must be mirrored before use."
        ),
        upstream_reference="EEGPT",
    ),
    BaselineCatalogEntry(
        model_id="labram_base",
        display_name="LaBraM-Base (External Pretrained)",
        status="unavailable",
        notes=(
            "External pretrained weights not vendored; integration pending contamination audit + task-compatibility "
            "adapter. Neural-token masked prediction with no raw-waveform decoder; TUSZ evaluation is contaminated "
            "by TUH pretraining."
        ),
        upstream_reference="LaBraM",
    ),
    BaselineCatalogEntry(
        model_id="cbramod",
        display_name="CBraMod (External Pretrained)",
        status="unavailable",
        notes=(
            "External pretrained weights not vendored; integration pending contamination audit + task-compatibility "
            "adapter. Masked patch reconstruction pretraining on TUEG; TUSZ evaluation is contaminated."
        ),
        upstream_reference="CBraMod",
    ),
    BaselineCatalogEntry(
        model_id="biot_prest16",
        display_name="BIOT EEG-PREST-16-channels (External Pretrained)",
        status="unavailable",
        notes=(
            "External pretrained weights not vendored; integration pending contamination audit + task-compatibility "
            "adapter. MGH resting-EEG pretraining only; the separate six-datasets checkpoint must never be "
            "substituted for it."
        ),
        upstream_reference="BIOT",
    ),
    BaselineCatalogEntry(
        model_id="biot_six_datasets",
        display_name="BIOT six-datasets (External Pretrained)",
        status="unavailable",
        notes=(
            "External pretrained weights not vendored; integration pending contamination audit + task-compatibility "
            "adapter. Pretrained on CHB-MIT and TUH corpora, so it is contaminated for CHB-MIT and TUSZ evaluation "
            "and is catalogued only to make that exclusion explicit."
        ),
        upstream_reference="BIOT",
    ),
)

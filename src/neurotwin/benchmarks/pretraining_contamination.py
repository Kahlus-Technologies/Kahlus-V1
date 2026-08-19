"""Pretraining-contamination registry for external pretrained EEG baselines.

`docs/audit/DATA_LEAKAGE_AUDIT.md` lists "Pretraining contamination" as a P2 leakage
class with the control "required before LaBraM/BIOT use". This module is that
requirement's machine-readable half: for every audited external pretrained model and
every evaluation dataset we intend to score it on, it records whether the model's
published pretraining corpus overlaps the evaluation data.

Scope and honesty boundary:

- This registry is *literature-derived metadata*, not a computed audit. Every status
  here comes from reading the upstream papers, repositories, and model cards. Nothing
  in this module hashes recordings, compares subject identifiers, or otherwise
  verifies overlap from data. A computed audit remains outstanding.
- Unknown model/dataset pairs fail closed: they resolve to ``UNVERIFIED``, never
  ``CLEAN``. Absence of evidence of contamination is not evidence of cleanliness.
- ``UNVERIFIED`` is also used where the published corpus list is incomplete or
  ambiguous, so it must be treated as blocking until a human clears it.

Dataset identifiers are the repo's canonical ids so this table can be joined against
adapter and audit artifacts without a translation layer.
"""

from __future__ import annotations

from dataclasses import dataclass


CLEAN = "clean"
CONTAMINATED = "contaminated"
UNVERIFIED = "unverified"

CONTAMINATION_STATUSES: tuple[str, ...] = (CLEAN, CONTAMINATED, UNVERIFIED)

# Canonical repo dataset ids.
# - neurotwin.adapters.sleep_edf.SLEEP_EDF_DATASET_ID
# - neurotwin.stf.public_data.CHB_MIT_DATASET_ID
# - "tuh_eeg" is the canonical TUH/TUSZ corpus id in neurotwin.adapters.registry;
#   the repo has no separate TUSZ-only adapter id (constants only, forecastability/m3).
SLEEP_EDF_EVAL_DATASET_ID = "sleep-edf-expanded-sleep-cassette"
CHB_MIT_EVAL_DATASET_ID = "chb_mit_physionet"
TUSZ_EVAL_DATASET_ID = "tuh_eeg"

EVAL_DATASET_IDS: tuple[str, ...] = (
    SLEEP_EDF_EVAL_DATASET_ID,
    CHB_MIT_EVAL_DATASET_ID,
    TUSZ_EVAL_DATASET_ID,
)

AUDITED_MODEL_IDS: tuple[str, ...] = (
    "eegpt",
    "cbramod",
    "labram_base",
    "biot_prest16",
    "biot_six_datasets",
)


@dataclass(frozen=True)
class ContaminationRecord:
    model_id: str
    dataset_id: str
    status: str
    reason: str

    def as_row(self) -> dict[str, str]:
        return {
            "model_id": self.model_id,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "reason": self.reason,
        }


_EEGPT_CORPUS = "PhysioNet-MI, HGD, TSU-Benchmark, SEED, M3CV"

CONTAMINATION_RECORDS: tuple[ContaminationRecord, ...] = (
    ContaminationRecord(
        model_id="eegpt",
        dataset_id=SLEEP_EDF_EVAL_DATASET_ID,
        status=CLEAN,
        reason=f"Published pretraining corpus ({_EEGPT_CORPUS}) contains no sleep-staging data and does not include Sleep-EDF.",
    ),
    ContaminationRecord(
        model_id="eegpt",
        dataset_id=CHB_MIT_EVAL_DATASET_ID,
        status=CLEAN,
        reason=f"Published pretraining corpus ({_EEGPT_CORPUS}) contains no seizure corpora and does not include CHB-MIT.",
    ),
    ContaminationRecord(
        model_id="eegpt",
        dataset_id=TUSZ_EVAL_DATASET_ID,
        status=CLEAN,
        reason=f"Published pretraining corpus ({_EEGPT_CORPUS}) contains no TUH-family data, so TUSZ is disjoint.",
    ),
    ContaminationRecord(
        model_id="cbramod",
        dataset_id=SLEEP_EDF_EVAL_DATASET_ID,
        status=CLEAN,
        reason="Pretrained only on TUEG clinical EEG (>9,000h); Sleep-EDF is a separate PhysioNet sleep corpus not included in TUEG.",
    ),
    ContaminationRecord(
        model_id="cbramod",
        dataset_id=CHB_MIT_EVAL_DATASET_ID,
        status=CLEAN,
        reason="Pretrained only on TUEG; CHB-MIT is a Boston Children's Hospital corpus that is not part of the TUH collection.",
    ),
    ContaminationRecord(
        model_id="cbramod",
        dataset_id=TUSZ_EVAL_DATASET_ID,
        status=CONTAMINATED,
        reason="Pretrained on TUEG (>9,000h), and TUEG is a superset of TUSZ, so TUSZ evaluation recordings were seen during pretraining.",
    ),
    ContaminationRecord(
        model_id="labram_base",
        dataset_id=SLEEP_EDF_EVAL_DATASET_ID,
        status=CLEAN,
        reason="Sleep-EDF does not appear in the published ~2,500h pretraining corpus list.",
    ),
    ContaminationRecord(
        model_id="labram_base",
        dataset_id=CHB_MIT_EVAL_DATASET_ID,
        status=UNVERIFIED,
        reason=(
            "CHB-MIT is not named in the published corpus list, but that list spans ~20 datasets plus self-collected "
            "data whose contents are not enumerated; overlap cannot be ruled out without contacting the authors."
        ),
    ),
    ContaminationRecord(
        model_id="labram_base",
        dataset_id=TUSZ_EVAL_DATASET_ID,
        status=CONTAMINATED,
        reason="The published ~2,500h pretraining mix includes TUH corpora, which cover TUSZ recordings.",
    ),
    ContaminationRecord(
        model_id="biot_prest16",
        dataset_id=SLEEP_EDF_EVAL_DATASET_ID,
        status=CLEAN,
        reason="The EEG-PREST-16-channels checkpoint is pretrained on MGH resting EEG only; Sleep-EDF is not included.",
    ),
    ContaminationRecord(
        model_id="biot_prest16",
        dataset_id=CHB_MIT_EVAL_DATASET_ID,
        status=CLEAN,
        reason="The EEG-PREST-16-channels checkpoint is pretrained on MGH resting EEG only; CHB-MIT is not included.",
    ),
    ContaminationRecord(
        model_id="biot_prest16",
        dataset_id=TUSZ_EVAL_DATASET_ID,
        status=CLEAN,
        reason="The EEG-PREST-16-channels checkpoint is pretrained on MGH resting EEG only; no TUH-family data is included.",
    ),
    ContaminationRecord(
        model_id="biot_six_datasets",
        dataset_id=SLEEP_EDF_EVAL_DATASET_ID,
        status=CLEAN,
        reason="The six-datasets pretraining mix (TUAB, TUEV, CHB-MIT, IIIC, PREST, SHHS) does not include Sleep-EDF; SHHS is a distinct sleep cohort.",
    ),
    ContaminationRecord(
        model_id="biot_six_datasets",
        dataset_id=CHB_MIT_EVAL_DATASET_ID,
        status=CONTAMINATED,
        reason="CHB-MIT is directly part of the six-datasets pretraining mix, so this checkpoint must never be evaluated on CHB-MIT.",
    ),
    ContaminationRecord(
        model_id="biot_six_datasets",
        dataset_id=TUSZ_EVAL_DATASET_ID,
        status=CONTAMINATED,
        reason=(
            "Pretraining includes TUAB and TUEV, which are TUH-family corpora drawn from the same clinical population "
            "as TUSZ; patient overlap with TUSZ is not excluded by the authors, so this is treated as contaminated."
        ),
    ),
)

_RECORD_INDEX: dict[tuple[str, str], ContaminationRecord] = {
    (record.model_id, record.dataset_id): record for record in CONTAMINATION_RECORDS
}


def contamination_status(model_id: str, dataset_id: str) -> str:
    """Return the contamination status for a model/dataset pair.

    Unknown pairs fail closed and return ``"unverified"``.
    """

    record = _RECORD_INDEX.get((model_id, dataset_id))
    return record.status if record is not None else UNVERIFIED


def contamination_reason(model_id: str, dataset_id: str) -> str:
    """Return the literature-derived reason string for a model/dataset pair."""

    record = _RECORD_INDEX.get((model_id, dataset_id))
    if record is not None:
        return record.reason
    return (
        "No audited pretraining-corpus evidence recorded for this model/dataset pair; "
        "fail closed until a human records the corpus overlap."
    )


def contamination_table() -> list[dict[str, str]]:
    """Return the full audited registry as machine-readable rows."""

    return [record.as_row() for record in CONTAMINATION_RECORDS]


def contaminated_pairs() -> list[dict[str, str]]:
    """Return only the rows that block evaluation outright."""

    return [record.as_row() for record in CONTAMINATION_RECORDS if record.status == CONTAMINATED]


def is_evaluation_permitted(model_id: str, dataset_id: str) -> bool:
    """Return True only when the pair is explicitly audited as clean."""

    return contamination_status(model_id, dataset_id) == CLEAN

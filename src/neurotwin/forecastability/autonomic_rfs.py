"""Autonomic RFS gate — residual peripheral forecastability for arousal beyond cortical spectral.

Flagship NeurIPS defendant: reformulated Y = micro-arousal in horizon h, beyond EEG spectral B.
Supports NSRR MESA (train) and SHHS (dataset-held-out test) when credentialed data is present.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.adapters.nsrr import (
    discover_nsrr_recordings,
    epoch_arousal_mask,
    load_nsrr_epoch_matrix,
    parse_nsrr_arousal_events,
)
from neurotwin.forecastability._rfs_eval import (
    horizon_payload,
    horizon_real_passes,
    horizon_synthetic_passes,
    write_json,
)
from neurotwin.forecastability.complexity_features import spectral_slope_block
from neurotwin.forecastability.m1 import _sigmoid, handcrafted_eeg_features

AUTONOMIC_RFS_SCHEMA = "kahlus.forecastability.autonomic_rfs.v1"
DEFAULT_HORIZONS = (1, 2, 4)
DEFAULT_EPOCH_SECONDS = 30.0
CLAIM_SCOPE = (
    "autonomic_residual_forecastability_micro_arousal_beyond_cortical_spectral_"
    "subject_held_out_nsrr_mesa_shhs_not_gastric_egg_not_coleman_data_not_clinical"
)


@dataclass(frozen=True)
class AutonomicRfsFixture:
    eeg_epochs: np.ndarray
    autonomic_epochs: np.ndarray
    y_by_horizon: dict[int, np.ndarray]
    subject: np.ndarray
    dataset: np.ndarray
    nuisance: np.ndarray
    base_rate: np.ndarray
    channels_used: dict[str, list[str]] | None = None


def run_autonomic_rfs_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    mesa_root: str | Path | None = None,
    shhs_root: str | Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    bootstrap_mode: str = "smoke",
    max_recordings: int | None = None,
    min_subjects: int = 8,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = make_autonomic_fixture(seed=seed, residual_signal=True, horizons=horizons)
    null = make_autonomic_fixture(seed=seed + 100, residual_signal=False, horizons=horizons)
    synthetic_known = _evaluate_fixture(known, seed=seed, horizons=horizons, bootstrap_mode=bootstrap_mode)
    synthetic_null = _evaluate_fixture(null, seed=seed + 100, horizons=horizons, bootstrap_mode=bootstrap_mode)

    mesa_status = "skipped"
    mesa_payload: dict[str, Any] | None = None
    mesa_failures: list[str] = []
    shhs_status = "skipped"
    shhs_payload: dict[str, Any] | None = None
    shhs_failures: list[str] = []

    if mesa_root is not None and Path(mesa_root).is_dir():
        try:
            mesa_fixture = load_nsrr_autonomic_fixture(
                mesa_root,
                dataset_code=0,
                dataset_name="mesa",
                horizons=horizons,
                max_recordings=max_recordings,
            )
            mesa_payload = _evaluate_fixture(
                mesa_fixture, seed=seed + 200, horizons=horizons, bootstrap_mode=bootstrap_mode
            )
            mesa_status = "evaluated"
        except Exception as exc:  # noqa: BLE001
            mesa_status = "failed"
            mesa_failures.append(f"mesa_load_failed:{exc}")
    else:
        mesa_failures.append("mesa_root_missing")

    if shhs_root is not None and Path(shhs_root).is_dir() and mesa_payload is not None:
        try:
            shhs_fixture = load_nsrr_autonomic_fixture(
                shhs_root,
                dataset_code=1,
                dataset_name="shhs",
                horizons=horizons,
                max_recordings=max_recordings,
            )
            shhs_payload = _evaluate_fixture(
                shhs_fixture, seed=seed + 300, horizons=horizons, bootstrap_mode=bootstrap_mode
            )
            shhs_status = "evaluated"
        except Exception as exc:  # noqa: BLE001
            shhs_status = "failed"
            shhs_failures.append(f"shhs_load_failed:{exc}")
    elif shhs_root is None or not Path(shhs_root).is_dir():
        shhs_failures.append("shhs_root_missing")

    synthetic_ok = horizon_synthetic_passes(synthetic_known, synthetic_null)
    mesa_ok = horizon_real_passes(mesa_payload, min_subjects=min_subjects, min_positive_events=100)
    if mesa_status == "evaluated":
        if mesa_ok:
            gate_passed = synthetic_ok and mesa_ok
            stop_reason = (
                "Autonomic RFS gate passed on synthetic known/null and powered MESA cohort"
                if gate_passed
                else "Autonomic RFS gate failed powered criteria on MESA; do not claim autonomic residual forecastability."
            )
        else:
            gate_passed = False
            stop_reason = (
                "Autonomic RFS honest negative on MESA under held-out evaluation; "
                "do not claim autonomic residual forecastability beyond cortical spectral baseline."
            )
    elif mesa_status in {"failed", "skipped"}:
        gate_passed = synthetic_ok and mesa_status != "failed"
        stop_reason = (
            "Synthetic fixture validated; MESA real cohort not available — "
            "obtain NSRR credentialed access before claim-grade autonomic RFS."
            if mesa_status == "skipped"
            else "MESA load failed; do not claim autonomic residual forecastability on real data."
        )
    else:
        gate_passed = synthetic_ok
        stop_reason = "Synthetic-only evaluation."

    gate = {
        "schema": AUTONOMIC_RFS_SCHEMA,
        "milestone": "autonomic_rfs_arousal",
        "claim_scope": CLAIM_SCOPE,
        "stop_reason": stop_reason,
        "gate_passed": gate_passed,
        "horizons_epochs": list(horizons),
        "epoch_seconds": DEFAULT_EPOCH_SECONDS,
        "bootstrap_mode": bootstrap_mode,
        "min_subjects": min_subjects,
        "synthetic_known": synthetic_known,
        "synthetic_null": synthetic_null,
        "mesa_status": mesa_status,
        "mesa_real": mesa_payload,
        "mesa_failures": mesa_failures,
        "shhs_status": shhs_status,
        "shhs_dataset_held_out": shhs_payload,
        "shhs_failures": shhs_failures,
        "estimand": {
            "y": "micro_arousal_within_horizon_epochs",
            "b": "cortical_spectral_plus_cycle_history",
            "z": "autonomic_hrv_resp_eog_emg_infraslow",
            "split": "subject_held_out_primary_dataset_held_out_secondary",
        },
    }
    write_json(out / "autonomic_rfs_report.json", gate)
    _write_report(out / "AUTONOMIC_RFS_EVIDENCE_REPORT.md", gate)
    return gate


def make_autonomic_fixture(
    *,
    seed: int,
    residual_signal: bool,
    strength: float = 1.0,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    n_subjects: int = 16,
    epochs_per_subject: int = 180,
) -> AutonomicRfsFixture:
    rng = np.random.default_rng(seed)
    eeg_epochs: list[np.ndarray] = []
    autonomic_epochs: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    datasets: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    max_h = max(horizons)
    samples = 128
    t_axis = np.linspace(0.0, 1.0, samples, dtype=np.float32)
    for subject in range(n_subjects):
        dataset_code = 0 if subject % 3 else 1
        precursor = 0.0
        arousal_events: list[int] = [0]
        raw_eeg: list[np.ndarray] = []
        raw_auto: list[np.ndarray] = []
        for epoch in range(epochs_per_subject + max_h):
            cycle = np.sin(2.0 * np.pi * epoch / 30.0)
            precursor = 0.85 * precursor + rng.normal(0.0, 0.4)
            visible = precursor if residual_signal else rng.normal(0.0, 1.0)
            # `strength` scales the injected autonomic->future-arousal coupling on
            # the residual path (default 1.0 = unchanged). strength=0 collapses it
            # to a null; graded values drive a calibration sweep.
            scale = strength if residual_signal else 1.0
            eeg = rng.normal(0.0, 0.15, size=(samples, 1)).astype(np.float32)
            eeg[:, 0] += 0.6 * cycle * np.sin(2.0 * np.pi * 4.0 * t_axis)
            auto = rng.normal(0.0, 0.12, size=(samples, 4)).astype(np.float32)
            auto[:, 0] += 1.2 * scale * visible * np.sin(2.0 * np.pi * 1.5 * t_axis)
            auto[:, 1] += 0.9 * scale * visible * np.cos(2.0 * np.pi * 0.8 * t_axis)
            auto[:, 2] += 0.7 * scale * visible * np.linspace(-1.0, 1.0, samples)
            auto[:, 3] += 0.8 * scale * visible * np.sin(2.0 * np.pi * 2.2 * t_axis)
            logit = -2.2 + 0.8 * cycle + 1.4 * float(np.mean(arousal_events[-6:])) + (
                2.8 * strength * precursor if residual_signal else 0.0
            )
            event = int(rng.random() < _sigmoid(logit))
            raw_eeg.append(eeg)
            raw_auto.append(auto)
            arousal_events.append(event)
        event_arr = np.asarray(arousal_events[1:], dtype=np.int64)
        recent_changes = [0]
        for epoch in range(epochs_per_subject):
            for h in horizons:
                y_maps[h].append(int(np.any(event_arr[epoch + 1 : epoch + h + 1])))
            eeg_epochs.append(raw_eeg[epoch])
            autonomic_epochs.append(raw_auto[epoch])
            subjects.append(subject)
            datasets.append(dataset_code)
            recent = float(np.mean(recent_changes[-12:]))
            cycle = np.sin(2.0 * np.pi * epoch / 30.0)
            nuisance.append(
                [1.0, cycle, np.cos(2.0 * np.pi * epoch / 30.0), recent, float(subject) / max(1, n_subjects)]
            )
            base_rates.append(float(np.mean(arousal_events[max(0, epoch - 15) : epoch + 1])))
            recent_changes.append(int(arousal_events[epoch + 1] != arousal_events[epoch]))
    return AutonomicRfsFixture(
        eeg_epochs=np.asarray(eeg_epochs, dtype=np.float32),
        autonomic_epochs=np.asarray(autonomic_epochs, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
    )


#: Which per-epoch mask from the adapter supplies Y for each requested class.
_AROUSAL_MASK_KEY = {
    "all": "arousal",
    "respiratory": "arousal_respiratory",
    "non_respiratory": "arousal_non_respiratory",
}


def load_slpdb_autonomic_fixture(
    *,
    records: tuple[str, ...] | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_records: int | None = None,
    dataset_code: int = 2,
    arousal_class: str = "all",
    data_dir: str | None = None,
    target: str = "arousal",
) -> AutonomicRfsFixture:
    """Build the autonomic fixture from MIT-BIH slpdb (open, no credentialing).

    Same estimand and nuisance construction as the NSRR path so the two are
    comparable. Subjects are grouped by slpdb's numeric prefix, so the two
    two-part recordings (``slp01a``/``slp01b``, ``slp02a``/``slp02b``) cannot put
    one person on both sides of a subject-held-out split.

    ``arousal_class`` selects which arousals become Y:

    ``non_respiratory``
        Bare and leg-movement arousals. The class where autonomic precedence is
        actually supported (Bonnet & Arand; Togo et al.; Allena et al.).
    ``respiratory``
        Arousals terminating a hypopnea or apnea. Precedence evidence here is
        weak, and Bahr-Hamm et al. (2022) report no autonomic coupling at all.
    ``all``
        Pooled. Reported for continuity with earlier runs, but it averages the
        two classes above and so is the least interpretable of the three.

    Y and its base rate follow the selected class; the nuisance block keeps the
    *pooled* recent-arousal history, since all arousals are observable to the
    baseline regardless of which class is being predicted. That makes B stronger
    and the residual test more conservative.
    """
    if arousal_class not in _AROUSAL_MASK_KEY:
        raise ValueError(
            f"arousal_class must be one of {sorted(_AROUSAL_MASK_KEY)}, got {arousal_class!r}"
        )
    if target not in {"arousal", "rem"}:
        raise ValueError(f"target must be 'arousal' or 'rem', got {target!r}")
    import wfdb

    from neurotwin.adapters.slpdb import load_slpdb_record, subject_id_from_record

    if data_dir is not None:
        # Only records whose signal and annotation files are both present locally.
        local = Path(data_dir)
        available = {p.stem for p in local.glob("*.dat")} & {p.stem for p in local.glob("*.st")}
        names = sorted(available) if records is None else [r for r in records if r in available]
    else:
        names = list(records) if records is not None else list(wfdb.get_record_list("slpdb"))
    if max_records is not None:
        names = names[:max_records]
    subject_codes: dict[str, int] = {}
    for name in names:
        subject_codes.setdefault(subject_id_from_record(name), len(subject_codes))

    eeg_epochs: list[np.ndarray] = []
    autonomic_epochs: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    datasets: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    channels_used: dict[str, list[str]] = {}
    skipped: list[str] = []
    max_h = max(horizons)

    for name in names:
        try:
            rec = load_slpdb_record(name, data_dir=data_dir)
        except Exception as exc:  # noqa: BLE001 - skip a bad record, keep the cohort
            skipped.append(f"{name}:{exc}")
            continue
        # Y is the requested class; the pooled mask still feeds the nuisance block
        # so B sees all observable arousal history regardless of class. That makes
        # B stronger and the residual test correspondingly more conservative.
        pooled_mask = rec["arousal"]
        n_epochs = rec["n_epochs"]
        if target == "rem":
            # Positive control. slpdb staging is R&K-scored from EEG (C4-A1) only,
            # so Z (ECG / BP / Resp) is disjoint from the scorer's input and a
            # certification here cannot be the model re-reading the scorer's own
            # evidence. Wake and movement-time epochs are dropped so the contrast
            # is REM vs NREM rather than REM vs everything.
            stages = rec["stages"]
            arousal_mask = np.asarray([int(s == "R") for s in stages], dtype=np.int64)
            sleep = np.asarray([s not in {"W", "MT", ""} for s in stages], dtype=bool)
        else:
            arousal_mask = rec[_AROUSAL_MASK_KEY[arousal_class]]
            sleep = np.ones(n_epochs, dtype=bool)
        if n_epochs < max_h + 4:
            skipped.append(f"{name}:too_few_epochs")
            continue
        channels_used.setdefault("slpdb", [])
        for label in rec["channels"]:
            if label not in channels_used["slpdb"]:
                channels_used["slpdb"].append(label)
        subj = subject_codes[rec["subject"]]
        recent_changes = [0]
        for epoch in range(n_epochs - max_h):
            if not sleep[epoch]:
                continue
            for h in horizons:
                y_maps[h].append(
                    int(arousal_mask[epoch])
                    if h == 0
                    else int(np.any(arousal_mask[epoch + 1 : epoch + h + 1]))
                )
            eeg_epochs.append(_resample_epoch(rec["eeg_epochs"][epoch], target_len=128))
            autonomic_epochs.append(_resample_epoch(rec["autonomic_epochs"][epoch], target_len=128))
            subjects.append(subj)
            datasets.append(dataset_code)
            recent = float(np.mean(recent_changes[-12:]))
            nuisance.append(
                [
                    1.0,
                    np.sin(2.0 * np.pi * epoch / max(1, n_epochs)),
                    np.cos(2.0 * np.pi * epoch / max(1, n_epochs)),
                    recent,
                    float(pooled_mask[epoch]),
                ]
            )
            base_rates.append(float(np.mean(arousal_mask[max(0, epoch - 10) : epoch + 1])))
            recent_changes.append(int(pooled_mask[epoch] != pooled_mask[max(0, epoch - 1)]))

    if not eeg_epochs:
        raise ValueError(f"no usable slpdb windows parsed (skipped: {skipped})")
    if len(set(subjects)) < 3:
        raise ValueError(f"need at least 3 slpdb subjects, got {len(set(subjects))}")
    if skipped:
        channels_used["skipped_records"] = skipped
    return AutonomicRfsFixture(
        eeg_epochs=np.asarray(eeg_epochs, dtype=np.float32),
        autonomic_epochs=np.asarray(autonomic_epochs, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
        channels_used=channels_used,
    )


def load_nsrr_autonomic_fixture(
    root: str | Path,
    *,
    dataset_code: int,
    dataset_name: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    max_recordings: int | None = None,
    epoch_seconds: float = DEFAULT_EPOCH_SECONDS,
) -> AutonomicRfsFixture:
    recordings = discover_nsrr_recordings(root, dataset=dataset_name)
    if max_recordings is not None:
        recordings = recordings[:max_recordings]
    if len(recordings) < 3:
        raise ValueError(f"need at least 3 NSRR recordings in {root}, found {len(recordings)}")
    subject_codes = {
        rec.subject_id: idx for idx, rec in enumerate({r.subject_id: r for r in recordings}.values())
    }
    eeg_epochs: list[np.ndarray] = []
    autonomic_epochs: list[np.ndarray] = []
    y_maps: dict[int, list[int]] = {h: [] for h in horizons}
    subjects: list[int] = []
    datasets: list[int] = []
    nuisance: list[list[float]] = []
    base_rates: list[float] = []
    channels_used: dict[str, list[str]] = {}
    max_h = max(horizons)
    # Montage is locked to the first recording so every window has the same channel
    # count (some SHHS1 nights carry an extra AIRFLOW channel). Recordings missing a
    # locked channel are skipped rather than zero-filled.
    locked: dict[str, tuple[str, ...]] | None = None
    skipped: list[str] = []
    for rec in recordings:
        events = parse_nsrr_arousal_events(rec.xml_path)
        try:
            matrix = load_nsrr_epoch_matrix(rec.edf_path, epoch_seconds=epoch_seconds, channel_groups=locked)
        except ValueError:
            skipped.append(rec.subject_id)
            continue
        if locked is None:
            locked = {g: tuple(labels) for g, labels in matrix["channels"].items()}
        elif {g: tuple(v) for g, v in matrix["channels"].items()} != locked:
            skipped.append(rec.subject_id)
            continue
        for group, labels in matrix["channels"].items():
            channels_used.setdefault(group, [])
            for label in labels:
                if label not in channels_used[group]:
                    channels_used[group].append(label)
        n_epochs = matrix["n_epochs"]
        arousal_mask = epoch_arousal_mask(events, n_epochs=n_epochs, epoch_seconds=epoch_seconds)
        eeg_stack = _stack_epoch_channels(matrix["epochs"], ("eeg",))
        auto_stack = _stack_epoch_channels(matrix["epochs"], ("ecg", "eog", "emg", "resp"))
        if eeg_stack.shape[0] < max_h + 4:
            continue
        subj = subject_codes[rec.subject_id]
        recent_changes = [0]
        for epoch in range(n_epochs - max_h):
            for h in horizons:
                y_maps[h].append(int(np.any(arousal_mask[epoch + 1 : epoch + h + 1])))
            eeg_epochs.append(_resample_epoch(eeg_stack[epoch], target_len=128))
            autonomic_epochs.append(_resample_epoch(auto_stack[epoch], target_len=128))
            subjects.append(subj)
            datasets.append(dataset_code)
            recent = float(np.mean(recent_changes[-12:]))
            nuisance.append(
                [
                    1.0,
                    np.sin(2.0 * np.pi * epoch / max(1, n_epochs)),
                    np.cos(2.0 * np.pi * epoch / max(1, n_epochs)),
                    recent,
                    float(arousal_mask[epoch]),
                ]
            )
            base_rates.append(float(np.mean(arousal_mask[max(0, epoch - 10) : epoch + 1])))
            recent_changes.append(int(arousal_mask[epoch] != arousal_mask[max(0, epoch - 1)]))
    if not eeg_epochs:
        raise ValueError(f"no usable NSRR autonomic windows parsed from {root}")
    if len(set(subjects)) < 3:
        raise ValueError("need at least 3 subjects with usable NSRR windows")
    return AutonomicRfsFixture(
        eeg_epochs=np.asarray(eeg_epochs, dtype=np.float32),
        autonomic_epochs=np.asarray(autonomic_epochs, dtype=np.float32),
        y_by_horizon={h: np.asarray(y_maps[h], dtype=np.int64) for h in horizons},
        subject=np.asarray(subjects, dtype=np.int64),
        dataset=np.asarray(datasets, dtype=np.int64),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        base_rate=np.asarray(base_rates, dtype=np.float32),
        channels_used=channels_used,
    )


def _stack_epoch_channels(epochs: dict[str, np.ndarray], keys: tuple[str, ...]) -> np.ndarray:
    """Stack modality groups on the channel axis -> (n_epochs, samples, total_channels)."""
    parts = [epochs[key] for key in keys if key in epochs]
    if not parts:
        raise ValueError(f"missing epoch channels for keys {keys}")
    return np.concatenate(parts, axis=2)


def _resample_epoch(epoch: np.ndarray, *, target_len: int) -> np.ndarray:
    """Resample ``(samples, channels)`` epoch to ``target_len`` samples."""
    if epoch.ndim != 2:
        raise ValueError(f"expected 2-D epoch (samples, channels), got shape {epoch.shape}")
    if epoch.shape[0] == target_len:
        return epoch.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, epoch.shape[0])
    x_new = np.linspace(0.0, 1.0, target_len)
    out = np.zeros((target_len, epoch.shape[1]), dtype=np.float32)
    for ch in range(epoch.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, epoch[:, ch])
    return out


def autonomic_feature_block(epochs: np.ndarray) -> np.ndarray:
    """Per-channel infraslow amplitude features (std, MAD, IQR, RMS-diff) for HRV/resp/EOG/EMG.

    Kept separate from handcrafted_eeg_features because autonomic Z needs
    time-domain envelope stats rather than spectral bandpower.
    """
    n = epochs.shape[0]
    feats: list[list[float]] = []
    for idx in range(n):
        window = epochs[idx]
        row: list[float] = []
        for ch in range(window.shape[1]):
            signal = window[:, ch].astype(np.float64)
            diff = np.diff(signal)
            row.extend(
                [
                    float(np.std(signal)),
                    float(np.mean(np.abs(diff))),
                    float(np.percentile(signal, 90) - np.percentile(signal, 10)),
                    float(np.sqrt(np.mean(diff**2))) if diff.size else 0.0,
                ]
            )
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def _evaluate_fixture(
    fixture: AutonomicRfsFixture,
    *,
    seed: int,
    horizons: tuple[int, ...],
    bootstrap_mode: str,
) -> dict[str, Any]:
    eeg_3d = fixture.eeg_epochs[:, :, None] if fixture.eeg_epochs.ndim == 2 else fixture.eeg_epochs
    b_matrix = np.concatenate([handcrafted_eeg_features(eeg_3d), spectral_slope_block(eeg_3d)], axis=1)
    z_matrix = autonomic_feature_block(fixture.autonomic_epochs)
    nuisance_b = np.concatenate([fixture.nuisance, fixture.base_rate[:, None], b_matrix], axis=1)
    rows = [
        horizon_payload(
            y=fixture.y_by_horizon[h],
            nuisance_b=nuisance_b,
            z=z_matrix,
            subject=fixture.subject,
            seed=seed + h,
            horizon=h,
            bootstrap_mode=bootstrap_mode,
        )
        for h in horizons
    ]
    return {
        "positive_events": int(sum(int(np.sum(fixture.y_by_horizon[h])) for h in horizons)),
        "n_windows": int(len(fixture.subject)),
        "n_subjects": int(len(set(fixture.subject.tolist()))),
        "n_datasets": int(len(set(fixture.dataset.tolist()))),
        "channels_used": fixture.channels_used,
        "horizons": rows,
    }


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Autonomic RFS arousal gate",
        "",
        f"- claim_scope: `{gate['claim_scope']}`",
        f"- stop_reason: {gate['stop_reason']}",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- mesa_status: `{gate['mesa_status']}`",
        f"- shhs_status: `{gate['shhs_status']}`",
        f"- bootstrap_mode: `{gate['bootstrap_mode']}`",
        f"- epoch_seconds: {gate['epoch_seconds']}",
        "",
        "## Estimand",
        "",
        f"- Y: {gate['estimand']['y']}",
        f"- B: {gate['estimand']['b']}",
        f"- Z: {gate['estimand']['z']}",
        "",
        "## Synthetic known / null",
        "",
        f"- known primary RFS bits: {gate['synthetic_known']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        f"- null primary RFS bits: {gate['synthetic_null']['horizons'][0]['residual_model']['rfs_bits']:.4f}",
        "",
        "## MESA real cohort",
        "",
    ]
    if gate["mesa_real"] is None:
        lines.append(f"- not run: {gate['mesa_failures']}")
    else:
        real = gate["mesa_real"]
        lines.append(f"- windows: {real['n_windows']} · subjects: {real['n_subjects']}")
        if real.get("channels_used"):
            lines.append(f"- channels: {real['channels_used']}")
        row = real["horizons"][0]
        lines.append(
            f"- h={row['horizon_epochs']}: positive_events={row['positive_events']} "
            f"residual_rfs_bits={row['residual_model']['rfs_bits']:.4f} "
            f"ci=[{row['residual_model'].get('rfs_ci_low', float('nan')):.4f}, "
            f"{row['residual_model'].get('rfs_ci_high', float('nan')):.4f}]"
        )
    lines.extend(["", "## SHHS dataset-held-out", ""])
    if gate["shhs_dataset_held_out"] is None:
        lines.append(f"- not run: {gate['shhs_failures']}")
    else:
        shhs = gate["shhs_dataset_held_out"]
        row = shhs["horizons"][0]
        lines.append(
            f"- windows: {shhs['n_windows']} · subjects: {shhs['n_subjects']} · "
            f"RFS bits: {row['residual_model']['rfs_bits']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

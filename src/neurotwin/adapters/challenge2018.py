"""PhysioNet/CinC Challenge 2018 adapter — "You Snooze, You Win".

Dataset: https://physionet.org/content/challenge-2018/1.0.0/
License: Open Data Commons Attribution License v1.0

994 training subjects (labels available), 989 test subjects (labels held out).
Each record: EEG (F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1), EOG (E1-M2),
EMG (Chin1-Chin2), respiration (ABD, CHEST, AIRFLOW), SaO2, ECG — all at 200 Hz.

Arousal vocabulary (from aux_note, prefix-stripped):
    arousal_spontaneous   — spontaneous cortical arousal; **primary target class**
    arousal_rera          — respiratory effort related arousal (RERA)
    resp_centralapnea     — central apnea (no autonomic pre-cursor expected)
    resp_obstructiveapnea — obstructive apnea
    resp_hypopnea         — hypopnea
    resp_mixedapnea       — mixed apnea

Design notes
------------
* This adapter **streams** raw signals and emits only epoch-level feature arrays
  (~2 kB/subject vs ~139 MB/subject raw). It never writes raw signal data to disk.
* Annotations use open/close bracket syntax: ``(arousal_spontaneous`` opens an
  interval, ``arousal_spontaneous)`` closes it. We convert to per-epoch binary
  labels by finding all epochs that overlap any arousal interval of each class.
* Subject ID is the filename stem (e.g. ``tr03-0005``). No two records share
  a subject — unlike slpdb there are no repeat-night duplicates.
* EEG channels: challenge uses referential montage (C3-M2, C4-M1 = central vs
  mastoid), equivalent to slpdb's C4-A1. We pool C3-M2 and C4-M1 by averaging.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

EPOCH_SECONDS = 30.0
EEG_TARGET_HZ = 32.0   # keeps spindle / sigma band resolvable after decimation
AUTONOMIC_HZ = 1.0      # HR, resp envelope on 1 Hz grid

# Arousal classes relevant to the autonomic-precedence hypothesis
SPONTANEOUS_TOKEN = "arousal_spontaneous"
RERA_TOKEN = "arousal_rera"
RESPIRATORY_TOKENS = frozenset({
    "resp_centralapnea",
    "resp_obstructiveapnea",
    "resp_hypopnea",
    "resp_mixedapnea",
})
NON_RESPIRATORY_TOKENS = frozenset({SPONTANEOUS_TOKEN, RERA_TOKEN})
ALL_AROUSAL_TOKENS = RESPIRATORY_TOKENS | NON_RESPIRATORY_TOKENS

# Canonical PhysioNet paths (pn_dir prefix)
_PN_VERSION = "challenge-2018/1.0.0"


def subject_id_from_record(rec_path: str) -> str:
    """Return the subject ID from a full record path like ``training/tr03-0005/tr03-0005``.

    The stem (``tr03-0005``) is unique per subject in this dataset.
    """
    return Path(rec_path).name


def list_training_records() -> list[str]:
    """Return all 994 training record paths from PhysioNet (network call)."""
    import wfdb
    recs = wfdb.get_record_list(_PN_VERSION)
    return [r for r in recs if r.startswith("training/")]


def _parse_arousal_epochs(
    ann_samples: np.ndarray,
    aux_notes: list[str],
    fs: float,
    n_epochs: int,
) -> dict[str, np.ndarray]:
    """Convert sample-level arousal interval annotations to per-epoch binary masks.

    Returns a dict with keys:
        ``arousal_spontaneous``, ``arousal_rera``, ``arousal_respiratory``,
        ``arousal_any`` — each a (n_epochs,) int64 array, 1 if the epoch
        overlaps the arousal interval, 0 otherwise.
    """
    epoch_samples = int(EPOCH_SECONDS * fs)

    spontaneous = np.zeros(n_epochs, dtype=np.int64)
    rera = np.zeros(n_epochs, dtype=np.int64)
    respiratory = np.zeros(n_epochs, dtype=np.int64)

    open_token: str | None = None
    open_sample: int = 0

    def _mark(token: str, start_s: int, end_s: int) -> None:
        ep_start = start_s // epoch_samples
        ep_end = min(n_epochs - 1, end_s // epoch_samples)
        if ep_start > ep_end or ep_start >= n_epochs:
            return
        if token == SPONTANEOUS_TOKEN:
            spontaneous[ep_start : ep_end + 1] = 1
        elif token == RERA_TOKEN:
            rera[ep_start : ep_end + 1] = 1
        elif token in RESPIRATORY_TOKENS:
            respiratory[ep_start : ep_end + 1] = 1

    for sample, note in zip(ann_samples, aux_notes):
        n = note.strip().strip("\x00")
        if n.startswith("("):
            # Opening bracket: begin interval
            token = n[1:]
            if token in ALL_AROUSAL_TOKENS:
                open_token = token
                open_sample = int(sample)
        elif open_token is not None and n == open_token + ")":
            # Closing bracket: mark all epochs in interval
            _mark(open_token, open_sample, int(sample))
            open_token = None
        # else: sleep stage label or unmatched token — ignore

    any_arousal = ((spontaneous | rera | respiratory) > 0).astype(np.int64)
    return {
        "arousal_spontaneous": spontaneous,
        "arousal_rera": rera,
        "arousal_respiratory": respiratory,
        "arousal_any": any_arousal,
    }


def _parse_stages(
    ann_samples: np.ndarray, aux_notes: list[str], n_epochs: int, fs: float = 200.0
) -> list[str]:
    """Map per-annotation stage labels onto per-epoch slots.

    The challenge annotates stages as standalone tokens (W, N1, N2, N3, R).
    Each annotation marks a transition: we apply it from that epoch onward until
    overwritten by a later annotation (forward-fill).
    """
    STAGE_TOKENS = frozenset({"W", "N1", "N2", "N3", "R"})
    epoch_samples = int(EPOCH_SECONDS * fs)
    # Collect (epoch_index, stage) pairs in order
    transitions: list[tuple[int, str]] = []
    for sample, note in zip(ann_samples, aux_notes):
        n = note.strip().strip("\x00")
        if n in STAGE_TOKENS:
            ep = min(int(sample) // epoch_samples, n_epochs - 1)
            transitions.append((ep, n))

    if not transitions:
        return ["?"] * n_epochs

    stages: list[str] = ["?"] * n_epochs
    # Apply each transition from its epoch to the next transition
    for k, (ep, stage) in enumerate(transitions):
        end = transitions[k + 1][0] if k + 1 < len(transitions) else n_epochs
        for i in range(ep, min(end, n_epochs)):
            stages[i] = stage

    # Back-fill any leading "?" before the first annotation
    if stages[0] == "?":
        first_stage = transitions[0][1]
        for i in range(transitions[0][0]):
            stages[i] = first_stage

    return stages


def _decimate(signal: np.ndarray, src_hz: float, target_hz: float) -> np.ndarray:
    """Mean-pool to ``target_hz`` (anti-aliases better than striding)."""
    factor = max(1, int(round(src_hz / target_hz)))
    usable = (len(signal) // factor) * factor
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return signal[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)


def _heart_rate_per_second(ecg: np.ndarray, fs: float, n_seconds: int) -> np.ndarray:
    """Instantaneous HR (bpm) on 1 Hz grid. Falls back to log-power envelope."""
    try:
        import wfdb.processing as wp
        beats = wp.xqrs_detect(sig=ecg, fs=fs, verbose=False)
        if len(beats) >= 10:
            times = beats / fs
            rate = 60.0 / np.diff(times)
            grid = np.interp(
                np.arange(n_seconds, dtype=np.float64),
                times[1:],
                rate,
                left=float(rate[0]),
                right=float(rate[-1]),
            )
            return grid.astype(np.float32)
    except Exception:  # noqa: BLE001
        pass
    env = _decimate(np.abs(ecg).astype(np.float32), fs, AUTONOMIC_HZ)
    out = np.zeros(n_seconds, dtype=np.float32)
    take = min(n_seconds, len(env))
    out[:take] = np.log(env[:take] + 1e-8)
    return out


def _envelope_per_second(signal: np.ndarray, fs: float, n_seconds: int) -> np.ndarray:
    env = _decimate(np.abs(signal).astype(np.float32), fs, AUTONOMIC_HZ)
    out = np.zeros(n_seconds, dtype=np.float32)
    take = min(n_seconds, len(env))
    out[:take] = np.log(env[:take] + 1e-8)
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    mean, std = float(np.mean(x)), float(np.std(x))
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def _pn_dir_for(rec_path: str) -> tuple[str, str]:
    """Split ``training/tr03-0005/tr03-0005`` into (pn_dir, rec_name).

    wfdb.rdrecord(rec_name, pn_dir=pn_dir) fetches from PhysioNet.
    """
    parts = rec_path.split("/")
    # e.g. training/tr03-0005 relative to the version root
    pn_dir = _PN_VERSION + "/" + "/".join(parts[:-1])
    return pn_dir, parts[-1]


def load_challenge2018_record(
    rec_path: str,
    *,
    data_dir: str | None = None,
    eeg_channels: list[str] | None = None,
) -> dict[str, Any]:
    """Stream one challenge-2018 record and return epoch-level feature arrays.

    Parameters
    ----------
    rec_path:
        Full record path as returned by ``list_training_records()``, e.g.
        ``"training/tr03-0005/tr03-0005"``.
    data_dir:
        If provided, read from a local directory that mirrors the PhysioNet
        layout (``<data_dir>/training/tr03-0005/tr03-0005.hea`` etc.).
        Otherwise streams from PhysioNet (slow but zero disk cost).
    eeg_channels:
        EEG channel names to average. Defaults to ``["C3-M2", "C4-M1"]``
        (central channels, comparable to slpdb's C4-A1).

    Returns
    -------
    dict with keys:
        record, subject, n_epochs,
        eeg_epochs    shape (n_epochs, eeg_samples_per_epoch, 1) float32,
        autonomic_epochs  shape (n_epochs, 30, 4) float32  [HR, ABD, CHEST, AIRFLOW],
        arousal_spontaneous, arousal_rera, arousal_respiratory, arousal_any
            each (n_epochs,) int64,
        stages  list[str],
        channels  list[str],
    """
    import wfdb

    if eeg_channels is None:
        eeg_channels = ["C3-M2", "C4-M1"]

    if data_dir is not None:
        # Local mirror: data_dir/training/tr03-0005/tr03-0005.hea
        rec_file = str(Path(data_dir) / rec_path)
        rec = wfdb.rdrecord(rec_file)
        ann = wfdb.rdann(rec_file, "arousal")
    else:
        pn_dir, rec_name = _pn_dir_for(rec_path)
        rec = wfdb.rdrecord(rec_name, pn_dir=pn_dir)
        ann = wfdb.rdann(rec_name, "arousal", pn_dir=pn_dir)

    fs = float(rec.fs)
    names = list(rec.sig_name)

    def channel(want: str) -> np.ndarray | None:
        for idx, name in enumerate(names):
            if name.lower() == want.lower():
                return np.nan_to_num(rec.p_signal[:, idx].astype(np.float32))
        return None

    def channel_contains(want: str) -> np.ndarray | None:
        for idx, name in enumerate(names):
            if want.lower() in name.lower():
                return np.nan_to_num(rec.p_signal[:, idx].astype(np.float32))
        return None

    # EEG: average requested central channels
    eeg_arrays = [channel(c) for c in eeg_channels]
    eeg_arrays = [a for a in eeg_arrays if a is not None]
    if not eeg_arrays:
        raise ValueError(f"{rec_path}: none of {eeg_channels} found in {names}")
    eeg = np.mean(np.stack(eeg_arrays, axis=0), axis=0).astype(np.float32)

    ecg = channel("ECG") or channel_contains("ECG")
    abd = channel("ABD")
    chest = channel("CHEST")
    airflow = channel("AIRFLOW")

    if ecg is None:
        raise ValueError(f"{rec_path}: ECG channel missing (have {names})")

    n_epochs = int(len(eeg) // (EPOCH_SECONDS * fs))
    if n_epochs < 20:
        raise ValueError(f"{rec_path}: only {n_epochs} epochs, too short")

    n_seconds = int(n_epochs * EPOCH_SECONDS)

    # Autonomic features on 1 Hz grid
    hr = _heart_rate_per_second(ecg, fs, n_seconds)
    abd_env = _envelope_per_second(abd, fs, n_seconds) if abd is not None else np.zeros(n_seconds, np.float32)
    chest_env = _envelope_per_second(chest, fs, n_seconds) if chest is not None else np.zeros(n_seconds, np.float32)
    airflow_env = _envelope_per_second(airflow, fs, n_seconds) if airflow is not None else np.zeros(n_seconds, np.float32)

    # EEG epochs at EEG_TARGET_HZ
    eeg_ds = _decimate(eeg, fs, EEG_TARGET_HZ)
    eeg_per_epoch = int(EPOCH_SECONDS * EEG_TARGET_HZ)
    needed = n_epochs * eeg_per_epoch
    if len(eeg_ds) < needed:
        eeg_ds = np.pad(eeg_ds, (0, needed - len(eeg_ds)))
    eeg_epochs = _zscore(eeg_ds[:needed]).reshape(n_epochs, eeg_per_epoch, 1)

    # Autonomic epochs (30 seconds × 4 channels)
    sec_per_epoch = int(EPOCH_SECONDS)
    auto = np.stack([
        _zscore(hr),
        _zscore(abd_env),
        _zscore(chest_env),
        _zscore(airflow_env),
    ], axis=1)  # (n_seconds, 4)
    auto_epochs = auto[: n_epochs * sec_per_epoch].reshape(n_epochs, sec_per_epoch, 4)

    # Labels
    arousal_masks = _parse_arousal_epochs(ann.sample, ann.aux_note, fs, n_epochs)
    stages = _parse_stages(ann.sample, ann.aux_note, n_epochs, fs)

    return {
        "record": rec_path,
        "subject": subject_id_from_record(rec_path),
        "n_epochs": n_epochs,
        "eeg_epochs": eeg_epochs.astype(np.float32),
        "autonomic_epochs": auto_epochs.astype(np.float32),
        "stages": stages,
        "channels": names,
        **arousal_masks,
    }

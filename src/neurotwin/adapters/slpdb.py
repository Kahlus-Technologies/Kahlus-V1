"""MIT-BIH Polysomnographic Database (slpdb) adapter for the autonomic RFS gate.

slpdb is the only fully open dataset with simultaneous cortical EEG and cardiac
signal plus scored arousals, so it can carry the F6 estimand (autonomic block Z
predicting a future arousal Y beyond cortical baseline B) while NSRR MESA access
is pending. 18 records, 250 Hz, ECG / BP / EEG (C4-A1) / Resp.

Two facts about this dataset drive the implementation:

1. **18 records but 16 subjects.** ``slp01a``/``slp01b`` and ``slp02a``/``slp02b``
   are the same person recorded twice. Grouping by record would put one person on
   both sides of a subject-held-out split -- the nights-as-identities leak. The
   subject id is therefore the numeric prefix, never the record name.

2. **Arousals are encoded as a suffix on the event token**, not a separate
   annotation. ``LA`` is a leg movement *with* arousal while ``L`` is one without;
   likewise ``HA``/``H`` for hypopnea and ``CAA``/``CA`` for central apnea. ``X``
   is obstructive apnea with arousal and ``A`` is a bare arousal. Only the
   arousal-marked tokens count toward Y.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

EPOCH_SECONDS = 30.0
#: Respiratory-event arousals. Autonomic precedence is weakly supported here --
#: airway reopening, hypoxia relief and sympathetic discharge land within seconds
#: of each other. Keep separate from the class below; see docs/research.
RESPIRATORY_AROUSAL_TOKENS = frozenset({"HA", "CAA", "X"})
#: Bare and leg-movement arousals. The class precedence evidence actually covers.
NON_RESPIRATORY_AROUSAL_TOKENS = frozenset({"A", "LA"})
#: Annotation tokens that mark an arousal of either class (see module docstring).
AROUSAL_TOKENS = RESPIRATORY_AROUSAL_TOKENS | NON_RESPIRATORY_AROUSAL_TOKENS
#: Effective EEG rate after decimation; keeps spindle/sigma band resolvable.
EEG_TARGET_HZ = 32.0
#: Autonomic features are computed on a 1 Hz grid (HR, BP, respiration envelopes).
AUTONOMIC_HZ = 1.0

_RECORD_SUBJECT = re.compile(r"^slp(\d+)")


def subject_id_from_record(record: str) -> str:
    """``slp01a`` and ``slp01b`` are one person: group on the numeric prefix."""
    match = _RECORD_SUBJECT.match(record)
    if match is None:
        raise ValueError(f"unrecognized slpdb record name: {record!r}")
    return match.group(1)


def parse_stage_and_arousal_classes(aux_note: str) -> tuple[str, bool, bool]:
    """Split one annotation into (stage, respiratory arousal, non-respiratory).

    Annotations arrive as e.g. ``"2 LA"`` or ``"4 L L"``; some carry trailing NUL
    bytes from the source files, which must be stripped before comparison.

    The two arousal classes are kept apart rather than pooled because the
    autonomic-precedence hypothesis is supported for one and contradicted for the
    other (see the token constants). Pooling averages a plausible effect against a
    documented null and yields a number that cannot be interpreted either way.
    """
    tokens = aux_note.replace("\x00", " ").split()
    if not tokens:
        return "", False, False
    stage, rest = tokens[0], tokens[1:]
    respiratory = any(token in RESPIRATORY_AROUSAL_TOKENS for token in rest)
    non_respiratory = any(token in NON_RESPIRATORY_AROUSAL_TOKENS for token in rest)
    return stage, respiratory, non_respiratory


def parse_stage_and_arousal(aux_note: str) -> tuple[str, bool]:
    """(stage, arousal of *either* class). Pooled view; prefer the class split."""
    stage, respiratory, non_respiratory = parse_stage_and_arousal_classes(aux_note)
    return stage, respiratory or non_respiratory


def _decimate(signal: np.ndarray, src_hz: float, target_hz: float) -> np.ndarray:
    """Mean-pool to ``target_hz``; anti-aliases better than plain striding."""
    factor = max(1, int(round(src_hz / target_hz)))
    usable = (len(signal) // factor) * factor
    if usable == 0:
        return np.zeros(0, dtype=np.float32)
    return signal[:usable].reshape(-1, factor).mean(axis=1).astype(np.float32)


def _heart_rate_per_second(ecg: np.ndarray, fs: float, n_seconds: int) -> np.ndarray:
    """Instantaneous heart rate (bpm) on a 1 Hz grid from QRS detection.

    Falls back to the ECG log-power envelope when detection is too sparse to give
    a usable rate, so one noisy record degrades rather than aborting the cohort.
    """
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
    except Exception:  # noqa: BLE001 - fall back to the envelope, recorded by caller
        pass
    env = _decimate(np.abs(ecg).astype(np.float32), fs, AUTONOMIC_HZ)
    out = np.zeros(n_seconds, dtype=np.float32)
    take = min(n_seconds, len(env))
    out[:take] = np.log(env[:take] + 1e-8)
    return out


def _envelope_per_second(signal: np.ndarray, fs: float, n_seconds: int) -> np.ndarray:
    """Log power envelope on a 1 Hz grid (BP and respiration are sub-1 Hz)."""
    env = _decimate(np.abs(signal).astype(np.float32), fs, AUTONOMIC_HZ)
    out = np.zeros(n_seconds, dtype=np.float32)
    take = min(n_seconds, len(env))
    out[:take] = np.log(env[:take] + 1e-8)
    return out


def _zscore(x: np.ndarray) -> np.ndarray:
    mean = float(np.mean(x))
    std = float(np.std(x))
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def load_slpdb_record(
    record: str, *, pn_dir: str = "slpdb", data_dir: str | None = None
) -> dict[str, Any]:
    """Load one slpdb record into per-epoch matrices plus the arousal masks.

    Returns EEG epochs at ``EEG_TARGET_HZ`` and autonomic epochs (heart rate,
    blood pressure, respiration) on a 1 Hz grid, both shaped
    ``(n_epochs, samples, n_channels)``.

    ``data_dir`` reads a local copy (see ``scripts/fetch_slpdb.py``); without it
    every record streams from PhysioNet, which is slow and reset-prone for these
    ~32 MB files.
    """
    import wfdb

    if data_dir is not None:
        base = str(Path(data_dir) / record)
        rec = wfdb.rdrecord(base)
        ann = wfdb.rdann(base, "st")
    else:
        rec = wfdb.rdrecord(record, pn_dir=pn_dir)
        ann = wfdb.rdann(record, "st", pn_dir=pn_dir)
    fs = float(rec.fs)
    names = list(rec.sig_name)

    def channel(want: str) -> np.ndarray | None:
        for idx, name in enumerate(names):
            if want.lower() in name.lower():
                return np.nan_to_num(rec.p_signal[:, idx].astype(np.float32))
        return None

    eeg = channel("EEG")
    ecg = channel("ECG")
    bp = channel("BP")
    resp = channel("Resp")
    if eeg is None or ecg is None:
        raise ValueError(f"{record}: missing EEG or ECG channel (have {names})")

    stages: list[str] = []
    arousals: list[int] = []
    respiratory_arousals: list[int] = []
    non_respiratory_arousals: list[int] = []
    for note in ann.aux_note:
        stage, respiratory, non_respiratory = parse_stage_and_arousal_classes(note)
        stages.append(stage)
        arousals.append(int(respiratory or non_respiratory))
        respiratory_arousals.append(int(respiratory))
        non_respiratory_arousals.append(int(non_respiratory))
    n_epochs = min(
        len(stages),
        int(len(eeg) // (EPOCH_SECONDS * fs)),
    )
    if n_epochs < 20:
        raise ValueError(f"{record}: only {n_epochs} usable epochs")

    n_seconds = int(n_epochs * EPOCH_SECONDS)
    hr = _heart_rate_per_second(ecg, fs, n_seconds)
    bp_env = _envelope_per_second(bp, fs, n_seconds) if bp is not None else np.zeros(n_seconds, np.float32)
    resp_env = _envelope_per_second(resp, fs, n_seconds) if resp is not None else np.zeros(n_seconds, np.float32)

    eeg_ds = _decimate(eeg, fs, EEG_TARGET_HZ)
    eeg_per_epoch = int(EPOCH_SECONDS * EEG_TARGET_HZ)
    needed = n_epochs * eeg_per_epoch
    if len(eeg_ds) < needed:
        eeg_ds = np.pad(eeg_ds, (0, needed - len(eeg_ds)))
    eeg_epochs = _zscore(eeg_ds[:needed]).reshape(n_epochs, eeg_per_epoch, 1)

    sec_per_epoch = int(EPOCH_SECONDS)
    autonomic = np.stack([_zscore(hr), _zscore(bp_env), _zscore(resp_env)], axis=1)
    autonomic_epochs = autonomic[: n_epochs * sec_per_epoch].reshape(n_epochs, sec_per_epoch, 3)

    return {
        "record": record,
        "subject": subject_id_from_record(record),
        "n_epochs": n_epochs,
        "eeg_epochs": eeg_epochs.astype(np.float32),
        "autonomic_epochs": autonomic_epochs.astype(np.float32),
        "arousal": np.asarray(arousals[:n_epochs], dtype=np.int64),
        "arousal_respiratory": np.asarray(respiratory_arousals[:n_epochs], dtype=np.int64),
        "arousal_non_respiratory": np.asarray(
            non_respiratory_arousals[:n_epochs], dtype=np.int64
        ),
        "stages": stages[:n_epochs],
        "channels": names,
    }

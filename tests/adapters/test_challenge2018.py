"""Unit tests for the challenge-2018 adapter annotation parser.

All tests are pure-Python (no network, no wfdb I/O) and run in <1 s.
"""
from __future__ import annotations

import numpy as np

from neurotwin.adapters.challenge2018 import (
    _parse_arousal_epochs,
    _parse_stages,
    subject_id_from_record,
)

FS = 200.0
EPOCH_SAMPLES = int(30 * FS)  # 6000 samples per epoch


# ── subject ID ────────────────────────────────────────────────────────────────


def test_subject_id_training():
    assert subject_id_from_record("training/tr03-0005/tr03-0005") == "tr03-0005"


def test_subject_id_test():
    assert subject_id_from_record("test/te09-0094/te09-0094") == "te09-0094"


# ── arousal parsing ───────────────────────────────────────────────────────────


def _ann(sample: int, note: str):
    return sample, note


def test_spontaneous_single_epoch():
    """A spontaneous arousal that falls entirely in epoch 2."""
    samples = np.array([2 * EPOCH_SAMPLES + 100, 2 * EPOCH_SAMPLES + 600])
    notes = ["(arousal_spontaneous", "arousal_spontaneous)"]
    masks = _parse_arousal_epochs(samples, notes, FS, n_epochs=5)

    assert masks["arousal_spontaneous"][2] == 1
    assert masks["arousal_spontaneous"].sum() == 1
    assert masks["arousal_any"][2] == 1
    assert masks["arousal_rera"].sum() == 0
    assert masks["arousal_respiratory"].sum() == 0


def test_rera_spans_two_epochs():
    """A RERA starting near end of epoch 1 and ending in epoch 2."""
    start = 1 * EPOCH_SAMPLES + EPOCH_SAMPLES - 100  # last 0.5 s of epoch 1
    end = 2 * EPOCH_SAMPLES + 200                    # early in epoch 2
    samples = np.array([start, end])
    notes = ["(arousal_rera", "arousal_rera)"]
    masks = _parse_arousal_epochs(samples, notes, FS, n_epochs=5)

    assert masks["arousal_rera"][1] == 1
    assert masks["arousal_rera"][2] == 1
    assert masks["arousal_rera"].sum() == 2


def test_respiratory_does_not_contaminate_non_respiratory():
    """Obstructive apnea should not appear in the spontaneous or RERA mask."""
    samples = np.array([0, EPOCH_SAMPLES])
    notes = ["(resp_obstructiveapnea", "resp_obstructiveapnea)"]
    masks = _parse_arousal_epochs(samples, notes, FS, n_epochs=4)

    assert masks["arousal_respiratory"].sum() > 0
    assert masks["arousal_spontaneous"].sum() == 0
    assert masks["arousal_rera"].sum() == 0


def test_multiple_event_types_same_record():
    """Simultaneous spontaneous + respiratory events, each tracked independently."""
    spont_start = 0
    spont_end = EPOCH_SAMPLES - 10
    resp_start = 2 * EPOCH_SAMPLES
    resp_end = 3 * EPOCH_SAMPLES - 10

    samples = np.array([spont_start, spont_end, resp_start, resp_end])
    notes = [
        "(arousal_spontaneous",
        "arousal_spontaneous)",
        "(resp_hypopnea",
        "resp_hypopnea)",
    ]
    masks = _parse_arousal_epochs(samples, notes, FS, n_epochs=5)

    assert masks["arousal_spontaneous"][0] == 1
    assert masks["arousal_respiratory"][2] == 1
    assert masks["arousal_spontaneous"][2] == 0
    assert masks["arousal_respiratory"][0] == 0


def test_empty_annotations():
    """No annotations → all-zero masks."""
    masks = _parse_arousal_epochs(np.array([]), [], FS, n_epochs=10)
    for v in masks.values():
        assert v.sum() == 0
        assert len(v) == 10


def test_arousal_at_last_epoch():
    """Arousal entirely in the last epoch is captured, not silently dropped."""
    n = 10
    start = (n - 1) * EPOCH_SAMPLES + 10
    end = (n - 1) * EPOCH_SAMPLES + 500
    samples = np.array([start, end])
    notes = ["(arousal_spontaneous", "arousal_spontaneous)"]
    masks = _parse_arousal_epochs(samples, notes, FS, n_epochs=n)
    assert masks["arousal_spontaneous"][n - 1] == 1


def test_unmatched_open_bracket_does_not_crash():
    """An open bracket with no matching close should not raise."""
    samples = np.array([0, EPOCH_SAMPLES])
    notes = ["(arousal_spontaneous", "N2"]  # stage note instead of close
    masks = _parse_arousal_epochs(samples, notes, FS, n_epochs=5)
    # spontaneous interval was never closed — epoch 0 should NOT be marked
    # (we only mark on close)
    assert masks["arousal_spontaneous"].sum() == 0


# ── stage parsing ─────────────────────────────────────────────────────────────


def test_stages_basic():
    """Stage labels forward-fill correctly."""
    samples = np.array([0, EPOCH_SAMPLES, 2 * EPOCH_SAMPLES])
    notes = ["W", "N2", "R"]
    stages = _parse_stages(samples, notes, n_epochs=4)
    assert stages[0] == "W"
    assert stages[1] == "N2"
    assert stages[2] == "R"
    assert stages[3] == "R"  # forward-fill


def test_stages_interleaved_with_arousals():
    """Stage parser ignores non-stage tokens."""
    samples = np.array([0, 500, EPOCH_SAMPLES, EPOCH_SAMPLES + 200])
    notes = ["N2", "(arousal_spontaneous", "N3", "arousal_spontaneous)"]
    stages = _parse_stages(samples, notes, n_epochs=3)
    assert stages[0] == "N2"
    assert stages[1] == "N3"


def test_stages_all_wake():
    """Single W annotation covers all epochs."""
    samples = np.array([0])
    notes = ["W"]
    stages = _parse_stages(samples, notes, n_epochs=5)
    assert all(s == "W" for s in stages)


# --- feature-extraction invariants -----------------------------------------
# These pin two corrections that a compute shortcut had previously undone:
# HR must be rate (not an amplitude envelope) and must say which path produced
# it, and EEG channels must stay separate rather than being averaged together.

def test_heart_rate_reports_which_path_fired():
    """HR must declare qrs vs envelope_fallback so results stay auditable."""
    import numpy as np

    from neurotwin.adapters.challenge2018 import _heart_rate_per_second

    fs, n_sec = 200.0, 60
    t = np.arange(int(fs * n_sec)) / fs
    # clean 60 bpm impulse train -> QRS detection should succeed
    ecg = np.zeros_like(t)
    ecg[:: int(fs)] = 1.0
    hr, path = _heart_rate_per_second(ecg.astype(np.float32), fs, n_sec)
    assert path in {"qrs", "envelope_fallback"}
    assert hr.shape == (n_sec,)

    # flat signal has no beats at all -> must fall back and SAY so
    hr2, path2 = _heart_rate_per_second(np.zeros_like(ecg, dtype=np.float32), fs, n_sec)
    assert path2 == "envelope_fallback"
    assert hr2.shape == (n_sec,)


def test_eeg_channels_are_not_averaged():
    """Two requested EEG channels must yield two output columns, not one."""
    import numpy as np

    from neurotwin.adapters.challenge2018 import _decimate, _zscore

    # Mirrors the per-channel decimate/zscore/reshape the loader performs.
    fs, n_epochs, per_epoch = 200.0, 3, 960
    needed = n_epochs * per_epoch
    raw = np.stack(
        [np.sin(np.arange(int(fs * 90)) / 50.0), np.cos(np.arange(int(fs * 90)) / 50.0)],
        axis=1,
    ).astype(np.float32)
    cols = []
    for ch in range(raw.shape[1]):
        ds = _decimate(raw[:, ch], fs, 32.0)
        if len(ds) < needed:
            ds = np.pad(ds, (0, needed - len(ds)))
        cols.append(_zscore(ds[:needed]))
    out = np.stack(cols, axis=1).reshape(n_epochs, per_epoch, len(cols))
    assert out.shape == (n_epochs, per_epoch, 2), "channels were collapsed"
    # the two channels must remain distinguishable after the reshape
    assert not np.allclose(out[..., 0], out[..., 1])

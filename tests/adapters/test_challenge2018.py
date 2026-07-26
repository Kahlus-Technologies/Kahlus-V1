"""Unit tests for the challenge-2018 adapter annotation parser.

All tests are pure-Python (no network, no wfdb I/O) and run in <1 s.
"""
from __future__ import annotations

import numpy as np
import pytest

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

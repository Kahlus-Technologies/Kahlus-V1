"""Every NSRR study skips an unreadable night, counts it by reason and keeps going.

The night readers are replaced with stubs so the tests need no PSG data; one test
checks that real broken files raise an error the studies treat as a bad night.
"""
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from neurotwin.adapters.nsrr import BAD_NIGHT_ERRORS, load_nsrr_epoch_matrix, parse_nsrr_events

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import event_locked_hr as elhr  # noqa: E402
import run_beat_warning as beat  # noqa: E402
import run_clas_replay as clas  # noqa: E402
import run_heart_arousal_benchmark as bench  # noqa: E402
import run_montage_stress as mont  # noqa: E402

IDS = ("200001", "200002", "200003", "200004")
# 200002 has a missing file, 200003 broken XML; the rest read fine.
ERRORS = {"200002": FileNotFoundError(2, "No such file or directory", "/data/shhs/edfs/shhs1-200002.edf"),
          "200003": ET.ParseError("no element found: line 1, column 3")}
EXPECTED = {"FileNotFoundError: [Errno 2] No such file or directory: '<path>'": 1,
            "ParseError: no element found: line 1, column 3": 1}


@pytest.fixture(autouse=True)
def _dev(monkeypatch):
    for k in ("KAHLUS_MESA", "NSRR_SHHS_IDS", "NSRR_MESA_IDS"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "shhs"
    for i in IDS:
        for path in (r / "edfs" / f"shhs1-{i}.edf", r / "annotations" / f"shhs1-{i}-nsrr.xml"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("stub")
    return r


def _raise_for(rec):
    if rec.subject_id in ERRORS:
        raise ERRORS[rec.subject_id]


def test_broken_files_raise_an_error_the_studies_skip(tmp_path):
    (tmp_path / "junk.edf").write_bytes(b"not an EDF header" * 30)
    (tmp_path / "bad.xml").write_text("<PSGAnnotation><ScoredEvents>")
    for call in (lambda: load_nsrr_epoch_matrix(tmp_path / "junk.edf", channel_groups={"eeg": ("EEG",)}),
                 lambda: load_nsrr_epoch_matrix(tmp_path / "missing.edf", channel_groups={"eeg": ("EEG",)}),
                 lambda: parse_nsrr_events(tmp_path / "bad.xml")):
        with pytest.raises(BAD_NIGHT_ERRORS):
            call()


def test_beat_warning_skips_and_counts(root, tmp_path, monkeypatch):
    def night(rec, cohort):
        _raise_for(rec)
        n = 5
        return {"y": np.zeros((n, 4), np.int8), "prior": np.ones((n, 6), np.float32),
                "beats": np.ones((n, 6), np.float32), "beats_shift": np.ones((n, 6), np.float32)}

    monkeypatch.setattr(beat, "night", night)
    data, n, skipped = beat.load("shhs", root, tmp_path / "cache")
    assert n == 2 and sorted(set(data["subject"].tolist())) == [0, 3]
    assert skipped.to_json() == EXPECTED
    assert sorted(p.name for p in (tmp_path / "cache").glob("*.npz")) == ["shhs-200001.npz", "shhs-200004.npz"]


def test_clas_replay_skips_and_counts(root, tmp_path, monkeypatch):
    def night(rec, cohort):
        _raise_for(rec)
        return {"clicks": np.arange(3.0), "bad": np.zeros(3, bool), "x_click": np.ones((3, 12), np.float32),
                "x_train": np.ones((4, 12), np.float32), "y_train": np.zeros(4, np.int8)}

    monkeypatch.setattr(clas, "night", night)
    nights, skipped = clas.load("shhs", root, tmp_path / "cache")
    assert len(nights) == 2 and skipped.to_json() == EXPECTED


def test_benchmark_skips_more_than_value_errors(root, tmp_path, monkeypatch):
    def build_recording(edf, xml, **kw):
        sid = Path(edf).stem.split("-")[-1]
        if sid in ERRORS:
            raise ERRORS[sid]
        n = 8
        return {"anchors": np.arange(n, dtype=float), "labels": np.zeros((n, 5), np.int8),
                "blocks": {"prior": np.ones((n, 6), np.float32), "cardiac": np.ones((n, 11), np.float32)}}

    monkeypatch.setattr(bench, "build_recording", build_recording)
    d = bench.load_cohort("shhs", root, tmp_path / "cache", None, "causal")
    assert d["n_nights"] == 2 and d["skipped"] == EXPECTED


def test_other_errors_still_stop_the_study(root, tmp_path, monkeypatch):
    def night(rec, cohort):
        raise RuntimeError("MESA final look already used")  # the seal's error type is never skipped

    monkeypatch.setattr(beat, "night", night)
    with pytest.raises(RuntimeError, match="already used"):
        beat.load("shhs", root, tmp_path / "cache")


def test_a_cohort_with_no_readable_night_stops_the_study(root, tmp_path, monkeypatch):
    def night(rec, cohort):
        raise OSError("unreadable")

    monkeypatch.setattr(clas, "night", night)
    with pytest.raises(RuntimeError, match="no readable shhs nights \\(4 skipped\\)"):
        clas.load("shhs", root, tmp_path / "cache")


def _eeg_night(rec):
    """Deterministic fake night: 20 min of noise EEG at 100 Hz, N2 throughout, 6 arousals."""
    _raise_for(rec)
    g = np.random.default_rng(int(rec.subject_id))
    fs, n_s = 100.0, 1200
    eeg = g.normal(0.0, 20.0, int(fs * n_s))
    events = [("Stage 2 sleep", float(s), 30.0) for s in range(0, n_s, 30)]
    for s in g.choice(np.arange(60, n_s - 60, 20), 6, replace=False):
        eeg[int(s * fs): int((s + 8) * fs)] *= 4.0  # arousals look different, so the detector learns something
        events.append(("Arousal (ASDA)", float(s), 8.0))
    return eeg, fs, events


def test_montage_skips_and_streams_nights_with_the_same_results(root, monkeypatch):
    calls = []

    def read_night(cohort, rec):
        calls.append(rec.subject_id)
        return _eeg_night(rec)

    monkeypatch.setattr(mont, "read_night", read_night)
    recs, clean, skipped = mont.load_clean("shhs", root)
    assert [r.subject_id for r in recs] == ["200001", "200004"] and skipped.to_json() == EXPECTED
    x, y = np.concatenate([n["x"] for n in clean]), np.concatenate([n["y"] for n in clean])
    clf = mont.make_model(0).fit(x[~np.isnan(x).all(axis=1)], y[~np.isnan(x).all(axis=1)])

    calls.clear()
    scored = mont.score_levels("shhs", recs, clean, clf)
    assert calls == ["200001", "200004"]  # each test night is read once for all levels
    new = {level: mont.score_nights(*scored[level], 0.5) for level in mont.LEVELS}

    raw = [_eeg_night(r) for r in recs]  # the old flow: every night's raw EEG held at once
    old = {}
    for level in mont.LEVELS:
        te = clean if level == "clean" else [mont.night_rows(e, fs, ev, level, i) for i, (e, fs, ev) in enumerate(raw)]
        old[level] = mont.score_nights(te, [clf.predict_proba(n["x"])[:, 1] for n in te], 0.5)
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)


def test_event_locked_hr_skips_and_counts(root, monkeypatch):
    def night_hr(rec, label):
        _raise_for(rec)
        bt = np.arange(1.0, 3600.0)
        return bt, np.ones(bt.size), [("Arousal (ASDA)", 1800.0, 10.0)]

    monkeypatch.setattr(elhr, "night_hr", night_hr)
    _, dropped, name, n_read, skipped = elhr.cohort_curves(root, "ECG")
    assert (name, n_read, dropped) == ("shhs", 2, 0) and skipped.to_json() == EXPECTED


def test_mesa_nights_are_not_named_in_skip_logs(capsys):
    from neurotwin.adapters.nsrr import NsrrRecording, SkippedNights

    s = SkippedNights("mesa")
    rec = NsrrRecording(subject_id="0042", edf_path=Path("/d/mesa-sleep-0042.edf"), xml_path=Path("/d/mesa-sleep-0042-nsrr.xml"), dataset="mesa")
    s.add(rec, OSError("cannot read mesa-sleep-0042.edf"))
    err = capsys.readouterr().err
    assert "0042" not in err and s.to_json() == {"OSError: cannot read <night>": 1}

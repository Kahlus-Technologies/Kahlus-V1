"""The cluster bootstraps give bit-identical results after the per-resample scans were removed.

_old_rfs and _old_mean are the code as it was before the change, kept here verbatim
as the reference. Every comparison is exact (==), not approximate.
"""
from __future__ import annotations

import numpy as np
import pytest

from neurotwin.forecastability.m1 import _cluster_bootstrap_rfs, _rfs_bits, _rows_by_group
from neurotwin.forecastability.m5 import _cluster_bootstrap_mean


def _old_rfs(y, baseline, pred, patient, *, seed, mode="smoke"):
    n_boot = 2_000 if mode == "claim" else 300
    rng = np.random.default_rng(seed)
    unique = np.unique(patient)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(patient == group) for group in chosen])
        samples.append(_rfs_bits(y[idx], baseline[idx], pred[idx]))
    return {
        "rfs_ci_low": float(np.percentile(samples, 2.5)),
        "rfs_ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_mode": mode,
        "bootstrap_samples": n_boot,
    }


def _old_mean(values, patient, *, seed, mode="smoke"):
    n_boot = 2_000 if mode == "claim" else 300
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(patient)
    unique = np.unique(groups)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in chosen])
        samples.append(float(np.mean(values[idx])))
    return {
        "pic_ci_low": float(np.percentile(samples, 2.5)),
        "pic_ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_mode": mode,
        "bootstrap_samples": n_boot,
    }


def _data(kind: str, seed: int):
    """Interleaved, unevenly sized groups; y, baseline and pred as the studies pass them."""
    g = np.random.default_rng(seed)
    sizes = g.integers(1, 60, size=int(g.integers(2, 40)))
    ids = {
        "int": np.arange(sizes.size) * 7 + 3,
        "str": np.array([f"shhs1-{200001 + i}" for i in range(sizes.size)]),
        "obj": np.array([f"s{i}" for i in range(sizes.size)], dtype=object),
        "float": np.arange(sizes.size) * 0.5,
    }[kind]
    patient = g.permutation(np.repeat(ids, sizes))  # rows of one subject are not contiguous
    n = patient.size
    y = (g.random(n) < 0.2).astype(np.int64)
    baseline = g.random(n)
    pred = np.clip(baseline + g.normal(0, 0.2, n), 0, 1).astype(np.float32)
    pred[g.random(n) < 0.05] = 0.0  # hits the clip in _nll
    return y, baseline, pred, patient


@pytest.mark.parametrize("kind", ["int", "str", "obj", "float"])
@pytest.mark.parametrize("seed", [0, 1, 17, 12345])
def test_rfs_bootstrap_matches_old_code_exactly(kind, seed):
    y, baseline, pred, patient = _data(kind, seed)
    for mode in ("smoke", "claim"):
        assert _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=seed, mode=mode) == \
            _old_rfs(y, baseline, pred, patient, seed=seed, mode=mode)


@pytest.mark.parametrize("kind", ["int", "str", "float"])
@pytest.mark.parametrize("seed", [0, 3, 17])
def test_mean_bootstrap_matches_old_code_exactly(kind, seed):
    _, values, _, patient = _data(kind, seed)
    for mode in ("smoke", "claim"):
        assert _cluster_bootstrap_mean(values, patient, seed=seed, mode=mode) == \
            _old_mean(values, patient, seed=seed, mode=mode)
        assert _cluster_bootstrap_mean(list(values), list(patient), seed=seed, mode=mode) == \
            _old_mean(list(values), list(patient), seed=seed, mode=mode)


def test_single_subject_matches():
    y, baseline, pred, patient = _data("int", 5)
    one = np.full(patient.size, 42)
    assert _cluster_bootstrap_rfs(y, baseline, pred, one, seed=1) == _old_rfs(y, baseline, pred, one, seed=1)


@pytest.mark.parametrize("kind", ["int", "str", "obj", "float"])
def test_rows_by_group_equals_one_scan_per_group(kind):
    patient = _data(kind, 9)[3]
    positions, rows = _rows_by_group(patient)
    unique = np.unique(patient)
    assert list(positions) == list(range(len(unique))) and len(rows) == len(unique)
    for u, r in zip(unique, rows):
        expect = np.flatnonzero(patient == u)
        assert r.dtype == expect.dtype and np.array_equal(r, expect)


def test_nan_group_matches_no_rows_as_before():
    patient = np.array([1.0, np.nan, 2.0, np.nan, 1.0])
    _, rows = _rows_by_group(patient)
    assert [r.tolist() for r in rows] == [np.flatnonzero(patient == u).tolist() for u in np.unique(patient)]

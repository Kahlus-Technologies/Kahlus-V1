"""The parallel cluster bootstrap gives the serial loop's samples, in the same order.

Every comparison is exact (==). The samples themselves are compared, not only the CI,
by recording what reaches np.percentile.
"""
from __future__ import annotations

import multiprocessing
from types import SimpleNamespace

import numpy as np
import pytest

from neurotwin.forecastability import m1
from neurotwin.forecastability.m1 import _cluster_bootstrap_rfs, _effective_workers, bootstrap_workers

needs_fork = pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(), reason="no fork here")


def _data(kind: str, seed: int):
    """Interleaved, unevenly sized subjects; y, baseline and pred as the studies pass them."""
    g = np.random.default_rng(seed)
    sizes = g.integers(1, 80, size=int(g.integers(20, 60)))
    ids = {
        "int": np.arange(sizes.size) * 7 + 3,
        "str": np.array([f"shhs1-{200001 + i}" for i in range(sizes.size)]),
        "float": np.arange(sizes.size) * 0.5,
    }[kind]
    patient = g.permutation(np.repeat(ids, sizes))
    n = patient.size
    y = (g.random(n) < 0.2).astype(np.int8)
    baseline = g.random(n)
    pred = np.clip(baseline + g.normal(0, 0.2, n), 0, 1).astype(np.float32)
    pred[g.random(n) < 0.05] = 0.0  # hits the clip in _nll
    return y, baseline, pred, patient


def _samples(monkeypatch, *args, **kwargs):
    """Run the bootstrap and return (result, the list of resample scores it built)."""
    seen = []
    real = np.percentile

    def record(a, q, *rest, **kw):
        seen.append(list(a))
        return real(a, q, *rest, **kw)

    monkeypatch.setattr(np, "percentile", record)
    try:
        out = _cluster_bootstrap_rfs(*args, **kwargs)
    finally:
        monkeypatch.setattr(np, "percentile", real)
    assert len(seen) == 2 and seen[0] == seen[1]
    return out, seen[0]


@needs_fork
@pytest.mark.parametrize("kind", ["int", "str", "float"])
@pytest.mark.parametrize("seed", [0, 7, 2024])
def test_parallel_matches_serial_exactly(monkeypatch, kind, seed):
    y, baseline, pred, patient = _data(kind, seed)
    for mode in ("smoke", "claim"):
        serial, serial_samples = _samples(monkeypatch, y, baseline, pred, patient, seed=seed, mode=mode, workers=1)
        assert len(serial_samples) == serial["bootstrap_samples"]
        for workers in (2, 16):
            out, samples = _samples(monkeypatch, y, baseline, pred, patient, seed=seed, mode=mode, workers=workers)
            assert samples == serial_samples
            assert out == serial
    assert m1._FORKED_DATA is None


@needs_fork
def test_default_path_uses_the_env_worker_count_and_matches_serial(monkeypatch):
    y, baseline, pred, patient = _data("str", 3)
    serial = _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=3, mode="claim", workers=1)
    monkeypatch.setattr(m1, "_PARALLEL_MIN_WORK", 0)
    monkeypatch.setenv("NEUROTWIN_BOOTSTRAP_WORKERS", "3")
    assert _effective_workers(None, len(y), 2_000) == 3
    assert _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=3, mode="claim") == serial


def test_worker_count_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("NEUROTWIN_BOOTSTRAP_WORKERS", raising=False)
    assert bootstrap_workers() == 16
    monkeypatch.setenv("NEUROTWIN_BOOTSTRAP_WORKERS", " 4 ")
    assert bootstrap_workers() == 4
    for bad in ("0", "-2", "four", "1.5"):
        monkeypatch.setenv("NEUROTWIN_BOOTSTRAP_WORKERS", bad)
        with pytest.raises(ValueError, match="positive integer"):
            bootstrap_workers()
    with pytest.raises(ValueError, match="at least 1"):
        _effective_workers(0, 10, 300)


@needs_fork
def test_when_the_pool_is_used(monkeypatch):
    monkeypatch.delenv("NEUROTWIN_BOOTSTRAP_WORKERS", raising=False)
    assert _effective_workers(None, 1_000, 300) == 1  # small input: serial
    assert _effective_workers(None, 100_000, 2_000) == 16
    assert _effective_workers(8, 10, 300) == 8  # an explicit count is used at any size
    assert _effective_workers(16, 10, 5) == 5  # never more workers than resamples
    monkeypatch.setattr(m1.multiprocessing, "current_process", lambda: SimpleNamespace(daemon=True))
    assert _effective_workers(16, 100_000, 2_000) == 1  # a daemon process may not start children


def test_no_fork_means_serial(monkeypatch):
    monkeypatch.setattr(m1.multiprocessing, "get_all_start_methods", lambda: ["spawn"])
    assert _effective_workers(16, 100_000, 2_000) == 1
    y, baseline, pred, patient = _data("int", 1)
    assert _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=1, workers=16) == \
        _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=1, workers=1)

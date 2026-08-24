"""Shared helpers for the Amrith-facing forecast diagnostics.

One home for the closed-form ridge, the Sleep-EDF channel loader, the AR(1)
self-test generator, and the kahlus-sweep-figure import resolver, so the
diagnostic scripts don't each carry their own copy (and don't each hardcode an
absolute path to the figure skill).
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge weights with a bias column. x:(n,L), y:(n,) or (n,k)."""
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    a = xb.T @ xb + alpha * np.eye(xb.shape[1])
    return np.linalg.solve(a, xb.T @ y)


def ridge_predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    return xb @ w


def load_sleep_edf_channel(
    path: Path, *, channel: str = "EEG Fpz-Cz", skip: int = 1_000_000,
    take: int = 300_000, zscore: bool = True,
) -> tuple[np.ndarray, str]:
    """Load one EEG channel from a Sleep-EDF PSG EDF. Returns (signal,
    resolved channel name). Falls back to the first channel whose name contains
    'EEG' if the requested one is absent.

    zscore=True uses the whole loaded clip, including future samples. Isolated
    forecast paths must pass zscore=False and normalize from train-split stats.
    """
    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    if channel not in raw.ch_names:
        channel = next((c for c in raw.ch_names if "EEG" in c), raw.ch_names[0])
    raw.pick([channel])
    raw.load_data(verbose="ERROR")
    sig = raw.get_data()[0].astype(np.float64)[skip:skip + take]
    if zscore:
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
    return sig.astype(np.float32), channel


def ar1_sequences(n: int, length: int, *, coef: float = 0.8, seed: int = 0) -> np.ndarray:
    """(n, length) array of AR(1) sequences x_t = coef*x_{t-1} + N(0,1). Used by
    the self-tests: a linear model must beat the trace mean on this."""
    rng = np.random.default_rng(seed)
    seq = np.zeros((n, length), dtype=np.float32)
    e = rng.normal(size=(n, length)).astype(np.float32)
    for t in range(1, length):
        seq[:, t] = coef * seq[:, t - 1] + e[:, t]
    return seq


def import_sweep_figure():
    """Return the kahlus-sweep-figure `sweep_figure` helper, resolving its path
    from the user's skills dir rather than a hardcoded absolute path."""
    candidate = Path.home() / ".claude/skills/kahlus-sweep-figure/scripts"
    if not (candidate / "sweep_figure.py").is_file():
        raise FileNotFoundError(
            f"kahlus-sweep-figure helper not found at {candidate}; install the "
            "skill or pass its scripts dir on sys.path"
        )
    sys.path.insert(0, str(candidate))
    from sweep_figure import sweep_figure

    return sweep_figure

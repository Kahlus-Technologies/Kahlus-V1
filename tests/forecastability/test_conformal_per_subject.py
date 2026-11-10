"""run_conformal_alarm's per-subject miss rates are exactly what the per-subject scans gave."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_conformal_alarm as rca  # noqa: E402


def _old(scores, y, subject, thr):
    return [
        1.0 - np.mean(scores[(subject == s) & (y == 1)] >= thr)
        for s in np.unique(subject)
        if ((subject == s) & (y == 1)).sum() >= 5
    ]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_per_subject_miss_matches_the_scans_exactly(seed):
    g = np.random.default_rng(seed)
    n = 4000
    subject = g.integers(0, 60, n)  # interleaved; some subjects have fewer than 5 positives
    y = (g.random(n) < g.uniform(0.0, 0.2, 60)[subject]).astype(np.int8)
    scores = g.random(n)
    pos_rows = [r for r in rca.positive_rows_by_subject(subject, y) if r.size >= 5]
    for thr in (*g.random(10), 0.0, 1.0):
        new = rca.per_subject_miss(scores, pos_rows, thr)
        old = _old(scores, y, subject, thr)
        assert new == old and all(type(a) is type(b) for a, b in zip(new, old))

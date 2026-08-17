"""TESTMODE: planted-signal fixtures with known correct verdicts.

Run the capacity harness against these BEFORE spending compute on real EEG. Each
fixture has a ground truth we derived rather than measured, so the harness either
reproduces it or the harness is broken. A sweep that cannot recover a planted
answer cannot be trusted to report an honest one.

This is the same discipline as F0: the arena is only worth anything because
synthetic known-signal fixtures pass and known-null fixtures fail.

| fixture     | generator                          | correct verdict                    |
| ----------- | ---------------------------------- | ---------------------------------- |
| `big_win`   | linear term + a term in |x|        | nonlinear beats linear, large gap  |
| `small_win` | AR(2) + mild cubic perturbation    | nonlinear beats linear, small gap  |
| `neutral`   | linear-Gaussian AR(2)              | nonlinear must NOT beat linear     |
| `no_signal` | i.i.d. white noise                 | everything at the variance floor   |

`neutral` and `no_signal` are the ones that catch a harness that manufactures
wins. `big_win` and `small_win` catch a harness too weak to detect a real effect
— the opposite failure, and the one that produces false negatives.

Each fixture emits the same structure `load_corpus` produces, so the harness
consumes it unchanged: {subject: (context (n,L), {horizon: target (n,)})}.

  PYTHONPATH=src python3 scripts/testmode_fixtures.py --selftest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from amrith_isolated_forecast_check import L  # noqa: E402

FIXTURES = ("big_win", "small_win", "neutral", "no_signal")

#: What each fixture must produce. Checked mechanically by the harness.
EXPECTED = {
    "big_win":   {"nonlinear_beats_linear": True,  "min_relative_gap": 0.15},
    "small_win": {"nonlinear_beats_linear": True,  "min_relative_gap": 0.02},
    "neutral":   {"nonlinear_beats_linear": False, "min_relative_gap": None},
    "no_signal": {"nonlinear_beats_linear": False, "min_relative_gap": None},
}


def _ar2(n: int, length: int, rng, a1: float = 1.2, a2: float = -0.5) -> np.ndarray:
    """Stable oscillatory AR(2). Its optimal predictor is LINEAR by construction."""
    e = rng.normal(size=(n, length))
    x = np.zeros_like(e)
    for t in range(2, length):
        x[:, t] = a1 * x[:, t - 1] + a2 * x[:, t - 2] + e[:, t]
    return x


def _generate(kind: str, n: int, length: int, rng) -> np.ndarray:
    if kind == "no_signal":
        return rng.normal(size=(n, length))

    if kind == "neutral":
        return _ar2(n, length, rng)

    if kind == "small_win":
        # AR(2) with a mild cubic term: linear captures most of it, a nonlinear
        # model should recover a small extra slice.
        e = rng.normal(size=(n, length))
        x = np.zeros_like(e)
        for t in range(2, length):
            x[:, t] = (1.2 * x[:, t - 1] - 0.5 * x[:, t - 2]
                       + 0.90 * np.tanh(x[:, t - 1]) ** 2
                       + e[:, t])
        return x

    if kind == "big_win":
        # A linear component the ridge can find, plus a term in |x| that it
        # provably cannot: for symmetric x, corr(x, |x|) = 0, so the absolute
        # value contributes nothing to any linear predictor while carrying real
        # information a nonlinear model can use.
        e = rng.normal(size=(n, length))
        x = np.zeros_like(e)
        for t in range(3, length):
            x[:, t] = (0.30 * x[:, t - 1]
                       + 1.25 * (np.abs(x[:, t - 2]) - 0.8)
                       + 0.22 * e[:, t])
        return x

    raise ValueError(f"unknown fixture {kind!r}")


def build_fixture(kind: str, *, horizons, n_subjects: int = 40,
                  windows_per_subject: int = 400, seed: int = 0) -> dict:
    """Return a corpus shaped exactly like `load_corpus` output."""
    if kind not in FIXTURES:
        raise ValueError(f"unknown fixture {kind!r}; choose from {FIXTURES}")
    rng = np.random.default_rng(seed)
    need = L + max(horizons)
    corpus: dict[str, tuple] = {}
    for i in range(n_subjects):
        raw = _generate(kind, windows_per_subject, need, rng)
        # z-score per "subject", matching how real recordings are handled
        raw = (raw - raw.mean()) / (raw.std() + 1e-9)
        ctx = raw[:, :L].astype(np.float32)
        tgt = {h: raw[:, L + h - 1].astype(np.float32) for h in horizons}
        corpus[f"{kind[:4].upper()}{i:03d}"] = (ctx, tgt)
    return corpus


def check_verdict(kind: str, ridge_mse: float, best_nonlinear_mse: float) -> dict:
    """Did the harness reproduce this fixture's known answer?"""
    exp = EXPECTED[kind]
    gap = (ridge_mse - best_nonlinear_mse) / ridge_mse if ridge_mse > 0 else 0.0
    beats = gap > 0.01
    ok = beats == exp["nonlinear_beats_linear"]
    if ok and exp["min_relative_gap"] is not None:
        ok = gap >= exp["min_relative_gap"]
    return {
        "fixture": kind, "ridge_mse": ridge_mse, "best_nonlinear_mse": best_nonlinear_mse,
        "relative_gap": gap, "expected_nonlinear_wins": exp["nonlinear_beats_linear"],
        "observed_nonlinear_wins": beats, "harness_correct": bool(ok),
    }


def selftest() -> None:
    """Each generator must have the statistical property it claims, before any
    model is involved. Catches a broken generator masquerading as a model result."""
    rng = np.random.default_rng(0)
    n, length = 4000, L + 2

    # no_signal: lag-1 autocorrelation ~ 0
    w = _generate("no_signal", n, length, rng)
    r = np.corrcoef(w[:, -2], w[:, -1])[0, 1]
    assert abs(r) < 0.05, f"no_signal should be uncorrelated, got r={r:.3f}"

    # neutral: strongly autocorrelated and linearly predictable
    a = _generate("neutral", n, length, rng)
    ra = np.corrcoef(a[:, -2], a[:, -1])[0, 1]
    assert abs(ra) > 0.5, f"neutral AR(2) should be autocorrelated, got r={ra:.3f}"

    # How much could a nonlinear model gain? Measure directly, without training
    # a network: fit linear on lags, then linear on lags PLUS |lags|. The drop is
    # nonlinear headroom that a linear predictor cannot reach by construction.
    def nonlinear_headroom(data: np.ndarray) -> float:
        lag1, lag2, y = data[:, -2], data[:, -3], data[:, -1]
        ones = np.ones(len(y))
        lin = np.stack([lag1, lag2, ones], axis=1)
        # Even (|x|, x^2) and odd (x^3) basis terms, so the probe is agnostic to
        # the form of the nonlinearity. Probing |x| alone cannot see a cubic.
        nl = np.stack([lag1, lag2, np.abs(lag1), np.abs(lag2),
                       lag1 ** 2, lag2 ** 2, lag1 ** 3, lag2 ** 3, ones], axis=1)
        r_lin = np.mean((y - lin @ np.linalg.lstsq(lin, y, rcond=None)[0]) ** 2)
        r_nl = np.mean((y - nl @ np.linalg.lstsq(nl, y, rcond=None)[0]) ** 2)
        return float((r_lin - r_nl) / r_lin)

    head = {k: nonlinear_headroom(_generate(k, n, length, rng)) for k in FIXTURES}
    assert head["big_win"] > 0.25, f"big_win headroom too small: {head['big_win']:.3f}"
    assert 0.03 < head["small_win"] < 0.15, (
        f"small_win headroom out of range: {head['small_win']:.3f}")
    assert head["neutral"] < 0.02, f"neutral must have ~no headroom: {head['neutral']:.3f}"
    assert head["no_signal"] < 0.02, f"no_signal headroom: {head['no_signal']:.3f}"
    assert head["big_win"] > 3 * head["small_win"], (
        "big_win and small_win must actually span a range, got "
        f"{head['big_win']:.3f} vs {head['small_win']:.3f}")

    # verdict checker: correct and incorrect harness outputs are distinguished
    assert check_verdict("neutral", 0.50, 0.499)["harness_correct"] is True
    assert check_verdict("neutral", 0.50, 0.20)["harness_correct"] is False
    assert check_verdict("big_win", 0.50, 0.20)["harness_correct"] is True
    assert check_verdict("big_win", 0.50, 0.499)["harness_correct"] is False

    print(f"selftest OK | no_signal r={r:+.3f} | neutral r={ra:+.3f}")
    print("  nonlinear headroom by fixture: " +
          "  ".join(f"{k}={head[k]:.3f}" for k in FIXTURES))


if __name__ == "__main__":
    selftest()

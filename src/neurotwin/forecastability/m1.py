from __future__ import annotations

from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TransitionFixture:
    windows: np.ndarray
    nuisance: np.ndarray
    y: np.ndarray
    patient: np.ndarray
    site: np.ndarray
    time_bucket: np.ndarray
    session: np.ndarray


def run_m1_gate(out_dir: str | Path, *, seed: int = 0) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = _run_fixture(make_transition_fixture(seed=seed, residual_signal=True), seed=seed)
    null = _run_fixture(make_transition_fixture(seed=seed + 100, residual_signal=False), seed=seed + 100)
    gate = {
        "milestone": "M1",
        "known_signal": known,
        "synthetic_null": null,
        "gate_passed": _known_passes(known) and _null_passes(null),
        "stop_reason": "M1 gate reached; do not proceed to M2 until this report is reviewed.",
    }
    _write_json(out / "m1_gate_report.json", gate)
    _write_report(out / "M1_EVIDENCE_REPORT.md", gate)
    return gate


def make_transition_fixture(
    *,
    seed: int,
    residual_signal: bool,
    n_patients: int = 12,
    steps_per_patient: int = 120,
    window: int = 32,
    channels: int = 4,
    horizon: int = 1,
) -> TransitionFixture:
    rng = np.random.default_rng(seed)
    windows: list[np.ndarray] = []
    nuisance: list[list[float]] = []
    labels: list[int] = []
    patients: list[int] = []
    sites: list[int] = []
    time_buckets: list[int] = []
    sessions: list[int] = []
    t_axis = np.linspace(0.0, 1.0, window, dtype=np.float32)
    for patient in range(n_patients):
        site = patient % 2
        precursor = 0.0
        events = [0]
        raw_windows: list[np.ndarray] = []
        logits: list[float] = []
        nuisance_rows: list[list[float]] = []
        for t in range(steps_per_patient + horizon):
            cycle = np.sin(2.0 * np.pi * t / 48.0)
            recent = float(np.mean(events[-12:]))
            precursor = 0.82 * precursor + rng.normal(0.0, 0.8)
            visible = precursor if residual_signal else rng.normal(0.0, 1.0)
            signal = rng.normal(0.0, 0.25, size=(window, channels)).astype(np.float32)
            signal[:, 0] += (0.6 + 0.28 * visible) * np.sin(2.0 * np.pi * (6.0 + 0.25 * cycle) * t_axis)
            signal[:, 1] += 0.18 * visible * np.linspace(-1.0, 1.0, window)
            signal[:, 2] += 0.15 * np.sin(2.0 * np.pi * 3.0 * t_axis + float(site))
            logit = -2.25 + 0.85 * cycle + 1.15 * recent + (1.75 * precursor if residual_signal else 0.0)
            event = int(rng.random() < _sigmoid(logit))
            raw_windows.append(signal)
            logits.append(logit)
            nuisance_rows.append([1.0, cycle, np.cos(2.0 * np.pi * t / 48.0), recent, float(t % 48) / 47.0])
            events.append(event)
        for t in range(steps_per_patient):
            windows.append(raw_windows[t])
            nuisance.append(nuisance_rows[t])
            labels.append(int(events[t + horizon]))
            patients.append(patient)
            sites.append(site)
            time_buckets.append((t % 48) // 12)
            sessions.append(t // max(1, steps_per_patient // 2))
    return TransitionFixture(
        windows=np.asarray(windows, dtype=np.float32),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        patient=np.asarray(patients, dtype=np.int64),
        site=np.asarray(sites, dtype=np.int64),
        time_bucket=np.asarray(time_buckets, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
    )


def handcrafted_eeg_features(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float32)
    diffs = np.diff(x, axis=1)
    line_length = np.mean(np.abs(diffs), axis=(1, 2), keepdims=False)[:, None]
    amplitude_variance = np.var(x, axis=(1, 2), keepdims=False)[:, None]
    centered_t = np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float32)
    trend = np.mean(x * centered_t[None, :, None], axis=(1, 2), keepdims=False)[:, None]
    spectrum = np.abs(np.fft.rfft(x, axis=1)) ** 2
    total_power = np.sum(spectrum, axis=(1, 2), keepdims=False)[:, None] + 1e-8
    bandpower = np.concatenate(
        [
            np.sum(spectrum[:, 1:4, :], axis=(1, 2), keepdims=False)[:, None] / total_power,
            np.sum(spectrum[:, 4:9, :], axis=(1, 2), keepdims=False)[:, None] / total_power,
            np.sum(spectrum[:, 9:, :], axis=(1, 2), keepdims=False)[:, None] / total_power,
        ],
        axis=1,
    )
    p = spectrum / (np.sum(spectrum, axis=1, keepdims=True) + 1e-8)
    spectral_entropy = -np.mean(np.sum(p * np.log(p + 1e-8), axis=1), axis=1)[:, None]
    coherence = np.asarray([_mean_abs_corr(row) for row in x], dtype=np.float32)[:, None]
    return np.concatenate([bandpower, line_length, spectral_entropy, coherence, amplitude_variance, trend], axis=1).astype(np.float32)


def discrete_survival_labels(events: np.ndarray, *, bins: int = 3) -> np.ndarray:
    y = np.asarray(events, dtype=np.int64)
    labels = np.zeros((len(y), bins), dtype=np.int64)
    for idx in range(len(y)):
        for horizon in range(1, bins + 1):
            if idx + horizon < len(y):
                labels[idx, horizon - 1] = y[idx + horizon]
    return labels


def _run_fixture(fixture: TransitionFixture, *, seed: int) -> dict[str, Any]:
    payload, _ = _run_fixture_with_traces(fixture, seed=seed)
    return payload


def _run_fixture_with_traces(
    fixture: TransitionFixture,
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    y = fixture.y
    z = handcrafted_eeg_features(fixture.windows)
    b = fixture.nuisance
    x_full = np.concatenate([b, z], axis=1)
    baseline = _crossfit_proba(b, y, fixture.patient, _logistic_factory)
    full = _crossfit_residual_proba(b, z, y, fixture.patient)
    gbm = _crossfit_proba(x_full, y, fixture.patient, _gbm_factory(seed))
    shuffled = _crossfit_residual_proba(b, z, y, fixture.patient, control="shuffle", seed=seed + 11)
    shifted = _crossfit_residual_proba(b, z, y, fixture.patient, control="time_shift", seed=seed + 12)
    moving = _moving_average_proba(y, fixture.patient)
    random_warning = np.full_like(full, float(np.mean(y)), dtype=np.float64)
    alarm_time = _within_patient_roll(full, fixture.patient, shift=17)
    gated_name, gated_baseline = _best_baseline(
        y,
        {
            "logistic_nuisance": baseline,
            "moving_average": moving,
            "random_warning": random_warning,
            "alarm_time_surrogate": alarm_time,
        },
    )
    survival = discrete_survival_labels(y, bins=3)
    payload = {
        "n": int(len(y)),
        "positive_events": int(np.sum(y)),
        "event_patients": int(len(set(fixture.patient[y == 1]))),
        "survival_label_shape": list(survival.shape),
        "gated_baseline_name": gated_name,
        "gated_baseline_nll": _nll(y, gated_baseline),
        "baseline_nll": _nll(y, baseline),
        "moving_average_nll": _nll(y, moving),
        "random_warning_nll": _nll(y, random_warning),
        "alarm_time_nll": _nll(y, alarm_time),
        "logistic_full": _rfs_payload(y, gated_baseline, full, fixture.patient, seed=seed),
        "gbm_full": _rfs_payload(y, gated_baseline, gbm, fixture.patient, seed=seed + 1),
        "shuffled_target_control": _rfs_payload(y, gated_baseline, shuffled, fixture.patient, seed=seed + 2),
        "time_shift_control": _rfs_payload(y, gated_baseline, shifted, fixture.patient, seed=seed + 3),
        "nuisance_probes": {
            "patient": _probe_accuracy(z, fixture.patient),
            "site": _probe_accuracy(z, fixture.site),
            "time_bucket": _probe_accuracy(z, fixture.time_bucket),
            "session": _probe_accuracy(z, fixture.session),
        },
    }
    traces = {
        "y": np.asarray(y, dtype=np.int64),
        "patient": np.asarray(fixture.patient),
        "gated_baseline": np.asarray(gated_baseline, dtype=np.float64),
        "logistic_full": np.asarray(full, dtype=np.float64),
    }
    return payload, traces


def _crossfit_proba(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    factory: Callable[[], Any],
    *,
    control: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float64)
    folds = GroupKFold(n_splits=min(4, len(set(groups))))
    rng = np.random.default_rng(seed)
    for train_idx, test_idx in folds.split(x, y, groups):
        y_train = y[train_idx].copy()
        if control == "shuffle":
            rng.shuffle(y_train)
        elif control == "time_shift":
            y_train = _shift_train_labels(y_train, groups[train_idx], shift=9)
        pred[test_idx] = _fit_predict(factory(), x[train_idx], y_train, x[test_idx])
    return np.clip(pred, 1e-5, 1.0 - 1e-5)


def _crossfit_residual_proba(
    b: np.ndarray,
    z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    control: str | None = None,
    seed: int = 0,
) -> np.ndarray:
    pred = np.zeros(len(y), dtype=np.float64)
    folds = GroupKFold(n_splits=min(4, len(set(groups))))
    rng = np.random.default_rng(seed)
    for train_idx, test_idx in folds.split(z, y, groups):
        y_train = y[train_idx].copy()
        if control == "shuffle":
            rng.shuffle(y_train)
        elif control == "time_shift":
            y_train = _shift_train_labels(y_train, groups[train_idx], shift=9)
        nuisance = _logistic_factory()
        q0_train = _fit_predict(nuisance, b[train_idx], y_train, b[train_idx])
        q0_test = _fit_predict(nuisance, b[train_idx], y_train, b[test_idx])
        pred[test_idx] = _fit_residual_offset_predict(z[train_idx], y_train, q0_train, z[test_idx], q0_test)
    return np.clip(pred, 1e-5, 1.0 - 1e-5)


def _fit_predict(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    if len(set(y_train)) < 2:
        return np.full(x_test.shape[0], float(np.mean(y_train)), dtype=np.float64)
    if model == "stable_logistic":
        return _fit_stable_logistic_predict(x_train, y_train, x_test)
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)


def _fit_stable_logistic_predict(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test_raw: np.ndarray,
    *,
    steps: int = 300,
    lr: float = 0.1,
    l2: float = 0.1,
) -> np.ndarray:
    x_train, x_test = _scaled_design(x_train_raw, x_test_raw)
    w = np.zeros(x_train.shape[1], dtype=np.float64)
    for _ in range(steps):
        w = np.nan_to_num(w, nan=0.0, posinf=20.0, neginf=-20.0)
        p = _sigmoid_array(np.clip(_safe_matmul(x_train, w), -50.0, 50.0))
        grad = _safe_matmul(x_train.T, p - y_train) / len(y_train)
        grad[1:] += l2 * w[1:]
        grad = np.nan_to_num(np.clip(grad, -5.0, 5.0), nan=0.0, posinf=5.0, neginf=-5.0)
        w = np.nan_to_num(np.clip(w - lr * grad, -20.0, 20.0), nan=0.0, posinf=20.0, neginf=-20.0)
    return _sigmoid_array(np.clip(_safe_matmul(x_test, w), -50.0, 50.0))


def _fit_residual_offset_predict(
    z_train: np.ndarray,
    y_train: np.ndarray,
    q0_train: np.ndarray,
    z_test: np.ndarray,
    q0_test: np.ndarray,
    *,
    steps: int = 40,
    l2: float = 0.2,
) -> np.ndarray:
    x_train, x_test = _scaled_design(z_train, z_test)
    w = np.zeros(x_train.shape[1], dtype=np.float64)
    offset = _logit(q0_train)
    for _ in range(steps):
        eta = np.clip(offset + _safe_matmul(x_train, w), -50.0, 50.0)
        p = _sigmoid_array(eta)
        weights = np.clip(p * (1.0 - p), 1e-5, None)
        grad = _safe_matmul(x_train.T, p - y_train) / len(y_train)
        grad[1:] += l2 * w[1:]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            hess = (x_train.T * weights) @ x_train / len(y_train)
        hess = np.nan_to_num(hess, nan=0.0, posinf=1e6, neginf=-1e6)
        hess[1:, 1:] += np.eye(x_train.shape[1] - 1) * l2
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hess) @ grad
        step = np.nan_to_num(step, nan=0.0, posinf=5.0, neginf=-5.0)
        old_obj = _offset_logistic_objective(x_train, y_train, offset, w, l2)
        for scale in (1.0, 0.5, 0.25, 0.125):
            candidate = w - scale * step
            if _offset_logistic_objective(x_train, y_train, offset, candidate, l2) <= old_obj:
                w = candidate
                break
        if float(np.linalg.norm(step)) < 1e-6:
            break
    return _sigmoid_array(np.clip(_logit(q0_test) + _safe_matmul(x_test, w), -50.0, 50.0))


def _offset_logistic_objective(x: np.ndarray, y: np.ndarray, offset: np.ndarray, w: np.ndarray, l2: float) -> float:
    eta = np.clip(offset + _safe_matmul(x, w), -50.0, 50.0)
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta) + 0.5 * l2 * float(np.dot(w[1:], w[1:])))


def _safe_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = left @ right
    return np.nan_to_num(out, nan=0.0, posinf=50.0, neginf=-50.0)


def _scaled_design(x_train_raw: np.ndarray, x_test_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    raw_train = np.nan_to_num(np.asarray(x_train_raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    raw_test = np.nan_to_num(np.asarray(x_test_raw, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x_train = np.nan_to_num(scaler.fit_transform(raw_train), nan=0.0, posinf=20.0, neginf=-20.0)
    x_test = np.nan_to_num(scaler.transform(raw_test), nan=0.0, posinf=20.0, neginf=-20.0)
    x_train = np.concatenate([np.ones((x_train.shape[0], 1)), np.clip(x_train, -20.0, 20.0)], axis=1)
    x_test = np.concatenate([np.ones((x_test.shape[0], 1)), np.clip(x_test, -20.0, 20.0)], axis=1)
    return x_train, x_test


def _rfs_payload(y: np.ndarray, baseline: np.ndarray, pred: np.ndarray, patient: np.ndarray, *, seed: int, bootstrap_mode: str = "smoke") -> dict[str, float | str]:
    values = _cluster_bootstrap_rfs(y, baseline, pred, patient, seed=seed, mode=bootstrap_mode)
    values["rfs_bits"] = _rfs_bits(y, baseline, pred)
    values["nll"] = _nll(y, pred)
    return values


def _rows_by_group(groups: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Positions 0..G-1 of the sorted unique groups and each group's row indices.

    rows[i] is exactly np.flatnonzero(groups == np.unique(groups)[i]): the same rows in
    the same ascending order, and no rows for a NaN group, since NaN == NaN is False.
    It takes one sort instead of one full scan per group, so a cluster bootstrap over
    G subjects stops costing G full scans per resample.

    Draw from the returned positions with the same rng call the old code made on the
    unique groups (rng.choice(positions, size=G, replace=True)). Generator.choice draws
    from the array's length only, so the resamples stay identical.
    """
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    rows = np.split(order, np.cumsum(np.bincount(inverse, minlength=len(unique)))[:-1])
    if unique.dtype.kind in "fc":
        for i in np.flatnonzero(np.isnan(unique)):
            rows[i] = order[:0]
    return np.arange(len(unique)), rows


# Processes that score bootstrap resamples. Override per run with the environment
# variable below or per call with workers=; 1 runs the serial loop.
BOOTSTRAP_WORKERS = 16
BOOTSTRAP_WORKERS_ENV = "NEUROTWIN_BOOTSTRAP_WORKERS"
# Below this many rows x resamples, starting the processes costs more than they save,
# so a call that didn't ask for workers explicitly stays serial.
_PARALLEL_MIN_WORK = 20_000_000


def bootstrap_workers() -> int:
    """The default worker count: NEUROTWIN_BOOTSTRAP_WORKERS if set, else 16."""
    raw = os.environ.get(BOOTSTRAP_WORKERS_ENV, "").strip()
    if not raw:
        return BOOTSTRAP_WORKERS
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"{BOOTSTRAP_WORKERS_ENV} must be a positive integer, got {raw!r}")
    return int(raw)


def _effective_workers(workers: int | None, n_rows: int, n_boot: int) -> int:
    if workers is None:
        workers = bootstrap_workers() if n_rows * n_boot >= _PARALLEL_MIN_WORK else 1
    elif workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    # Workers are forked so they share the arrays without copying them. Without fork,
    # or inside a daemon process (which may not start children), stay serial.
    if "fork" not in multiprocessing.get_all_start_methods() or multiprocessing.current_process().daemon:
        return 1
    return min(workers, n_boot)


def _resample_rfs(chosen: np.ndarray, y: np.ndarray, baseline: np.ndarray, pred: np.ndarray, rows: list[np.ndarray]) -> float:
    idx = np.concatenate([rows[i] for i in chosen.tolist()])
    return _rfs_bits(y[idx], baseline[idx], pred[idx])


_FORKED_DATA: tuple | None = None  # set only while a worker pool is open; children inherit it


def _resample_rfs_forked(chosen: np.ndarray) -> float:
    return _resample_rfs(chosen, *_FORKED_DATA)


def _cluster_bootstrap_rfs(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    patient: np.ndarray,
    *,
    seed: int,
    mode: str = "smoke",
    workers: int | None = None,
) -> dict[str, float | str]:
    """Subject-clustered bootstrap CI of the residual bits of pred over baseline.

    workers > 1 scores the resamples in forked processes. The parent draws every
    resample first, from the same generator in the same order as the serial loop, and
    pool.map returns the scores in that order, so the samples and the CI are
    bit-identical to workers=1. workers=None uses bootstrap_workers() for large inputs
    and the serial loop for small ones.
    """
    global _FORKED_DATA
    n_boot = 2_000 if mode == "claim" else 300
    rng = np.random.default_rng(seed)
    positions, rows = _rows_by_group(patient)
    n_workers = _effective_workers(workers, len(y), n_boot)
    if n_workers == 1:
        samples = []
        for _ in range(n_boot):
            chosen = rng.choice(positions, size=len(positions), replace=True)
            samples.append(_resample_rfs(chosen, y, baseline, pred, rows))
    else:
        draws = [rng.choice(positions, size=len(positions), replace=True) for _ in range(n_boot)]
        _FORKED_DATA = (y, baseline, pred, rows)
        try:
            with multiprocessing.get_context("fork").Pool(n_workers) as pool:
                samples = pool.map(_resample_rfs_forked, draws, chunksize=max(1, n_boot // (4 * n_workers)))
        finally:
            _FORKED_DATA = None
    return {
        "rfs_ci_low": float(np.percentile(samples, 2.5)),
        "rfs_ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_mode": mode,
        "bootstrap_samples": n_boot,
    }


def _rfs_bits(y: np.ndarray, baseline: np.ndarray, pred: np.ndarray) -> float:
    return float((_nll(y, baseline) - _nll(y, pred)) / np.log(2.0))


def _best_baseline(y: np.ndarray, candidates: dict[str, np.ndarray]) -> tuple[str, np.ndarray]:
    name = min(candidates, key=lambda key: _nll(y, candidates[key]))
    return name, candidates[name]


def _nll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _moving_average_proba(y: np.ndarray, patient: np.ndarray, *, window: int = 20) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float64)
    base = float(np.mean(y))
    for group in np.unique(patient):
        idx = np.flatnonzero(patient == group)
        history: list[int] = []
        for row in idx:
            out[row] = float(np.mean(history[-window:])) if history else base
            history.append(int(y[row]))
    return np.clip(out, 1e-5, 1.0 - 1e-5)


def _within_patient_roll(values: np.ndarray, patient: np.ndarray, *, shift: int) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    for group in np.unique(patient):
        idx = np.flatnonzero(patient == group)
        out[idx] = np.roll(values[idx], shift)
    return np.clip(out, 1e-5, 1.0 - 1e-5)


def _shift_train_labels(y: np.ndarray, groups: np.ndarray, *, shift: int) -> np.ndarray:
    out = y.copy()
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        out[idx] = np.roll(out[idx], shift)
    return out


def _probe_accuracy(z: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {"accuracy": 1.0, "chance": 1.0}
    train = np.arange(len(labels)) % 3 != 0
    test = ~train
    x = np.nan_to_num(np.asarray(z, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    centers = np.asarray([np.mean(x[train & (labels == label)], axis=0) for label in classes], dtype=np.float64)
    distances = np.sum((x[test, None, :] - centers[None, :, :]) ** 2, axis=2)
    pred = classes[np.argmin(distances, axis=1)]
    accuracy = float(np.mean(pred == labels[test]))
    return {"accuracy": accuracy, "chance": float(1.0 / len(classes))}


def _logistic_factory() -> Any:
    return "stable_logistic"


def _gbm_factory(seed: int) -> Callable[[], Any]:
    def factory() -> Any:
        return HistGradientBoostingClassifier(max_iter=64, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)

    return factory


def _known_passes(payload: dict[str, Any]) -> bool:
    full = payload["logistic_full"]
    shuffled = payload["shuffled_target_control"]
    shifted = payload["time_shift_control"]
    return bool(
        payload["positive_events"] >= 40
        and payload["event_patients"] >= 8
        and full["rfs_bits"] > 0.03
        and full["rfs_ci_low"] > 0.0
        and shuffled["rfs_bits"] < full["rfs_bits"] * 0.4
        and shifted["rfs_bits"] < full["rfs_bits"] * 0.4
    )


def _null_passes(payload: dict[str, Any]) -> bool:
    full = payload["logistic_full"]
    return bool(payload["positive_events"] >= 20 and full["rfs_ci_low"] <= 0.0 <= full["rfs_ci_high"] and abs(full["rfs_bits"]) < 0.03)


def _mean_abs_corr(window: np.ndarray) -> float:
    if window.shape[1] < 2:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(window, rowvar=False)
    tri = corr[np.triu_indices_from(corr, k=1)]
    finite = np.abs(tri[np.isfinite(tri)])
    return float(np.mean(finite)) if finite.size else 0.0


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def _sigmoid_array(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(p / (1.0 - p))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    known = gate["known_signal"]
    null = gate["synthetic_null"]
    lines = [
        "# Kahlus Forecastability Trial 0 - M1 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        "",
        "## Synthetic Known Signal",
        "",
        _fixture_lines(known),
        "",
        "## Synthetic Null",
        "",
        _fixture_lines(null),
        "",
        "M1 stops here. M2 should not start until this instrumentation gate is reviewed.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fixture_lines(payload: dict[str, Any]) -> str:
    full = payload["logistic_full"]
    shuffled = payload["shuffled_target_control"]
    shifted = payload["time_shift_control"]
    return "\n".join(
        [
            f"- rows/events/event-patients: `{payload['n']}` / `{payload['positive_events']}` / `{payload['event_patients']}`",
            f"- logistic RFS bits: `{full['rfs_bits']:.6f}` CI `[ {full['rfs_ci_low']:.6f}, {full['rfs_ci_high']:.6f} ]`",
            f"- GBM RFS bits: `{payload['gbm_full']['rfs_bits']:.6f}`",
            f"- shuffled-target RFS bits: `{shuffled['rfs_bits']:.6f}`",
            f"- time-shift RFS bits: `{shifted['rfs_bits']:.6f}`",
            f"- gated baseline: `{payload.get('gated_baseline_name', 'logistic_nuisance')}` NLL `{payload.get('gated_baseline_nll', payload['baseline_nll']):.6f}`",
            f"- baseline/moving/random/alarm NLL: `{payload['baseline_nll']:.6f}` / `{payload['moving_average_nll']:.6f}` / `{payload['random_warning_nll']:.6f}` / `{payload['alarm_time_nll']:.6f}`",
        ]
    )

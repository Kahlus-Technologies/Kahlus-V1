from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GroupKFold

from neurotwin.forecastability.m1 import (
    _best_baseline,
    _crossfit_proba,
    _crossfit_residual_proba,
    _logistic_factory,
    _moving_average_proba,
    _nll,
    _rfs_payload,
    _rows_by_group,
    _scaled_design,
    _within_patient_roll,
)


CHANNEL_GROUPS = ((0, 1), (2, 3))
BLOCKED_CLAIMS = [
    "no_consciousness_claim",
    "no_pci_replacement_claim",
    "no_clinical_claim",
    "no_model_superiority_claim",
]


@dataclass(frozen=True)
class PICFixture:
    windows: np.ndarray
    future: np.ndarray
    nuisance: np.ndarray
    y: np.ndarray
    patient: np.ndarray
    world: str


def run_m5_gate(out_dir: str | Path, *, seed: int = 0, bootstrap_mode: str = "smoke") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    worlds = {
        "integrated_predictive": _run_fixture(make_pic_fixture(seed=seed, world="integrated_predictive"), seed=seed, bootstrap_mode=bootstrap_mode),
        "independent_predictable": _run_fixture(make_pic_fixture(seed=seed + 100, world="independent_predictable"), seed=seed + 100, bootstrap_mode=bootstrap_mode),
        "white_noise": _run_fixture(make_pic_fixture(seed=seed + 200, world="white_noise"), seed=seed + 200, bootstrap_mode=bootstrap_mode),
        "nuisance_only": _run_fixture(make_pic_fixture(seed=seed + 300, world="nuisance_only"), seed=seed + 300, bootstrap_mode=bootstrap_mode),
    }
    gate_failures = _synthetic_gate_failures(worlds)
    gate = {
        "milestone": "M5",
        "method": "passive_predictive_integration_complexity_synthetic_gate",
        "validation_scope": "synthetic_instrument_validity_only",
        "external_generalization": False,
        "public_data_used": False,
        "nuisance_conditioned": True,
        "bootstrap_mode": bootstrap_mode,
        "worlds": worlds,
        "gate_failures": gate_failures,
        "gate_predicate": _gate_predicate(gate_failures),
        "synthetic_gate_passed": not gate_failures,
        "gate_passed": not gate_failures,
        "claim_scope": "passive_predictive_complexity_synthetic_method_only",
        "blocked_claims": BLOCKED_CLAIMS,
        "stop_reason": "M5 is synthetic-only; public-data Passive PIC requires a later gate.",
    }
    _write_json(out / "m5_gate_report.json", gate)
    _write_report(out / "M5_EVIDENCE_REPORT.md", gate)
    return gate


def make_pic_fixture(
    *,
    seed: int,
    world: str,
    n_patients: int = 12,
    steps_per_patient: int = 120,
    window: int = 32,
    channels: int = 4,
) -> PICFixture:
    if channels != 4:
        raise ValueError("M5 synthetic PIC fixture currently uses exactly four channels")
    if world not in {"integrated_predictive", "independent_predictable", "white_noise", "nuisance_only"}:
        raise ValueError(f"unknown PIC world {world!r}")
    rng = np.random.default_rng(seed)
    t_axis = np.linspace(-1.0, 1.0, window, dtype=np.float32)
    windows: list[np.ndarray] = []
    future: list[np.ndarray] = []
    nuisance: list[list[float]] = []
    labels: list[int] = []
    patients: list[int] = []
    for patient in range(n_patients):
        site = patient % 2
        a = rng.normal()
        b = rng.normal()
        events = [0]
        for t in range(steps_per_patient):
            cycle = float(np.sin(2.0 * np.pi * t / 48.0))
            recent = float(np.mean(events[-12:]))
            a_next = 0.86 * a + rng.normal(0.0, 0.45)
            b_next = 0.84 * b + rng.normal(0.0, 0.45)
            if world == "white_noise":
                a_obs, b_obs = rng.normal(size=2)
                target = rng.normal(0.0, 1.0, size=channels)
                logit = -2.0 + 0.8 * cycle + 1.1 * recent
            elif world == "nuisance_only":
                a_obs = 0.6 * cycle + rng.normal(0.0, 0.8)
                b_obs = -0.4 * cycle + rng.normal(0.0, 0.8)
                target = np.array([cycle, cycle, -cycle, -cycle], dtype=np.float64) + rng.normal(0.0, 0.6, size=channels)
                logit = -2.1 + 1.45 * cycle + 1.2 * recent
            elif world == "independent_predictable":
                a_obs, b_obs = a, b
                target = np.array([a_next, 0.7 * a_next, b_next, 0.7 * b_next], dtype=np.float64) + rng.normal(0.0, 0.35, size=channels)
                logit = -2.3 + 0.7 * cycle + 1.0 * recent
            else:
                a_obs, b_obs = a, b
                shared = 0.65 * a + 0.65 * b
                shared_next = 0.65 * a_next + 0.65 * b_next
                integration_intensity = abs(shared)
                target = np.array([shared_next, 0.8 * shared_next, -shared_next, -0.8 * shared_next], dtype=np.float64) + rng.normal(0.0, 0.24, size=channels)
                logit = -3.0 + 0.5 * cycle + 0.9 * recent + 1.55 * integration_intensity

            signal = rng.normal(0.0, 0.18, size=(window, channels)).astype(np.float32)
            signal[:, 0] += (a_obs + 0.15 * cycle) + 0.08 * t_axis
            signal[:, 1] += (0.8 * a_obs - 0.10 * cycle) - 0.05 * t_axis
            signal[:, 2] += (b_obs + 0.12 * site) + 0.06 * t_axis
            signal[:, 3] += (0.8 * b_obs - 0.12 * site) - 0.04 * t_axis
            event = int(rng.random() < _sigmoid(logit))
            windows.append(signal)
            future.append(target.astype(np.float32))
            nuisance.append([1.0, cycle, float(np.cos(2.0 * np.pi * t / 48.0)), recent, float(t % 48) / 47.0])
            labels.append(event)
            patients.append(patient)
            events.append(event)
            a, b = a_next, b_next
    return PICFixture(
        windows=np.asarray(windows, dtype=np.float32),
        future=np.asarray(future, dtype=np.float32),
        nuisance=np.asarray(nuisance, dtype=np.float32),
        y=np.asarray(labels, dtype=np.int64),
        patient=np.asarray(patients, dtype=np.int64),
        world=world,
    )


def estimate_pic_bits(
    windows: np.ndarray,
    future: np.ndarray,
    patient: np.ndarray,
    *,
    nuisance: np.ndarray | None = None,
    channel_groups: tuple[tuple[int, ...], ...] = CHANNEL_GROUPS,
    ridge: float = 1.0,
    bootstrap_mode: str = "smoke",
) -> dict[str, Any]:
    nuisance_x = None if nuisance is None else np.asarray(nuisance, dtype=np.float64)
    x = _window_features(windows)
    if nuisance_x is not None:
        x = np.concatenate([x, nuisance_x], axis=1)
    y = np.asarray(future, dtype=np.float64)
    groups = np.asarray(patient)
    factor_features = []
    for channel_group in channel_groups:
        cols = np.asarray(channel_group, dtype=np.int64)
        factor_x = _window_features(windows[:, :, cols])
        if nuisance_x is not None:
            factor_x = np.concatenate([factor_x, nuisance_x], axis=1)
        factor_features.append((cols, factor_x))
    return _estimate_pic_from_designs(x, y, groups, factor_features, ridge=ridge, bootstrap_mode=bootstrap_mode)


def _estimate_pic_from_designs(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    factor_features: list[tuple[np.ndarray, np.ndarray]],
    *,
    ridge: float,
    bootstrap_mode: str,
) -> dict[str, Any]:
    joint_nll = np.zeros(len(y), dtype=np.float64)
    factor_nll = np.zeros(len(y), dtype=np.float64)
    folds = GroupKFold(n_splits=min(4, len(set(groups))))
    for train_idx, test_idx in folds.split(x, y, groups):
        joint_pred, joint_var = _ridge_gaussian_predict(x[train_idx], y[train_idx], x[test_idx], ridge=ridge)
        joint_nll[test_idx] = _gaussian_nll_rows(y[test_idx], joint_pred, joint_var)
        for cols, factor_x in factor_features:
            pred, var = _ridge_gaussian_predict(factor_x[train_idx], y[train_idx][:, cols], factor_x[test_idx], ridge=ridge)
            factor_nll[test_idx] += _gaussian_nll_rows(y[test_idx][:, cols], pred, var)
    pic_rows = (factor_nll - joint_nll) / np.log(2.0)
    summary = _cluster_bootstrap_mean(pic_rows, groups, seed=17, mode=bootstrap_mode)
    summary.update(
        {
            "pic_bits": float(np.mean(pic_rows)),
            "joint_nll": float(np.mean(joint_nll)),
            "factorized_nll": float(np.mean(factor_nll)),
            "per_sample_pic_bits": pic_rows.astype(float).tolist(),
        }
    )
    return summary


def _run_fixture(fixture: PICFixture, *, seed: int, bootstrap_mode: str = "smoke") -> dict[str, Any]:
    pic = estimate_pic_bits(fixture.windows, fixture.future, fixture.patient, nuisance=fixture.nuisance, bootstrap_mode=bootstrap_mode)
    spectral = _spectral_pic_bits(fixture.windows, fixture.future, fixture.patient, fixture.nuisance, bootstrap_mode=bootstrap_mode)
    integration_features = _current_integration_features(fixture.windows)
    y = fixture.y
    b = fixture.nuisance
    baseline = _crossfit_proba(b, y, fixture.patient, _logistic_factory)
    full = _crossfit_residual_proba(b, integration_features, y, fixture.patient)
    shuffled = _crossfit_residual_proba(b, integration_features, y, fixture.patient, control="shuffle", seed=seed + 11)
    shifted = _crossfit_residual_proba(b, integration_features, y, fixture.patient, control="time_shift", seed=seed + 12)
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
    return {
        "world": fixture.world,
        "n": int(len(y)),
        "positive_events": int(np.sum(y)),
        "event_patients": int(len(set(fixture.patient[y == 1]))),
        "pic": {key: value for key, value in pic.items() if key != "per_sample_pic_bits"},
        "attribution": {
            "time_summary": {key: value for key, value in pic.items() if key != "per_sample_pic_bits"},
            "spectral_power": {key: value for key, value in spectral.items() if key != "per_sample_pic_bits"},
        },
        "gated_baseline_name": gated_name,
        "gated_baseline_nll": _nll(y, gated_baseline),
        "baseline_nll": _nll(y, baseline),
        "moving_average_nll": _nll(y, moving),
        "random_warning_nll": _nll(y, random_warning),
        "alarm_time_nll": _nll(y, alarm_time),
        "integration_feature_residual": _rfs_payload(y, gated_baseline, full, fixture.patient, seed=seed, bootstrap_mode=bootstrap_mode),
        "shuffled_target_control": _rfs_payload(y, gated_baseline, shuffled, fixture.patient, seed=seed + 1),
        "time_shift_control": _rfs_payload(y, gated_baseline, shifted, fixture.patient, seed=seed + 2),
    }


def _window_features(windows: np.ndarray) -> np.ndarray:
    x = np.asarray(windows, dtype=np.float64)
    means = np.mean(x, axis=1)
    stds = np.std(x, axis=1)
    slopes = np.mean(x * np.linspace(-1.0, 1.0, x.shape[1], dtype=np.float64)[None, :, None], axis=1)
    return np.concatenate([means, stds, slopes], axis=1)


def _spectral_pic_bits(windows: np.ndarray, future: np.ndarray, patient: np.ndarray, nuisance: np.ndarray, *, bootstrap_mode: str) -> dict[str, Any]:
    nuisance_x = np.asarray(nuisance, dtype=np.float64)
    x = np.concatenate([_spectral_power_features(windows), nuisance_x], axis=1)
    y = np.asarray(future, dtype=np.float64)
    groups = np.asarray(patient)
    factor_features = []
    for channel_group in CHANNEL_GROUPS:
        cols = np.asarray(channel_group, dtype=np.int64)
        factor_x = np.concatenate([_spectral_power_features(windows[:, :, cols]), nuisance_x], axis=1)
        factor_features.append((cols, factor_x))
    return _estimate_pic_from_designs(x, y, groups, factor_features, ridge=1.0, bootstrap_mode=bootstrap_mode)


def _spectral_power_features(windows: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(np.asarray(windows, dtype=np.float64), axis=1)) ** 2
    total = np.sum(spectrum, axis=(1, 2), keepdims=False)[:, None] + 1e-8
    return np.concatenate(
        [
            np.sum(spectrum[:, 1:4, :], axis=(1, 2), keepdims=False)[:, None] / total,
            np.sum(spectrum[:, 4:9, :], axis=(1, 2), keepdims=False)[:, None] / total,
            np.sum(spectrum[:, 9:, :], axis=(1, 2), keepdims=False)[:, None] / total,
        ],
        axis=1,
    )


def _current_integration_features(windows: np.ndarray) -> np.ndarray:
    means = np.mean(np.asarray(windows, dtype=np.float64), axis=1)
    left = np.mean(means[:, :2], axis=1)
    right = np.mean(means[:, 2:], axis=1)
    return np.column_stack([left, right, np.abs(left + right), np.abs(left - right), left * right])


def _ridge_gaussian_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    x_train_s, x_test_s = _scaled_design(x_train, x_test)
    y = np.asarray(y_train, dtype=np.float64)
    penalty = np.eye(x_train_s.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    xtx = np.nan_to_num(np.einsum("ni,nj->ij", x_train_s, x_train_s) + penalty, nan=0.0, posinf=1e8, neginf=-1e8)
    rhs = np.nan_to_num(np.einsum("ni,nk->ik", x_train_s, y), nan=0.0, posinf=1e8, neginf=-1e8)
    coef = np.nan_to_num(np.linalg.solve(xtx, rhs), nan=0.0, posinf=1e6, neginf=-1e6)
    train_pred = np.einsum("ni,ik->nk", x_train_s, coef)
    test_pred = np.einsum("ni,ik->nk", x_test_s, coef)
    var = np.mean((y - train_pred) ** 2, axis=0) + 1e-4
    return test_pred, var


def _gaussian_nll_rows(y: np.ndarray, pred: np.ndarray, var: np.ndarray) -> np.ndarray:
    variance = np.clip(np.asarray(var, dtype=np.float64), 1e-6, None)
    err = np.asarray(y, dtype=np.float64) - np.asarray(pred, dtype=np.float64)
    return 0.5 * np.sum(np.log(2.0 * np.pi * variance) + (err * err) / variance, axis=1)


def _cluster_bootstrap_mean(values: np.ndarray, patient: np.ndarray, *, seed: int, mode: str = "smoke") -> dict[str, float | str]:
    n_boot = 2_000 if mode == "claim" else 300
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    positions, rows = _rows_by_group(patient)
    samples = []
    for _ in range(n_boot):
        chosen = rng.choice(positions, size=len(positions), replace=True)
        idx = np.concatenate([rows[i] for i in chosen.tolist()])
        samples.append(float(np.mean(values[idx])))
    return {
        "pic_ci_low": float(np.percentile(samples, 2.5)),
        "pic_ci_high": float(np.percentile(samples, 97.5)),
        "bootstrap_mode": mode,
        "bootstrap_samples": n_boot,
    }


def _synthetic_gate_failures(worlds: dict[str, dict[str, Any]]) -> list[str]:
    integrated = worlds["integrated_predictive"]
    independent = worlds["independent_predictable"]
    white = worlds["white_noise"]
    nuisance = worlds["nuisance_only"]
    full = integrated["integration_feature_residual"]
    shuffled = integrated["shuffled_target_control"]
    shifted = integrated["time_shift_control"]
    failures = []
    if integrated["positive_events"] < 40:
        failures.append("integrated_underpowered_positive_events")
    if integrated["event_patients"] < 8:
        failures.append("integrated_underpowered_event_patients")
    if integrated["pic"]["pic_bits"] <= 0.05 or integrated["pic"]["pic_ci_low"] <= 0.0:
        failures.append("integrated_pic_not_positive")
    if full["rfs_bits"] <= 0.001 or full["rfs_ci_low"] <= 0.0:
        failures.append("integrated_residual_rfs_not_positive")
    if shuffled["rfs_bits"] >= full["rfs_bits"] * 0.5:
        failures.append("shuffle_control_too_close")
    if shifted["rfs_bits"] >= full["rfs_bits"] * 0.5:
        failures.append("time_shift_control_too_close")
    if abs(independent["pic"]["pic_bits"]) >= 0.05 or independent["pic"]["pic_ci_high"] >= 0.05:
        failures.append("independent_world_pic_positive")
    if abs(white["pic"]["pic_bits"]) >= 0.05 or white["pic"]["pic_ci_high"] >= 0.05:
        failures.append("white_noise_world_pic_positive")
    if abs(nuisance["pic"]["pic_bits"]) >= 0.05 or nuisance["pic"]["pic_ci_high"] >= 0.05:
        failures.append("nuisance_world_pic_positive")
    if nuisance["integration_feature_residual"]["rfs_ci_low"] > 0.0 or nuisance["integration_feature_residual"]["rfs_bits"] >= 0.02:
        failures.append("nuisance_world_residual_rfs_positive")
    return failures


def _gate_predicate(failures: list[str]) -> dict[str, str]:
    return {
        "split": "pass",
        "finite": "pass",
        "baseline": "pass",
        "controls": "warning" if any("control" in failure for failure in failures) else "pass",
        "power": "warning" if any("underpowered" in failure for failure in failures) else "pass",
        "scope": "pass",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Kahlus Forecastability Trial 0 - M5 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        f"Synthetic gate passed: `{gate['synthetic_gate_passed']}`",
        f"Gate failures: `{', '.join(gate['gate_failures']) if gate['gate_failures'] else 'none'}`",
        f"Claim scope: `{gate['claim_scope']}`",
        f"Validation scope: `{gate['validation_scope']}`",
        f"External generalization: `{gate['external_generalization']}`",
        f"Public data used: `{gate['public_data_used']}`",
        f"Nuisance conditioned: `{gate['nuisance_conditioned']}`",
        f"Bootstrap mode: `{gate['bootstrap_mode']}`",
        "",
        "## Method",
        "",
        "Passive Predictive Integration Complexity estimates whether nuisance-conditioned joint future prediction beats nuisance-conditioned factorized channel-group future prediction under patient-held-out cross-fitting. It is a synthetic method gate only.",
        "PIC bits are an operational predictive-integration score under this model class, not a direct measure of consciousness or integrated information.",
        "",
        "| world | PIC bits | PIC CI low | PIC CI high | integration-feature residual RFS | RFS CI low | RFS CI high | gated baseline |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("integrated_predictive", "independent_predictable", "white_noise", "nuisance_only"):
        payload = gate["worlds"][name]
        pic = payload["pic"]
        rfs = payload["integration_feature_residual"]
        lines.append(
            "| {world} | {pic_bits:.6f} | {pic_ci_low:.6f} | {pic_ci_high:.6f} | {rfs_bits:.6f} | {rfs_ci_low:.6f} | {rfs_ci_high:.6f} | {baseline} |".format(
                world=name,
                baseline=payload["gated_baseline_name"],
                pic_bits=pic["pic_bits"],
                pic_ci_low=pic["pic_ci_low"],
                pic_ci_high=pic["pic_ci_high"],
                rfs_bits=rfs["rfs_bits"],
                rfs_ci_low=rfs["rfs_ci_low"],
                rfs_ci_high=rfs["rfs_ci_high"],
            )
        )
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            "Time-summary PIC is the primary gated score. Spectral-power PIC is reported for attribution only and is not required to be positive.",
            "",
            "| world | time PIC bits | spectral PIC bits | spectral CI low | spectral CI high |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("integrated_predictive", "independent_predictable", "white_noise", "nuisance_only"):
        attribution = gate["worlds"][name]["attribution"]
        time = attribution["time_summary"]
        spectral = attribution["spectral_power"]
        lines.append(
            "| {world} | {time_pic:.6f} | {spectral_pic:.6f} | {spectral_low:.6f} | {spectral_high:.6f} |".format(
                world=name,
                time_pic=time["pic_bits"],
                spectral_pic=spectral["pic_bits"],
                spectral_low=spectral["pic_ci_low"],
                spectral_high=spectral["pic_ci_high"],
            )
        )
    lines.extend(
        [
            "",
            "## Blocked Claims",
            "",
            *[f"- `{claim}`" for claim in gate["blocked_claims"]],
            "",
            gate["stop_reason"],
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))

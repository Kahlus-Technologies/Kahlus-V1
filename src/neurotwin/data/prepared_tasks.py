from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.data.forecast_contract import (
    FORECAST_PROTOCOL_V2_NONOVERLAP,
    ForecastProtocolError,
    ForecastTaskSpec,
    ResolvedForecastTaskSpec,
    WindowExampleProvenance,
    assert_forecast_scoring_is_leakage_safe,
    strictly_future_metric_mask,
)
from neurotwin.data.schemas import NeuralEventBatch
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest
from neurotwin.data.windows import WindowSpec, batch_to_windows


@dataclass(frozen=True)
class PreparedSuiteConfig:
    event_manifest: str | Path
    split_manifest: str | Path
    window_length: int = 8
    stride: int = 8
    seed: int = 0
    train_steps: int = 5
    require_ci: bool = True
    max_windows_per_split: int | None = None
    model_ids: tuple[str, ...] | None = None
    forecast_task: ForecastTaskSpec | None = None


@dataclass(frozen=True)
class SupervisedWindowTask:
    task_id: str
    source_modality: str
    target_modality: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    forecast_spec: ResolvedForecastTaskSpec | None = None
    train_provenance: tuple[WindowExampleProvenance, ...] = ()
    val_provenance: tuple[WindowExampleProvenance, ...] = ()
    test_provenance: tuple[WindowExampleProvenance, ...] = ()
    metric_mask: np.ndarray | None = None
    train_metric_mask: np.ndarray | None = None
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    val_metric_mask: np.ndarray | None = None
    notes: tuple[str, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate_provenance("train", self.x_train, self.train_provenance)
        self._validate_provenance("test", self.x_test, self.test_provenance)
        if self.x_val is not None:
            self._validate_provenance("val", self.x_val, self.val_provenance)
        elif self.val_provenance:
            raise ValueError("val provenance exists without validation inputs")
        if self.forecast_spec is not None:
            for rows in (self.train_provenance, self.val_provenance, self.test_provenance):
                for row in rows:
                    row.validate_against(self.forecast_spec)
            if self.forecast_spec.claim_eligible and not (self.train_provenance and self.test_provenance):
                raise ValueError("claim-eligible forecast tasks require train and test provenance")

    @staticmethod
    def _validate_provenance(
        split: str,
        values: np.ndarray,
        provenance: tuple[WindowExampleProvenance, ...],
    ) -> None:
        if provenance and len(provenance) != int(np.asarray(values).shape[0]):
            raise ValueError(f"{split} provenance count does not match supervised examples")


def build_prepared_window_tasks(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
    seed: int = 0,
    max_windows_per_split: int | None = None,
    forecast_task: ForecastTaskSpec | None = None,
) -> tuple[tuple[SupervisedWindowTask, ...], list[dict[str, str]]]:
    skipped: list[dict[str, str]] = []
    windows_by_split = prepared_windows_by_split(
        batches,
        split,
        window_length=window_length,
        stride=stride,
        max_windows_per_split=max_windows_per_split,
    )
    split_keys = _split_record_keys(split)
    for batch in batches:
        if _record_id(batch) not in split_keys:
            skipped.append({"task_id": "all", "reason": f"event not present in split manifest: {_record_id(batch)}"})

    tasks: list[SupervisedWindowTask] = []
    future = (
        _nonoverlapping_future_task_from_batches(
            batches,
            split,
            forecast_task=forecast_task,
            max_windows_per_split=max_windows_per_split,
        )
        if forecast_task is not None
        else _future_task_from_windows(windows_by_split)
    )
    if future is not None:
        tasks.append(future)
    else:
        skipped.append({"task_id": "future_state_forecasting", "reason": "need train and test windows for one modality"})

    masked = _masked_task_from_windows(windows_by_split, seed=seed)
    if masked is not None:
        tasks.append(masked)
    else:
        skipped.append({"task_id": "masked_neural_reconstruction", "reason": "need train and test windows for one modality"})

    cross_modal = _cross_modal_task_from_windows(windows_by_split)
    if cross_modal is not None:
        tasks.append(cross_modal)
    else:
        skipped.append({"task_id": "cross_modal_translation", "reason": "need paired train/test windows for two modalities"})

    stimulus_fmri = _stimulus_fmri_task_from_windows(windows_by_split)
    if stimulus_fmri is not None:
        tasks.append(stimulus_fmri)
    else:
        skipped.append({"task_id": "stimulus_to_fmri_response", "reason": "need fMRI train/test windows with aligned stimulus embeddings"})
    return tuple(tasks), skipped


def prepared_windows_by_split(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    window_length: int,
    stride: int,
    max_windows_per_split: int | None = None,
) -> dict[str, list[NeuralEventBatch]]:
    if max_windows_per_split is not None and int(max_windows_per_split) <= 0:
        raise ValueError("max_windows_per_split must be positive when provided")
    max_windows = int(max_windows_per_split) if max_windows_per_split is not None else None
    spec = WindowSpec(length=window_length, stride=stride)
    split_keys = _split_record_keys(split)
    windows_by_split: dict[str, list[NeuralEventBatch]] = {"train": [], "val": [], "test": []}
    for batch in batches:
        split_name = split_keys.get(_record_id(batch))
        if split_name is None:
            continue
        if max_windows is not None:
            remaining = max_windows - len(windows_by_split[split_name])
            if remaining <= 0:
                continue
            windows_by_split[split_name].extend(batch_to_windows(batch, spec)[:remaining])
        else:
            windows_by_split[split_name].extend(batch_to_windows(batch, spec))
    return windows_by_split


def first_prepared_modality_with_splits(windows_by_split: dict[str, list[NeuralEventBatch]]) -> str | None:
    train_modalities = {window.modality for window in windows_by_split["train"]}
    test_modalities = {window.modality for window in windows_by_split["test"]}
    for preferred in ("eeg", "fmri", "meg", "spikes"):
        if preferred in train_modalities and preferred in test_modalities:
            return preferred
    shared = sorted(train_modalities & test_modalities)
    return shared[0] if shared else None


def build_future_forecasting_task_from_windows(
    windows_by_split: dict[str, list[NeuralEventBatch]],
    *,
    task_id: str = "future_state_forecasting",
    notes: tuple[str, ...] = (),
    horizon: int = 1,
) -> SupervisedWindowTask | None:
    modality = first_prepared_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    train = [window.signal for window in windows_by_split["train"] if window.modality == modality]
    val = [window.signal for window in windows_by_split["val"] if window.modality == modality]
    test = [window.signal for window in windows_by_split["test"] if window.modality == modality]
    x_train, y_train = _legacy_overlapping_future_xy(train, horizon=horizon)
    x_val, y_val = _legacy_overlapping_future_xy(val, horizon=horizon)
    x_test, y_test = _legacy_overlapping_future_xy(test, horizon=horizon)
    if x_train is None or x_test is None or y_train is None or y_test is None:
        return None
    train_mask = strictly_future_metric_mask(y_train, forecast_horizon=horizon, input_length=x_train.shape[1])
    test_mask = strictly_future_metric_mask(y_test, forecast_horizon=horizon, input_length=x_test.shape[1])
    val_mask = (
        strictly_future_metric_mask(y_val, forecast_horizon=horizon, input_length=x_val.shape[1])
        if y_val is not None and x_val is not None
        else None
    )
    assert_forecast_scoring_is_leakage_safe(
        window_length=int(x_test.shape[1]),
        forecast_horizon=int(horizon),
        metric_mask=test_mask,
        context="prepared future forecasting task",
    )
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metric_mask=test_mask,
        train_metric_mask=train_mask,
        x_val=x_val,
        y_val=y_val,
        val_metric_mask=val_mask,
        notes=notes
        or (
            f"prepared {modality} next-state windows",
            "metric_mask scores only strictly-future target positions",
        ),
        metadata={
            "forecast_protocol_id": "kahlus.forecast.v1_overlap",
            "claim_eligible": False,
            "forecast_horizon": int(horizon),
            "strictly_future_metric_mask": True,
        },
    )


def _future_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    return build_future_forecasting_task_from_windows(windows_by_split)


def _nonoverlapping_future_task_from_batches(
    batches: list[NeuralEventBatch],
    split: SplitManifest,
    *,
    forecast_task: ForecastTaskSpec,
    max_windows_per_split: int | None,
) -> SupervisedWindowTask | None:
    """Build v2 forecast pairs directly from recordings, before any window slicing."""

    if forecast_task.protocol_id != FORECAST_PROTOCOL_V2_NONOVERLAP:
        raise ForecastProtocolError("the direct prepared-data builder only accepts the v2 non-overlap protocol")

    split_keys = _split_record_keys(split)
    available_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for batch in batches:
        split_name = split_keys.get(_record_id(batch))
        if split_name is not None:
            available_by_split[split_name].add(batch.modality)
    shared_modalities = available_by_split["train"] & available_by_split["test"]
    modality = next((name for name in ("eeg", "fmri", "meg", "spikes") if name in shared_modalities), None)
    if modality is None:
        return None

    records_by_id = {record.record_id: record for record in split.all_records}
    arrays: dict[str, list[np.ndarray]] = {name: [] for name in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")}
    provenance: dict[str, list[WindowExampleProvenance]] = {"train": [], "val": [], "test": []}
    resolved_spec: ResolvedForecastTaskSpec | None = None
    limit = int(max_windows_per_split) if max_windows_per_split is not None else None
    if limit is not None and limit <= 0:
        raise ValueError("max_windows_per_split must be positive when provided")

    for batch in batches:
        if batch.modality != modality:
            continue
        record_id = _record_id(batch)
        split_name = split_keys.get(record_id)
        record = records_by_id.get(record_id)
        if split_name is None or record is None:
            continue
        sampling_rate_hz = batch.sampling_rate
        if sampling_rate_hz is None:
            raise ForecastProtocolError(
                f"v2 forecasting requires sampling_rate metadata for record {record_id!r}"
            )
        current_spec = forecast_task.resolve(sampling_rate_hz)
        if resolved_spec is None:
            resolved_spec = current_spec
        elif current_spec != resolved_spec:
            raise ForecastProtocolError(
                "a supervised forecast task requires recordings with one resolved sampling grid; "
                "partition mixed sampling rates before building the task"
            )
        remaining = None if limit is None else limit - len(provenance[split_name])
        if remaining is not None and remaining <= 0:
            continue
        x, y, rows = _nonoverlapping_recording_windows(
            batch,
            record=record,
            split=split_name,
            spec=current_spec,
            limit=remaining,
        )
        if x.size == 0:
            continue
        arrays[f"{split_name}_x"].append(x)
        arrays[f"{split_name}_y"].append(y)
        provenance[split_name].extend(rows)

    if resolved_spec is None:
        return None
    x_train, y_train = _stack_supervised(arrays["train_x"]), _stack_supervised(arrays["train_y"])
    x_val, y_val = _stack_supervised(arrays["val_x"]), _stack_supervised(arrays["val_y"])
    x_test, y_test = _stack_supervised(arrays["test_x"]), _stack_supervised(arrays["test_y"])
    if x_train.size == 0 or x_test.size == 0:
        return None
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val if x_val.size else None,
        y_val=y_val if y_val.size else None,
        forecast_spec=resolved_spec,
        train_provenance=tuple(provenance["train"]),
        val_provenance=tuple(provenance["val"]),
        test_provenance=tuple(provenance["test"]),
        notes=("prepared non-overlapping forecast windows built directly from source recordings",),
        metadata={
            "forecast_protocol_id": resolved_spec.spec.protocol_id,
            "forecast_schema_version": resolved_spec.spec.schema_version,
            "claim_eligible": resolved_spec.claim_eligible,
            "context_samples": resolved_spec.context_samples,
            "target_samples": resolved_spec.target_samples,
            "gap_samples": resolved_spec.gap_samples,
            "stride_samples": resolved_spec.stride_samples,
            "sampling_rate_hz": resolved_spec.sampling_rate_hz,
        },
    )


def _nonoverlapping_recording_windows(
    batch: NeuralEventBatch,
    *,
    record: RecordingRecord,
    split: str,
    spec: ResolvedForecastTaskSpec,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray, list[WindowExampleProvenance]]:
    signal = np.asarray(batch.signal, dtype=np.float32)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    rows: list[WindowExampleProvenance] = []
    max_start = signal.shape[0] - (spec.context_samples + spec.gap_samples + spec.target_samples)
    for start in range(0, max_start + 1, spec.stride_samples):
        input_start, input_stop, target_start, target_stop = spec.ranges(start)
        xs.append(signal[input_start:input_stop])
        ys.append(signal[target_start:target_stop])
        rows.append(
            WindowExampleProvenance(
                dataset_id=batch.dataset_id,
                subject_id=record.subject_id,
                session_id=record.session_id,
                site_id=record.site_id,
                record_id=record.record_id,
                source_hash=batch.source_hash,
                split=split,
                input_start=input_start,
                input_stop=input_stop,
                target_start=target_start,
                target_stop=target_stop,
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    if not xs:
        empty_x = np.empty((0, spec.context_samples, signal.shape[-1]), dtype=np.float32)
        empty_y = np.empty((0, spec.target_samples, signal.shape[-1]), dtype=np.float32)
        return empty_x, empty_y, []
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), rows


def _stack_supervised(parts: list[np.ndarray]) -> np.ndarray:
    if not parts:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _masked_task_from_windows(
    windows_by_split: dict[str, list[NeuralEventBatch]],
    seed: int,
) -> SupervisedWindowTask | None:
    modality = first_prepared_modality_with_splits(windows_by_split)
    if modality is None:
        return None
    train = np.asarray([window.signal for window in windows_by_split["train"] if window.modality == modality], dtype=np.float32)
    val = np.asarray([window.signal for window in windows_by_split["val"] if window.modality == modality], dtype=np.float32)
    test = np.asarray([window.signal for window in windows_by_split["test"] if window.modality == modality], dtype=np.float32)
    if train.size == 0 or test.size == 0:
        return None
    rng = np.random.default_rng(seed + 31)
    train_mask = rng.random(train.shape) < 0.2
    val_mask = rng.random(val.shape) < 0.2 if val.size else None
    test_mask = rng.random(test.shape) < 0.2
    x_train = train.copy()
    x_val = val.copy() if val.size else None
    x_test = test.copy()
    x_train[train_mask] = 0.0
    if x_val is not None and val_mask is not None:
        x_val[val_mask] = 0.0
    x_test[test_mask] = 0.0
    return SupervisedWindowTask(
        task_id="masked_neural_reconstruction",
        source_modality=modality,
        target_modality=modality,
        x_train=x_train,
        y_train=train,
        x_test=x_test,
        y_test=test,
        metric_mask=test_mask,
        x_val=x_val,
        y_val=val if val.size else None,
        val_metric_mask=val_mask,
        notes=(f"prepared {modality} masked reconstruction",),
    )


def _cross_modal_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    modalities = sorted({window.modality for split_windows in windows_by_split.values() for window in split_windows})
    if len(modalities) < 2:
        return None
    source = "eeg" if "eeg" in modalities else modalities[0]
    target_candidates = [modality for modality in modalities if modality != source]
    target = "fmri" if "fmri" in target_candidates else target_candidates[0]
    train_pairs = _paired_windows(windows_by_split["train"], source, target)
    val_pairs = _paired_windows(windows_by_split["val"], source, target)
    test_pairs = _paired_windows(windows_by_split["test"], source, target)
    if not train_pairs or not test_pairs:
        return None
    x_train, y_train = _stack_pairs(train_pairs)
    x_val, y_val = _stack_pairs(val_pairs) if val_pairs else (None, None)
    x_test, y_test = _stack_pairs(test_pairs)
    return SupervisedWindowTask(
        task_id="cross_modal_translation",
        source_modality=source,
        target_modality=target,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(f"prepared paired {source}->{target} windows",),
    )


def _stimulus_fmri_task_from_windows(windows_by_split: dict[str, list[NeuralEventBatch]]) -> SupervisedWindowTask | None:
    train = _stimulus_fmri_xy(windows_by_split["train"])
    val = _stimulus_fmri_xy(windows_by_split["val"])
    test = _stimulus_fmri_xy(windows_by_split["test"])
    if train is None or test is None:
        return None
    x_train, y_train = train
    x_test, y_test = test
    x_val, y_val = val if val is not None else (None, None)
    evidence = _stimulus_feature_evidence(
        [
            *windows_by_split["train"],
            *windows_by_split["val"],
            *windows_by_split["test"],
        ]
    )
    return SupervisedWindowTask(
        task_id="stimulus_to_fmri_response",
        source_modality="stimulus",
        target_modality="fmri",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        notes=(
            "prepared stimulus->fMRI response windows",
            "tribe_style is a clean_room_approximation, not an exact TRIBE v2 reproduction",
            str(evidence["claim_note"]),
        ),
        metadata={"stimulus_evidence": evidence},
    )


def _legacy_overlapping_future_xy(
    signals: list[np.ndarray],
    *,
    horizon: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Historical shifted-window task retained only for artifact compatibility.

    ``horizon`` defaults to 1 (the historical bug). Pair with
    ``strictly_future_metric_mask`` so reported scores cannot be inflated by the
    copyable overlap.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    usable = [np.asarray(signal, dtype=np.float32) for signal in signals if signal.shape[0] >= horizon + 1]
    if not usable:
        return None, None
    return (
        np.asarray([signal[:-horizon] for signal in usable], dtype=np.float32),
        np.asarray([signal[horizon:] for signal in usable], dtype=np.float32),
    )


def _paired_windows(
    windows: list[NeuralEventBatch],
    source: str,
    target: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[str, str, str, str, int], dict[str, NeuralEventBatch]] = {}
    for window in windows:
        key = (
            window.dataset,
            window.subject_id,
            window.session_id,
            window.site_id,
            int(window.metadata.get("window_start_index", 0)),
        )
        grouped.setdefault(key, {})[window.modality] = window
    pairs = []
    for group in grouped.values():
        if source in group and target in group:
            source_signal = group[source].signal
            target_signal = group[target].signal
            n_time = min(source_signal.shape[0], target_signal.shape[0])
            pairs.append((source_signal[:n_time], target_signal[:n_time]))
    return pairs


def _stimulus_fmri_xy(windows: list[NeuralEventBatch]) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = [
        (np.asarray(window.stimulus_embedding, dtype=np.float32), np.asarray(window.signal, dtype=np.float32))
        for window in windows
        if window.modality == "fmri"
        and window.stimulus_embedding is not None
        and window.stimulus_embedding.shape[0] == window.signal.shape[0]
    ]
    if not pairs:
        return None
    return (
        np.asarray([stimulus for stimulus, _ in pairs], dtype=np.float32),
        np.asarray([signal for _, signal in pairs], dtype=np.float32),
    )


def _stimulus_feature_evidence(windows: list[NeuralEventBatch]) -> dict[str, object]:
    used = [
        window
        for window in windows
        if window.modality == "fmri"
        and window.stimulus_embedding is not None
        and window.stimulus_embedding.shape[0] == window.signal.shape[0]
    ]
    if not used:
        return {
            "status": "missing",
            "require_real_stimulus": False,
            "claim_eligible": False,
            "claim_note": "stimulus-to-fMRI unavailable: no aligned stimulus embeddings",
            "sources": [],
            "modalities": [],
            "hashes": [],
        }
    sources = sorted({_optional_text(window.metadata.get("stimulus_feature_source")) for window in used if window.metadata.get("stimulus_feature_source")})
    hashes = sorted({_optional_text(window.metadata.get("stimulus_feature_hash")) for window in used if window.metadata.get("stimulus_feature_hash")})
    modalities = sorted(
        {
            str(item)
            for window in used
            for item in _metadata_list(window.metadata.get("stimulus_feature_modalities"))
        }
    )
    statuses = sorted({_optional_text(window.metadata.get("stimulus_feature_status")) for window in used if window.metadata.get("stimulus_feature_status")})
    source_artifacts = sorted(
        {
            str(item)
            for window in used
            for item in _stimulus_source_artifacts(window.metadata)
        }
    )
    require_real = all(bool(window.metadata.get("require_real_stimulus")) for window in used)
    source_text = " ".join(sources).lower()
    status_text = " ".join(statuses).lower()
    looks_hash_only = not sources or "transcript_hash" in source_text or ("hash" in source_text and not modalities)
    looks_synthetic = any(token in f"{source_text} {status_text}" for token in ("synthetic", "default", "placeholder", "plumbing", "hash_only"))
    real_status = bool(statuses) and all(_real_stimulus_status(status) for status in statuses)
    source_verified = all(_stimulus_source_verified(window) for window in used)
    source_artifact_hash_verified = all(_stimulus_source_artifact_hash_verified(window) for window in used)
    embedding_hashes = sorted({digest for window in used for digest in _stimulus_embedding_hashes(window)})
    failure_reasons = _stimulus_evidence_failure_reasons(
        require_real=require_real,
        sources=sources,
        hashes=hashes,
        modalities=modalities,
        statuses=statuses,
        source_artifacts=source_artifacts,
        looks_hash_only=looks_hash_only,
        looks_synthetic=looks_synthetic,
        real_status=real_status,
        source_verified=source_verified,
        source_artifact_hash_verified=source_artifact_hash_verified,
    )
    claim_eligible = not failure_reasons
    status = _stimulus_evidence_status(
        claim_eligible=claim_eligible,
        require_real=require_real,
        looks_hash_only=looks_hash_only,
        looks_synthetic=looks_synthetic,
        hashes=hashes,
        source_artifacts=source_artifacts,
        source_verified=source_verified,
        source_artifact_hash_verified=source_artifact_hash_verified,
    )
    note = (
        "stimulus-to-fMRI uses real precomputed stimulus features"
        if claim_eligible
        else "stimulus-to-fMRI is plumbing only: " + "; ".join(failure_reasons)
    )
    return {
        "status": status,
        "require_real_stimulus": bool(require_real),
        "claim_eligible": bool(claim_eligible),
        "claim_note": note,
        "sources": sources,
        "modalities": modalities,
        "hashes": hashes,
        "feature_statuses": statuses,
        "source_artifacts": source_artifacts,
        "source_artifact_hash_verified": bool(source_artifact_hash_verified),
        "hash_verified": bool(source_artifact_hash_verified),
        "stimulus_embedding_hash": embedding_hashes[0] if len(embedding_hashes) == 1 else None,
        "stimulus_embedding_hashes": embedding_hashes,
        "failure_reasons": failure_reasons,
    }


def _stack_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([source for source, _ in pairs], dtype=np.float32),
        np.asarray([target for _, target in pairs], dtype=np.float32),
    )


def _record_id(batch: NeuralEventBatch) -> str:
    return str(batch.metadata.get("record_id") or batch.metadata.get("source_record_id"))


def _split_record_keys(split: SplitManifest) -> dict[str, str]:
    keys = {}
    for split_name in ("train", "val", "test"):
        for record in getattr(split, split_name):
            keys[record.record_id] = split_name
    return keys


def _metadata_list(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return tuple(value)
    return (value,)


def _optional_text(value: object) -> str:
    return str(value)


def _stimulus_evidence_failure_reasons(
    *,
    require_real: bool,
    sources: list[str],
    hashes: list[str],
    modalities: list[str],
    statuses: list[str],
    source_artifacts: list[str],
    looks_hash_only: bool,
    looks_synthetic: bool,
    real_status: bool,
    source_verified: bool,
    source_artifact_hash_verified: bool,
) -> list[str]:
    reasons: list[str] = []
    if not require_real:
        reasons.append("require_real_stimulus is false")
    if not sources:
        reasons.append("stimulus_feature_source is missing")
    if not modalities:
        reasons.append("stimulus_feature_modalities is missing")
    if not hashes:
        reasons.append("stimulus_feature_hash is missing")
    if not statuses:
        reasons.append("stimulus_feature_status is missing")
    elif not real_status:
        reasons.append("stimulus_feature_status is not real/precomputed")
    if not source_artifacts:
        reasons.append("stimulus feature path/manifest/uri is missing")
    elif not source_verified:
        reasons.append("stimulus feature source artifact is not verifiable")
    if looks_hash_only:
        reasons.append("stimulus_feature_source looks hash-derived")
    if looks_synthetic:
        reasons.append("stimulus feature source/status looks synthetic")
    if hashes and not source_artifact_hash_verified:
        reasons.append("stimulus_feature_hash is not verified")
    return reasons


def _real_stimulus_status(status: str) -> bool:
    text = status.lower()
    return any(token in text for token in ("real", "precomputed")) and not any(
        token in text for token in ("synthetic", "default", "placeholder", "plumbing", "hash_only")
    )


def _stimulus_evidence_status(
    *,
    claim_eligible: bool,
    require_real: bool,
    looks_hash_only: bool,
    looks_synthetic: bool,
    hashes: list[str],
    source_artifacts: list[str],
    source_verified: bool,
    source_artifact_hash_verified: bool,
) -> str:
    if claim_eligible:
        return "real_stimulus_features"
    if looks_hash_only or looks_synthetic or not require_real:
        return "plumbing_only"
    if hashes and source_artifacts and source_verified and not source_artifact_hash_verified:
        return "hash_mismatch"
    return "unverified"


def _stimulus_source_artifacts(metadata: dict[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for key in (
        "stimulus_feature_path",
        "stimulus_feature_file",
        "stimulus_feature_manifest",
        "stimulus_feature_manifest_path",
        "stimulus_feature_uri",
    ):
        for value in _metadata_list(metadata.get(key)):
            if value not in (None, ""):
                values.append(str(value))
    return tuple(values)


def _stimulus_source_verified(window: NeuralEventBatch) -> bool:
    artifacts = _stimulus_source_artifacts(window.metadata)
    if not artifacts:
        return False
    for artifact in artifacts:
        path = _local_stimulus_artifact_path(artifact)
        if path.is_file():
            return True
    return False


def _stimulus_source_artifact_hash_verified(window: NeuralEventBatch) -> bool:
    expected = {_normalize_hash(str(value)) for value in _metadata_list(window.metadata.get("stimulus_feature_hash")) if value not in (None, "")}
    expected.discard("")
    if not expected:
        return False
    for digest in _stimulus_artifact_hashes(window):
        if digest in expected:
            return True
    return False


def _stimulus_artifact_hashes(window: NeuralEventBatch) -> tuple[str, ...]:
    hashes: list[str] = []
    for artifact in _stimulus_source_artifacts(window.metadata):
        path = _local_stimulus_artifact_path(artifact)
        if not path.is_file():
            continue
        try:
            hashes.append(_sha256_bytes(path.read_bytes()))
        except OSError:
            continue
    return tuple(hashes)


def _local_stimulus_artifact_path(value: str) -> Path:
    text = str(value)
    if text.startswith("file://"):
        return Path(text[len("file://") :])
    return Path(text)


def _stimulus_embedding_hashes(window: NeuralEventBatch) -> tuple[str, ...]:
    if window.stimulus_embedding is None:
        return ()
    return (_sha256_bytes(np.ascontiguousarray(window.stimulus_embedding).tobytes()),)


def _normalize_hash(value: str) -> str:
    text = value.strip().lower()
    for prefix in ("sha256:", "sha256=", "hash:"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

"""Versioned contracts for leakage-audited neural signal forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np


FORECAST_PROTOCOL_V1_OVERLAP = "kahlus.forecast.v1_overlap"
FORECAST_PROTOCOL_V2_NONOVERLAP = "kahlus.forecast.v2_nonoverlap"


class ForecastProtocolError(ValueError):
    """Raised when a forecasting task cannot support its declared claim scope."""


@dataclass(frozen=True)
class ForecastTaskSpec:
    """Time-based forecasting protocol before resolution to recording sample indices."""

    protocol_id: str
    schema_version: int
    context_seconds: float
    target_seconds: float
    gap_seconds: float
    stride_seconds: float
    claim_eligible: bool

    def __post_init__(self) -> None:
        if self.protocol_id not in {FORECAST_PROTOCOL_V1_OVERLAP, FORECAST_PROTOCOL_V2_NONOVERLAP}:
            raise ForecastProtocolError(f"unsupported forecast protocol {self.protocol_id!r}")
        if self.schema_version < 1:
            raise ForecastProtocolError("schema_version must be positive")
        for name in ("context_seconds", "target_seconds", "stride_seconds"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ForecastProtocolError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.gap_seconds)) or self.gap_seconds < 0.0:
            raise ForecastProtocolError("gap_seconds must be finite and non-negative")
        if self.claim_eligible and self.protocol_id != FORECAST_PROTOCOL_V2_NONOVERLAP:
            raise ForecastProtocolError("only the v2 non-overlap protocol can be claim eligible")
        if self.claim_eligible and self.gap_seconds <= 0.0:
            raise ForecastProtocolError("claim-eligible v2 forecasting requires a positive gap_seconds")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ForecastTaskSpec":
        return cls(
            protocol_id=str(values.get("protocol_id", FORECAST_PROTOCOL_V2_NONOVERLAP)),
            schema_version=int(values.get("schema_version", 2)),
            context_seconds=float(values["context_seconds"]),
            target_seconds=float(values["target_seconds"]),
            gap_seconds=float(values["gap_seconds"]),
            stride_seconds=float(values["stride_seconds"]),
            claim_eligible=bool(values.get("claim_eligible", True)),
        )

    def resolve(self, sampling_rate_hz: float) -> "ResolvedForecastTaskSpec":
        sampling_rate_hz = float(sampling_rate_hz)
        if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0.0:
            raise ForecastProtocolError("sampling_rate_hz must be finite and positive")
        return ResolvedForecastTaskSpec(
            spec=self,
            sampling_rate_hz=sampling_rate_hz,
            context_samples=_seconds_to_samples(self.context_seconds, sampling_rate_hz, "context_seconds"),
            target_samples=_seconds_to_samples(self.target_seconds, sampling_rate_hz, "target_seconds"),
            gap_samples=_seconds_to_samples(self.gap_seconds, sampling_rate_hz, "gap_seconds", allow_zero=True),
            stride_samples=_seconds_to_samples(self.stride_seconds, sampling_rate_hz, "stride_seconds"),
        )


@dataclass(frozen=True)
class ResolvedForecastTaskSpec:
    """Forecast protocol resolved to one recording's sampling grid."""

    spec: ForecastTaskSpec
    sampling_rate_hz: float
    context_samples: int
    target_samples: int
    gap_samples: int
    stride_samples: int
    target_offset_samples: int | None = None

    def __post_init__(self) -> None:
        if min(self.context_samples, self.target_samples, self.stride_samples) < 1:
            raise ForecastProtocolError("context, target, and stride sample counts must be positive")
        if self.gap_samples < 0:
            raise ForecastProtocolError("gap_samples must be non-negative")
        if self.spec.claim_eligible and self.gap_samples < 1:
            raise ForecastProtocolError("claim-eligible v2 forecasting requires at least one gap sample")

    @property
    def resolved_target_offset_samples(self) -> int:
        if self.target_offset_samples is not None:
            return int(self.target_offset_samples)
        return self.context_samples + self.gap_samples

    @property
    def claim_eligible(self) -> bool:
        return self.spec.claim_eligible and self.spec.protocol_id == FORECAST_PROTOCOL_V2_NONOVERLAP

    def ranges(self, input_start: int) -> tuple[int, int, int, int]:
        input_start = int(input_start)
        if input_start < 0:
            raise ForecastProtocolError("input_start must be non-negative")
        input_stop = input_start + self.context_samples
        target_start = input_start + self.resolved_target_offset_samples
        target_stop = target_start + self.target_samples
        return input_start, input_stop, target_start, target_stop

    def assert_claim_eligible(self) -> None:
        if not self.claim_eligible:
            raise ForecastProtocolError(
                f"forecast protocol {self.spec.protocol_id!r} is historical or otherwise ineligible for claim-bearing evidence"
            )


@dataclass(frozen=True)
class WindowExampleProvenance:
    """Immutable recording-level provenance for one supervised forecast example."""

    dataset_id: str
    subject_id: str
    session_id: str
    site_id: str
    record_id: str
    source_hash: str | None
    split: str
    input_start: int
    input_stop: int
    target_start: int
    target_stop: int

    def __post_init__(self) -> None:
        for name in ("dataset_id", "subject_id", "session_id", "site_id", "record_id", "split"):
            if not str(getattr(self, name)).strip():
                raise ForecastProtocolError(f"{name} must be non-empty")
        if self.input_start < 0 or self.target_start < 0:
            raise ForecastProtocolError("window starts must be non-negative")
        if self.input_stop <= self.input_start or self.target_stop <= self.target_start:
            raise ForecastProtocolError("window stops must be greater than starts")

    @property
    def target_overlaps_input(self) -> bool:
        return self.input_start < self.target_stop and self.target_start < self.input_stop

    def validate_against(self, spec: ResolvedForecastTaskSpec) -> None:
        expected = spec.ranges(self.input_start)
        actual = (self.input_start, self.input_stop, self.target_start, self.target_stop)
        if actual != expected:
            raise ForecastProtocolError(f"window ranges {actual} do not match protocol ranges {expected}")
        if spec.claim_eligible and self.target_overlaps_input:
            raise ForecastProtocolError("claim-eligible forecast provenance cannot overlap its input")


def assert_forecast_scoring_is_leakage_safe(
    *,
    window_length: int,
    forecast_horizon: int,
    metric_mask: np.ndarray | None,
    context: str = "forecast task",
) -> None:
    """Fail construction of a forecast task whose score could be inflated by copying.

    A shifted-window task with ``forecast_horizon < window_length`` leaves
    ``window_length - forecast_horizon`` target positions already present in the
    input, so a model that copies the input scores near-zero on them. That is
    only admissible when a strictly-future ``metric_mask`` restricts scoring (and
    training) to the positions the input does not contain.

    Passes when the windows are disjoint by construction
    (``forecast_horizon >= window_length``) or when a non-empty mask is attached.
    """

    if forecast_horizon >= window_length:
        return
    if metric_mask is not None and bool(np.asarray(metric_mask).any()):
        return
    raise ForecastProtocolError(
        f"{context}: forecast_horizon={forecast_horizon} < window_length={window_length} "
        f"leaves {window_length - forecast_horizon} of {window_length} target positions "
        "inside the input window, and no strictly-future metric_mask was attached. "
        "Scores from this task would reward copying rather than forecasting. Either set "
        "forecast_horizon >= window_length for disjoint windows, or attach a "
        "strictly_future_metric_mask()."
    )


def strictly_future_metric_mask(
    y: np.ndarray,
    *,
    forecast_horizon: int,
    input_length: int | None = None,
) -> np.ndarray:
    """Boolean mask that is True only on target positions absent from the input.

    Assumes the target window starts ``forecast_horizon`` samples after the input
    start (the historical shifted-window convention). Positions whose absolute
    index still falls inside the input window are copyable and must not score.
    """

    y_arr = np.asarray(y)
    if y_arr.ndim < 2:
        raise ForecastProtocolError("y must have at least a batch and time axis")
    if forecast_horizon < 1:
        raise ForecastProtocolError("forecast_horizon must be >= 1")
    target_length = int(y_arr.shape[1])
    context = int(target_length if input_length is None else input_length)
    if context < 1:
        raise ForecastProtocolError("input_length must be positive")
    time_mask = np.zeros(target_length, dtype=bool)
    for index in range(target_length):
        absolute = forecast_horizon + index
        if absolute >= context:
            time_mask[index] = True
    if not bool(time_mask.any()):
        raise ForecastProtocolError(
            "strictly-future mask is empty; increase forecast_horizon or shorten the input window"
        )
    mask = np.zeros(y_arr.shape, dtype=bool)
    mask[:, time_mask, ...] = True
    return mask


def legacy_overlapping_forecast_spec(
    *,
    window_samples: int,
    forecast_horizon_samples: int,
    stride_samples: int,
    sampling_rate_hz: float = 1.0,
) -> ResolvedForecastTaskSpec:
    """Describe historical one-shift forecasting without making it claim eligible."""

    if window_samples < 2:
        raise ForecastProtocolError("legacy window_samples must be at least two")
    if forecast_horizon_samples < 1 or stride_samples < 1:
        raise ForecastProtocolError("legacy horizon and stride must be positive")
    sampling_rate_hz = float(sampling_rate_hz)
    spec = ForecastTaskSpec(
        protocol_id=FORECAST_PROTOCOL_V1_OVERLAP,
        schema_version=1,
        context_seconds=float(window_samples) / sampling_rate_hz,
        target_seconds=float(window_samples) / sampling_rate_hz,
        gap_seconds=0.0,
        stride_seconds=float(stride_samples) / sampling_rate_hz,
        claim_eligible=False,
    )
    return ResolvedForecastTaskSpec(
        spec=spec,
        sampling_rate_hz=sampling_rate_hz,
        context_samples=window_samples,
        target_samples=window_samples,
        gap_samples=0,
        stride_samples=stride_samples,
        target_offset_samples=forecast_horizon_samples,
    )


def _seconds_to_samples(seconds: float, sampling_rate_hz: float, name: str, *, allow_zero: bool = False) -> int:
    resolved = float(seconds) * sampling_rate_hz
    samples = int(round(resolved))
    if not math.isclose(resolved, samples, rel_tol=0.0, abs_tol=1e-8):
        raise ForecastProtocolError(f"{name}={seconds} does not resolve to an integer sample count at {sampling_rate_hz} Hz")
    if samples < 0 or (samples == 0 and not allow_zero):
        raise ForecastProtocolError(f"{name} resolves to an invalid sample count {samples}")
    return samples

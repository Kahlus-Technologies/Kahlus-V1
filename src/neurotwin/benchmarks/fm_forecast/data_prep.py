"""Per-window conversion of harness forecast arrays into model-ready tensors.

Shapes this module is written against, read off ``baseline_suite``/``prepared_tasks``
rather than assumed:

- ``task.x_train`` / ``x_val`` / ``x_test`` are ``(N, context_samples, C)`` ``float32``.
- ``task.y_train`` / ``y_val`` / ``y_test`` are ``(N, target_samples, C)`` ``float32``.
- A runner's prediction must equal ``task.y_test.shape`` exactly and be finite
  (``baseline_suite._validate_prediction``).
- For the F8-comparable Sleep-EDF lane: ``C = 1``, ``context_samples = 127`` at 100 Hz,
  ``target_samples = 1``, and horizon ``h`` is carried by ``gap_samples = h - 1`` in the
  resolved forecast spec - so the target is the single sample at ``i + 127 + h - 1``.

Leakage hygiene, which is the reason this module exists at all:

- **Every transform here is per-window and stateless.** No mean, sd, min, max, or filter
  state is ever computed across windows or shared between train, validation, and test.
  ``resample_windows(x)[k]`` is bit-identical to ``resample_windows(x[k : k + 1])[0]``,
  and there is a test that says so.
- **No normalisation happens here.** The harness already z-scores each recording, and
  plan section 4.1 makes the normalisation convention a single shared decision taken
  upstream of any model. Adding a second normalisation inside an adapter would silently
  give one entrant a different convention from the rest.

Resampling is FFT-based (the ``scipy.signal.resample`` algorithm, reimplemented on
``numpy.fft``) because SciPy is **not** a core dependency of this package - it appears in
``pyproject.toml`` only under the ``docs`` extra. Depending on it here would make the
resampler's availability, and therefore its exact numerics, environment-dependent. The
frequency-domain form also preserves a constant (DC-only) window exactly, which the
edge-zero-padded polyphase form does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: The harness task id these arrays come from (``prepared_tasks``/``baseline_suite``).
FORECAST_TASK_ID = "future_state_forecasting"

#: Sampling rate of the Sleep-EDF forecasting arena (plan section 3.1).
FORECAST_SAMPLING_RATE_HZ = 100.0


def _as_windows(x: np.ndarray, *, name: str) -> tuple[np.ndarray, bool]:
    """Return ``(N, T, C)`` float64 windows and whether a channel axis was added."""

    values = np.asarray(x)
    if values.dtype == object or not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be a numeric array, got dtype {values.dtype}")
    if values.ndim == 2:
        return values.astype(np.float64, copy=False)[:, :, None], True
    if values.ndim == 3:
        return values.astype(np.float64, copy=False), False
    raise ValueError(f"{name} must be (N, T) or (N, T, C), got shape {values.shape}")


def _fft_resample(x: np.ndarray, num: int) -> np.ndarray:
    """Fourier-method resample of real ``(N, T, C)`` windows along axis 1.

    Mirrors ``scipy.signal.resample`` for real input with no window: truncate or
    zero-pad the rFFT, fixing up the Nyquist bin so a real signal stays real and
    correctly scaled, then inverse-transform to ``num`` samples.
    """

    n_in = x.shape[1]
    if num == n_in:
        return x.copy()
    if num < 1:
        raise ValueError("resampled length must be positive")

    spectrum = np.fft.rfft(x, axis=1)
    n_keep = min(num, n_in)
    nyq = n_keep // 2 + 1
    out_bins = num // 2 + 1
    resampled_spectrum = np.zeros((x.shape[0], out_bins, x.shape[2]), dtype=spectrum.dtype)
    resampled_spectrum[:, :nyq, :] = spectrum[:, :nyq, :]
    if n_keep % 2 == 0:
        # The bin at +N/2 is a Nyquist component of one of the two grids. Downsampling
        # folds a cosine-only bin back into a full component (x2); upsampling splits the
        # source Nyquist component across +/-N/2 (x0.5).
        if num < n_in:
            resampled_spectrum[:, n_keep // 2, :] *= 2.0
        else:
            resampled_spectrum[:, n_keep // 2, :] *= 0.5
    resampled = np.fft.irfft(resampled_spectrum, n=num, axis=1)
    return resampled * (float(num) / float(n_in))


def _align_length(x: np.ndarray, n_samples_out: int) -> np.ndarray:
    """Trim or edge-pad ``(N, T, C)`` windows to ``n_samples_out``, keeping the tail.

    Patch-based encoders need a length that is a whole number of patches, which almost
    never equals ``round(T * sr_out / sr_in)``. Plan section 3.2 fixes the rule: each
    model gets the largest patch-aligned context **not exceeding** the incumbent's, so a
    too-long resampled window is trimmed **from the front**. That keeps the most recent
    history and, critically, keeps the window ending at the same instant, so no entrant
    ever sees samples closer to the target than another does.
    """

    n_have = x.shape[1]
    if n_samples_out == n_have:
        return x
    if n_samples_out < n_have:
        return x[:, n_have - n_samples_out :, :]
    pad = np.repeat(x[:, :1, :], n_samples_out - n_have, axis=1)
    return np.concatenate([pad, x], axis=1)


def resample_windows(
    x: np.ndarray,
    *,
    sr_in: float = FORECAST_SAMPLING_RATE_HZ,
    sr_out: float,
    n_samples_out: int | None = None,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Resample forecast windows from ``sr_in`` to ``sr_out``, per window.

    Args:
        x: ``(N, T)`` or ``(N, T, C)`` windows. A 2-D input is treated as ``C = 1`` and
            returned 2-D.
        sr_in: input sampling rate; 100 Hz for this benchmark's arena.
        sr_out: the pretrained encoder's native rate (256 Hz for EEGPT, 200 Hz for
            CBraMod/LaBraM/BIOT).
        n_samples_out: exact output length. When given and different from the natural
            resampled length ``round(T * sr_out / sr_in)``, the window is trimmed from the
            front (or edge-padded at the front, if short) per plan section 3.2. When
            ``None``, the natural length is used.
        dtype: output dtype, ``float32`` to match the harness arrays.

    Returns:
        Resampled windows with the input's dimensionality and the requested dtype.

    Deterministic and stateless: no randomness, no cross-window statistics, no filter
    state carried between calls.
    """

    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("sampling rates must be positive")
    windows, squeezed = _as_windows(x, name="x")
    if windows.shape[1] < 1:
        raise ValueError("forecast windows must have at least one time sample")
    if n_samples_out is not None and int(n_samples_out) < 1:
        raise ValueError("n_samples_out must be positive when provided")

    natural = int(round(windows.shape[1] * float(sr_out) / float(sr_in)))
    natural = max(natural, 1)
    resampled = _fft_resample(windows, natural)
    if n_samples_out is not None:
        resampled = _align_length(resampled, int(n_samples_out))
    out = np.ascontiguousarray(resampled.astype(dtype, copy=False))
    return out[:, :, 0] if squeezed else out


def to_channels_first(x: np.ndarray) -> np.ndarray:
    """``(N, T, C)`` (harness layout) -> ``(N, C, T)`` (encoder layout).

    A ``(N, T)`` input is promoted to ``(N, 1, T)``.
    """

    windows, _ = _as_windows(x, name="x")
    return np.ascontiguousarray(np.transpose(windows, (0, 2, 1)).astype(np.float32, copy=False))


def prepare_windows(
    x: np.ndarray,
    *,
    sr_in: float = FORECAST_SAMPLING_RATE_HZ,
    sr_out: float,
    n_samples_out: int | None = None,
) -> np.ndarray:
    """Harness windows -> model-ready ``(N, C, T)`` float32 tensor input."""

    resampled = resample_windows(x, sr_in=sr_in, sr_out=sr_out, n_samples_out=n_samples_out)
    return to_channels_first(resampled)


def flatten_targets(y: np.ndarray) -> np.ndarray:
    """``(N, target_samples, C)`` -> ``(N, target_samples * C)`` float32.

    The bolt-on head predicts a flat vector; for the F8 lane that vector has length 1.
    """

    values = np.asarray(y)
    if values.ndim == 1:
        return values.astype(np.float32, copy=False).reshape(-1, 1)
    return np.ascontiguousarray(values.astype(np.float32, copy=False).reshape(values.shape[0], -1))


def unflatten_targets(flat: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """``(N, D)`` head output -> the harness target shape, which a runner must match.

    ``target_shape`` is the shape of the split being predicted (e.g. ``task.y_test.shape``);
    the leading axis is taken from ``flat`` so a partial batch is still an error rather
    than a silent reshape.
    """

    values = np.asarray(flat, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    expected = int(np.prod(target_shape[1:])) if len(target_shape) > 1 else 1
    if values.shape[0] != int(target_shape[0]) or values.shape[1] != expected:
        raise ValueError(
            f"cannot reshape predictions {values.shape} into harness target shape {tuple(target_shape)}"
        )
    return np.ascontiguousarray(values.reshape(tuple(target_shape)))


@dataclass(frozen=True)
class ForecastArrays:
    """Model-ready arrays for one entrant, one lane, one horizon.

    ``x_*`` are ``(N, C, T)`` encoder inputs; ``y_*`` are ``(N, D)`` flat targets.
    ``test_target_shape`` is the harness ``y_test`` shape a runner's prediction must match.

    ``y_test`` is carried for scoring **outside** the training path. Nothing in
    ``conditions.run_condition`` or ``head.fit_head`` reads it, and there is a test that
    replaces it with NaNs and asserts predictions are unchanged.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    test_target_shape: tuple[int, ...]
    y_test: np.ndarray | None = None
    sr_in: float = FORECAST_SAMPLING_RATE_HZ
    sr_out: float = FORECAST_SAMPLING_RATE_HZ
    n_samples_out: int | None = None

    @property
    def n_channels(self) -> int:
        return int(self.x_train.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.x_train.shape[2])

    @property
    def output_dim(self) -> int:
        return int(self.y_train.shape[1])


def prepare_forecast_arrays(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray | None = None,
    sr_in: float = FORECAST_SAMPLING_RATE_HZ,
    sr_out: float = FORECAST_SAMPLING_RATE_HZ,
    n_samples_out: int | None = None,
) -> ForecastArrays:
    """Convert a harness forecast task's arrays into :class:`ForecastArrays`.

    Validation arrays are mandatory: every admissible condition selects on validation
    MSE (plan section 3.3), and a task without a validation split cannot be run under
    this protocol without looking at test.
    """

    if x_val is None or y_val is None:
        raise ValueError(
            "forecast conditions select on validation MSE, so a validation split is required; "
            "a task with x_val/y_val=None cannot be run without touching test"
        )
    if y_test is None:
        raise ValueError("y_test is required to fix the harness prediction shape a runner must match")

    target_shape = tuple(int(dim) for dim in np.asarray(y_test).shape)
    prepared = {
        name: prepare_windows(values, sr_in=sr_in, sr_out=sr_out, n_samples_out=n_samples_out)
        for name, values in (("x_train", x_train), ("x_val", x_val), ("x_test", x_test))
    }
    if int(prepared["x_test"].shape[0]) != target_shape[0]:
        raise ValueError(
            f"x_test has {prepared['x_test'].shape[0]} windows but y_test has {target_shape[0]}"
        )
    flat_train = flatten_targets(y_train)
    flat_val = flatten_targets(y_val)
    flat_test = flatten_targets(y_test)
    if flat_val.shape[1] != flat_train.shape[1] or flat_test.shape[1] != flat_train.shape[1]:
        raise ValueError("train, validation, and test targets must have the same flattened width")
    return ForecastArrays(
        x_train=prepared["x_train"],
        y_train=flat_train,
        x_val=prepared["x_val"],
        y_val=flat_val,
        x_test=prepared["x_test"],
        test_target_shape=target_shape,
        y_test=np.asarray(y_test, dtype=np.float32),
        sr_in=float(sr_in),
        sr_out=float(sr_out),
        n_samples_out=None if n_samples_out is None else int(n_samples_out),
    )

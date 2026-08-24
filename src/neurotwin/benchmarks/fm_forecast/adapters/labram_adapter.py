"""LaBraM-Base as a Kahlus forecast baseline runner.

LaBraM (`labram_base_patch200_200`, 12 layers, embed dim 200) is a bidirectional
neural transformer over 200-sample "patches": input is ``[B, electrodes, patches, 200]``
and ``forward_features`` with mean pooling returns one 200-dim vector per window
(``external/vendor/labram/modeling_finetune.py:373-388``, ``:467-472``).

Decisions this module freezes, and why:

- **Context window.** ``SR_OUT = 200`` Hz, ``N_SAMPLES_OUT = 200``: exactly one complete
  patch, i.e. 1.0 s. The F8 Sleep-EDF lane gives 127 samples at 100 Hz = 1.27 s, which
  resamples to 254 samples at 200 Hz. LaBraM's patch grid is structurally fixed at 200
  samples, so 254 is not representable; per plan section 3.2 each entrant gets the
  largest patch-aligned context **not exceeding** the incumbent's, hence 200 samples
  (1 patch), trimmed from the front by ``data_prep`` so the window still ends at the
  same instant as everybody else's.
- **Channels.** ``get_input_chans`` maps electrode names through
  ``standard_1020.index(name) + 1`` (``external/vendor/labram/utils.py:713-717``), but
  ``pos_embed`` only has ``128 + 1`` slots (``modeling_finetune.py:282-286``) while
  ``standard_1020`` lists 136 names, so any name at list index >= 128 indexes out of
  bounds - a known upstream bug. The Sleep-EDF lane has no LaBraM-legal montage anyway
  (Fpz-Cz / Pz-Oz bipolar derivations are not in ``standard_1020``), so this adapter
  uses *positional* channel ids ``range(C + 1)``, which is exactly the precedent set by
  the upstream Sleep-EDF finetune script
  (``external/vendor/eegpt/downstream/finetune_LaBraM_SleepEDF.py:71``), and hard-refuses
  any ``C`` that would push an index past slot 128.
- **Scaling.** Upstream divides microvolts by 100, i.e. feeds units of 0.1 mV
  (``engine_for_finetuning.py:64-65``). The harness hands us per-recording z-scored,
  unitless windows, so there is no honest conversion factor. ``input_scale`` is therefore
  exposed with default ``1.0`` (no silent rescaling) and is a *validation-tunable*
  hyperparameter: tune it on val MSE like any other, never on test.
- **Random-init twin.** ``load_encoder(None)`` builds the same architecture with no
  checkpoint. That is the pretraining-value control: same capacity, same context, same
  head, zero pretraining.

Import hygiene: nothing here imports ``torch`` or vendored code at module import time,
and the vendored module is loaded by file path under the namespaced ``sys.modules`` key
``kahlus_vendor_labram_modeling_finetune`` rather than by putting
``external/vendor/labram`` on ``sys.path``. The vendor trees collide on top-level names
(``labram/utils.py`` vs ``eegpt/downstream/utils.py``, ``model``, ``datasets``) and the
whole benchmark runs in one pytest process, so a bare ``import modeling_finetune`` would
make the winner depend on test ordering. ``modeling_finetune.py`` imports only ``math``,
``functools``, ``torch``, ``timm``, and ``einops`` - no intra-repo imports - so the
single namespaced load is sufficient.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from neurotwin.benchmarks.fm_forecast.data_prep import (
    FORECAST_SAMPLING_RATE_HZ,
    prepare_forecast_arrays,
)
from neurotwin.benchmarks.fm_forecast.vendor_import import (
    VendoredUpstreamMissing,
    require_vendored,
)
from neurotwin.benchmarks.fm_forecast.weights_manifest import resolve_checkpoint
from neurotwin.benchmarks.pretraining_contamination import (
    SLEEP_EDF_EVAL_DATASET_ID,
    is_evaluation_permitted,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import torch

    from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
    from neurotwin.data.prepared_tasks import SupervisedWindowTask


#: Canonical repo model id: joins against ``pretraining_contamination`` and
#: ``weights_manifest`` (whose ``upstream_id`` alias is ``labram``).
MODEL_ID = "labram_base"

#: ``external/upstreams.lock.json`` key / vendor directory name.
UPSTREAM_ID = "labram"

#: LaBraM's native rate; upstream resamples every corpus to 200 Hz.
SR_OUT = 200.0

#: Structurally fixed patch width: the temporal conv, the 200-dim embedding, and the
#: ``[B, C, P, 200]`` reshape all assume it (``modeling_finetune.py:390-397``).
PATCH_SAMPLES = 200

#: Context length handed to the encoder: exactly one complete patch (1.0 s).
N_SAMPLES_OUT = 200

#: ``forward_features`` mean-pooled output width for the Base model.
EMBED_DIM = 200

#: ``pos_embed`` is ``(1, 128 + 1, embed_dim)``: slot 0 is CLS, slots 1..128 are
#: electrodes (``modeling_finetune.py:282-286``). ``utils.standard_1020`` nevertheless
#: lists 136 names, so names at list index >= 128 produce ``input_chans`` values of
#: 129..136 and index out of bounds. This adapter refuses that rather than reproducing
#: the upstream bug.
MAX_CHANNEL_INDEX = 128

#: ``time_embed`` has 16 positions and is sliced, never interpolated
#: (``modeling_finetune.py:286``, ``:363-366``), capping a window at 16 patches / 16 s.
MAX_TIME_PATCHES = 16

#: Namespaced ``sys.modules`` key for the vendored encoder module.
VENDOR_MODULE_NAME = "kahlus_vendor_labram_modeling_finetune"

_VENDOR_MODULE_RELPATH = "modeling_finetune.py"

# Construction kwargs, chosen so the pinned checkpoint loads with **zero unexpected**
# keys and only zero-valued missing ones. Cross-checked against
# ``run_class_finetuning.get_models`` (:193-208) and the checkpoint's own key set:
#   num_classes=0        -> ``head`` becomes ``nn.Identity``; we only call
#                           ``forward_features``, and a classifier head would be dead
#                           weight in a forecasting probe.
#   use_abs_pos_emb=True -> the checkpoint ships ``student.pos_embed``; upstream's
#                           argparse default of False is wrong for this checkpoint and
#                           would crash the moment ``input_chans`` is passed.
#   use_mean_pooling=True-> ``fc_norm`` + mean over non-CLS tokens, the documented
#                           200-dim embedding.
#   init_values=0.1      -> upstream ``--layer_scale_init_value`` default; the checkpoint
#                           contains ``gamma_1``/``gamma_2`` per block.
#   qkv_bias=True        -> upstream default. ``q_bias``/``v_bias`` are absent from the
#                           checkpoint but zero-initialised, so this is numerically a
#                           no-op at load time (see ``_EXPECTED_MISSING_SUFFIXES``).
#   drop_path_rate=0.0   -> stochastic depth is a fine-tuning regulariser, not part of
#                           the pretrained function; 0.0 keeps embeddings deterministic.
_MODEL_KWARGS: dict[str, Any] = {
    "num_classes": 0,
    "use_abs_pos_emb": True,
    "use_rel_pos_bias": False,
    "use_mean_pooling": True,
    "init_values": 0.1,
    "qkv_bias": True,
    "drop_rate": 0.0,
    "attn_drop_rate": 0.0,
    "drop_path_rate": 0.0,
}

#: Checkpoint prefix stripped by upstream's loader (``run_class_finetuning.py:350-358``):
#: the pretrained file is a DINO-style student/teacher bundle.
_CHECKPOINT_PREFIX = "student."

#: Model parameters legitimately absent from the pretrained checkpoint. ``fc_norm`` is
#: the mean-pooling LayerNorm, which only exists in the finetune model and is created
#: as an identity-parameterised LayerNorm (weight 1, bias 0); ``q_bias``/``v_bias`` are
#: zero tensors. Every other missing key means the surgery is wrong.
_EXPECTED_MISSING_KEYS = frozenset({"fc_norm.weight", "fc_norm.bias"})
_EXPECTED_MISSING_SUFFIXES = (".attn.q_bias", ".attn.v_bias")

_ENCODER_CLASS: type | None = None


class LaBraMAdapterError(RuntimeError):
    """Raised when a LaBraM encoder cannot be built, loaded, or lawfully evaluated."""


def _load_vendored_module(*, vendor_dir_path: Path | str | None = None):
    """Import the vendored ``modeling_finetune`` under a namespaced ``sys.modules`` key.

    Deliberately *not* ``vendor_import.ensure_on_sys_path``: putting
    ``external/vendor/labram`` on ``sys.path`` would make the bare names ``utils``,
    ``model``, and ``datasets`` resolvable to LaBraM's files for the rest of the
    process, and the other three adapters ship files with those same names.
    """

    cached = sys.modules.get(VENDOR_MODULE_NAME)
    if cached is not None:
        return cached

    root = require_vendored(UPSTREAM_ID, vendor_dir_path=vendor_dir_path)
    source = root / _VENDOR_MODULE_RELPATH
    if not source.is_file():
        raise VendoredUpstreamMissing(
            f"vendored upstream {UPSTREAM_ID!r} is present at {root} but "
            f"{_VENDOR_MODULE_RELPATH!r} is missing, so the LaBraM encoder cannot be "
            "built. Re-vendor with `python3 scripts/vendor_upstreams.py --ids labram "
            "--include-restricted`."
        )

    spec = importlib.util.spec_from_file_location(VENDOR_MODULE_NAME, source)
    if spec is None or spec.loader is None:
        raise LaBraMAdapterError(f"cannot build an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution on purpose: timm's ``register_model`` decorator looks
    # the defining module up via ``sys.modules[fn.__module__]`` while the module body is
    # still running.
    sys.modules[VENDOR_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(VENDOR_MODULE_NAME, None)
        raise
    return module


def _validate_geometry(n_channels: int, n_samples: int) -> int:
    """Check the window against LaBraM's fixed patch grid; return the patch count.

    Runs before any model is constructed, so an illegal montage or window costs no
    parameter allocation.
    """

    channels = int(n_channels)
    samples = int(n_samples)
    if channels < 1:
        raise ValueError(f"n_channels must be >= 1, got {channels}")

    # ``input_chans = range(C + 1)`` reserves 0 for CLS, so the largest index used is C.
    # pos_embed has 129 slots (CLS + 128 electrodes, modeling_finetune.py:282-286) even
    # though utils.standard_1020 lists 136 names, so indices above 128 are an upstream
    # out-of-bounds bug we refuse to reproduce.
    max_input_chan = channels
    if max_input_chan > MAX_CHANNEL_INDEX:
        raise ValueError(
            f"LaBraM has {MAX_CHANNEL_INDEX} learned electrode positions plus CLS "
            f"(pos_embed is (1, {MAX_CHANNEL_INDEX + 1}, embed_dim)), but {channels} "
            f"channels would index pos_embed slot {max_input_chan}. Note that "
            "utils.standard_1020 lists 136 names: every name at list index >= "
            f"{MAX_CHANNEL_INDEX} is unusable, a known upstream out-of-bounds bug."
        )

    if samples < PATCH_SAMPLES or samples % PATCH_SAMPLES:
        raise ValueError(
            f"LaBraM's patch width is structurally fixed at {PATCH_SAMPLES} samples "
            f"(1.0 s at {SR_OUT:g} Hz), so n_samples must be a positive multiple of "
            f"{PATCH_SAMPLES}; got {samples}."
        )
    n_patches = samples // PATCH_SAMPLES
    if n_patches > MAX_TIME_PATCHES:
        raise ValueError(
            f"time_embed has {MAX_TIME_PATCHES} positions and is sliced without "
            f"interpolation, so at most {MAX_TIME_PATCHES} patches "
            f"({MAX_TIME_PATCHES * PATCH_SAMPLES} samples) fit; got {n_patches}."
        )
    return n_patches


def _encoder_class() -> type:
    """Define (once) the ``(N, C, T) -> (N, 200)`` wrapper around ``NeuralTransformer``.

    Defined lazily because it subclasses ``torch.nn.Module`` and this module must be
    importable without importing torch.
    """

    global _ENCODER_CLASS
    if _ENCODER_CLASS is not None:
        return _ENCODER_CLASS

    from torch import nn

    class LaBraMEncoder(nn.Module):
        """Harness-shaped wrapper: ``(N, C, T)`` float32 -> ``(N, EMBED_DIM)``."""

        def __init__(
            self,
            model: nn.Module,
            *,
            n_channels: int,
            n_patches: int,
            input_scale: float = 1.0,
            checkpoint_report: dict[str, Any] | None = None,
        ) -> None:
            super().__init__()
            self.model = model
            self.n_channels = int(n_channels)
            self.n_patches = int(n_patches)
            self.n_samples = int(n_patches) * PATCH_SAMPLES
            # Positional channel ids, CLS first; see the module docstring for why names
            # are not used on this lane.
            self.input_chans = list(range(self.n_channels + 1))
            self.input_scale = float(input_scale)
            self.checkpoint_report = dict(checkpoint_report or {})

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            if x.ndim != 3:
                raise ValueError(f"expected (N, C, T) input, got shape {tuple(x.shape)}")
            n, channels, samples = x.shape
            if int(channels) != self.n_channels:
                raise ValueError(
                    f"this encoder was built for {self.n_channels} channel(s) "
                    f"(pos_embed slots are per-channel), got {int(channels)}"
                )
            if int(samples) != self.n_samples:
                raise ValueError(
                    f"this encoder was built for {self.n_samples} samples "
                    f"({self.n_patches} x {PATCH_SAMPLES}), got {int(samples)}"
                )
            patched = (x * self.input_scale).reshape(n, self.n_channels, self.n_patches, PATCH_SAMPLES)
            return self.model.forward_features(patched, input_chans=self.input_chans)

    _ENCODER_CLASS = LaBraMEncoder
    return _ENCODER_CLASS


def _load_pretrained_state(model, checkpoint_path: Path) -> dict[str, Any]:
    """Apply upstream's key surgery and load the checkpoint; return a load report.

    Mirrors ``run_class_finetuning.py:334-371``: pick ``model``/``module``, keep only
    ``student.*`` and strip that prefix, drop ``relative_position_index`` buffers, drop
    keys the finetune model does not have or that disagree on shape. Unlike upstream -
    which only *prints* missing and unexpected keys - this raises on anything outside
    the documented allowlist, because a silently half-loaded encoder would be scored as
    if it were pretrained.
    """

    import torch

    # ``weights_only=True`` with a closed allowlist: the checkpoint is a training bundle
    # whose non-tensor pickled objects are an ``argparse.Namespace``, numpy scalars, and
    # their reconstruction helpers - the complete set of GLOBAL opcodes in the pickle,
    # all of which rebuild data rather than run code. The SHA-256 pin in
    # ``weights_manifest.resolve_checkpoint`` still runs first; this keeps a tampered
    # file from reaching the full pickle VM even if that pin were ever bypassed.
    import _codecs
    import argparse

    import numpy

    try:  # numpy >= 2 moved the module; the pickle still says ``numpy.core.multiarray``
        from numpy._core.multiarray import scalar as _numpy_scalar
    except ImportError:  # pragma: no cover - numpy < 2
        from numpy.core.multiarray import scalar as _numpy_scalar

    with torch.serialization.safe_globals(
        [
            numpy.dtype,
            # The pickled ``numpy.dtype`` instances rebuild as concrete DType-class
            # objects, whose type must itself be allowlisted (numpy >= 2 class layout).
            numpy.dtypes.Float64DType,
            _codecs.encode,
            argparse.Namespace,
            (_numpy_scalar, "numpy.core.multiarray.scalar"),
        ]
    ):
        raw = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)

    checkpoint_model = None
    selected_key = None
    if isinstance(raw, dict):
        for model_key in ("model", "module"):
            if model_key in raw and isinstance(raw[model_key], dict):
                checkpoint_model, selected_key = raw[model_key], model_key
                break
    if checkpoint_model is None:
        checkpoint_model = raw
    if not isinstance(checkpoint_model, dict):
        raise LaBraMAdapterError(
            f"checkpoint {checkpoint_path} does not contain a state dict under "
            "'model'/'module' nor at the top level"
        )

    total_entries = len(checkpoint_model)
    stripped = {
        key[len(_CHECKPOINT_PREFIX) :]: value
        for key, value in checkpoint_model.items()
        if key.startswith(_CHECKPOINT_PREFIX)
    }
    if not stripped:
        raise LaBraMAdapterError(
            f"checkpoint {checkpoint_path} has no {_CHECKPOINT_PREFIX!r}-prefixed "
            "entries, so it is not the DINO-style LaBraM pretraining bundle this "
            "adapter's key surgery is written against"
        )
    student_entries = len(stripped)

    stripped = {key: value for key, value in stripped.items() if "relative_position_index" not in key}
    model_state = model.state_dict()
    dropped = sorted(
        key
        for key, value in stripped.items()
        if key not in model_state or tuple(model_state[key].shape) != tuple(value.shape)
    )
    loadable = {key: value for key, value in stripped.items() if key not in dropped}

    incompatible = model.load_state_dict(loadable, strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    missing = sorted(incompatible.missing_keys)
    surprising = [
        key
        for key in missing
        if key not in _EXPECTED_MISSING_KEYS and not key.endswith(_EXPECTED_MISSING_SUFFIXES)
    ]
    if unexpected or surprising:
        raise LaBraMAdapterError(
            f"LaBraM checkpoint {checkpoint_path} did not load cleanly: "
            f"unexpected={unexpected}, unexplained missing={surprising}. Expected "
            "missing keys are only fc_norm.* (mean-pooling LayerNorm, finetune-only, "
            "identity-initialised) and blocks.*.attn.{q,v}_bias (zero-initialised)."
        )

    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_key": selected_key,
        "checkpoint_entries": int(total_entries),
        "student_entries": int(student_entries),
        "model_entries": int(len(model_state)),
        "loaded_entries": int(len(loadable)),
        "dropped_keys": dropped,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
    }


def load_encoder(
    checkpoint_path: Path | str | None,
    *,
    n_channels: int = 1,
    n_samples: int = N_SAMPLES_OUT,
    device: str = "cpu",
    input_scale: float = 1.0,
):
    """Build a LaBraM-Base encoder, optionally loading the pretrained checkpoint.

    Args:
        checkpoint_path: verified path from ``weights_manifest.resolve_checkpoint``, or
            ``None`` for the **random-init twin** - identical architecture with no
            pretraining, which is the control that makes "pretraining bought X" a
            measurable claim rather than an assumption.
        n_channels: channels in the harness window; ``input_chans`` becomes
            ``range(n_channels + 1)`` and must not exceed ``MAX_CHANNEL_INDEX``.
        n_samples: context length at 200 Hz; must be a multiple of ``PATCH_SAMPLES`` and
            at most ``MAX_TIME_PATCHES`` patches.
        device: where the encoder is placed; ``run_condition`` may move it again.
        input_scale: multiplier applied to the window before patching. Default ``1.0``:
            the harness's z-scored windows are unitless, upstream's ``/100`` convention
            assumes microvolts, and silently applying a conversion that does not exist
            would be a hidden preprocessing choice. Tune on validation MSE if at all.

    Returns:
        A ``torch.nn.Module`` mapping ``(N, C, T)`` float32 to ``(N, EMBED_DIM)``, with
        the load report available as ``.checkpoint_report``.
    """

    n_patches = _validate_geometry(n_channels, n_samples)
    vendored = _load_vendored_module()
    model = vendored.labram_base_patch200_200(**_MODEL_KWARGS)

    report: dict[str, Any] = {"pretrained": checkpoint_path is not None}
    if checkpoint_path is not None:
        report.update(_load_pretrained_state(model, Path(checkpoint_path)))

    encoder = _encoder_class()(
        model,
        n_channels=int(n_channels),
        n_patches=n_patches,
        input_scale=float(input_scale),
        checkpoint_report=report,
    )
    encoder.to(device)
    encoder.eval()
    return encoder


def _require_permitted_evaluation() -> None:
    """Fail closed unless this model is audited clean for the Sleep-EDF arena."""

    if not is_evaluation_permitted(MODEL_ID, SLEEP_EDF_EVAL_DATASET_ID):
        raise LaBraMAdapterError(
            f"evaluation of {MODEL_ID!r} on {SLEEP_EDF_EVAL_DATASET_ID!r} is not "
            "permitted by neurotwin.benchmarks.pretraining_contamination: the pair is "
            "either contaminated, unaudited, or the model is slate-excluded. Refusing "
            "to produce a number that could not be reported."
        )


def make_runner(
    weights_dir: Path | str,
    condition: "Condition",
    *,
    config: "ConditionConfig | None" = None,
    pretrained: bool = True,
    input_scale: float = 1.0,
) -> Callable[["SupervisedWindowTask", int, int], np.ndarray]:
    """Build a ``(task, seed, train_steps) -> predictions`` runner for LaBraM-Base.

    The returned callable matches ``baseline_runners.BaselineRunner`` exactly, so it can
    be wrapped in an ``ExecutableBaselineRunner`` beside ``gru``/``tiny_ssm`` and scored
    by the same code path. ``train_steps`` is interpreted as the epoch budget for the
    condition (the fit loop's own gradient-step count is recorded in the run log), which
    is how the local torch baselines spend their ``steps`` argument; ``seed`` is threaded
    into ``ConditionConfig`` so it drives head init, shuffling, and the encoder's random
    init in the un-pretrained arm.

    Args:
        weights_dir: directory holding the staged, SHA-verified checkpoints.
        condition: ``Condition.FROZEN`` or ``Condition.FULL_FT``.
            ``Condition.ACAUSAL_INFILL`` is refused downstream by ``run_condition``.
        config: tuning-grid point; ``condition``, ``seed``, and ``epochs`` are overridden
            per call.
        pretrained: ``False`` builds the random-init twin and never touches
            ``weights_dir``.
        input_scale: see :func:`load_encoder`.
    """

    _require_permitted_evaluation()
    from neurotwin.benchmarks.fm_forecast import conditions as _conditions

    weights_root = Path(weights_dir)
    base_config = config if config is not None else _conditions.ConditionConfig(condition=condition)

    def runner(task: "SupervisedWindowTask", seed: int, train_steps: int) -> np.ndarray:
        _require_permitted_evaluation()

        epochs = int(train_steps) if int(train_steps) > 0 else int(base_config.epochs)
        run_config = replace(
            base_config,
            condition=condition,
            seed=int(seed),
            epochs=epochs,
        )

        arrays = prepare_forecast_arrays(
            x_train=task.x_train,
            y_train=task.y_train,
            x_val=task.x_val,
            y_val=task.y_val,
            x_test=task.x_test,
            y_test=task.y_test,
            sr_in=FORECAST_SAMPLING_RATE_HZ,
            sr_out=SR_OUT,
            n_samples_out=N_SAMPLES_OUT,
        )

        checkpoint = resolve_checkpoint(MODEL_ID, weights_root) if pretrained else None
        # The random-init twin draws its parameters from the global torch RNG, and
        # ``run_condition`` only seeds *after* the encoder exists. Without this, two
        # calls with the same ``seed`` would compare two different control encoders.
        import torch

        torch.manual_seed(int(seed))
        encoder = load_encoder(
            checkpoint,
            n_channels=arrays.n_channels,
            n_samples=arrays.n_samples,
            input_scale=input_scale,
        )

        result = _conditions.run_condition(encoder, arrays, run_config)
        target_shape = tuple(int(dim) for dim in np.asarray(task.y_test).shape)
        return np.asarray(result.predictions, dtype=np.float32).reshape(target_shape)

    return runner

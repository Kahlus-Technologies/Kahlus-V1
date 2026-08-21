"""CBraMod as a Kahlus forecast baseline runner.

**Input scaling.** Upstream CBraMod trains on microvolt-scale EEG divided by 100, i.e.
units of 0.1 mV (``CBraMod/preprocessing/*``, and the same ``/100`` convention LaBraM
documents in ``engine_for_finetuning.py:64-65``; both upstreams inherit it from the
TUEG pipeline). The Kahlus harness hands adapters per-recording z-scored, *unitless*
windows, so there is no honest microvolt conversion factor to apply. ``input_scale`` is
therefore exposed with default ``1.0`` -- no silent rescaling -- and is a
*validation-tunable* hyperparameter, mirroring ``labram_adapter``: sweep it on validation
MSE like any other axis, never on test.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable

import numpy as np

from neurotwin.benchmarks.fm_forecast.data_prep import (
    prepare_forecast_arrays,
    unflatten_targets,
)
from neurotwin.benchmarks.fm_forecast.vendor_import import require_vendored
from neurotwin.benchmarks.fm_forecast.weights_manifest import resolve_checkpoint
from neurotwin.benchmarks.pretraining_contamination import (
    SLEEP_EDF_EVAL_DATASET_ID,
    has_model_selection_exposure,
    is_evaluation_permitted,
)


if TYPE_CHECKING:
    import torch

    from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
    from neurotwin.data.prepared_tasks import SupervisedWindowTask


MODEL_ID = "cbramod"
SR_OUT = 200.0
N_SAMPLES_OUT = 200
EMBED_DIM = 200

_PATCH_SAMPLES = 200
_VENDOR_TRANSFORMER_KEY = "kahlus_vendor_cbramod_criss_cross_transformer"
_VENDOR_MODEL_KEY = "kahlus_vendor_cbramod_model"


def _load_module_from_path(module_key: str, path: Path) -> ModuleType:
    existing = sys.modules.get(module_key)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot construct an import spec for vendored CBraMod at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_key] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(module_key) is module:
            del sys.modules[module_key]
        raise
    return module


def _load_vendor_cbramod_module() -> ModuleType:
    """Load CBraMod under private keys without retaining its bare ``models`` package."""

    existing = sys.modules.get(_VENDOR_MODEL_KEY)
    if existing is not None:
        return existing

    vendor_root = require_vendored("cbramod")
    models_dir = vendor_root / "models"
    transformer = _load_module_from_path(
        _VENDOR_TRANSFORMER_KEY,
        models_dir / "criss_cross_transformer.py",
    )

    model_path = models_dir / "cbramod.py"
    spec = importlib.util.spec_from_file_location(_VENDOR_MODEL_KEY, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot construct an import spec for vendored CBraMod at {model_path}")
    module = importlib.util.module_from_spec(spec)

    bare_names = ("models", "models.criss_cross_transformer")
    previous = {name: sys.modules[name] for name in bare_names if name in sys.modules}
    models_package = ModuleType("models")
    models_package.__path__ = [str(models_dir)]
    models_package.criss_cross_transformer = transformer
    sys.modules["models"] = models_package
    sys.modules["models.criss_cross_transformer"] = transformer
    sys.modules[_VENDOR_MODEL_KEY] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(_VENDOR_MODEL_KEY) is module:
            del sys.modules[_VENDOR_MODEL_KEY]
        raise
    finally:
        for name in bare_names:
            sys.modules.pop(name, None)
        sys.modules.update(previous)
    return module


def load_encoder(
    checkpoint_path: Path | None,
    *,
    n_channels: int = 1,
    n_samples: int = 200,
    device: str = "cpu",
    input_scale: float = 1.0,
) -> torch.nn.Module:
    """Build the CBraMod encoder or its architecture-matched random twin.

    Args:
        checkpoint_path: verified path from ``weights_manifest.resolve_checkpoint``, or
            ``None`` for the random-init twin.
        n_channels: channels in the harness window.
        n_samples: context length at :data:`SR_OUT`; must be a multiple of the 200-sample
            patch.
        device: where the encoder is placed; ``run_condition`` may move it again.
        input_scale: multiplier applied to the window before patching. Default ``1.0``
            (no rescaling); see the module docstring for why upstream's ``/100``
            microvolt convention is *not* silently applied to unitless z-scored windows.
    """

    import torch

    vendor_module = _load_vendor_cbramod_module()
    backbone = vendor_module.CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
    )

    load_stats: dict[str, object] | None = None
    if checkpoint_path is not None:
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        incompatible = backbone.load_state_dict(state_dict, strict=True)
        load_stats = {
            "matched_keys": len(state_dict),
            "missing_keys": tuple(incompatible.missing_keys),
            "unexpected_keys": tuple(incompatible.unexpected_keys),
        }
    backbone.proj_out = torch.nn.Identity()

    class CBraModForecastEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = backbone
            self.n_channels = int(n_channels)
            self.n_samples = int(n_samples)
            self.input_scale = float(input_scale)
            self.checkpoint_load_stats = load_stats
            self.pretrained = checkpoint_path is not None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            patches = (x.to(dtype=torch.float32) * self.input_scale).reshape(
                x.shape[0],
                self.n_channels,
                self.n_samples // _PATCH_SAMPLES,
                _PATCH_SAMPLES,
            )
            features = self.backbone(patches)
            return features.mean(dim=(1, 2))

    return CBraModForecastEncoder().to(device).eval()


def make_runner(
    weights_dir: Path,
    condition: Condition,
    *,
    config: ConditionConfig | None = None,
    pretrained: bool = True,
    input_scale: float = 1.0,
) -> Callable[[SupervisedWindowTask, int, int], np.ndarray]:
    """Return a baseline-suite runner with the standard ``(task, seed, steps)`` API.

    ``input_scale`` is threaded to :func:`load_encoder`; see the module docstring.
    """

    if not is_evaluation_permitted(MODEL_ID, SLEEP_EDF_EVAL_DATASET_ID):
        raise RuntimeError(
            f"pretraining contamination gate refuses {MODEL_ID!r} on "
            f"{SLEEP_EDF_EVAL_DATASET_ID!r}"
        )

    from neurotwin.benchmarks.fm_forecast.conditions import ConditionConfig, run_condition

    checkpoint_path = resolve_checkpoint(MODEL_ID, weights_dir) if pretrained else None
    base_config = config or ConditionConfig()

    def runner(task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        import torch

        torch.manual_seed(int(seed))
        arrays = prepare_forecast_arrays(
            x_train=task.x_train,
            y_train=task.y_train,
            x_val=task.x_val,
            y_val=task.y_val,
            x_test=task.x_test,
            y_test=task.y_test,
            sr_out=SR_OUT,
            n_samples_out=N_SAMPLES_OUT,
        )
        encoder = load_encoder(
            checkpoint_path,
            n_channels=arrays.n_channels,
            n_samples=arrays.n_samples,
            device="cpu",
            input_scale=float(input_scale),
        )
        run_config = replace(
            base_config,
            condition=condition,
            seed=int(seed),
            epochs=int(train_steps),
        )
        result = run_condition(encoder, arrays, run_config)
        flat_predictions = np.asarray(result.predictions, dtype=np.float32).reshape(
            result.predictions.shape[0], -1
        )
        return unflatten_targets(flat_predictions, tuple(task.y_test.shape))

    runner.model_id = MODEL_ID  # type: ignore[attr-defined]
    runner.condition = condition  # type: ignore[attr-defined]
    runner.pretrained = bool(pretrained)  # type: ignore[attr-defined]
    runner.checkpoint_path = checkpoint_path  # type: ignore[attr-defined]
    runner.input_scale = float(input_scale)  # type: ignore[attr-defined]
    runner.model_selection_exposure = has_model_selection_exposure(  # type: ignore[attr-defined]
        MODEL_ID, SLEEP_EDF_EVAL_DATASET_ID
    )
    return runner

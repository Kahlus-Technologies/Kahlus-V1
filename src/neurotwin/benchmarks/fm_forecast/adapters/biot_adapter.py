"""BIOT PREST-16 adapter for the forecast baseline harness."""

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
    is_evaluation_permitted,
)


if TYPE_CHECKING:
    import torch

    from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
    from neurotwin.data.prepared_tasks import SupervisedWindowTask


MODEL_ID = "biot_prest16"
SR_OUT = 200.0
N_SAMPLES_OUT = 200
EMBED_DIM = 256

# At this 200-sample context, n_fft=200 and center=False yield exactly one STFT
# token per data channel. This single-token context is a reported BIOT handicap.
_CHECKPOINT_CHANNEL_SLOTS = 16
_VENDOR_MODULE_KEY = "kahlus_vendor_biot_model"


def _load_vendor_biot_module() -> ModuleType:
    """Load BIOT's model file under a private, collision-free module name."""

    existing = sys.modules.get(_VENDOR_MODULE_KEY)
    if existing is not None:
        return existing

    model_path = require_vendored("biot") / "model" / "biot.py"
    spec = importlib.util.spec_from_file_location(_VENDOR_MODULE_KEY, model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot construct an import spec for vendored BIOT at {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VENDOR_MODULE_KEY] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if sys.modules.get(_VENDOR_MODULE_KEY) is module:
            del sys.modules[_VENDOR_MODULE_KEY]
        raise
    return module


def load_encoder(
    checkpoint_path: Path | None,
    *,
    n_channels: int = 1,
    n_samples: int = 200,
    device: str = "cpu",
) -> torch.nn.Module:
    """Build the PREST-16 encoder or its architecture-matched random twin."""

    import torch

    vendor_module = _load_vendor_biot_module()
    encoder = vendor_module.BIOTEncoder(
        emb_size=EMBED_DIM,
        heads=8,
        depth=4,
        n_fft=200,
        hop_length=100,
        # PREST-16 learned 16 positional channel slots. A one-channel forecast uses
        # slot zero but must retain the full table for an exact checkpoint key fit.
        n_channels=_CHECKPOINT_CHANNEL_SLOTS,
    )

    load_stats: dict[str, object] | None = None
    if checkpoint_path is not None:
        state_dict = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        incompatible = encoder.load_state_dict(state_dict, strict=False)
        load_stats = {
            "matched_keys": len(encoder.state_dict()) - len(incompatible.missing_keys),
            "missing_keys": tuple(incompatible.missing_keys),
            "unexpected_keys": tuple(incompatible.unexpected_keys),
        }
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"BIOT PREST-16 checkpoint does not fit the encoder: {load_stats}")

    class BIOTForecastEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            self.n_channels = int(n_channels)
            self.n_samples = int(n_samples)
            self.checkpoint_load_stats = load_stats

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.encoder(x.to(dtype=torch.float32))

    return BIOTForecastEncoder().to(device)


def make_runner(
    weights_dir: Path,
    condition: Condition,
    *,
    config: ConditionConfig | None = None,
    pretrained: bool = True,
) -> Callable[[SupervisedWindowTask, int, int], np.ndarray]:
    """Return a baseline-suite runner with the standard ``(task, seed, steps)`` API."""

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

    return runner

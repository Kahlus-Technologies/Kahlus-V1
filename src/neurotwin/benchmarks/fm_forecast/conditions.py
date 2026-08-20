"""The per-entrant condition framework: frozen probe, full fine-tune, and the acausal slot.

Plan section 3.3 defines exactly three conditions per external model, and the label
discipline between them is load-bearing:

===========================  ==========================================================
``Condition.FROZEN``         Encoder in eval mode under ``no_grad``; only the bolt-on
                             head trains. Causal, admissible. The cleanest measurement
                             of what pretraining bought.
``Condition.FULL_FT``        Encoder and head train end to end, encoder on a lower
                             learning rate. Causal, admissible. The condition a
                             reviewer will insist on.
``Condition.ACAUSAL_INFILL`` Mask the future window and let the bidirectional model
                             infill it. **Not implemented here on purpose.** It sees
                             samples after the target, so it is *infilling*, never
                             *forecasting*, it is inadmissible in the ranking claim, and
                             it is adapter-specific (each upstream masks differently).
                             :func:`run_condition` refuses it rather than silently
                             routing an acausal number through the causal path.
===========================  ==========================================================

The ranking claim is computed over each model's best **admissible** condition, selected
by ``argmin`` validation MSE (:func:`select_best_condition`). Selecting on test MSE would
be tuning on test; nothing in this module reads ``ForecastArrays.y_test``.

Every run returns a JSON-serialisable log - configuration, per-epoch history, gradient
steps, wall clock, parameter counts - because plan section 5.3 requires every attempt to
be logged and section 3.4 requires budget symmetry to be checkable from the artifact
rather than asserted.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable, Protocol

import numpy as np
import torch
from torch import Tensor, nn

from neurotwin.benchmarks.fm_forecast.data_prep import ForecastArrays, unflatten_targets
from neurotwin.benchmarks.fm_forecast.head import (
    FitResult,
    ForecastHead,
    HeadConfig,
    TrainConfig,
    fit_head,
    predict,
    resolve_device,
)


class Condition(str, Enum):
    """The three conditions of plan section 3.3."""

    FROZEN = "frozen"
    FULL_FT = "full_ft"
    ACAUSAL_INFILL = "acausal_infill"

    @property
    def is_admissible(self) -> bool:
        """True for causal conditions, i.e. the ones a forecasting ranking may use."""

        return self is not Condition.ACAUSAL_INFILL

    @property
    def label(self) -> str:
        """The word that must appear next to this condition's numbers in any table."""

        return "infilling" if self is Condition.ACAUSAL_INFILL else "forecasting"


#: Conditions eligible for the ranking claim, in reporting order.
ADMISSIBLE_CONDITIONS: tuple[Condition, ...] = (Condition.FROZEN, Condition.FULL_FT)


class EncoderFn(Protocol):
    """Minimal encoder contract: ``(N, C, T)`` float32 input -> ``(N, E)`` embedding.

    An extra token axis is tolerated and flattened, so an encoder returning
    ``(N, tokens, E)`` does not need a per-model wrapper just to reshape.
    """

    def __call__(self, x: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class ConditionConfig:
    """One point in the symmetric tuning grid, for one entrant and one horizon."""

    condition: Condition = Condition.FROZEN
    lr: float = 1e-3
    encoder_lr: float | None = None
    encoder_lr_scale: float = 0.1
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 64
    patience: int = 5
    min_delta: float = 0.0
    grad_clip: float | None = None
    seed: int = 0
    device: str = "auto"
    head: HeadConfig = field(default_factory=HeadConfig)

    def resolved_encoder_lr(self) -> float:
        """Encoder learning rate: explicit if given, else ``lr * encoder_lr_scale``.

        Fine-tuning a pretrained encoder at the head's learning rate is the standard way
        to destroy it in the first few steps, so the default is deliberately lower.
        """

        if self.encoder_lr is not None:
            return float(self.encoder_lr)
        return float(self.lr) * float(self.encoder_lr_scale)

    def train_config(self) -> TrainConfig:
        return TrainConfig(
            lr=float(self.lr),
            encoder_lr=self.resolved_encoder_lr(),
            weight_decay=float(self.weight_decay),
            epochs=int(self.epochs),
            batch_size=int(self.batch_size),
            patience=int(self.patience),
            min_delta=float(self.min_delta),
            grad_clip=None if self.grad_clip is None else float(self.grad_clip),
            seed=int(self.seed),
            device=str(self.device),
        )

    def as_log(self) -> dict[str, object]:
        payload: dict[str, object] = {
            key: value for key, value in asdict(self).items() if key not in {"condition", "head"}
        }
        payload["condition"] = self.condition.value
        payload["resolved_encoder_lr"] = self.resolved_encoder_lr()
        payload["head"] = asdict(self.head)
        return payload


@dataclass(frozen=True)
class ConditionResult:
    """Predictions plus the full training log for one condition of one entrant."""

    condition: Condition
    predictions: np.ndarray
    log: dict[str, object]
    fit: FitResult

    @property
    def best_val_mse(self) -> float:
        return float(self.fit.best_val_mse)

    @property
    def is_admissible(self) -> bool:
        return self.condition.is_admissible


class _EncoderHead(nn.Module):
    """Encoder and bolt-on head as one trainable module, for the full fine-tune."""

    def __init__(self, encoder: nn.Module, head: ForecastHead) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        return self.head(_flatten_embedding(self.encoder(x)))


def _flatten_embedding(embedding: Tensor) -> Tensor:
    if embedding.ndim == 1:
        return embedding.reshape(1, -1)
    if embedding.ndim == 2:
        return embedding
    return embedding.reshape(embedding.shape[0], -1)


def _embed(
    encoder: EncoderFn | Callable[[Tensor], Tensor],
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> Tensor:
    """Frozen forward pass: eval mode, ``no_grad``, batched, on ``device``."""

    was_training = bool(getattr(encoder, "training", False))
    if isinstance(encoder, nn.Module):
        encoder.to(device)
        encoder.eval()
    inputs = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    chunks: list[Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, int(inputs.shape[0]), int(batch_size)):
                chunks.append(_flatten_embedding(encoder(inputs[start : start + int(batch_size)])).detach())
    finally:
        if isinstance(encoder, nn.Module) and was_training:
            encoder.train()
    if not chunks:
        raise ValueError("cannot embed an empty split")
    return torch.cat(chunks, dim=0).to(dtype=torch.float32)


def _parameter_counts(encoder: object) -> dict[str, int]:
    if not isinstance(encoder, nn.Module):
        return {"encoder_parameters": 0, "encoder_trainable_parameters": 0}
    total = sum(parameter.numel() for parameter in encoder.parameters())
    trainable = sum(parameter.numel() for parameter in encoder.parameters() if parameter.requires_grad)
    return {"encoder_parameters": int(total), "encoder_trainable_parameters": int(trainable)}


def run_condition(
    encoder: EncoderFn | Callable[[Tensor], Tensor] | nn.Module,
    arrays: ForecastArrays,
    config: ConditionConfig | None = None,
) -> ConditionResult:
    """Run one condition end to end and return predictions plus a training log.

    Args:
        encoder: a ``torch.nn.Module`` (required for ``FULL_FT``) or any callable mapping
            ``(N, C, T)`` float32 to ``(N, E)``, as produced by a per-model adapter.
        arrays: model-ready arrays from ``data_prep.prepare_forecast_arrays``. Only
            ``x_train``/``y_train``/``x_val``/``y_val``/``x_test`` are read;
            ``y_test`` is never touched.
        config: the tuning-grid point to run.

    Returns:
        A :class:`ConditionResult` whose ``predictions`` already match
        ``arrays.test_target_shape``, i.e. the harness ``task.y_test.shape`` that
        ``baseline_suite`` validates a runner against.

    Raises:
        NotImplementedError: for ``Condition.ACAUSAL_INFILL``, which is adapter-specific
            and inadmissible as a forecast.
    """

    config = config or ConditionConfig()
    if config.condition is Condition.ACAUSAL_INFILL:
        raise NotImplementedError(
            "acausal masked infilling is a documented slot, not an implementation: it is "
            "adapter-specific (each upstream masks differently) and it sees samples after "
            "the target, so its numbers are 'infilling', never 'forecasting', and are "
            "inadmissible in the ranking claim (benchmark plan section 3.3). Implement it "
            "in the per-model adapter and report it in a separate table with an explicit "
            "acausality warning."
        )
    if config.condition not in ADMISSIBLE_CONDITIONS:
        raise ValueError(f"unknown condition {config.condition!r}")

    device = resolve_device(config.device)
    torch.manual_seed(int(config.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))

    train_config = config.train_config()
    started = time.perf_counter()

    if config.condition is Condition.FROZEN:
        embed_batch = max(int(config.batch_size), 1)
        train_features = _embed(encoder, arrays.x_train, device=device, batch_size=embed_batch)
        val_features = _embed(encoder, arrays.x_val, device=device, batch_size=embed_batch)
        test_features = _embed(encoder, arrays.x_test, device=device, batch_size=embed_batch)
        embed_dim = int(train_features.shape[1])
        head = ForecastHead(embed_dim, arrays.output_dim, config.head)
        fit = fit_head(
            head,
            x_train=train_features,
            y_train=arrays.y_train,
            x_val=val_features,
            y_val=arrays.y_val,
            config=train_config,
        )
        flat_predictions = predict(head, test_features, device=device, batch_size=max(embed_batch, 1))
        trainable_module: nn.Module = head
    else:
        if not isinstance(encoder, nn.Module):
            raise TypeError(
                "the full fine-tune condition trains the encoder, so it requires a "
                "torch.nn.Module; a plain embedding callable can only be used FROZEN"
            )
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
        with torch.no_grad():
            probe = _flatten_embedding(
                encoder.to(device).eval()(
                    torch.as_tensor(np.asarray(arrays.x_train[:1]), dtype=torch.float32, device=device)
                )
            )
        embed_dim = int(probe.shape[1])
        head = ForecastHead(embed_dim, arrays.output_dim, config.head)
        model = _EncoderHead(encoder, head)
        param_groups = [
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": config.resolved_encoder_lr(),
            },
            {"params": [p for p in model.head.parameters() if p.requires_grad], "lr": float(config.lr)},
        ]
        fit = fit_head(
            model,
            x_train=arrays.x_train,
            y_train=arrays.y_train,
            x_val=arrays.x_val,
            y_val=arrays.y_val,
            config=train_config,
            param_groups=param_groups,
        )
        flat_predictions = predict(model, arrays.x_test, device=device, batch_size=max(int(config.batch_size), 1))
        trainable_module = model

    predictions = unflatten_targets(flat_predictions, arrays.test_target_shape)
    head_parameters = int(sum(parameter.numel() for parameter in head.parameters()))
    trainable_parameters = int(
        sum(parameter.numel() for parameter in trainable_module.parameters() if parameter.requires_grad)
    )
    log: dict[str, object] = {
        "condition": config.condition.value,
        "condition_label": config.condition.label,
        "admissible_for_ranking": bool(config.condition.is_admissible),
        "config": config.as_log(),
        "embed_dim": embed_dim,
        "output_dim": int(arrays.output_dim),
        "n_channels": int(arrays.n_channels),
        "n_samples": int(arrays.n_samples),
        "n_train_windows": int(arrays.x_train.shape[0]),
        "n_val_windows": int(arrays.x_val.shape[0]),
        "n_test_windows": int(arrays.x_test.shape[0]),
        "sr_in_hz": float(arrays.sr_in),
        "sr_out_hz": float(arrays.sr_out),
        "n_samples_out": arrays.n_samples_out,
        "head_parameters": head_parameters,
        "trainable_parameters": trainable_parameters,
        "wall_clock_seconds": float(time.perf_counter() - started),
        "test_targets_read": False,
        "fit": fit.as_log(),
        **_parameter_counts(encoder),
    }
    return ConditionResult(condition=config.condition, predictions=predictions, log=log, fit=fit)


def select_best_condition(results: list[ConditionResult] | tuple[ConditionResult, ...]) -> ConditionResult:
    """Return the admissible condition with the lowest **validation** MSE.

    This is the plan section 3.3 selection rule. Test MSE is not an input, so the
    selected condition is fixed before any test number is looked at.
    """

    admissible = [result for result in results if result.is_admissible]
    if not admissible:
        raise ValueError(
            "no admissible condition to select from: the ranking claim may only use "
            "causal conditions (frozen, full_ft), never acausal infilling"
        )
    finite = [result for result in admissible if np.isfinite(result.best_val_mse)]
    if not finite:
        raise ValueError("every admissible condition produced a non-finite validation MSE")
    return min(finite, key=lambda result: (result.best_val_mse, result.condition.value))

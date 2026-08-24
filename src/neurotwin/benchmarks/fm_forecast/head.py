"""The identical bolt-on forecast head, and its seeded, validation-selected training loop.

Plan section 3.3 requires "an identical bolt-on prediction head on every model" - the
same module, the same optimiser, the same selection rule for every entrant, so a ranking
difference is a difference between encoders and not between head implementations. This
module is that head.

Selection discipline, enforced structurally rather than by convention:

- :func:`fit_head` takes train and validation tensors and **no test tensors at all**.
  There is no argument through which test data could reach the training loop.
- Model selection is ``argmin`` validation MSE over epochs; the best epoch's weights are
  restored into the module before returning, so the caller cannot accidentally predict
  with the last epoch instead of the selected one.
- Early stopping is on validation MSE with a patience, and the full per-epoch history is
  returned for the plan section 5.3 attempts ledger.

Determinism: the seed drives ``torch.manual_seed`` (parameter init) and a dedicated CPU
``torch.Generator`` for batch shuffling, so two runs with the same seed on the same
device produce identical predictions. Shuffling is generated on CPU even when training on
CUDA, so the batch order does not depend on the device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class HeadConfig:
    """Shape of the bolt-on head.

    ``hidden_dim=0`` gives the linear probe, which is the default and the cleanest
    reading of "what did pretraining buy". A small MLP is available so head width can
    take part in the symmetric tuning grid of plan section 3.4.
    """

    hidden_dim: int = 0
    dropout: float = 0.0
    bias: bool = True


class ForecastHead(nn.Module):
    """Encoder embedding ``(N, E)`` -> horizon prediction ``(N, D)``."""

    def __init__(self, embed_dim: int, output_dim: int, config: HeadConfig | None = None) -> None:
        super().__init__()
        config = config or HeadConfig()
        if embed_dim < 1 or output_dim < 1:
            raise ValueError("embed_dim and output_dim must be positive")
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim)
        self.config = config
        if config.hidden_dim and int(config.hidden_dim) > 0:
            layers: list[nn.Module] = [
                nn.Linear(self.embed_dim, int(config.hidden_dim), bias=config.bias),
                nn.GELU(),
            ]
            if config.dropout:
                layers.append(nn.Dropout(float(config.dropout)))
            layers.append(nn.Linear(int(config.hidden_dim), self.output_dim, bias=config.bias))
            self.net: nn.Module = nn.Sequential(*layers)
        else:
            self.net = nn.Linear(self.embed_dim, self.output_dim, bias=config.bias)

    def forward(self, embedding: Tensor) -> Tensor:
        if embedding.ndim != 2:
            embedding = embedding.reshape(embedding.shape[0], -1)
        return self.net(embedding)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation budget for one configuration of one entrant.

    Every field is logged into the run artifact so plan section 3.4's budget-symmetry
    claim can be checked rather than trusted.
    """

    lr: float = 1e-3
    encoder_lr: float | None = None
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 64
    patience: int = 5
    min_delta: float = 0.0
    grad_clip: float | None = None
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_mse: float
    val_mse: float
    gradient_steps: int

    def as_row(self) -> dict[str, float | int]:
        return {
            "epoch": int(self.epoch),
            "train_mse": float(self.train_mse),
            "val_mse": float(self.val_mse),
            "gradient_steps": int(self.gradient_steps),
        }


@dataclass(frozen=True)
class FitResult:
    """Outcome of one training run, with the selected epoch already restored."""

    best_val_mse: float
    best_epoch: int
    epochs_ran: int
    gradient_steps: int
    stopped_early: bool
    device: str
    history: tuple[EpochRecord, ...] = field(default_factory=tuple)

    def as_log(self) -> dict[str, object]:
        return {
            "best_val_mse": float(self.best_val_mse),
            "best_epoch": int(self.best_epoch),
            "epochs_ran": int(self.epochs_ran),
            "gradient_steps": int(self.gradient_steps),
            "stopped_early": bool(self.stopped_early),
            "device": str(self.device),
            "history": [record.as_row() for record in self.history],
        }


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``"auto"`` to CUDA when available, else CPU. Explicit names pass through."""

    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _to_tensor(values: np.ndarray | Tensor, device: torch.device) -> Tensor:
    if isinstance(values, Tensor):
        return values.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)


def _mse(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target) ** 2)


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().to("cpu").clone() for key, value in module.state_dict().items()}


def fit_head(
    model: nn.Module,
    *,
    x_train: np.ndarray | Tensor,
    y_train: np.ndarray | Tensor,
    x_val: np.ndarray | Tensor,
    y_val: np.ndarray | Tensor,
    config: TrainConfig | None = None,
    param_groups: Iterable[dict[str, object]] | None = None,
) -> FitResult:
    """Train ``model`` on ``(x_train, y_train)``, selecting the epoch by validation MSE.

    ``model`` maps an input batch to ``(N, D)`` predictions; for the frozen condition it
    is the head alone reading precomputed embeddings, for the full fine-tune it is the
    encoder and head composed. ``param_groups`` lets the caller give the encoder a lower
    learning rate than the head; when omitted, every trainable parameter uses
    ``config.lr``.

    Returns a :class:`FitResult` after loading the best epoch's weights back into
    ``model``. Test data is not an argument and is never seen here.
    """

    config = config or TrainConfig()
    if config.epochs < 1:
        raise ValueError("epochs must be >= 1")
    if config.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if config.patience < 1:
        raise ValueError("patience must be >= 1")

    device = resolve_device(config.device)
    torch.manual_seed(int(config.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(config.seed))

    model = model.to(device)
    train_inputs = _to_tensor(x_train, device)
    train_targets = _to_tensor(y_train, device)
    val_inputs = _to_tensor(x_val, device)
    val_targets = _to_tensor(y_val, device)
    if train_inputs.shape[0] != train_targets.shape[0]:
        raise ValueError("x_train and y_train must have the same number of windows")
    if val_inputs.shape[0] != val_targets.shape[0]:
        raise ValueError("x_val and y_val must have the same number of windows")
    if train_inputs.shape[0] < 1 or val_inputs.shape[0] < 1:
        raise ValueError("training and validation splits must be non-empty")

    groups = list(param_groups) if param_groups is not None else None
    if groups is None:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("model has no trainable parameters")
        groups = [{"params": trainable, "lr": float(config.lr)}]
    optimizer = torch.optim.Adam(groups, lr=float(config.lr), weight_decay=float(config.weight_decay))

    # Batch order is drawn on CPU so it is identical on CPU and CUDA for a given seed.
    shuffler = torch.Generator()
    shuffler.manual_seed(int(config.seed))

    n_train = int(train_inputs.shape[0])
    best_val = float("inf")
    best_epoch = 0
    best_state = _cpu_state_dict(model)
    history: list[EpochRecord] = []
    gradient_steps = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        order = torch.randperm(n_train, generator=shuffler).to(device)
        epoch_loss = 0.0
        epoch_count = 0
        for start in range(0, n_train, int(config.batch_size)):
            index = order[start : start + int(config.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            loss = _mse(model(train_inputs[index]), train_targets[index])
            loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for group in optimizer.param_groups for parameter in group["params"]],
                    float(config.grad_clip),
                )
            optimizer.step()
            gradient_steps += 1
            epoch_loss += float(loss.detach()) * int(index.shape[0])
            epoch_count += int(index.shape[0])

        model.eval()
        with torch.no_grad():
            val_mse = float(_mse(model(val_inputs), val_targets).detach())
        train_mse = epoch_loss / max(epoch_count, 1)
        history.append(
            EpochRecord(epoch=epoch, train_mse=train_mse, val_mse=val_mse, gradient_steps=gradient_steps)
        )

        if np.isfinite(val_mse) and val_mse < best_val - float(config.min_delta):
            best_val = val_mse
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config.patience):
                stopped_early = True
                break

    model.load_state_dict(best_state)
    model.to(device)
    return FitResult(
        best_val_mse=best_val,
        best_epoch=best_epoch,
        epochs_ran=len(history),
        gradient_steps=gradient_steps,
        stopped_early=stopped_early,
        device=str(device),
        history=tuple(history),
    )


def predict(
    model: nn.Module,
    x: np.ndarray | Tensor,
    *,
    device: str | torch.device = "auto",
    batch_size: int = 256,
) -> np.ndarray:
    """Batched, no-grad forward pass returning ``(N, D)`` float32 predictions."""

    resolved = resolve_device(device) if isinstance(device, str) else device
    model = model.to(resolved)
    model.eval()
    inputs = _to_tensor(x, resolved)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, int(inputs.shape[0]), int(batch_size)):
            chunk = model(inputs[start : start + int(batch_size)])
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)
            outputs.append(chunk.detach().to("cpu").numpy().astype(np.float32))
    if not outputs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(outputs, axis=0)

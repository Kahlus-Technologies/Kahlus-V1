"""EEGPT as a Kahlus forecast baseline runner.

Wraps the vendored upstream ``EEGTransformer`` (the 4-second ``eegpt_mcae_58chs_4s_large4E``
encoder) so it can be scored on the Sleep-EDF future-state forecasting lane through the
shared ``fm_forecast`` infrastructure: :func:`~neurotwin.benchmarks.fm_forecast.data_prep.prepare_forecast_arrays`
for the input transform, :func:`~neurotwin.benchmarks.fm_forecast.conditions.run_condition`
for the identical bolt-on head, and
:func:`~neurotwin.benchmarks.fm_forecast.weights_manifest.resolve_checkpoint` for the
hash-pinned weights.

Design decisions worth stating rather than burying:

**Window.** ``SR_OUT = 256 Hz`` is EEGPT's native rate (the published 4-s path is 1024
samples). ``N_SAMPLES_OUT = 320`` is 1.25 s, i.e. exactly 5 complete 64-sample patches -
the largest patch-aligned context **not exceeding** the incumbent's 1.27 s context (plan
section 3.2). ``EEGTransformer.forward`` asserts the patch count matches the constructor's
``img_size``, so the encoder is rebuilt at this size rather than the published 1024; every
parameter tensor in the checkpoint is independent of ``img_size`` (the patch embedding is a
``(1, 64)`` Conv2d, channel identity comes from a 62-entry ``nn.Embedding``, and there is no
learned temporal position embedding), so ``strict=True`` loading still holds.

**Pooling.** ``target_encoder(x, chan_ids)`` returns ``[B, N, embed_num, embed_dim]``
= ``[B, 5, 4, 512]``. This adapter means over the patch axis ``N`` and flattens the four
summary tokens, giving ``EMBED_DIM = 4 * 512 = 2048``. Mean-over-tokens is the pooling the
other three entrants' upstreams use for a single feature vector (LaBraM's ``forward_features``,
BIOT's encoder, CBraMod's average-pooling head), so pooling is not a place where the
entrants silently differ; flattening the summary tokens rather than averaging them keeps
the upstream probe's ``[B, N, 2048]`` view of a patch (linear_probe_EEGPT_BCIC2A.py:86-90).

**Random-init twin.** ``load_encoder(None)`` builds the identical architecture with no
pretrained weights. That is the pretraining-value control: the difference between it and
the loaded encoder, under an identical head and budget, is what the checkpoint bought.

**Scale.** No normalisation happens here. ``data_prep`` documents why (the harness already
z-scores each recording and the normalisation convention is one shared upstream decision);
EEGPT's own downstream harnesses feed µV-scaled data, so this adapter runs the encoder
off its training scale on purpose, symmetrically with every other entrant.

**Contamination.** ``eegpt``/Sleep-EDF is CLEAN but carries a model-selection exposure
(``pretraining_contamination.MODEL_SELECTION_EXPOSURE_PAIRS``): Sleep-EDFx was in EEGPT's
published downstream suite, so its architecture and hyperparameters were chosen with this
dataset in view. :func:`make_runner` records that on the returned runner as
``model_selection_exposure`` so a results table can carry the column.

Imports here are lazy on purpose. ``external/vendor/`` is gitignored and absent in CI, and
several vendored upstreams ship colliding top-level module names (``utils``, ``Modules``,
``model``). Nothing in this module imports ``torch`` or upstream code at import time, and
the vendored file is loaded by path under a unique ``sys.modules`` key so that importing
this adapter never puts a bare ``utils``/``Modules`` on the module table or mutates
``sys.path``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import numpy as np
    import torch

    from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
    from neurotwin.data.prepared_tasks import SupervisedWindowTask


#: Contamination-registry / weights-manifest id for this entrant.
MODEL_ID = "eegpt"

#: ``external/upstreams.lock.json`` key and vendor directory name.
UPSTREAM_ID = "eegpt"

#: EEGPT's native sampling rate (the published 4-s path is 1024 samples).
SR_OUT = 256.0

#: 1.25 s = 5 complete 64-sample patches: the largest patch-aligned context that does not
#: exceed the incumbent's 1.27 s F8 context (benchmark design doc section 3.2).
N_SAMPLES_OUT = 320

#: Published encoder geometry for ``eegpt_mcae_58chs_4s_large4E``.
PATCH_SIZE = 64
ENCODER_EMBED_DIM = 512
EMBED_NUM = 4
DEPTH = 8
NUM_HEADS = 8

#: Embedding width this adapter hands the bolt-on head: mean over the 5 patches, then the
#: 4 summary tokens flattened.
EMBED_DIM = EMBED_NUM * ENCODER_EMBED_DIM  # 2048

#: Electrode name for the Lane-A single channel. Sleep-EDF Fpz-Cz is a bipolar derivation
#: with no entry in EEGPT's 10-20 ``CHANNEL_DICT``; upstream's own Sleep-EDF harness
#: resolves this exactly the same way, listing monopolar names including ``'FPZ'``
#: (external/vendor/eegpt/downstream/finetune_EEGPT_SleepEDF.py:33), so the derivation is
#: mapped onto its leading electrode's learned channel embedding.
LANE_A_CHANNEL_NAMES: tuple[str, ...] = ("FPZ",)

#: Checkpoint key prefix for the EMA target encoder (linear_probe_EEGPT_BCIC2A.py:39-69).
CHECKPOINT_ENCODER_PREFIX = "target_encoder."

#: Unique ``sys.modules`` key for the vendored model file. Never ``Modules.models.*``:
#: four upstreams are imported into one pytest process and ``Modules``/``utils``/``model``
#: collide across them.
_VENDOR_MODULE_KEY = "kahlus_vendor_eegpt_mcae_finetune"

#: Path of the vendored model file inside the ``eegpt`` checkout. This file is self
#: contained - it imports only ``torch``/``numpy``/stdlib, no intra-repo modules - so
#: loading it by path needs no further alias resolution.
_VENDOR_MODULE_RELPATH = "downstream/Modules/models/EEGPT_mcae_finetune.py"


def _vendored_module() -> Any:
    """Import the vendored ``EEGPT_mcae_finetune`` module hermetically, by path.

    Uses ``importlib.util.spec_from_file_location`` under :data:`_VENDOR_MODULE_KEY` so no
    bare upstream name (``Modules``, ``utils``, ``model``) is ever registered and
    ``sys.path`` is untouched. Idempotent: a second call reuses the cached module.
    """

    import importlib.util
    import sys

    cached = sys.modules.get(_VENDOR_MODULE_KEY)
    if cached is not None:
        return cached

    from neurotwin.benchmarks.fm_forecast.vendor_import import (
        VendoredUpstreamMissing,
        require_vendored,
        vendor_command,
    )

    root = require_vendored(UPSTREAM_ID)
    path = root / _VENDOR_MODULE_RELPATH
    if not path.is_file():
        raise VendoredUpstreamMissing(
            f"vendored upstream {UPSTREAM_ID!r} is present at {root} but "
            f"{_VENDOR_MODULE_RELPATH} is missing, so the EEGPT encoder cannot be built. "
            f"Re-vendor with `{vendor_command(UPSTREAM_ID)}`."
        )

    spec = importlib.util.spec_from_file_location(_VENDOR_MODULE_KEY, path)
    if spec is None or spec.loader is None:
        raise VendoredUpstreamMissing(f"cannot build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_VENDOR_MODULE_KEY] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Never leave a half-executed module behind for the next caller to find.
        sys.modules.pop(_VENDOR_MODULE_KEY, None)
        raise
    return module


def _build_transformer(n_channels: int, n_samples: int) -> torch.nn.Module:
    """Instantiate the published ``EEGTransformer`` geometry at ``[n_channels, n_samples]``."""

    from functools import partial

    from torch import nn

    vendored = _vendored_module()
    return vendored.EEGTransformer(
        img_size=[int(n_channels), int(n_samples)],
        patch_size=PATCH_SIZE,
        embed_num=EMBED_NUM,
        embed_dim=ENCODER_EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        init_std=0.02,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )


def load_pretrained_state(transformer: torch.nn.Module, checkpoint_path: Path | str) -> int:
    """Load the EMA target encoder from a Lightning checkpoint into ``transformer``.

    The recipe is upstream's (linear_probe_EEGPT_BCIC2A.py:39-69): read
    ``checkpoint["state_dict"]``, keep only the ``target_encoder.*`` entries, strip that
    15-character prefix, and load. The sibling ``encoder.*``, ``predictor.*``, and
    ``reconstructor.*`` groups in the same file are the pretraining scaffolding and are
    deliberately dropped. Loading is ``strict=True``: a key that survives the surgery but
    does not belong to the encoder, or an encoder parameter the checkpoint does not
    supply, is a wiring bug, not something to tolerate.

    Returns the number of tensors loaded.
    """

    import torch

    # ``weights_only=False`` is unavoidable here, and the exposure is documented rather
    # than hand-waved. The Lightning checkpoint pickles optimizer/scheduler training
    # state next to the weights, and the stream contains a ``POP`` opcode (from
    # ``OneCycleLR``'s pickled ``anneal_func`` bound method) that torch's weights-only
    # unpickler structurally rejects ("Unsupported operand 48") regardless of any
    # ``safe_globals`` allowlist - so the safe path cannot parse this file at all.
    # Mitigations: (1) the SHA-256 pin in ``weights_manifest.resolve_checkpoint`` has
    # already run by now, so only the byte-exact mirrored file reaches this line;
    # (2) that exact file's pickle GLOBAL opcodes were audited via ``pickletools`` -
    # beyond the default tensor rebuilders it references only ``AdamW``, ``OneCycleLR``,
    # ``defaultdict``, ``dict``, and ``getattr``, i.e. no code-execution imports.
    # The residual trust is trust-on-first-use of the 2026-08-19 mirror: upstream
    # publishes no digest to anchor against (figshare share link only).
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{checkpoint_path} is not an EEGPT Lightning checkpoint: expected a dict with "
            "a 'state_dict' key (linear_probe_EEGPT_BCIC2A.py:39-69)"
        )
    state_dict = checkpoint["state_dict"]
    encoder_state = {
        key[len(CHECKPOINT_ENCODER_PREFIX) :]: value
        for key, value in state_dict.items()
        if key.startswith(CHECKPOINT_ENCODER_PREFIX)
    }
    if not encoder_state:
        raise ValueError(
            f"{checkpoint_path} contains no {CHECKPOINT_ENCODER_PREFIX!r} entries, so there "
            "is no EEGPT target encoder to load"
        )
    transformer.load_state_dict(encoder_state, strict=True)
    return len(encoder_state)


def _encoder_wrapper_class() -> type:
    """Build (once) the ``nn.Module`` wrapper that turns EEGPT into an ``EncoderFn``."""

    import torch
    from torch import nn

    class EEGPTForecastEncoder(nn.Module):
        """``(N, C, T)`` float32 -> ``(N, EMBED_DIM)`` float32.

        Holds the vendored ``EEGTransformer`` plus the fixed channel-id tensor, and pools
        ``[B, N, embed_num, embed_dim]`` down to ``[B, embed_num * embed_dim]`` by
        averaging over patches and flattening the summary tokens.
        """

        def __init__(
            self,
            transformer: nn.Module,
            chan_ids: torch.Tensor,
            *,
            channel_names: tuple[str, ...],
            n_channels: int,
            n_samples: int,
            pretrained: bool,
        ) -> None:
            super().__init__()
            self.encoder = transformer
            # Non-persistent: the channel ids are configuration, not learned state, and
            # must not appear in this wrapper's state_dict.
            self.register_buffer("chan_ids", chan_ids, persistent=False)
            self.channel_names = tuple(channel_names)
            self.n_channels = int(n_channels)
            self.n_samples = int(n_samples)
            self.pretrained = bool(pretrained)
            self.embed_dim = EMBED_DIM

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 3:
                raise ValueError(f"expected (N, C, T) input, got shape {tuple(x.shape)}")
            if int(x.shape[1]) != self.n_channels or int(x.shape[2]) != self.n_samples:
                raise ValueError(
                    f"EEGPT encoder was built for (C={self.n_channels}, T={self.n_samples}) and "
                    f"asserts that geometry internally, got {tuple(x.shape)}"
                )
            tokens = self.encoder(x.to(dtype=torch.float32), self.chan_ids)
            # tokens: (B, n_patches, embed_num, embed_dim) -> mean over patches, flatten tokens.
            pooled = tokens.mean(dim=1)
            return pooled.reshape(pooled.shape[0], -1)

    return EEGPTForecastEncoder


def load_encoder(
    checkpoint_path: Path | None,
    *,
    n_channels: int = 1,
    n_samples: int = N_SAMPLES_OUT,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """Build the EEGPT forecast encoder, optionally with pretrained weights.

    Args:
        checkpoint_path: hash-verified path from ``weights_manifest.resolve_checkpoint``,
            or ``None`` for the **random-init twin** - the identical architecture with no
            pretrained weights, which is the pretraining-value control.
        n_channels: channels in the input. Only the Lane-A single-channel case is wired,
            because :data:`LANE_A_CHANNEL_NAMES` is what fixes the channel-embedding rows.
        n_samples: samples per window; must be a whole number of 64-sample patches, since
            ``EEGTransformer.forward`` asserts the patch count.
        device: where the returned module lives.

    Returns:
        A ``torch.nn.Module`` mapping ``(N, C, T)`` float32 to ``(N, EMBED_DIM)`` float32.
    """

    import torch

    n_channels = int(n_channels)
    n_samples = int(n_samples)
    if n_channels != len(LANE_A_CHANNEL_NAMES):
        raise ValueError(
            f"this adapter wires the Lane-A montage {LANE_A_CHANNEL_NAMES} "
            f"({len(LANE_A_CHANNEL_NAMES)} channel(s)), so n_channels={n_channels} has no "
            "electrode names to map onto EEGPT's channel embedding; extend "
            "LANE_A_CHANNEL_NAMES with CHANNEL_DICT-legal names to add channels"
        )
    if n_samples < PATCH_SIZE or n_samples % PATCH_SIZE:
        raise ValueError(
            f"n_samples must be a positive multiple of the {PATCH_SIZE}-sample patch "
            f"(EEGTransformer.forward asserts the patch count), got {n_samples}"
        )

    transformer = _build_transformer(n_channels, n_samples)
    if checkpoint_path is not None:
        load_pretrained_state(transformer, checkpoint_path)

    chan_ids = transformer.prepare_chan_ids(list(LANE_A_CHANNEL_NAMES)).long()
    wrapper = _encoder_wrapper_class()(
        transformer,
        chan_ids,
        channel_names=LANE_A_CHANNEL_NAMES,
        n_channels=n_channels,
        n_samples=n_samples,
        pretrained=checkpoint_path is not None,
    )
    return wrapper.to(torch.device(device) if isinstance(device, str) else device).eval()


def make_runner(
    weights_dir: Path,
    condition: Condition,
    *,
    config: ConditionConfig | None = None,
    pretrained: bool = True,
) -> Callable[[SupervisedWindowTask, int, int], np.ndarray]:
    """Return a ``baseline_runners.BaselineRunner`` for EEGPT under one condition.

    The returned callable has the harness signature ``(task, seed, train_steps) -> ndarray``
    and returns predictions whose shape is exactly ``task.y_test.shape``.

    ``train_steps`` mirrors the ``gru``/``ssm_fallback`` runner's semantics
    (``baseline_suite._fit_torch_sequence_model``): it is the *gradient-budget* knob, one
    step per pass over the training split. Here that becomes the epoch budget of the shared
    head trainer, ``epochs = max(1, train_steps)``; with the default batch size the
    correspondence is one optimiser step per epoch per batch, and the realised
    ``gradient_steps`` is recorded in the condition log either way. ``seed`` overrides
    ``config.seed`` so a multi-seed sweep re-seeds encoder init, head init, and batch order
    together.

    Args:
        weights_dir: directory holding the staged checkpoints (hash-verified on the way in).
        condition: ``Condition.FROZEN`` or ``Condition.FULL_FT``; ``ACAUSAL_INFILL`` is
            refused downstream by ``run_condition`` because it is not a forecast.
        config: tuning-grid point. ``condition``, ``seed`` and ``epochs`` are overridden
            per call; everything else is taken from it.
        pretrained: ``False`` builds the random-init twin and never touches the manifest.

    Raises:
        RuntimeError: if ``pretraining_contamination.is_evaluation_permitted`` refuses this
            model on the Sleep-EDF evaluation dataset. The gate is checked here, before any
            checkpoint IO, so a slate-excluded or contaminated model cannot be constructed
            into a runnable object at all.
    """

    from neurotwin.benchmarks import pretraining_contamination as contamination
    from neurotwin.benchmarks.fm_forecast.weights_manifest import resolve_checkpoint

    dataset_id = contamination.SLEEP_EDF_EVAL_DATASET_ID
    if not contamination.is_evaluation_permitted(MODEL_ID, dataset_id):
        raise RuntimeError(
            f"evaluation of {MODEL_ID!r} on {dataset_id!r} is not permitted: "
            f"status={contamination.contamination_status(MODEL_ID, dataset_id)!r}, "
            f"slate_excluded={MODEL_ID in contamination.SLATE_EXCLUDED_MODEL_IDS}. "
            "Refusing to build a runner rather than producing a number that would have to "
            "be retracted (pretraining_contamination.is_evaluation_permitted)."
        )
    exposure = contamination.has_model_selection_exposure(MODEL_ID, dataset_id)

    checkpoint_path = resolve_checkpoint(MODEL_ID, weights_dir) if pretrained else None

    def run_eegpt(task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        import dataclasses

        import numpy as np
        import torch

        from neurotwin.benchmarks.fm_forecast.conditions import ConditionConfig, run_condition
        from neurotwin.benchmarks.fm_forecast.data_prep import prepare_forecast_arrays
        from neurotwin.benchmarks.fm_forecast.head import resolve_device

        base_config = config or ConditionConfig()
        run_config = dataclasses.replace(
            base_config,
            condition=condition,
            seed=int(seed),
            epochs=max(1, int(train_steps)),
        )

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

        # Seed before construction so the random-init twin is reproducible per seed, and
        # so a pretrained run's head init is drawn from the same stream as the twin's.
        torch.manual_seed(int(seed))
        encoder = load_encoder(
            checkpoint_path,
            n_channels=arrays.n_channels,
            n_samples=arrays.n_samples,
            device=resolve_device(run_config.device),
        )
        result = run_condition(encoder, arrays, run_config)

        predictions = np.ascontiguousarray(np.asarray(result.predictions, dtype=np.float32))
        expected = tuple(int(dim) for dim in np.asarray(task.y_test).shape)
        if predictions.shape != expected:
            raise ValueError(
                f"{MODEL_ID} runner produced predictions of shape {predictions.shape}, "
                f"but the harness requires task.y_test.shape {expected}"
            )
        return predictions

    run_eegpt.model_id = MODEL_ID  # type: ignore[attr-defined]
    run_eegpt.condition = condition  # type: ignore[attr-defined]
    run_eegpt.pretrained = bool(pretrained)  # type: ignore[attr-defined]
    run_eegpt.checkpoint_path = checkpoint_path  # type: ignore[attr-defined]
    # Reportable-but-not-disqualifying: EEGPT's authors evaluated on Sleep-EDFx.
    run_eegpt.model_selection_exposure = bool(exposure)  # type: ignore[attr-defined]
    return run_eegpt

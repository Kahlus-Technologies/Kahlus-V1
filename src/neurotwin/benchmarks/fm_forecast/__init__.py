"""Shared infrastructure for evaluating external pretrained EEG models as forecasters.

This package is the mechanics half of
``docs/research/eeg_fm_benchmark_plan_2026-08-18.md`` sections 3 and 4: the pieces every
per-model adapter (EEGPT, CBraMod, LaBraM, BIOT) needs before it can be registered as an
``ExecutableBaselineRunner``. It deliberately contains **no** model-specific code and
**no** results.

Boundaries this package enforces, because the plan requires them:

- **No network access, ever.** ``weights_manifest`` verifies locally staged checkpoints
  against frozen SHA-256 digests and fails closed; it never downloads. ``huggingface_hub``
  is quarantined repo-wide (``tests/artifacts/test_docs_contracts.py``) and must not be
  imported from anywhere in this package.
- **No top-level import of vendored upstream code.** ``external/vendor/`` is gitignored
  and may be absent; use ``vendor_import.require_vendored`` lazily, inside the adapter
  that needs it.
- **No leakage.** ``data_prep`` transforms are per-window and stateless: nothing here
  computes a statistic across windows or across splits. ``head.fit_head`` selects on
  validation MSE only and never reads test targets.
- **Label discipline.** ``conditions.Condition.ACAUSAL_INFILL`` is a documented slot with
  no implementation; an acausal infill number is not a forecast and must never be scored
  through the causal path (plan section 3.3).
"""

from __future__ import annotations

from neurotwin.benchmarks.fm_forecast.conditions import (
    ADMISSIBLE_CONDITIONS,
    Condition,
    ConditionConfig,
    ConditionResult,
    run_condition,
    select_best_condition,
)
from neurotwin.benchmarks.fm_forecast.data_prep import (
    FORECAST_TASK_ID,
    ForecastArrays,
    flatten_targets,
    prepare_forecast_arrays,
    prepare_windows,
    resample_windows,
    to_channels_first,
    unflatten_targets,
)
from neurotwin.benchmarks.fm_forecast.head import (
    EpochRecord,
    FitResult,
    ForecastHead,
    HeadConfig,
    TrainConfig,
    fit_head,
    predict,
    resolve_device,
)
from neurotwin.benchmarks.fm_forecast.vendor_import import (
    VENDORED_UPSTREAM_IDS,
    VendoredUpstreamMissing,
    ensure_on_sys_path,
    is_vendored,
    require_vendored,
    vendored_root,
)
from neurotwin.benchmarks.fm_forecast.weights_manifest import (
    WEIGHTS_MANIFEST,
    CheckpointChecksumError,
    CheckpointNotStagedError,
    CheckpointSpec,
    WeightsManifestError,
    checkpoint_spec,
    manifest_model_ids,
    manifest_rows,
    resolve_checkpoint,
    sha256_file,
    verify_checkpoint,
)


__all__ = [
    "ADMISSIBLE_CONDITIONS",
    "FORECAST_TASK_ID",
    "VENDORED_UPSTREAM_IDS",
    "WEIGHTS_MANIFEST",
    "CheckpointChecksumError",
    "CheckpointNotStagedError",
    "CheckpointSpec",
    "Condition",
    "ConditionConfig",
    "ConditionResult",
    "EpochRecord",
    "FitResult",
    "ForecastArrays",
    "ForecastHead",
    "HeadConfig",
    "TrainConfig",
    "VendoredUpstreamMissing",
    "WeightsManifestError",
    "checkpoint_spec",
    "ensure_on_sys_path",
    "fit_head",
    "flatten_targets",
    "is_vendored",
    "manifest_model_ids",
    "manifest_rows",
    "predict",
    "prepare_forecast_arrays",
    "prepare_windows",
    "require_vendored",
    "resample_windows",
    "resolve_checkpoint",
    "resolve_device",
    "run_condition",
    "select_best_condition",
    "sha256_file",
    "to_channels_first",
    "unflatten_targets",
    "vendored_root",
    "verify_checkpoint",
]

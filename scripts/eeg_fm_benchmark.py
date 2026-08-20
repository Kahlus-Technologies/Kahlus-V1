"""Run the external-pretrained-EEG-model vs Kahlus-incumbent forecasting benchmark.

This is the runnable half of ``docs/research/eeg_fm_benchmark_plan_2026-08-18.md``
(Lane A, the F8-comparable single-channel Sleep-EDF lane). It places CBraMod, EEGPT,
LaBraM-Base and BIOT PREST-16 -- each under the frozen-encoder and full-fine-tune
conditions, each against its own randomly-initialised architectural twin -- alongside
the incumbents (``kahlus_gru``, a tuned ridge-AR, persistence) under one protocol, one
normalisation convention, one symmetric tuning budget and one contamination audit.

Nothing here decides whether a claim may be made. It writes a dossier; a human reads it.

Protocol (plan section 3.1, byte-identical to F8 ``scripts/forecast_scaleup_paired.py``)
----------------------------------------------------------------------------------------
* PhysioNet Sleep-EDF Expanded sleep-cassette, ``EEG Fpz-Cz`` at 100 Hz.
* Subject id is the first five characters of the PSG filename, so both nights of one
  person land in the same split.
* Per recording: skip 1,000,000 samples, take 300,000.
* Context ``L = 127``; the target for horizon ``h`` is the single sample at
  ``i + 127 + h - 1``. Zero shared samples at every horizon, asserted per horizon by
  the index audit control.
* Shuffled subject-disjoint 39 / 12 / 27 split at seed 0, via F8's ``split_subjects``
  (imported, not re-derived, so it cannot drift).
* Horizons ``h in {1, 2, 4, 8, 16, 32, 64, 128, 256}``.

The transductive z-scoring fix (plan section 4.1 condition 1, a blocker)
------------------------------------------------------------------------
F8 z-scored each recording with that recording's **own full-series** mean and standard
deviation. Those statistics are computed over samples that include every test target, so
the normalisation is transductive: it leaks a summary of the test targets into the
inputs. F8 tolerated it because a GRU and a ridge see the identical convention, but the
plan makes it intolerable across models whose pretraining assumed different input
scaling, because a transductive statistic can interact differently with a pretrained
encoder than with a linear model.

This driver therefore computes each recording's mean and standard deviation from a
**prefix segment that is disjoint from every context and target window used downstream**,
and applies them causally to the evaluated segment:

* Default (``causal_prefix``): the statistics window is the 100,000 samples immediately
  **before** the 1,000,000-sample skip, i.e. absolute indices ``[900000, 1000000)``,
  while every context/target index lives in ``[1000000, 1300000)``. Disjoint by
  construction, and strictly in the past of everything it normalises.
* Fallback, used when the recording cannot supply a pre-skip prefix: the first
  ``prefix`` samples of the taken segment become the statistics window and the usable
  window starts after them, so disjointness still holds by construction.
* ``--legacy-normalization`` restores F8's full-series behaviour for the ablation the
  plan asks for ("run the benchmark under both and show the ranking is invariant"). It
  is **off** by default and the mode is recorded in every artifact.

``plan_normalization`` and ``window_span`` are pure index arithmetic precisely so the
disjointness property is a unit-testable fact rather than a comment.

Test-set discipline (plan sections 3.3 and 5.3)
-----------------------------------------------
Hyperparameter selection runs through :func:`select_configuration`, which is handed a
*selection task* whose ``x_test``/``y_test`` fields have been replaced by the validation
arrays. The real test arrays are not reachable from that object, so no configuration can
be chosen with any knowledge of test MSE, and a NaN-poisoned ``y_test`` provably cannot
change a selection. Test predictions are produced exactly once per entrant per horizon
by :func:`evaluate_selected`, and the count of test evaluations is reported.

Symmetric tuning budget (plan section 3.4)
-------------------------------------------
Every gradient-trained entrant -- incumbent GRU and every foundation model alike --
receives the same number of validation configurations, the same epoch budget, the same
patience, the same optimiser family, and the same validation-selection rule, because
they all train through ``fm_forecast.head.fit_head``. The tuned ridge receives F8's
full 14-point alpha grid (closed form, zero gradient steps). Persistence is
parameter-free and its degenerate one-point grid is labelled as such in the artifact
rather than padded to look symmetric. Configurations tried, gradient steps and wall
clock are all logged so a reader can check the symmetry claim instead of trusting it.

Known limitation: FM entrants are driven through their adapters' ``make_runner`` surface,
which returns predictions only, so the artifact records per-attempt configuration, seed,
validation MSE and wall clock rather than the per-epoch loss history that
``conditions.run_condition`` produces internally.

Parallelism and devices
------------------------
Single process by design. ``--device auto`` resolves to CUDA when visible; the process
respects whatever ``CUDA_VISIBLE_DEVICES`` the operator set and never edits it. To use
several GPUs, launch several processes with disjoint ``--models`` and disjoint
``--out-dir``; there is deliberately no multiprocessing inside this script.

Run::

    PYTHONPATH=src python3 scripts/eeg_fm_benchmark.py \\
        --data-root /path/to/sleep-edfx/sleep-cassette \\
        --weights-dir /path/to/staged/weights \\
        --out-dir artifacts/eeg_fm_benchmark --smoke
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS_DIR.parent
for _candidate in (str(_SCRIPTS_DIR), str(_REPO_ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from neurotwin.benchmarks.pretraining_contamination import (  # noqa: E402
    SLEEP_EDF_EVAL_DATASET_ID,
    contamination_status,
    contamination_table,
    has_model_selection_exposure,
    is_evaluation_permitted,
)
from neurotwin.data.forecast_contract import (  # noqa: E402
    FORECAST_PROTOCOL_V2_NONOVERLAP,
    ForecastTaskSpec,
    WindowExampleProvenance,
)
from neurotwin.data.prepared_tasks import SupervisedWindowTask  # noqa: E402
from neurotwin.gates import evaluate_gate  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402


ARTIFACT_SCHEMA = "kahlus.eeg_fm_benchmark.v1"

#: The claim scope this benchmark's dossier is evaluated under. Human-approved
#: (Krish Garg, 2026-08-19); see ``neurotwin.gates.unified_gate.NARROW_CLAIM_SCOPES``.
CLAIM_SCOPE = "external_pretrained_eeg_baseline_ranking_sleep_edf_forecast"

#: Gate branch for the v1 Sleep-EDF forecasting lane.
GATE_BRANCH = "v1"

DATASET_ID = SLEEP_EDF_EVAL_DATASET_ID

# ---------------------------------------------------------------------------
# Protocol constants, all transcribed from F8 (scripts/forecast_scaleup_paired.py)
# and plan section 3.1. Changing any of them breaks comparability with the
# published incumbent row, so they are constants rather than defaults.
# ---------------------------------------------------------------------------

SAMPLING_RATE_HZ = 100.0
CONTEXT_SAMPLES = 127
TARGET_SAMPLES = 1
CHANNEL = "EEG Fpz-Cz"
SKIP_SAMPLES = 1_000_000
TAKE_SAMPLES = 300_000
NORMALIZATION_PREFIX_SAMPLES = 100_000
STRIDE_SAMPLES = 256
MAX_WINDOWS_PER_RECORDING = 400
N_SUBJECTS = 78
SPLIT_SEED = 0
HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_N_BOOT = 1000

#: F8's 14-point ridge alpha grid, which runs high because long horizons want heavy
#: regularisation. Its size fixes N for the symmetric budget of plan section 3.4.
RIDGE_ALPHAS: tuple[float, ...] = (
    0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0, 2e5, 1e6, 5e6, 2e7,
)
TUNING_GRID_SIZE = len(RIDGE_ALPHAS)

#: Shared learning-rate axis for every gradient-trained entrant, so the budget symmetry
#: is visible in the grid itself rather than asserted in prose.
LEARNING_RATES: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
HEAD_HIDDEN_DIMS: tuple[int, ...] = (0, 128)
GRU_HIDDEN_DIMS: tuple[int, ...] = (32, 128)

FM_MODEL_IDS: tuple[str, ...] = ("eegpt", "cbramod", "labram_base", "biot_prest16")
CONDITIONS: tuple[str, ...] = ("frozen", "full_ft")

INCUMBENT_IDS: tuple[str, ...] = ("persistence", "tuned_ridge_ar", "kahlus_gru")

FAMILY_INCUMBENT = "incumbent"
FAMILY_EXTERNAL = "external_pretrained"
FAMILY_RANDOM_TWIN = "random_init_twin"

NORMALIZATION_CAUSAL_PREFIX = "causal_prefix"
NORMALIZATION_CAUSAL_HEAD = "causal_head_prefix"
NORMALIZATION_LEGACY = "legacy_full_series_transductive"

_ADAPTER_MODULES: dict[str, str] = {
    "eegpt": "neurotwin.benchmarks.fm_forecast.adapters.eegpt_adapter",
    "cbramod": "neurotwin.benchmarks.fm_forecast.adapters.cbramod_adapter",
    "labram_base": "neurotwin.benchmarks.fm_forecast.adapters.labram_adapter",
    "biot_prest16": "neurotwin.benchmarks.fm_forecast.adapters.biot_adapter",
}


class AdapterUnavailable(RuntimeError):
    """One model's vendored upstream or staged checkpoint is missing.

    Raised per model so a missing vendor checkout or an unstaged checkpoint blocks only
    that entrant; the rest of the benchmark still runs and the artifact records why the
    model is absent. It never degrades into a numeric result: an entrant that cannot be
    built is reported as absent, never as a large MSE.
    """


class EntrantFailed(RuntimeError):
    """Every configuration of one entrant failed at one horizon.

    Distinct from :class:`AdapterUnavailable` on purpose: the model was constructible and
    simply produced no finite validation MSE anywhere on its grid. Scoring it would
    invent a number, so the driver records the failure and blocks the gate instead.
    """


# ---------------------------------------------------------------------------
# Normalisation planning: pure index arithmetic (see module docstring)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationPlan:
    """Where a recording's z-scoring statistics come from, in absolute sample indices.

    ``[stats_start, stats_stop)`` is the segment whose mean and standard deviation are
    used. ``[usable_start, usable_stop)`` is the segment context and target windows may
    be drawn from. ``read_start`` is the first absolute sample the loader must read.

    Under either causal mode the two intervals are disjoint by construction and the
    statistics window lies strictly in the past, which is exactly what
    :meth:`is_disjoint_from` reports and what the driver asserts before scoring.
    """

    mode: str
    stats_start: int
    stats_stop: int
    usable_start: int
    usable_stop: int
    read_start: int

    @property
    def is_transductive(self) -> bool:
        """True when the statistics window can contain evaluated targets (legacy mode)."""

        return self.mode == NORMALIZATION_LEGACY

    @property
    def stats_offset(self) -> int:
        """Offset of the statistics window inside the buffer the loader reads."""

        return self.stats_start - self.read_start

    @property
    def usable_offset(self) -> int:
        """Offset of the usable window inside the buffer the loader reads."""

        return self.usable_start - self.read_start

    def is_disjoint_from(self, start: int, stop: int) -> bool:
        """True when ``[start, stop)`` shares no sample with the statistics window."""

        return int(stop) <= self.stats_start or int(start) >= self.stats_stop

    def as_log(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "transductive": self.is_transductive,
            "stats_window_absolute": [int(self.stats_start), int(self.stats_stop)],
            "usable_window_absolute": [int(self.usable_start), int(self.usable_stop)],
            "read_start_absolute": int(self.read_start),
        }


def plan_normalization(
    *,
    skip: int = SKIP_SAMPLES,
    take: int = TAKE_SAMPLES,
    prefix: int = NORMALIZATION_PREFIX_SAMPLES,
    legacy: bool = False,
) -> NormalizationPlan:
    """Choose the statistics window for one recording.

    Args:
        skip: leading samples of the recording that F8 discards.
        take: samples F8 evaluates on, starting at ``skip``.
        legacy: reproduce F8's transductive full-series z-scoring for the ablation.

    Returns:
        A :class:`NormalizationPlan`. In the causal modes the statistics window never
        overlaps the usable window, so no context or target sample can influence the
        statistics that normalise it.
    """

    skip, take, prefix = int(skip), int(take), int(prefix)
    if take < 1:
        raise ValueError("take must be positive")
    if prefix < 1:
        raise ValueError("prefix must be positive")

    if legacy:
        # F8's original convention, kept only so the ranking can be shown invariant
        # (or not) to it. The statistics window IS the usable window: transductive.
        return NormalizationPlan(
            mode=NORMALIZATION_LEGACY,
            stats_start=skip,
            stats_stop=skip + take,
            usable_start=skip,
            usable_stop=skip + take,
            read_start=skip,
        )

    if skip >= prefix:
        # Preferred: statistics come from the discarded pre-skip region, so the whole
        # 300k evaluated segment stays available and nothing evaluated is ever seen.
        return NormalizationPlan(
            mode=NORMALIZATION_CAUSAL_PREFIX,
            stats_start=skip - prefix,
            stats_stop=skip,
            usable_start=skip,
            usable_stop=skip + take,
            read_start=skip - prefix,
        )

    if take <= prefix:
        raise ValueError(
            f"cannot carve a {prefix}-sample causal statistics prefix out of a "
            f"{take}-sample segment with only {skip} samples of history available"
        )
    # Fallback: burn the head of the evaluated segment as the statistics window and
    # start the usable window after it. Still disjoint, still causal, just shorter.
    return NormalizationPlan(
        mode=NORMALIZATION_CAUSAL_HEAD,
        stats_start=skip,
        stats_stop=skip + prefix,
        usable_start=skip + prefix,
        usable_stop=skip + take,
        read_start=skip,
    )


def window_span(start: int, *, context: int = CONTEXT_SAMPLES, horizon: int) -> tuple[int, int, int, int]:
    """Absolute ``(ctx_start, ctx_stop, target_start, target_stop)`` for one example.

    F8's index rule: given ``signal[i : i + context]`` predict ``signal[i + context + h - 1]``.
    The context and target intervals are disjoint for every ``h >= 1``.
    """

    start, context, horizon = int(start), int(context), int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    target_start = start + context + horizon - 1
    return start, start + context, target_start, target_start + TARGET_SAMPLES


def window_starts(
    plan: NormalizationPlan,
    *,
    context: int = CONTEXT_SAMPLES,
    max_horizon: int,
    stride: int = STRIDE_SAMPLES,
    max_windows: int = MAX_WINDOWS_PER_RECORDING,
    seed: int = 0,
) -> np.ndarray:
    """Absolute context-window start indices for one recording, F8's rule.

    Every returned start satisfies ``plan.is_disjoint_from(*span)`` for the whole
    context+target extent at every horizon up to ``max_horizon`` whenever ``plan`` is
    one of the causal modes; the driver asserts this rather than assuming it.
    """

    need = int(context) + int(max_horizon)
    starts = np.arange(int(plan.usable_start), int(plan.usable_stop) - need, int(stride))
    if starts.size == 0:
        return starts.astype(np.int64)
    if starts.size > int(max_windows):
        rng = np.random.default_rng(int(seed))
        starts = np.sort(rng.choice(starts, size=int(max_windows), replace=False))
    return starts.astype(np.int64)


def assert_normalization_disjoint(
    plan: NormalizationPlan,
    starts: Sequence[int] | np.ndarray,
    *,
    context: int = CONTEXT_SAMPLES,
    horizons: Sequence[int] = HORIZONS,
) -> None:
    """Fail loudly if any evaluated sample also fed the normalisation statistics."""

    if plan.is_transductive:
        return
    for start in np.asarray(starts, dtype=np.int64).tolist():
        for horizon in horizons:
            ctx_start, _, _, target_stop = window_span(start, context=context, horizon=int(horizon))
            if not plan.is_disjoint_from(ctx_start, target_stop):
                raise ValueError(
                    f"normalisation statistics window {[plan.stats_start, plan.stats_stop]} "
                    f"overlaps the window [{ctx_start}, {target_stop}) at horizon {horizon}: "
                    "the causal normaliser would be transductive"
                )


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingWindows:
    """One PSG recording's windows, already normalised under ``plan``."""

    subject: str
    record_id: str
    starts: np.ndarray
    context: np.ndarray
    targets: dict[int, np.ndarray]
    stats_mean: float
    stats_std: float


def load_recording(
    path: Path,
    plan: NormalizationPlan,
    *,
    channel: str = CHANNEL,
    take: int = TAKE_SAMPLES,
) -> tuple[np.ndarray, float, float]:
    """Load one channel and normalise it with statistics drawn from ``plan``.

    Returns the usable segment (absolute indices ``[plan.usable_start, plan.usable_stop)``)
    as float32, plus the mean and standard deviation actually applied.
    """

    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    resolved = channel
    if resolved not in raw.ch_names:
        resolved = next((name for name in raw.ch_names if "EEG" in name), raw.ch_names[0])
    raw.pick([resolved])
    raw.load_data(verbose="ERROR")
    values = raw.get_data()[0].astype(np.float64)

    stop = plan.usable_stop
    buffer = values[plan.read_start : stop]
    if buffer.size < (plan.usable_stop - plan.read_start):
        raise ValueError(f"{path.name}: recording is too short for the F8 skip/take protocol")

    stats = buffer[plan.stats_offset : plan.stats_offset + (plan.stats_stop - plan.stats_start)]
    if plan.is_transductive:
        stats = buffer[plan.usable_offset :]
    mean = float(stats.mean())
    std = float(stats.std())
    usable = buffer[plan.usable_offset :]
    normalised = ((usable - mean) / (std + 1e-8)).astype(np.float32)
    if normalised.size < int(take) and not plan.is_transductive and plan.mode == NORMALIZATION_CAUSAL_PREFIX:
        raise ValueError(f"{path.name}: usable segment is shorter than the requested take")
    return normalised, mean, std


def cut_windows(
    signal: np.ndarray,
    plan: NormalizationPlan,
    *,
    horizons: Sequence[int],
    context: int = CONTEXT_SAMPLES,
    stride: int = STRIDE_SAMPLES,
    max_windows: int = MAX_WINDOWS_PER_RECORDING,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Cut a normalised recording into contexts and strictly-future single-sample targets."""

    max_horizon = int(max(horizons))
    starts = window_starts(
        plan,
        context=context,
        max_horizon=max_horizon,
        stride=stride,
        max_windows=max_windows,
        seed=seed,
    )
    usable_len = int(signal.shape[0])
    starts = starts[(starts - plan.usable_start + context + max_horizon) <= usable_len]
    if starts.size == 0:
        return starts, np.zeros((0, context), dtype=np.float32), {int(h): np.zeros((0,), dtype=np.float32) for h in horizons}
    assert_normalization_disjoint(plan, starts, context=context, horizons=horizons)

    local = starts - plan.usable_start
    contexts = np.stack([signal[i : i + context] for i in local]).astype(np.float32)
    targets = {
        int(h): np.asarray([signal[i + context + int(h) - 1] for i in local], dtype=np.float32)
        for h in horizons
    }
    return starts, contexts, targets


def load_corpus(
    data_root: Path,
    plan: NormalizationPlan,
    *,
    horizons: Sequence[int],
    n_subjects: int = N_SUBJECTS,
    stride: int = STRIDE_SAMPLES,
    max_windows: int = MAX_WINDOWS_PER_RECORDING,
    channel: str = CHANNEL,
    take: int = TAKE_SAMPLES,
) -> dict[str, list[RecordingWindows]]:
    """Load every sleep-cassette PSG recording, grouped by subject (F8's grouping)."""

    psg = sorted(Path(data_root).glob("*-PSG.edf"))
    if not psg:
        raise FileNotFoundError(f"no *-PSG.edf recordings under {data_root}")
    by_subject: dict[str, list[Path]] = {}
    for candidate in psg:
        # SC4001E0 -> "SC400": both nights of one person share a subject id.
        by_subject.setdefault(candidate.name[:5], []).append(candidate)
    subjects = sorted(by_subject)[: int(n_subjects)]

    corpus: dict[str, list[RecordingWindows]] = {}
    for index, subject in enumerate(subjects):
        for record_index, record_path in enumerate(sorted(by_subject[subject])):
            signal, mean, std = load_recording(record_path, plan, channel=channel, take=take)
            starts, contexts, targets = cut_windows(
                signal,
                plan,
                horizons=horizons,
                stride=stride,
                max_windows=max_windows,
                seed=index * 1000 + record_index,
            )
            if starts.size == 0:
                continue
            corpus.setdefault(subject, []).append(
                RecordingWindows(
                    subject=subject,
                    record_id=record_path.stem,
                    starts=starts,
                    context=contexts,
                    targets=targets,
                    stats_mean=mean,
                    stats_std=std,
                )
            )
    return corpus


def split_subjects(subjects: Sequence[str], seed: int = SPLIT_SEED) -> tuple[list[str], list[str], list[str]]:
    """F8's deterministic subject-disjoint partition, imported so it cannot drift."""

    from forecast_scaleup_paired import split_subjects as f8_split_subjects

    return f8_split_subjects(list(subjects), seed)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForecastTaskBundle:
    """One horizon's harness task, plus the selection view that cannot see test."""

    horizon: int
    task: SupervisedWindowTask
    selection_task: SupervisedWindowTask
    test_subject_bounds: dict[str, tuple[int, int]]
    claim_eligible: bool
    n_train: int
    n_val: int
    n_test: int


def _forecast_spec(horizon: int) -> tuple[ForecastTaskSpec, bool]:
    """Resolve the v2 non-overlap spec for one horizon.

    ``gap_samples = h - 1`` reproduces F8's index rule exactly. The v2 contract reserves
    ``claim_eligible`` for a strictly positive gap, so ``h = 1`` (the adjacent-sample
    forecast, still zero shared samples) is declared ineligible rather than relaxing the
    contract. The index audit control checks disjointness at every horizon independently.
    """

    claim_eligible = horizon >= 2
    spec = ForecastTaskSpec(
        protocol_id=FORECAST_PROTOCOL_V2_NONOVERLAP,
        schema_version=2,
        context_seconds=CONTEXT_SAMPLES / SAMPLING_RATE_HZ,
        target_seconds=TARGET_SAMPLES / SAMPLING_RATE_HZ,
        gap_seconds=(horizon - 1) / SAMPLING_RATE_HZ,
        stride_seconds=STRIDE_SAMPLES / SAMPLING_RATE_HZ,
        claim_eligible=claim_eligible,
    )
    return spec, claim_eligible


def _pool(
    corpus: Mapping[str, list[RecordingWindows]],
    subjects: Sequence[str],
    horizon: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray, list[WindowExampleProvenance], dict[str, tuple[int, int]]]:
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    provenance: list[WindowExampleProvenance] = []
    bounds: dict[str, tuple[int, int]] = {}
    cursor = 0
    for subject in subjects:
        start_cursor = cursor
        for record in corpus[subject]:
            contexts.append(record.context)
            targets.append(record.targets[int(horizon)])
            for start in record.starts.tolist():
                ctx_start, ctx_stop, target_start, target_stop = window_span(start, horizon=horizon)
                provenance.append(
                    WindowExampleProvenance(
                        dataset_id=DATASET_ID,
                        subject_id=subject,
                        session_id=record.record_id,
                        site_id="physionet-sleep-cassette",
                        record_id=record.record_id,
                        source_hash=None,
                        split=split,
                        input_start=ctx_start,
                        input_stop=ctx_stop,
                        target_start=target_start,
                        target_stop=target_stop,
                    )
                )
            cursor += int(record.context.shape[0])
        bounds[subject] = (start_cursor, cursor)
    x = np.concatenate(contexts).astype(np.float32)[:, :, None]
    y = np.concatenate(targets).astype(np.float32)[:, None, None]
    return x, y, provenance, bounds


def build_task_bundle(
    corpus: Mapping[str, list[RecordingWindows]],
    *,
    horizon: int,
    train_subjects: Sequence[str],
    val_subjects: Sequence[str],
    test_subjects: Sequence[str],
) -> ForecastTaskBundle:
    """Assemble one horizon's :class:`SupervisedWindowTask` and its selection view."""

    spec, claim_eligible = _forecast_spec(horizon)
    resolved = spec.resolve(SAMPLING_RATE_HZ)

    x_train, y_train, train_prov, _ = _pool(corpus, train_subjects, horizon, "train")
    x_val, y_val, val_prov, _ = _pool(corpus, val_subjects, horizon, "val")
    x_test, y_test, test_prov, test_bounds = _pool(corpus, test_subjects, horizon, "test")

    task = SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_val=x_val,
        y_val=y_val,
        forecast_spec=resolved,
        train_provenance=tuple(train_prov),
        val_provenance=tuple(val_prov),
        test_provenance=tuple(test_prov),
        notes=(f"eeg_fm_benchmark lane A horizon {horizon}",),
        metadata={"horizon": int(horizon), "sampling_rate_hz": SAMPLING_RATE_HZ},
    )
    return ForecastTaskBundle(
        horizon=int(horizon),
        task=task,
        selection_task=make_selection_task(task),
        test_subject_bounds=test_bounds,
        claim_eligible=bool(claim_eligible),
        n_train=int(x_train.shape[0]),
        n_val=int(x_val.shape[0]),
        n_test=int(x_test.shape[0]),
    )


def make_selection_task(task: SupervisedWindowTask) -> SupervisedWindowTask:
    """Return a view of ``task`` whose test arrays have been replaced by validation ones.

    This is the structural half of the test-set discipline. Entrant code called during
    hyperparameter search receives *this* object, so the real ``y_test`` is not reachable
    from anywhere in the selection path -- not merely unread by convention.
    """

    return replace(
        task,
        x_test=np.asarray(task.x_val, dtype=np.float32),
        y_test=np.asarray(task.y_val, dtype=np.float32),
        test_provenance=task.val_provenance,
        notes=task.notes + ("selection view: test arrays replaced by validation arrays",),
    )


def index_audit(bundle: ForecastTaskBundle) -> dict[str, Any]:
    """Plan section 4 index audit: zero shared samples between context and target."""

    shared = 0
    checked = 0
    for rows in (bundle.task.train_provenance, bundle.task.val_provenance, bundle.task.test_provenance):
        for row in rows:
            checked += 1
            if row.target_overlaps_input:
                shared += 1
    return {
        "horizon": bundle.horizon,
        "windows_checked": checked,
        "windows_with_shared_samples": shared,
        "passed": shared == 0,
        "falsification_threshold": "any window sharing a sample between context and target fails",
    }


# ---------------------------------------------------------------------------
# Entrants
# ---------------------------------------------------------------------------


EntrantRunner = Callable[[SupervisedWindowTask, int, int], np.ndarray]
RunnerFactory = Callable[[Mapping[str, Any]], EntrantRunner]


@dataclass(frozen=True)
class Entrant:
    """One competitor: an id, a contamination-joinable model id, and a tuning grid."""

    entrant_id: str
    family: str
    model_id: str | None
    condition: str | None
    pretrained: bool
    grid: tuple[dict[str, Any], ...]
    factory: RunnerFactory
    budget_note: str = ""
    ranked: bool = True

    @property
    def configs_tried(self) -> int:
        return len(self.grid)

    def as_log(self) -> dict[str, Any]:
        model_id = self.model_id
        return {
            "entrant_id": self.entrant_id,
            "family": self.family,
            "model_id": model_id,
            "condition": self.condition,
            "pretrained": bool(self.pretrained),
            "ranked": bool(self.ranked),
            "configs_tried": self.configs_tried,
            "budget_note": self.budget_note,
            "contamination_status": (
                contamination_status(model_id, DATASET_ID) if model_id else "not_applicable_no_pretraining"
            ),
            "model_selection_exposure": bool(model_id and has_model_selection_exposure(model_id, DATASET_ID)),
        }


def _flat_targets(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32)
    return values.reshape(values.shape[0], -1)


def _persistence_factory(_config: Mapping[str, Any]) -> EntrantRunner:
    """Reuse ``baseline_suite``'s registered persistence runner verbatim."""

    from neurotwin.benchmarks.baseline_suite import EXECUTABLE_BASELINE_RUNNERS

    spec = next(runner for runner in EXECUTABLE_BASELINE_RUNNERS if runner.model_id == "persistence")

    def runner(task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        return spec.predict(task, seed, train_steps)

    return runner


def _ridge_factory(config: Mapping[str, Any]) -> EntrantRunner:
    """Closed-form ridge-AR on the flattened context, at one grid alpha."""

    from _forecast_common import ridge_fit, ridge_predict

    alpha = float(config["alpha"])

    def runner(task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        x_train = np.asarray(task.x_train, dtype=np.float64).reshape(task.x_train.shape[0], -1)
        x_test = np.asarray(task.x_test, dtype=np.float64).reshape(task.x_test.shape[0], -1)
        weights = ridge_fit(x_train, _flat_targets(task.y_train).astype(np.float64), alpha)
        prediction = ridge_predict(weights, x_test)
        return prediction.reshape(task.y_test.shape).astype(np.float32)

    return runner


def _gru_factory(config: Mapping[str, Any]) -> EntrantRunner:
    """The F8 incumbent GRU, trained through the shared ``fit_head`` protocol.

    Deliberately *not* F8's fixed-gradient-step recipe: to make the plan section 3.4
    budget symmetry real, the incumbent GRU gets the identical epoch budget, patience,
    optimiser and validation-selection rule as every foundation-model head. Its
    architecture (single-layer GRU over the 127-sample context, linear read-out of the
    final state, direct horizon regression) is unchanged from F8.
    """

    import torch
    from torch import nn

    from neurotwin.benchmarks.fm_forecast.head import TrainConfig, fit_head, predict

    hidden = int(config["hidden"])
    lr = float(config["lr"])
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    patience = int(config["patience"])
    device = str(config.get("device", "auto"))

    class _GRUForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(1, hidden, batch_first=True)
            self.readout = nn.Linear(hidden, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            output, _ = self.gru(x)
            return self.readout(output[:, -1, :])

    def runner(task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        torch.manual_seed(int(seed))
        model = _GRUForecaster()
        fit_head(
            model,
            x_train=np.asarray(task.x_train, dtype=np.float32),
            y_train=_flat_targets(task.y_train),
            x_val=np.asarray(task.x_val, dtype=np.float32),
            y_val=_flat_targets(task.y_val),
            config=TrainConfig(
                lr=lr,
                epochs=min(epochs, int(train_steps)) if train_steps else epochs,
                batch_size=batch_size,
                patience=patience,
                seed=int(seed),
                device=device,
            ),
        )
        flat = predict(model, np.asarray(task.x_test, dtype=np.float32), device=device)
        return flat.reshape(task.y_test.shape).astype(np.float32)

    return runner


def load_adapter(model_id: str):
    """Import one model's adapter lazily, so a missing vendor checkout blocks only it."""

    module_path = _ADAPTER_MODULES.get(model_id)
    if module_path is None:
        raise AdapterUnavailable(f"{model_id!r} is not one of {sorted(_ADAPTER_MODULES)}")
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - any import failure is a per-model block
        raise AdapterUnavailable(
            f"adapter for {model_id!r} could not be imported ({type(exc).__name__}: {exc}). "
            f"Vendor its upstream under external/vendor/ and stage its checkpoint; only "
            f"{model_id!r} is blocked, the rest of the benchmark still runs."
        ) from exc


def _fm_factory(model_id: str, condition: str, weights_dir: Path, *, pretrained: bool, device: str) -> RunnerFactory:
    """Build a factory over one model's adapter ``make_runner`` surface."""

    def factory(config: Mapping[str, Any]) -> EntrantRunner:
        from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
        from neurotwin.benchmarks.fm_forecast.head import HeadConfig
        from neurotwin.benchmarks.fm_forecast.vendor_import import VendoredUpstreamMissing
        from neurotwin.benchmarks.fm_forecast.weights_manifest import WeightsManifestError

        adapter = load_adapter(model_id)
        condition_config = ConditionConfig(
            condition=Condition(condition),
            lr=float(config["lr"]),
            encoder_lr_scale=float(config.get("encoder_lr_scale", 0.1)),
            epochs=int(config["epochs"]),
            batch_size=int(config["batch_size"]),
            patience=int(config["patience"]),
            device=device,
            head=HeadConfig(hidden_dim=int(config["head_hidden"])),
        )
        try:
            return adapter.make_runner(
                Path(weights_dir),
                Condition(condition),
                config=condition_config,
                pretrained=bool(pretrained),
            )
        except (VendoredUpstreamMissing, WeightsManifestError) as exc:
            # A missing vendor checkout or an unstaged/mismatched checkpoint is a
            # staging problem, not a score. Re-raised as AdapterUnavailable so the
            # driver drops this model with a message instead of scoring it as bad.
            raise AdapterUnavailable(
                f"{model_id!r} cannot be built from {weights_dir}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    return factory


def _gradient_grid(n_configs: int, *, epochs: int, batch_size: int, patience: int, device: str, hidden_axis: Sequence[int], hidden_key: str) -> tuple[dict[str, Any], ...]:
    grid = [
        {
            "lr": lr,
            hidden_key: hidden,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "device": device,
        }
        for hidden in hidden_axis
        for lr in LEARNING_RATES
    ]
    return tuple(grid[: int(n_configs)])


def build_entrants(
    *,
    models: Sequence[str],
    conditions: Sequence[str],
    weights_dir: Path,
    include_random_twin: bool,
    n_configs: int,
    max_train_epochs: int,
    batch_size: int,
    patience: int,
    device: str,
) -> tuple[list[Entrant], list[dict[str, str]]]:
    """Assemble the entrant slate, skipping models whose contamination gate refuses them."""

    entrants: list[Entrant] = [
        Entrant(
            entrant_id="persistence",
            family=FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=({"kind": "parameter_free"},),
            factory=_persistence_factory,
            budget_note=(
                "parameter-free floor: a one-point grid is its complete configuration space, "
                "so it is reported as 1 rather than padded to N to look symmetric"
            ),
        ),
        Entrant(
            entrant_id="tuned_ridge_ar",
            family=FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=tuple({"alpha": alpha} for alpha in RIDGE_ALPHAS[: int(n_configs)]),
            factory=_ridge_factory,
            budget_note="F8's 14-point alpha grid to 2e7; closed form, zero gradient steps",
        ),
        Entrant(
            entrant_id="kahlus_gru",
            family=FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=_gradient_grid(
                n_configs,
                epochs=max_train_epochs,
                batch_size=batch_size,
                patience=patience,
                device=device,
                hidden_axis=GRU_HIDDEN_DIMS,
                hidden_key="hidden",
            ),
            factory=_gru_factory,
            budget_note="same lr axis, epoch budget, patience and val-selection rule as every FM head",
        ),
    ]

    skipped: list[dict[str, str]] = []
    for model_id in models:
        if not is_evaluation_permitted(model_id, DATASET_ID):
            skipped.append(
                {
                    "model_id": model_id,
                    "reason": (
                        f"pretraining contamination gate refuses {model_id!r} on {DATASET_ID!r} "
                        f"(status {contamination_status(model_id, DATASET_ID)!r})"
                    ),
                }
            )
            continue
        for condition in conditions:
            variants = [(True, FAMILY_EXTERNAL, "")]
            if include_random_twin:
                variants.append((False, FAMILY_RANDOM_TWIN, "__random_twin"))
            for pretrained, family, suffix in variants:
                entrants.append(
                    Entrant(
                        entrant_id=f"{model_id}__{condition}{suffix}",
                        family=family,
                        model_id=model_id,
                        condition=condition,
                        pretrained=pretrained,
                        grid=_gradient_grid(
                            n_configs,
                            epochs=max_train_epochs,
                            batch_size=batch_size,
                            patience=patience,
                            device=device,
                            hidden_axis=HEAD_HIDDEN_DIMS,
                            hidden_key="head_hidden",
                        ),
                        factory=_fm_factory(
                            model_id, condition, weights_dir, pretrained=pretrained, device=device
                        ),
                        budget_note=(
                            "identical bolt-on head, lr axis, epoch budget, patience and "
                            "val-selection rule as the incumbent GRU"
                        ),
                        # Random twins are the plan section 4 control, not competitors.
                        ranked=pretrained,
                    )
                )
    return entrants, skipped


# ---------------------------------------------------------------------------
# Selection and evaluation (structurally separated; see module docstring)
# ---------------------------------------------------------------------------


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(diff**2))


@dataclass
class AttemptLedger:
    """Plan section 5.3 iteration ledger: every attempt, every test evaluation."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(self, **payload: Any) -> None:
        row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), **payload}
        self.rows.append(row)

    @property
    def n_attempts(self) -> int:
        return len(self.rows)

    @property
    def n_test_evaluations(self) -> int:
        return sum(1 for row in self.rows if row.get("phase") == "test")

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return path


def select_configuration(
    entrant: Entrant,
    selection_task: SupervisedWindowTask,
    *,
    horizon: int,
    seeds: Sequence[int],
    train_steps: int,
    ledger: AttemptLedger,
) -> dict[str, Any]:
    """Choose ``entrant``'s configuration on validation MSE alone.

    ``selection_task`` must be a :func:`make_selection_task` view: its ``y_test`` *is*
    ``y_val``, so the real test targets are not reachable from this function's inputs and
    no amount of code here can tune on test.
    """

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for config_index, config in enumerate(entrant.grid):
        per_seed: list[float] = []
        for seed in seeds:
            started = time.perf_counter()
            try:
                runner = entrant.factory(config)
                prediction = runner(selection_task, int(seed), int(train_steps))
                value = mse(selection_task.y_test, prediction)
                error = ""
            except AdapterUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - a failed config is data, not a crash
                value = float("inf")
                error = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - started
            per_seed.append(value)
            ledger.record(
                phase="validation",
                entrant_id=entrant.entrant_id,
                horizon=int(horizon),
                config_index=config_index,
                config=dict(config),
                seed=int(seed),
                val_mse=value,
                wall_clock_seconds=round(elapsed, 4),
                error=error,
            )
        finite = [value for value in per_seed if np.isfinite(value)]
        scored.append((float(np.mean(finite)) if finite else float("inf"), config_index, dict(config)))

    best_val, best_index, best_config = min(scored, key=lambda item: (item[0], item[1]))
    if not np.isfinite(best_val):
        raise EntrantFailed(
            f"every one of {entrant.configs_tried} configurations of {entrant.entrant_id!r} "
            f"produced a non-finite validation MSE at horizon {horizon}; refusing to run a "
            "test evaluation, because there is no selected configuration to evaluate"
        )
    return {
        "config": best_config,
        "config_index": best_index,
        "val_mse": best_val,
        "grid_edge_selected": bool(best_index in (0, len(entrant.grid) - 1)) and len(entrant.grid) > 1,
        "configs_tried": len(entrant.grid),
        "val_mse_by_config": [value for value, _, _ in scored],
    }


def evaluate_selected(
    entrant: Entrant,
    task: SupervisedWindowTask,
    selection: Mapping[str, Any],
    *,
    horizon: int,
    seeds: Sequence[int],
    train_steps: int,
    test_subject_bounds: Mapping[str, tuple[int, int]],
    ledger: AttemptLedger,
) -> dict[str, Any]:
    """The single test evaluation for one entrant at one horizon (plan section 5.3).

    Called once per entrant per horizon, after :func:`select_configuration` has already
    fixed the configuration. Nothing here may feed back into a selection.
    """

    per_seed_subject_mse: list[dict[str, float]] = []
    for seed in seeds:
        started = time.perf_counter()
        runner = entrant.factory(selection["config"])
        prediction = np.asarray(runner(task, int(seed), int(train_steps)), dtype=np.float32)
        if prediction.shape != task.y_test.shape:
            raise ValueError(
                f"{entrant.entrant_id} prediction shape {prediction.shape} does not match "
                f"target {task.y_test.shape}"
            )
        by_subject = {
            subject: mse(task.y_test[low:high], prediction[low:high])
            for subject, (low, high) in test_subject_bounds.items()
            if high > low
        }
        per_seed_subject_mse.append(by_subject)
        ledger.record(
            phase="test",
            entrant_id=entrant.entrant_id,
            horizon=int(horizon),
            config=dict(selection["config"]),
            seed=int(seed),
            test_mse_mean=float(np.mean(list(by_subject.values()))) if by_subject else float("nan"),
            wall_clock_seconds=round(time.perf_counter() - started, 4),
        )

    subjects = sorted(per_seed_subject_mse[0]) if per_seed_subject_mse else []
    # Average each subject's error across seeds: the claim is about the method, not
    # about an ensemble we never proposed (F8's convention).
    averaged = {
        subject: float(np.mean([run[subject] for run in per_seed_subject_mse])) for subject in subjects
    }
    seed_pooled = [
        float(np.mean([run[subject] for subject in subjects])) for run in per_seed_subject_mse
    ]
    return {
        "per_subject_test_mse": averaged,
        "test_mse_mean": float(np.mean(list(averaged.values()))) if averaged else float("nan"),
        "seed_pooled_test_mse": seed_pooled,
        "seed_spread": float(np.std(seed_pooled)) if seed_pooled else 0.0,
        "n_test_evaluations": len(per_seed_subject_mse),
    }


def bootstrap_subject_ci(
    per_subject: Mapping[str, float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = SPLIT_SEED,
) -> dict[str, float]:
    """Seeded subject-level bootstrap CI on the mean per-subject MSE."""

    values = np.asarray([per_subject[key] for key in sorted(per_subject)], dtype=np.float64)
    if values.size == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "n_boot": int(n_boot)}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(values, size=(int(n_boot), values.size), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "n_boot": int(n_boot),
        "n_subjects": int(values.size),
    }


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def label_shuffle_control(
    entrant: Entrant,
    bundle: ForecastTaskBundle,
    selection: Mapping[str, Any],
    *,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Permute the training targets and check the entrant's validation skill collapses."""

    rng = np.random.default_rng(int(seed))
    task = bundle.selection_task
    shuffled = np.asarray(task.y_train, dtype=np.float32).copy()
    shuffled = shuffled[rng.permutation(shuffled.shape[0])]
    poisoned = replace(task, y_train=shuffled)
    runner = entrant.factory(selection["config"])
    prediction = runner(poisoned, int(seed), int(selection["config"].get("epochs", 1)))
    shuffled_mse = mse(task.y_test, prediction)
    honest_mse = float(selection["val_mse"])
    return {
        "entrant_id": entrant.entrant_id,
        "horizon": bundle.horizon,
        "val_mse_honest": honest_mse,
        "val_mse_label_shuffled": shuffled_mse,
        "passed": bool(np.isfinite(shuffled_mse) and shuffled_mse >= honest_mse),
        "falsification_threshold": "shuffled-label validation MSE must not beat the honest one",
    }


def random_twin_controls(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Plan section 4's most important control: did pretraining buy anything at all?"""

    by_key = {(row["entrant_id"], row["horizon"]): row for row in results}
    controls: list[dict[str, Any]] = []
    for (entrant_id, horizon), row in sorted(by_key.items()):
        if not entrant_id.endswith("__random_twin"):
            continue
        pretrained_id = entrant_id[: -len("__random_twin")]
        pretrained = by_key.get((pretrained_id, horizon))
        if pretrained is None:
            continue
        gap = float(row["test_mse_mean"]) - float(pretrained["test_mse_mean"])
        controls.append(
            {
                "model_id": pretrained.get("model_id"),
                "condition": pretrained.get("condition"),
                "horizon": horizon,
                "pretrained_test_mse": float(pretrained["test_mse_mean"]),
                "random_twin_test_mse": float(row["test_mse_mean"]),
                "gap_random_minus_pretrained": gap,
                "pretraining_helped": bool(gap > 0.0),
                "falsification_threshold": (
                    "a pretrained model that does not beat its own random-init twin shows "
                    "pretraining bought nothing on this task"
                ),
            }
        )
    return controls


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=False
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "clean_worktree": not bool(status),
        "dirty_paths": sorted(line[3:] for line in status.splitlines()) if status else [],
    }


def _environment_payload() -> dict[str, Any]:
    versions: dict[str, str] = {"numpy": np.__version__}
    for name in ("torch", "mne"):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001 - a missing optional package is reportable, not fatal
            versions[name] = "not installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "package_versions": versions,
    }


def contamination_dump() -> list[dict[str, Any]]:
    """The contamination table with the plan's model-selection-exposure column added."""

    return [
        {**row, "model_selection_exposure": has_model_selection_exposure(row["model_id"], row["dataset_id"])}
        for row in contamination_table()
    ]


def build_ranking(results: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> list[dict[str, Any]]:
    """Per horizon, rank ranked entrants, taking each FM's best admissible condition.

    "Best admissible condition" is argmin **validation** MSE over {frozen, full_ft}
    (plan section 3.3); the selected condition's test number is the only one that enters
    the ranking. Random-init twins are controls and never appear here.
    """

    ranking: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = [row for row in results if row["horizon"] == int(horizon) and row.get("ranked")]
        best_by_model: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            key = row.get("model_id") or row["entrant_id"]
            incumbent = best_by_model.get(key)
            if incumbent is None or float(row["val_mse"]) < float(incumbent["val_mse"]):
                best_by_model[key] = row
        ordered = sorted(best_by_model.values(), key=lambda row: (float(row["test_mse_mean"]), row["entrant_id"]))
        ranking.append(
            {
                "horizon": int(horizon),
                "rows": [
                    {
                        "rank": index + 1,
                        "entrant_id": row["entrant_id"],
                        "model_id": row.get("model_id"),
                        "condition": row.get("condition"),
                        "condition_selected_on": "validation_mse",
                        "val_mse": float(row["val_mse"]),
                        "test_mse_mean": float(row["test_mse_mean"]),
                        "test_mse_ci95_low": float(row["test_mse_ci95_low"]),
                        "test_mse_ci95_high": float(row["test_mse_ci95_high"]),
                        "contamination_status": row["contamination_status"],
                        "model_selection_exposure": row["model_selection_exposure"],
                        "grid_edge_selected": row["grid_edge_selected"],
                        "configs_tried": row["configs_tried"],
                    }
                    for index, row in enumerate(ordered)
                ],
            }
        )
    return ranking


def build_gate_dossier(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the unified gate under the benchmark's narrow claim scope.

    Fails closed: every benchmark-specific precondition that is not demonstrably True
    becomes an explicit failure reason, so a partially-run benchmark can never emerge
    with ``scientific_claim_allowed=True``.
    """

    results = list(payload.get("results", []))
    controls = payload.get("controls", {})
    ranking = list(payload.get("ranking", []))

    finite_metrics = bool(results) and all(
        np.isfinite(float(row["test_mse_mean"])) and np.isfinite(float(row["val_mse"])) for row in results
    )
    split = payload.get("split", {})
    split_sets = [set(split.get(name, [])) for name in ("train", "val", "test")]
    split_audit_passed = bool(all(split_sets)) and not (
        (split_sets[0] & split_sets[1]) or (split_sets[0] & split_sets[2]) or (split_sets[1] & split_sets[2])
    )
    index_audit_passed = bool(controls.get("index_audit")) and all(
        row["passed"] for row in controls.get("index_audit", [])
    )
    normalization_passed = not bool(payload.get("normalization", {}).get("transductive", True))
    label_shuffle_passed = bool(controls.get("label_shuffle")) and all(
        row["passed"] for row in controls.get("label_shuffle", [])
    )
    random_twin_present = bool(controls.get("random_init_twin"))
    contamination_passed = all(
        row["contamination_status"] == "clean"
        for row in results
        if row.get("model_id") and row.get("ranked")
    ) and not payload.get("skipped_models")
    ranked_ids = {row["entrant_id"] for row in results if row.get("ranked")}
    incumbents_present = set(INCUMBENT_IDS).issubset(ranked_ids)
    no_grid_edge = all(not row["grid_edge_selected"] for row in results)
    calibration_checked = bool(ranking) and all(
        np.isfinite(row["test_mse_ci95_low"]) and np.isfinite(row["test_mse_ci95_high"])
        for block in ranking
        for row in block["rows"]
    )

    reasons: list[str] = []
    if not index_audit_passed:
        reasons.append("index audit did not pass at every horizon")
    if not normalization_passed:
        reasons.append(
            "normalisation is transductive (legacy full-series mode): the ranking is an "
            "ablation, not a claim-bearing run"
        )
    if not label_shuffle_passed:
        reasons.append("label-shuffle control did not collapse")
    if not random_twin_present:
        reasons.append("random-init architectural twin control is missing")
    if not contamination_passed:
        reasons.append("not every ranked external entrant is 'clean' on Sleep-EDF")
    if not incumbents_present:
        reasons.append(f"incumbent ladder incomplete; required {sorted(INCUMBENT_IDS)}")
    if not no_grid_edge:
        reasons.append("a selected hyperparameter sits on a grid edge; extend the grid and re-run")
    if payload.get("unavailable_models"):
        reasons.append(
            "one or more requested models were unavailable: "
            + ", ".join(sorted(row["model_id"] for row in payload["unavailable_models"]))
        )
    if payload.get("failed_entrants"):
        reasons.append(
            "one or more entrants produced no finite validation MSE on any configuration: "
            + ", ".join(sorted({row["entrant_id"] for row in payload["failed_entrants"]}))
        )
    if not bool(payload.get("git", {}).get("clean_worktree")):
        reasons.append("dirty worktree: committed artifacts would not match the code that produced them")

    gate = evaluate_gate(
        branch=GATE_BRANCH,
        dataset=DATASET_ID,
        split_audit_passed=split_audit_passed,
        baseline_table_present=bool(ranking),
        finite_metrics=finite_metrics,
        calibration_checked=calibration_checked,
        claim_scope=CLAIM_SCOPE,
        extra_failure_reasons=reasons,
    )
    gate.update(
        {
            "index_audit_passed": index_audit_passed,
            "normalization_causal": normalization_passed,
            "label_shuffle_collapsed": label_shuffle_passed,
            "random_twin_control_present": random_twin_present,
            "contamination_audit_passed": contamination_passed,
            "incumbent_ladder_complete": incumbents_present,
            "tuning_budget_symmetric": no_grid_edge,
            "controls_pass": bool(
                index_audit_passed and label_shuffle_passed and random_twin_present and normalization_passed
            ),
            "human_sign_off": "claim scope approved by Krish Garg 2026-08-19",
            "stop_reason": (
                "EEG-FM benchmark dossier written; read it before writing any ranking sentence. "
                "This scope permits a leakage-controlled Sleep-EDF forecasting ranking only -- "
                "never SOTA, clinical, or generic-superiority language."
            ),
        }
    )
    return gate


def write_markdown_report(path: Path, payload: Mapping[str, Any]) -> Path:
    """Human-readable report generated from the JSON; never hand-edited."""

    gate = payload.get("gate", {})
    lines: list[str] = [
        "# External Pretrained EEG Models vs Kahlus Incumbents — Sleep-EDF Forecasting",
        "",
        f"- Schema: `{payload.get('schema')}`",
        f"- Created: {payload.get('created_utc')}",
        f"- Commit: `{payload.get('git', {}).get('commit')}` "
        f"(clean worktree: {payload.get('git', {}).get('clean_worktree')})",
        f"- Dataset: `{payload.get('dataset_id')}`",
        f"- Normalisation: `{payload.get('normalization', {}).get('mode')}` "
        f"(transductive: {payload.get('normalization', {}).get('transductive')})",
        f"- Claim scope: `{gate.get('claim_scope')}` — "
        f"scientific_claim_allowed = **{gate.get('scientific_claim_allowed')}**",
        "",
        "> This table ranks entrants on one corpus, one task, one lane. It says nothing about",
        "> these models' published capabilities, nothing about SOTA, and nothing clinical.",
        "> The external models are evaluated **off-task**: they are masked-infilling encoders,",
        "> not forecasters, so a poor number is evidence about transfer, not about the model.",
        "",
    ]
    if gate.get("failure_reasons"):
        lines += ["## Gate failures", ""] + [f"- {reason}" for reason in gate["failure_reasons"]] + [""]

    lines += ["## Ranking by horizon", ""]
    for block in payload.get("ranking", []):
        lines += [
            f"### h = {block['horizon']}",
            "",
            "| rank | entrant | condition | val MSE | test MSE | 95% CI | contamination | selection exposure | grid edge |",
            "|---:|---|---|---:|---:|---|---|---|---|",
        ]
        for row in block["rows"]:
            lines.append(
                f"| {row['rank']} | `{row['entrant_id']}` | {row['condition'] or '-'} | "
                f"{row['val_mse']:.6f} | {row['test_mse_mean']:.6f} | "
                f"[{row['test_mse_ci95_low']:.6f}, {row['test_mse_ci95_high']:.6f}] | "
                f"{row['contamination_status']} | {row['model_selection_exposure']} | "
                f"{row['grid_edge_selected']} |"
            )
        lines.append("")

    lines += ["## Controls", ""]
    for name, rows in (payload.get("controls") or {}).items():
        lines.append(f"### {name}")
        lines.append("")
        for row in rows:
            passed = row.get("passed", row.get("pretraining_helped"))
            lines.append(f"- `{json.dumps(row, sort_keys=True, default=str)}` → {passed}")
        lines.append("")

    budget = payload.get("tuning_budget", {})
    lines += [
        "## Tuning budget",
        "",
        f"- configurations per gradient-trained entrant: {budget.get('n_configs_per_entrant')}",
        f"- seeds: {budget.get('seeds')}",
        f"- epoch budget: {budget.get('max_train_epochs')}",
        f"- attempts logged: {payload.get('attempts', {}).get('n_attempts')}",
        f"- test-set evaluations: {payload.get('attempts', {}).get('n_test_evaluations')}",
        "",
        "## Checkpoints",
        "",
    ]
    for row in payload.get("checkpoints", []):
        lines.append(f"- `{row['model_id']}` — `{row['filename']}` sha256 `{row['sha256']}` ({row['license_note'][:80]}…)")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_artifacts(out_dir: Path, payload: Mapping[str, Any], ledger: AttemptLedger) -> dict[str, Path]:
    """Write the JSON dossier, the gate, the ledger and the Markdown report."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "benchmark_json": write_json(out / "eeg_fm_benchmark.json", payload),
        "gate_json": write_json(out / "eeg_fm_benchmark_gate.json", payload["gate"]),
        "contamination_json": write_json(out / "contamination.json", payload["contamination_table"]),
        "checkpoints_json": write_json(out / "checkpoints.json", payload["checkpoints"]),
        "controls_json": write_json(out / "controls.json", payload["controls"]),
        "environment_json": write_json(out / "environment.json", payload["environment"]),
        "git_state_json": write_json(out / "git_state.json", payload["git"]),
        "attempts_log": ledger.write(out / "attempts.log"),
        "report_md": write_markdown_report(out / "EEG_FM_BENCHMARK_REPORT.md", payload),
    }
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="eeg_fm_benchmark",
        description=(
            "Leakage-controlled Sleep-EDF forecasting benchmark of external pretrained EEG "
            "models against the Kahlus incumbent ladder. Writes a dossier; makes no claim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True, help="Sleep-EDF sleep-cassette directory")
    parser.add_argument("--weights-dir", type=Path, required=True, help="directory of hand-staged checkpoints")
    parser.add_argument("--out-dir", type=Path, required=True, help="artifact output directory")
    parser.add_argument(
        "--models",
        default="all",
        help=f"comma-separated subset of {','.join(FM_MODEL_IDS)}, or 'all'",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(h) for h in HORIZONS),
        help="comma-separated horizons in samples at 100 Hz",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(CONDITIONS),
        help="comma-separated admissible conditions (frozen, full_ft)",
    )
    parser.add_argument(
        "--include-random-twin",
        dest="include_random_twin",
        action="store_true",
        default=True,
        help="run each FM's randomly-initialised architectural twin (plan section 4)",
    )
    parser.add_argument(
        "--no-include-random-twin",
        dest="include_random_twin",
        action="store_false",
        help="skip the random-init twin control (blocks the gate)",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated seeds per selected configuration",
    )
    parser.add_argument("--subjects", type=int, default=N_SUBJECTS, help="number of subjects to load")
    parser.add_argument(
        "--max-windows-per-recording", type=int, default=MAX_WINDOWS_PER_RECORDING, help="window cap per recording"
    )
    parser.add_argument("--stride", type=int, default=STRIDE_SAMPLES, help="window stride in samples")
    parser.add_argument("--n-configs", type=int, default=TUNING_GRID_SIZE, help="configurations per entrant")
    parser.add_argument("--max-train-epochs", type=int, default=50, help="shared epoch budget")
    parser.add_argument("--batch-size", type=int, default=64, help="shared batch size")
    parser.add_argument("--patience", type=int, default=5, help="shared early-stopping patience")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="subject-level bootstrap draws")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"), help="torch device")
    parser.add_argument(
        "--normalization-prefix",
        type=int,
        default=NORMALIZATION_PREFIX_SAMPLES,
        help="samples used for the causal normalisation statistics",
    )
    parser.add_argument(
        "--legacy-normalization",
        action="store_true",
        help=(
            "reproduce F8's transductive full-series z-scoring for the ablation; OFF by "
            "default and blocks the gate when on"
        ),
    )
    parser.add_argument(
        "--smoke",
        nargs="?",
        const="on",
        default="off",
        choices=("on", "off"),
        help=(
            "tiny CPU-friendly run: 6 subjects, 1 horizon, 2 epochs, 2 configs, 1 seed. "
            "Bare --smoke means on; the explicit on/off form exists so ops/jobs.yaml can "
            "pass it as a typed parameter (the job runner has no boolean type)"
        ),
    )
    return parser


@dataclass(frozen=True)
class RunConfig:
    """The fully resolved run configuration, echoed verbatim into the artifact."""

    data_root: Path
    weights_dir: Path
    out_dir: Path
    models: tuple[str, ...]
    horizons: tuple[int, ...]
    conditions: tuple[str, ...]
    include_random_twin: bool
    seeds: tuple[int, ...]
    subjects: int
    max_windows_per_recording: int
    stride: int
    n_configs: int
    max_train_epochs: int
    batch_size: int
    patience: int
    n_boot: int
    device: str
    normalization_prefix: int
    legacy_normalization: bool
    smoke: bool

    def as_log(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "weights_dir": str(self.weights_dir),
            "out_dir": str(self.out_dir),
            "models": list(self.models),
            "horizons": list(self.horizons),
            "conditions": list(self.conditions),
            "include_random_twin": self.include_random_twin,
            "seeds": list(self.seeds),
            "subjects": self.subjects,
            "max_windows_per_recording": self.max_windows_per_recording,
            "stride": self.stride,
            "n_configs": self.n_configs,
            "max_train_epochs": self.max_train_epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "n_boot": self.n_boot,
            "device": self.device,
            "normalization_prefix": self.normalization_prefix,
            "legacy_normalization": self.legacy_normalization,
            "smoke": self.smoke,
        }


def _csv_ints(raw: str, name: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError(f"--{name} must list at least one value")
    return values


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """Turn parsed arguments into the resolved, smoke-adjusted run configuration."""

    models = FM_MODEL_IDS if str(args.models).strip() == "all" else tuple(
        token.strip() for token in str(args.models).split(",") if token.strip()
    )
    unknown = [model for model in models if model not in FM_MODEL_IDS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; known: {list(FM_MODEL_IDS)}")

    conditions = tuple(token.strip() for token in str(args.conditions).split(",") if token.strip())
    bad_conditions = [condition for condition in conditions if condition not in CONDITIONS]
    if bad_conditions:
        raise ValueError(
            f"unknown or inadmissible conditions {bad_conditions}; only {list(CONDITIONS)} are "
            "causal and admissible. Acausal masked infilling is 'infilling', never 'forecasting'."
        )

    horizons = _csv_ints(args.horizons, "horizons")
    seeds = _csv_ints(args.seeds, "seeds")
    subjects = int(args.subjects)
    max_windows = int(args.max_windows_per_recording)
    n_configs = int(args.n_configs)
    max_train_epochs = int(args.max_train_epochs)
    device = str(args.device)

    smoke = str(args.smoke) == "on"
    if smoke:
        subjects = min(subjects, 6)
        horizons = horizons[:1]
        max_windows = min(max_windows, 32)
        n_configs = min(n_configs, 2)
        max_train_epochs = min(max_train_epochs, 2)
        seeds = seeds[:1]
        device = "cpu" if device == "auto" else device

    if n_configs < 1:
        raise ValueError("--n-configs must be >= 1")
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("horizons must be >= 1")

    return RunConfig(
        data_root=Path(args.data_root),
        weights_dir=Path(args.weights_dir),
        out_dir=Path(args.out_dir),
        models=tuple(models),
        horizons=horizons,
        conditions=conditions,
        include_random_twin=bool(args.include_random_twin),
        seeds=seeds,
        subjects=subjects,
        max_windows_per_recording=max_windows,
        stride=int(args.stride),
        n_configs=n_configs,
        max_train_epochs=max_train_epochs,
        batch_size=int(args.batch_size),
        patience=int(args.patience),
        n_boot=int(args.n_boot),
        device=device,
        normalization_prefix=int(args.normalization_prefix),
        legacy_normalization=bool(args.legacy_normalization),
        smoke=smoke,
    )


def set_determinism(seed: int) -> None:
    """Seed every generator this driver can reach, before any entrant runs."""

    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        torch.use_deterministic_algorithms(False)
    except Exception:  # noqa: BLE001 - torch absence is reported in environment.json
        pass


def run_benchmark(config: RunConfig) -> dict[str, Any]:
    """Load, tune on validation, evaluate once on test, and write the dossier."""

    from neurotwin.benchmarks.fm_forecast.weights_manifest import manifest_rows

    set_determinism(SPLIT_SEED)
    plan = plan_normalization(
        skip=SKIP_SAMPLES,
        take=TAKE_SAMPLES,
        prefix=config.normalization_prefix,
        legacy=config.legacy_normalization,
    )
    print(f"normalisation: {plan.mode} (transductive={plan.is_transductive})", flush=True)

    corpus = load_corpus(
        config.data_root,
        plan,
        horizons=config.horizons,
        n_subjects=config.subjects,
        stride=config.stride,
        max_windows=config.max_windows_per_recording,
    )
    subjects = sorted(corpus)
    if len(subjects) < 6:
        raise ValueError(f"need at least 6 subjects for a subject-disjoint split, got {len(subjects)}")
    train_subjects, val_subjects, test_subjects = split_subjects(subjects, SPLIT_SEED)
    print(
        f"split: train {len(train_subjects)}  val {len(val_subjects)}  test {len(test_subjects)}",
        flush=True,
    )

    entrants, skipped = build_entrants(
        models=config.models,
        conditions=config.conditions,
        weights_dir=config.weights_dir,
        include_random_twin=config.include_random_twin,
        n_configs=config.n_configs,
        max_train_epochs=config.max_train_epochs,
        batch_size=config.batch_size,
        patience=config.patience,
        device=config.device,
    )

    ledger = AttemptLedger()
    results: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    failed: list[dict[str, Any]] = []
    controls: dict[str, list[dict[str, Any]]] = {"index_audit": [], "label_shuffle": []}
    blocked_entrants: set[str] = set()

    for horizon in config.horizons:
        bundle = build_task_bundle(
            corpus,
            horizon=horizon,
            train_subjects=train_subjects,
            val_subjects=val_subjects,
            test_subjects=test_subjects,
        )
        controls["index_audit"].append(index_audit(bundle))
        print(
            f"h={horizon}: train {bundle.n_train}  val {bundle.n_val}  test {bundle.n_test}",
            flush=True,
        )

        for entrant in entrants:
            if entrant.entrant_id in blocked_entrants:
                continue
            try:
                selection = select_configuration(
                    entrant,
                    bundle.selection_task,
                    horizon=horizon,
                    seeds=config.seeds,
                    train_steps=config.max_train_epochs,
                    ledger=ledger,
                )
                evaluation = evaluate_selected(
                    entrant,
                    bundle.task,
                    selection,
                    horizon=horizon,
                    seeds=config.seeds,
                    train_steps=config.max_train_epochs,
                    test_subject_bounds=bundle.test_subject_bounds,
                    ledger=ledger,
                )
            except AdapterUnavailable as exc:
                blocked_entrants.add(entrant.entrant_id)
                if entrant.model_id and not any(row["model_id"] == entrant.model_id for row in unavailable):
                    unavailable.append({"model_id": entrant.model_id, "reason": str(exc)})
                print(f"  SKIP {entrant.entrant_id}: {exc}", flush=True)
                continue
            except EntrantFailed as exc:
                failed.append(
                    {"entrant_id": entrant.entrant_id, "horizon": int(horizon), "reason": str(exc)}
                )
                print(f"  FAIL {entrant.entrant_id}: {exc}", flush=True)
                continue

            interval = bootstrap_subject_ci(
                evaluation["per_subject_test_mse"], n_boot=config.n_boot, seed=SPLIT_SEED + horizon
            )
            row = {
                **entrant.as_log(),
                "horizon": int(horizon),
                "claim_eligible_protocol": bundle.claim_eligible,
                "selected_config": selection["config"],
                "grid_edge_selected": selection["grid_edge_selected"],
                "val_mse": selection["val_mse"],
                "val_mse_by_config": selection["val_mse_by_config"],
                "test_mse_ci95_low": interval["ci95_low"],
                "test_mse_ci95_high": interval["ci95_high"],
                "bootstrap": interval,
                **evaluation,
            }
            results.append(row)
            print(
                f"  {entrant.entrant_id:<34} val {selection['val_mse']:.6f}  "
                f"test {evaluation['test_mse_mean']:.6f}",
                flush=True,
            )

            if entrant.entrant_id == "tuned_ridge_ar":
                controls["label_shuffle"].append(label_shuffle_control(entrant, bundle, selection))

    controls["random_init_twin"] = random_twin_controls(results)

    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "lane": "lane_a_single_channel_fpz_cz",
        "config": config.as_log(),
        "environment": _environment_payload(),
        "git": _git_state(),
        "protocol": {
            "protocol_id": FORECAST_PROTOCOL_V2_NONOVERLAP,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "channel": CHANNEL,
            "context_samples": CONTEXT_SAMPLES,
            "target_samples": TARGET_SAMPLES,
            "skip_samples": SKIP_SAMPLES,
            "take_samples": TAKE_SAMPLES,
            "horizons": list(config.horizons),
            "target_index_rule": "signal[i + context + h - 1]",
            "claim_eligible_note": (
                "the v2 contract reserves claim_eligible for a strictly positive gap, so h=1 "
                "(gap 0, still zero shared samples) is reported as protocol-ineligible"
            ),
        },
        "normalization": plan.as_log(),
        "split": {
            "seed": SPLIT_SEED,
            "train": list(train_subjects),
            "val": list(val_subjects),
            "test": list(test_subjects),
        },
        "tuning_budget": {
            "n_configs_per_entrant": config.n_configs,
            "seeds": list(config.seeds),
            "max_train_epochs": config.max_train_epochs,
            "batch_size": config.batch_size,
            "patience": config.patience,
            "learning_rate_axis": list(LEARNING_RATES),
            "ridge_alpha_grid": list(RIDGE_ALPHAS),
            "notes": [entrant.budget_note for entrant in entrants if entrant.budget_note],
        },
        "entrants": [entrant.as_log() for entrant in entrants],
        "results": results,
        "contamination_table": contamination_dump(),
        "checkpoints": manifest_rows(),
        "controls": controls,
        "skipped_models": skipped,
        "unavailable_models": unavailable,
        "failed_entrants": failed,
        "attempts": {
            "n_attempts": ledger.n_attempts,
            "n_test_evaluations": ledger.n_test_evaluations,
            "log_file": "attempts.log",
        },
    }
    payload["ranking"] = build_ranking(results, config.horizons)
    payload["gate"] = build_gate_dossier(payload)

    written = write_artifacts(config.out_dir, payload, ledger)
    for name, path in sorted(written.items()):
        print(f"wrote {name}: {path}", flush=True)
    print(
        f"scientific_claim_allowed = {payload['gate']['scientific_claim_allowed']}; "
        f"failure_reasons = {payload['gate']['failure_reasons']}",
        flush=True,
    )
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    run_benchmark(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

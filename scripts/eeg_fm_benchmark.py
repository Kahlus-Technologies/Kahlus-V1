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

Sharding and merging
--------------------
Each of those per-GPU invocations writes a dossier that is a *fragment*: its incumbent
ladder may be incomplete, its slate is missing the models the other shards carried, and
its gate says so. The claim, if there is one, is about the union, so ``--merge`` is a
first-class mode rather than a shell script over the JSONs. It recomputes ranking, paired
statistics, Holm families, power, the beat-best-baseline verdict, protocol completeness,
budget symmetry and the gate from the union of the shards' per-subject test errors, and it
hard-errors before writing anything if the shards do not describe *one* experiment.

Exactly one shard may carry the incumbent ladder (``--incumbents off`` on the others):
running the ladder twice would put the same ``(entrant, horizon)`` through two held-out
test evaluations, which the merge refuses as a repeated test look.

Run::

    PYTHONPATH=src python3 scripts/eeg_fm_benchmark.py \\
        --data-root /path/to/sleep-edfx/sleep-cassette \\
        --weights-dir /path/to/staged/weights \\
        --out-dir artifacts/eeg_fm_benchmark --smoke

    PYTHONPATH=src python3 scripts/eeg_fm_benchmark.py \\
        --merge artifacts/shard_a artifacts/shard_b \\
        --out-dir artifacts/eeg_fm_benchmark_merged
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

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

#: Audited exact channel names for the F8 Fpz-Cz derivation. Membership is checked by
#: exact string match and nothing else: silently falling back to "the first channel whose
#: name contains EEG" (what F8's loader does) would let a Pz-Oz recording be scored as an
#: Fpz-Cz one, which is a different experiment wearing the same label. Both spellings were
#: audited against the PhysioNet sleep-cassette EDF headers; anything else is a hard error.
CHANNEL_ALIASES: tuple[str, ...] = ("EEG Fpz-Cz", "Fpz-Cz")

#: The protocol's sampling rate. A recording at any other rate has different context
#: seconds per window, so it is refused rather than silently resampled.
EXPECTED_SAMPLING_RATE_HZ = 100.0
SAMPLING_RATE_TOLERANCE_HZ = 1e-6

SKIP_SAMPLES = 1_000_000
TAKE_SAMPLES = 300_000
NORMALIZATION_PREFIX_SAMPLES = 100_000
STRIDE_SAMPLES = 256
MAX_WINDOWS_PER_RECORDING = 400
N_SUBJECTS = 78
SPLIT_SEED = 0
HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Plan section 3.1: "10,000-draw bootstrap of the mean per-subject delta".
DEFAULT_N_BOOT = 10_000
#: Smoke runs may lower it; a smoke run can never pass the gate anyway (see
#: :func:`protocol_completeness`).
SMOKE_N_BOOT = 200
MIN_CLAIMABLE_N_BOOT = 10_000

#: F8's split, byte-identical (plan section 3.1). A run on any other split sizes is a
#: different experiment and may not carry the ranking claim.
F8_SPLIT_SIZES: tuple[int, int, int] = (39, 12, 27)
F8_TEST_SUBJECTS = F8_SPLIT_SIZES[2]

#: Plan section 3.4 rule 4: "at least 3 seeds per selected configuration".
MIN_SEEDS = 3

#: Two-sided significance level for the paired tests, before Holm correction.
ALPHA_LEVEL = 0.05

#: Minimum detectable effect for the ``powered`` gate term, in z-scored-MSE units: the
#: widest paired-delta 95% CI a claim-bearing run may have against the best external
#: entrant.
#:
#: **The design document does not name an MDE.** Plan section 5.1 requires only that the
#: paired CI exclude zero, which a two-subject study can satisfy by luck. 0.10 is declared
#: *here*, before the data, as roughly a tenth of the z-scored-MSE dynamic range the F8
#: crossover table spans (``docs/results/forecast_crossover_2026-07-27.md:48-58``).
#: It is an engineering placeholder standing in for a power calculation, and it is
#: surfaced in the artifact as ``mde_source``. Ratified at 0.10 by Krish Garg on
#: 2026-08-21, before any full-scale test evaluation had been merged or read.
MDE_ZSCORED_MSE = 0.10
MDE_SOURCE = "declared_in_driver_pre_data__ratified_by_krish_garg_2026-08-21"

#: Preregistered control margins (plan section 4). A control "collapses" only if the
#: corrupted run is *materially* worse, not merely not-better: an entrant whose shuffled
#: score ties its honest score has not demonstrated that it was using the labels.
LABEL_SHUFFLE_MIN_DEGRADATION = 0.20
TIME_SHIFT_MIN_DEGRADATION = 0.20
#: Windows to roll the training targets by for the time-shift control. Windows are
#: ``STRIDE_SAMPLES`` apart, so 7 windows is 17.92 s off the true lag at 100 Hz --
#: far outside any horizon in the grid.
TIME_SHIFT_WINDOWS = 7
#: The AR(2) harness sanity task (plan section 4, F8 T5): on a stationary AR(2) process
#: the minimum-MSE predictor is linear, so a tuned ridge must beat the mean predictor by
#: at least this fraction or the harness itself is broken.
AR2_SANITY_MIN_IMPROVEMENT = 0.20
#: The synthetic NULL (plan section 4): on a signal with no predictable structure nothing
#: may score. A ridge may fit noise slightly better than the mean in-sample, so the
#: threshold is a small out-of-sample tolerance, not zero.
SYNTHETIC_NULL_MAX_IMPROVEMENT = 0.05
#: phi = (0.9, -0.2) is chosen so the 20% margin is a real test rather than a coin flip.
#: For an AR(2) with unit innovation variance the one-step-optimal MSE is 1 and the
#: unconditional variance is ``(1 - phi2) / ((1 + phi2) * ((1 - phi2)^2 - phi1^2))``, which
#: is 2.38 here: a perfect linear predictor improves on the mean predictor by 58%, leaving
#: ample headroom above the threshold for finite-sample estimation error. The obvious
#: textbook choice phi = (0.6, -0.3) has an unconditional variance of only 1.40, i.e. a
#: *theoretical ceiling* of 28% improvement, so a 20% preregistered margin there would be
#: within noise of the best achievable score and would fail for healthy harnesses.
AR2_COEFFICIENTS: tuple[float, float] = (0.9, -0.2)

#: Plan section 4 controls, and which of them this driver implements. Anything in
#: ``PLAN_SECTION_4_CONTROLS`` that is not in ``IMPLEMENTED_CONTROLS`` is emitted as an
#: explicit failure reason: an absent control must never be an invisible one.
PLAN_SECTION_4_CONTROLS: tuple[str, ...] = (
    "random_init_twin",
    "label_shuffle",
    "time_shift",
    "subject_identity_probe",
    "synthetic_null",
    "ar2_sanity",
    "index_audit",
    "cluster_permutation_null",
)
IMPLEMENTED_CONTROLS: tuple[str, ...] = (
    "random_init_twin",
    "label_shuffle",
    "time_shift",
    "synthetic_null",
    "ar2_sanity",
    "index_audit",
)
UNIMPLEMENTED_CONTROL_REASONS: dict[str, str] = {
    "subject_identity_probe": (
        "plan section 4 requires a probe of whether each entrant's representation predicts "
        "test-subject identity above chance + 0.20 (Identity Trap, arXiv:2606.06647). It is "
        "not implemented: the adapters expose a (task, seed, steps) -> predictions surface "
        "with no way to read out per-window embeddings, so the probe cannot be fitted "
        "without an embedding-export API on every adapter. Fails closed."
    ),
    "cluster_permutation_null": (
        "the AGENTS.md control list names a cluster-permutation null. It is not implemented: "
        "this task emits one scalar target per window rather than a time-cluster, so the "
        "cluster-forming step has no definition here and inventing one would be a new "
        "statistical method, not a control. Fails closed."
    ),
}

#: F8's 14-point ridge alpha grid, which runs high because long horizons want heavy
#: regularisation. Its size fixes N for the symmetric budget of plan section 3.4.
RIDGE_ALPHAS: tuple[float, ...] = (
    0.1, 0.5, 1.0, 5.0, 20.0, 100.0, 500.0, 2000.0, 10000.0, 50000.0, 2e5, 1e6, 5e6, 2e7,
)
TUNING_GRID_SIZE = len(RIDGE_ALPHAS)

#: Shared learning-rate axis for every gradient-trained entrant, so the budget symmetry
#: is visible in the grid itself rather than asserted in prose. Five decade-spaced points:
#: the frozen probes want the middle of this range and full fine-tuning of a pretrained
#: encoder wants the bottom of it, so a single axis has to reach both without leaving
#: either optimum on an endpoint.
LEARNING_RATES: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
HEAD_HIDDEN_DIMS: tuple[int, ...] = (0, 64, 128)
GRU_HIDDEN_DIMS: tuple[int, ...] = (32, 64, 128)

#: Gradient-boosting rung of the AGENTS.md baseline ladder, on the same 5 x 3 shape as
#: every other tuned entrant so :func:`_budget_grid` gives it the identical 14-point budget.
GBM_LEARNING_RATES: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 0.5)
GBM_MAX_LEAF_NODES: tuple[int, ...] = (15, 31, 63)
GBM_MAX_ITER = 48

#: Input-scale axis for the two upstreams that document a ``/100`` microvolt convention
#: (LaBraM ``engine_for_finetuning.py:64-65``; CBraMod inherits the same TUEG-pipeline
#: convention). The harness feeds unitless z-scored windows, so neither the identity nor
#: the 1/100 factor is obviously right -- which makes it a hyperparameter, swept on
#: validation like any other. It replaces the head-width axis for those two models so the
#: 14-configuration budget is unchanged; the substitution is recorded per entrant in
#: ``axes`` and called out in the report. Three points, so "the winner sat on an edge" is
#: a question with an answer: 1.0 (identity, the harness's own units), 0.01 (the documented
#: upstream factor) and 0.1 between them.
INPUT_SCALES: tuple[float, ...] = (0.01, 0.1, 1.0)
INPUT_SCALE_MODEL_IDS: tuple[str, ...] = ("cbramod", "labram_base")

FM_MODEL_IDS: tuple[str, ...] = ("eegpt", "cbramod", "labram_base", "biot_prest16")
CONDITIONS: tuple[str, ...] = ("frozen", "full_ft")

#: Per-model context length in seconds, from each adapter's ``N_SAMPLES_OUT / SR_OUT``.
#: Plan section 3.2: every FM gets the largest patch-aligned context *not exceeding* F8's
#: 1.27 s, so no FM ever sees more history than the incumbent.
CONTEXT_SECONDS_BY_MODEL: dict[str, float] = {
    "eegpt": 320 / 256.0,
    "cbramod": 200 / 200.0,
    "labram_base": 200 / 200.0,
    "biot_prest16": 200 / 200.0,
}
INCUMBENT_CONTEXT_SECONDS = CONTEXT_SAMPLES / SAMPLING_RATE_HZ

INCUMBENT_IDS: tuple[str, ...] = ("persistence", "tuned_ridge_ar", "gbm", "kahlus_gru")

#: The design document's incumbent ladder (section 3.5), rung by rung, with how this
#: driver realises each one. A rung that is neither ranked nor explicitly realised by
#: another entrant is a gate failure, so the ladder cannot silently lose a rung -- the
#: exact pathology plan section 8 flags for ``gbm``.
INCUMBENT_LADDER: tuple[dict[str, str], ...] = (
    {
        "rung": "persistence",
        "realised_by": "persistence",
        "note": "the mandatory naive floor (Beck et al. 2025)",
    },
    {
        "rung": "linear_ridge",
        "realised_by": "tuned_ridge_ar",
        "note": (
            "Lane A is single-channel, so the harness window flattens to a 127-lag vector "
            "and a plain ridge on that vector IS the autoregressive ridge -- the same "
            "design matrix, the same closed form, the same alpha grid. The rung is "
            "realised by tuned_ridge_ar rather than duplicated as a second identical row, "
            "which would inflate the entrant count without adding an estimator. In a "
            "multi-channel Lane B the two rungs separate and must both be run."
        ),
    },
    {
        "rung": "autoregressive_ridge",
        "realised_by": "tuned_ridge_ar",
        "note": "F8's 14-point per-horizon alpha sweep to 2e7, run to convergence",
    },
    {
        "rung": "gbm",
        "realised_by": "gbm",
        "note": (
            "required by the AGENTS.md ladder and by plan section 8's 'gbm present in the "
            "produced table' blocker; run here on the flattened context with its own "
            "14-point (learning rate x leaf count) grid"
        ),
    },
    {
        "rung": "kahlus_gru",
        "realised_by": "kahlus_gru",
        "note": "the F8 incumbent, re-tuned under the symmetric budget",
    },
)

#: Plan section 3.4 rule 2 asks for an unfreezing-depth axis in the full-fine-tune
#: condition. Recorded as a decision rather than swept -- see ``UNFREEZE_DEPTH_DECISION``.
UNFREEZE_DEPTH_DECISION: dict[str, Any] = {
    "axis": "unfreeze_depth",
    "swept": False,
    "realised_value": "full_encoder",
    "reason": (
        "'last N blocks' has no model-agnostic definition across these four upstreams: the "
        "adapters wrap heterogeneous encoders (CBraMod criss-cross transformer, EEGPT "
        "target-encoder, LaBraM ViT, BIOT linear-attention) whose block containers are named "
        "and nested differently, so partial unfreezing needs per-adapter surgery rather than "
        "a shared parameter-group rule. Adding it would also double the grid to 28 "
        "configurations for the fine-tuned rows only, breaking the equal-configuration-count "
        "rule of plan section 3.4 that the whole benchmark rests on. The full-fine-tune rows "
        "therefore unfreeze the whole encoder, which is the *most* favourable setting for the "
        "external models, so the omission cannot flatter the incumbents."
    ),
}

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


class DataStrictnessError(RuntimeError):
    """A recording does not supply exactly the channel and rate the protocol declares.

    Deliberately fatal rather than per-recording skippable. F8's loader falls back to "the
    first channel whose name contains EEG", which on a Sleep-EDF cassette PSG silently
    yields ``EEG Pz-Oz`` when ``EEG Fpz-Cz`` is missing -- a different derivation scored
    under the Fpz-Cz label. A corpus that cannot supply the declared channel at the
    declared rate is the wrong corpus, and the run must stop, not adapt.
    """


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
    resolved_channel: str = CHANNEL
    sampling_rate_hz: float = SAMPLING_RATE_HZ

    def as_log(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "record_id": self.record_id,
            "resolved_channel": self.resolved_channel,
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "n_windows": int(self.starts.size),
            "stats_mean": float(self.stats_mean),
            "stats_std": float(self.stats_std),
        }


def resolve_channel(channel_names: Sequence[str], *, aliases: Sequence[str] = CHANNEL_ALIASES) -> str:
    """Return the one audited alias present in ``channel_names``, or raise.

    Exact string match only, and exactly one match is required: no substring search, no
    "contains EEG" fallback, no case folding. A recording that offers two audited aliases
    is ambiguous and is refused too, because picking one would be an undocumented choice.
    """

    present = [name for name in aliases if name in tuple(channel_names)]
    if len(present) == 1:
        return present[0]
    if not present:
        raise DataStrictnessError(
            f"none of the audited Fpz-Cz channel names {list(aliases)} is present; the "
            f"recording offers {sorted(channel_names)}. Refusing to fall back to another "
            "channel: scoring Pz-Oz under an Fpz-Cz label is a different experiment."
        )
    raise DataStrictnessError(
        f"the recording offers more than one audited Fpz-Cz alias ({present}), so the "
        "channel to score is ambiguous. Resolve it in the corpus, not in the driver."
    )


def load_recording(
    path: Path,
    plan: NormalizationPlan,
    *,
    channel: str = CHANNEL,
    take: int = TAKE_SAMPLES,
) -> tuple[np.ndarray, float, float, dict[str, Any]]:
    """Load the audited Fpz-Cz channel and normalise it with ``plan``'s statistics.

    Returns the usable segment (absolute indices ``[plan.usable_start, plan.usable_stop)``)
    as float32, the mean and standard deviation actually applied, and a provenance record
    naming the resolved channel and the verified sampling rate.

    Raises:
        DataStrictnessError: if no audited channel alias is present, or the recording is
            not at :data:`EXPECTED_SAMPLING_RATE_HZ`.
    """

    import mne

    raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
    aliases = (channel,) if channel not in CHANNEL_ALIASES else CHANNEL_ALIASES
    try:
        resolved = resolve_channel(raw.ch_names, aliases=aliases)
    except DataStrictnessError as exc:
        raise DataStrictnessError(f"{path.name}: {exc}") from exc
    rate = float(raw.info["sfreq"])
    if abs(rate - EXPECTED_SAMPLING_RATE_HZ) > SAMPLING_RATE_TOLERANCE_HZ:
        raise DataStrictnessError(
            f"{path.name}: sampling rate is {rate} Hz, but the protocol declares "
            f"{EXPECTED_SAMPLING_RATE_HZ} Hz. Context length, horizons and the target "
            "index rule are all defined in samples, so a different rate silently changes "
            "every horizon in seconds. Refusing to resample."
        )
    offered = sorted(raw.ch_names)
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
    provenance = {
        "record_id": path.stem,
        "resolved_channel": resolved,
        "sampling_rate_hz": rate,
        "channel_names_offered": offered,
        "n_samples_usable": int(normalised.size),
    }
    return normalised, mean, std, provenance


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
            signal, mean, std, provenance = load_recording(
                record_path, plan, channel=channel, take=take
            )
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
                    resolved_channel=str(provenance["resolved_channel"]),
                    sampling_rate_hz=float(provenance["sampling_rate_hz"]),
                )
            )
    return corpus


def recording_provenance(corpus: Mapping[str, list[RecordingWindows]]) -> list[dict[str, Any]]:
    """Per-recording resolved channel and verified sampling rate, for the artifact."""

    return [record.as_log() for subject in sorted(corpus) for record in corpus[subject]]


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
    """One competitor: an id, a contamination-joinable model id, and a tuning grid.

    ``axes`` declares the sweep axes the grid was built from, each in ascending order.
    Grid-edge detection is *per axis* (:func:`axis_edge_report`) rather than
    "is the selected config first or last in the flattened grid", which is what plan
    section 3.4 rule 2 actually asks for: an lr at the top of its axis is an edge even when
    it sits in the middle of the flattened product.

    ``gradient_trained`` marks the entrants that share the ``fit_head`` budget, and is what
    the budget-symmetry check is computed over; ``fits_targets`` is False for
    parameter-free entrants, whose fit cannot be corrupted by a label-shuffle because there
    is nothing fitted to corrupt.
    """

    entrant_id: str
    family: str
    model_id: str | None
    condition: str | None
    pretrained: bool
    grid: tuple[dict[str, Any], ...]
    factory: RunnerFactory
    budget_note: str = ""
    ranked: bool = True
    gradient_trained: bool = False
    fits_targets: bool = True
    axes: Mapping[str, tuple[Any, ...]] = field(default_factory=dict)

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
            "gradient_trained": bool(self.gradient_trained),
            "fits_targets": bool(self.fits_targets),
            "configs_tried": self.configs_tried,
            "axes": {name: list(values) for name, values in dict(self.axes).items()},
            "budget_note": self.budget_note,
            "context_seconds": (
                CONTEXT_SECONDS_BY_MODEL.get(model_id, INCUMBENT_CONTEXT_SECONDS)
                if model_id
                else INCUMBENT_CONTEXT_SECONDS
            ),
            "contamination_status": (
                contamination_status(model_id, DATASET_ID) if model_id else "not_applicable_no_pretraining"
            ),
            "model_selection_exposure": bool(model_id and has_model_selection_exposure(model_id, DATASET_ID)),
        }


AXIS_TOO_SMALL_NOTE = "axis too small for edge detection (needs >= 3 points)"


def axis_edge_report(axes: Mapping[str, Sequence[Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    """Per-axis grid-edge report for one selected configuration.

    An axis with fewer than three points cannot distinguish "the optimum is interior" from
    "the optimum is at the edge", so it is reported as *undetectable* rather than silently
    passed -- a two-point axis that always reports "no edge" would let a truncated grid
    look converged.
    """

    per_axis: dict[str, Any] = {}
    for name, raw_values in dict(axes).items():
        if name not in config:
            continue
        values = tuple(raw_values)
        selected = config[name]
        if len(values) < 3:
            per_axis[name] = {
                "n_points": len(values),
                "selected": selected,
                "at_edge": None,
                "note": AXIS_TOO_SMALL_NOTE,
            }
            continue
        per_axis[name] = {
            "n_points": len(values),
            "selected": selected,
            "at_edge": bool(selected == values[0] or selected == values[-1]),
            "note": "",
        }
    undetectable = sorted(name for name, row in per_axis.items() if row["at_edge"] is None)
    return {
        "per_axis": per_axis,
        "any_axis_at_edge": any(row["at_edge"] for row in per_axis.values() if row["at_edge"] is not None),
        "axes_without_edge_detection": undetectable,
    }


def _grid_from_axes(axes: Mapping[str, Sequence[Any]], **extras: Any) -> tuple[dict[str, Any], ...]:
    """Cartesian product of ``axes`` (declaration order), with ``extras`` on every point."""

    names = list(axes)
    return tuple(
        {**dict(zip(names, values)), **extras}
        for values in product(*(tuple(axes[name]) for name in names))
    )


#: Stated here in writing, before any run, because "which 14 of the 15 cells" is exactly
#: the kind of choice that must not be made per entrant after seeing results.
GRID_SUBSET_RULE = (
    "5 (learning rate) x 3 (second axis) = 15 candidate cells, minus the single corner at "
    "(largest learning rate, largest second-axis value), giving exactly 14"
)


def _budget_grid(axes: Mapping[str, Sequence[Any]], **extras: Any) -> tuple[dict[str, Any], ...]:
    """One entrant's fixed 14-configuration budget, from a 5 x 3 axis product.

    Every tuned entrant sweeps a 5-point learning-rate axis against a 3-point second axis.
    Three points is the *minimum* at which "the selected value sits on a grid edge" is a
    question with an answer -- on a two-point axis every selection is an endpoint, so plan
    section 3.4's no-edge rule can never be satisfied and can never be checked. But
    5 x 3 = 15 is one cell more than the 14-configuration budget that section 3.4 fixes
    from F8's ridge alpha grid, so exactly one cell has to go, by :data:`GRID_SUBSET_RULE`:
    the corner at the largest learning rate *and* the largest second-axis value. That cell
    is the biggest step into the widest head, i.e. the one most likely to diverge and least
    likely to win a validation selection; and because it is a corner, every value of every
    axis still appears in other cells, so per-axis edge detection is unaffected by the drop.
    """

    grid = _grid_from_axes(axes, **extras)
    corner = {name: max(values) for name, values in axes.items()}
    kept = tuple(point for point in grid if any(point[name] != value for name, value in corner.items()))
    if len(kept) != TUNING_GRID_SIZE:
        raise ValueError(
            f"axes {({name: list(values) for name, values in axes.items()})} yield "
            f"{len(kept)} configurations after {GRID_SUBSET_RULE}, but the budget is "
            f"{TUNING_GRID_SIZE}. The budget is symmetric by construction, so an axis "
            "shape that does not land on it is a design error, not something to pad."
        )
    return kept


def _flat_targets(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y, dtype=np.float32)
    return values.reshape(values.shape[0], -1)


# ---------------------------------------------------------------------------
# Training-log capture
#
# Adapters expose only ``(task, seed, steps) -> predictions``, so without a side channel
# the realised epoch count, gradient-step count and checkpoint load report of every FM
# attempt would be unobservable from here -- and plan section 3.4 requires budget symmetry
# to be *checkable from the artifact*. ``conditions.capture_condition_logs`` publishes the
# per-run log of every FM attempt; the local GRU entrant publishes the same shape itself.
# ---------------------------------------------------------------------------

_LOCAL_TRAINING_LOGS: list[list[dict[str, Any]]] = []


def _publish_local_log(log: dict[str, Any]) -> None:
    """Publish a training log from an entrant this driver trains itself (the GRU)."""

    if _LOCAL_TRAINING_LOGS:
        _LOCAL_TRAINING_LOGS[-1].append(log)


@contextmanager
def capture_training_logs() -> Iterator[list[dict[str, Any]]]:
    """Collect every training log produced inside the block, FM adapters included."""

    collected: list[dict[str, Any]] = []
    _LOCAL_TRAINING_LOGS.append(collected)
    sink_context = None
    try:
        from neurotwin.benchmarks.fm_forecast import conditions as _conditions

        sink_context = _conditions.capture_condition_logs(collected.append)
        sink_context.__enter__()
    except Exception:  # noqa: BLE001 - torch/vendor absence must not disable the driver
        sink_context = None
    try:
        yield collected
    finally:
        if sink_context is not None:
            sink_context.__exit__(None, None, None)
        _LOCAL_TRAINING_LOGS.pop()


def summarise_training_logs(logs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Reduce captured training logs to the budget-symmetry fields the ledger records."""

    if not logs:
        return {"captured": False}
    fits = [dict(log.get("fit") or {}) for log in logs]
    return {
        "captured": True,
        "n_runs": len(logs),
        "epoch_budget": [int(dict(log.get("config") or {}).get("epochs", 0)) for log in logs],
        "epochs_ran": [int(fit.get("epochs_ran", 0)) for fit in fits],
        "gradient_steps": [int(fit.get("gradient_steps", 0)) for fit in fits],
        "best_epoch": [int(fit.get("best_epoch", 0)) for fit in fits],
        "stopped_early": [bool(fit.get("stopped_early", False)) for fit in fits],
        "trainable_parameters": [int(log.get("trainable_parameters", 0)) for log in logs],
    }


def checkpoint_reports_from_logs(logs: Sequence[Mapping[str, Any]]) -> list[Any]:
    """Every non-null encoder checkpoint load report the captured logs carry."""

    return [log["encoder_checkpoint_report"] for log in logs if log.get("encoder_checkpoint_report")]


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
        fit = fit_head(
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
        # Publish the same log shape the FM adapters publish through
        # ``conditions.capture_condition_logs``, so budget symmetry is computed from one
        # vocabulary across every gradient-trained entrant.
        _publish_local_log(
            {
                "condition": "incumbent_gru",
                "config": {"epochs": epochs, "lr": lr, "batch_size": batch_size, "patience": patience},
                "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "encoder_checkpoint_report": None,
                "fit": fit.as_log(),
            }
        )
        flat = predict(model, np.asarray(task.x_test, dtype=np.float32), device=device)
        return flat.reshape(task.y_test.shape).astype(np.float32)

    return runner


def _gbm_factory(config: Mapping[str, Any]) -> EntrantRunner:
    """The AGENTS.md ladder's gradient-boosting rung, on the flattened context.

    Registered in ``baseline_suite.EXECUTABLE_BASELINE_RUNNERS`` as ``gbm``, but that
    runner reshapes its prediction to ``(N, T, F)`` -- the sequence-to-sequence shape --
    which does not match this task's single-sample ``(N, 1, 1)`` target. It is therefore
    re-expressed here against the forecasting shape rather than adapted at a distance;
    the estimator, iteration count and leaf-size floor are the registered runner's.
    """

    from sklearn.ensemble import HistGradientBoostingRegressor

    learning_rate = float(config["gbm_learning_rate"])
    max_leaf_nodes = int(config["gbm_max_leaf_nodes"])

    def runner(task: SupervisedWindowTask, seed: int, train_steps: int) -> np.ndarray:
        model = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=GBM_MAX_ITER,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=10,
            early_stopping=False,
            random_state=int(seed),
        )
        x_train = np.asarray(task.x_train, dtype=np.float64).reshape(task.x_train.shape[0], -1)
        x_test = np.asarray(task.x_test, dtype=np.float64).reshape(task.x_test.shape[0], -1)
        model.fit(x_train, _flat_targets(task.y_train).ravel().astype(np.float64))
        return model.predict(x_test).reshape(task.y_test.shape).astype(np.float32)

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
        kwargs: dict[str, Any] = {}
        if model_id in INPUT_SCALE_MODEL_IDS:
            # Only the two upstreams that document a /100 microvolt convention expose
            # this axis; passing it to the others would be a silent no-op at best.
            kwargs["input_scale"] = float(config.get("input_scale", 1.0))
        try:
            return adapter.make_runner(
                Path(weights_dir),
                Condition(condition),
                config=condition_config,
                pretrained=bool(pretrained),
                **kwargs,
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


def gradient_axes(model_id: str | None) -> dict[str, tuple[Any, ...]]:
    """The two-axis sweep for one gradient-trained entrant: 5 x 3, minus one corner.

    Every gradient entrant sweeps the shared learning-rate axis plus exactly one second
    axis, so the configuration count is identical across entrants by construction rather
    than by truncation (see :func:`_budget_grid` for the 15-to-14 rule). The second axis is
    the head width, except for the two models whose upstreams document a ``/100`` input
    convention, where the input scale is the more consequential unknown and takes the slot
    (see :data:`INPUT_SCALES`).
    """

    if model_id is None:
        return {"hidden": GRU_HIDDEN_DIMS, "lr": LEARNING_RATES}
    if model_id in INPUT_SCALE_MODEL_IDS:
        return {"input_scale": INPUT_SCALES, "lr": LEARNING_RATES}
    return {"head_hidden": HEAD_HIDDEN_DIMS, "lr": LEARNING_RATES}


def _gradient_grid(
    n_configs: int,
    *,
    epochs: int,
    batch_size: int,
    patience: int,
    device: str,
    axes: Mapping[str, Sequence[Any]],
    extras: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    grid = _budget_grid(
        axes,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        device=device,
        **dict(extras or {}),
    )
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
    truncate_closed_form: int | None = None,
    include_incumbents: bool = True,
) -> tuple[list[Entrant], list[dict[str, str]]]:
    """Assemble the entrant slate, skipping models whose contamination gate refuses them.

    ``n_configs`` truncates the *gradient* grids only. The ridge alpha grid and the GBM
    grid are closed-form or near-free, so truncating them would weaken a baseline for no
    compute saving -- exactly the McGreivy & Hakim pathology plan section 3.4 exists to
    prevent -- and ``--n-configs`` therefore cannot reach them. ``truncate_closed_form``
    is set only by the smoke path, which can never pass the gate anyway, and its use is
    reported as ``closed_form_grid_truncated`` so it cannot go unnoticed.

    ``include_incumbents`` exists for sharding, and only for sharding (``--incumbents
    off``). The ladder is cheap next to a fine-tuned foundation model, so running it on
    every shard would look like a harmless duplication -- but each duplicate is a second
    held-out test evaluation of the same ``(entrant, horizon)`` family, which plan section
    5.3 forbids and which :func:`validate_shards` refuses to merge. Exactly one shard
    therefore carries the ladder, and the shards that do not simply report the missing
    rows: an incumbent-free slate fails its own ladder and protocol-completeness checks,
    and the merged union restores them.
    """

    def _closed_form(grid: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
        return grid if truncate_closed_form is None else grid[: int(truncate_closed_form)]

    incumbents: list[Entrant] = [
        Entrant(
            entrant_id="persistence",
            family=FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=({"kind": "parameter_free"},),
            factory=_persistence_factory,
            fits_targets=False,
            axes={},
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
            grid=_closed_form(tuple({"alpha": alpha} for alpha in RIDGE_ALPHAS)),
            factory=_ridge_factory,
            axes={"alpha": RIDGE_ALPHAS},
            budget_note=(
                "F8's full 14-point alpha grid to 2e7, never truncated by --n-configs; "
                "closed form, zero gradient steps. Realises both the linear_ridge and the "
                "autoregressive_ridge rungs of the design-document ladder, which coincide "
                "on a single-channel flattened context (see incumbent_ladder)"
            ),
        ),
        Entrant(
            entrant_id="gbm",
            family=FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=_closed_form(
                _budget_grid(
                    {"gbm_max_leaf_nodes": GBM_MAX_LEAF_NODES, "gbm_learning_rate": GBM_LEARNING_RATES}
                )
            ),
            factory=_gbm_factory,
            axes={
                "gbm_max_leaf_nodes": GBM_MAX_LEAF_NODES,
                "gbm_learning_rate": GBM_LEARNING_RATES,
            },
            budget_note=(
                "AGENTS.md ladder rung, 14-point (leaf count x learning rate) grid, never "
                "truncated by --n-configs; boosted trees, so its budget is iterations not "
                "epochs and it is excluded from the gradient-entrant symmetry check"
            ),
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
                axes=gradient_axes(None),
            ),
            factory=_gru_factory,
            gradient_trained=True,
            axes=gradient_axes(None),
            budget_note="same lr axis, epoch budget, patience and val-selection rule as every FM head",
        ),
    ]

    entrants: list[Entrant] = list(incumbents) if include_incumbents else []
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
            axes = gradient_axes(model_id)
            extras = {"head_hidden": 0} if "head_hidden" not in axes else {}
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
                            axes=axes,
                            extras=extras,
                        ),
                        factory=_fm_factory(
                            model_id, condition, weights_dir, pretrained=pretrained, device=device
                        ),
                        gradient_trained=True,
                        axes=axes,
                        budget_note=(
                            "identical bolt-on head, lr axis, epoch budget, patience and "
                            "val-selection rule as the incumbent GRU; second sweep axis is "
                            + ("input_scale" if model_id in INPUT_SCALE_MODEL_IDS else "head width")
                        ),
                        # Random twins are the plan section 4 control, not competitors.
                        ranked=pretrained,
                    )
                )
    return entrants, skipped


def ladder_report(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Which design-document ladder rungs actually produced a row, and which did not."""

    ranked_ids = {row["entrant_id"] for row in results if row.get("ranked")}
    rungs: list[dict[str, Any]] = []
    for rung in INCUMBENT_LADDER:
        realised_by = rung["realised_by"]
        rungs.append(
            {
                **rung,
                "ranked_entrant_present": realised_by in ranked_ids,
                "distinct_row": realised_by == rung["rung"],
            }
        )
    missing = sorted(rung["rung"] for rung in rungs if not rung["ranked_entrant_present"])
    return {"rungs": rungs, "missing_rungs": missing, "complete": not missing}


# ---------------------------------------------------------------------------
# Selection and evaluation (structurally separated; see module docstring)
# ---------------------------------------------------------------------------


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = np.asarray(y_true, dtype=np.float64).ravel() - np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(diff**2))


class RepeatTestEvaluation(RuntimeError):
    """The output directory already records a test evaluation of a planned family.

    Plan section 5.3 allows *one* held-out test evaluation per configuration family, and
    a ledger that is rewritten on every invocation cannot enforce that: a second run into
    the same directory would erase the evidence of the first. The ledger is therefore
    append-only across invocations and the driver refuses to start a run that would take a
    second look, unless a human passes ``--allow-repeat-test`` and accepts that the
    resulting dossier is not claim-bearing.
    """


def _family_key(entrant_id: str, horizon: int) -> str:
    return f"{entrant_id}@h{int(horizon)}"


@dataclass
class AttemptLedger:
    """Plan section 5.3 iteration ledger: every attempt, every test evaluation.

    Append-only across invocations. Every row carries ``run_id`` (UTC timestamp, pid and
    git revision) so the rows of two runs into the same directory stay distinguishable,
    and ``prior_test_families`` carries forward what earlier runs already looked at, which
    is what makes the cumulative test-evaluation count in the artifact honest.
    """

    run_id: str = "unknown-run"
    rows: list[dict[str, Any]] = field(default_factory=list)
    prior_test_families: dict[str, int] = field(default_factory=dict)
    prior_rows: int = 0

    def record(self, **payload: Any) -> None:
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **payload,
        }
        self.rows.append(row)

    @property
    def n_attempts(self) -> int:
        return len(self.rows)

    @property
    def n_test_evaluations(self) -> int:
        return sum(1 for row in self.rows if row.get("phase") == "test")

    @property
    def n_test_evaluations_cumulative(self) -> int:
        return self.n_test_evaluations + sum(self.prior_test_families.values())

    def repeated_test_families(self) -> list[str]:
        """Families this run test-evaluated that an earlier run had already evaluated."""

        seen = {
            _family_key(row["entrant_id"], row["horizon"])
            for row in self.rows
            if row.get("phase") == "test" and row.get("entrant_id") and row.get("horizon") is not None
        }
        return sorted(seen & set(self.prior_test_families))

    def as_log(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "n_attempts": self.n_attempts,
            "n_attempts_including_prior_runs": self.n_attempts + self.prior_rows,
            "n_test_evaluations": self.n_test_evaluations,
            "n_test_evaluations_cumulative": self.n_test_evaluations_cumulative,
            "prior_test_families": dict(sorted(self.prior_test_families.items())),
            "repeated_test_families": self.repeated_test_families(),
            "log_file": "attempts.log",
            "append_only": True,
        }

    def write(self, path: Path) -> Path:
        """Append this run's rows; never truncate a ledger another run wrote."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        return path


def read_prior_ledger(path: Path) -> tuple[dict[str, int], int]:
    """Test-evaluated families and row count already recorded at ``path``.

    A malformed line is skipped rather than fatal: a truncated final line from a killed
    run must not make the directory permanently unusable, and skipping can only *under*
    count prior looks, which the gate treats conservatively anyway.
    """

    families: dict[str, int] = {}
    total = 0
    if not Path(path).is_file():
        return families, total
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        total += 1
        if row.get("phase") != "test" or not row.get("entrant_id") or row.get("horizon") is None:
            continue
        key = _family_key(str(row["entrant_id"]), int(row["horizon"]))
        families[key] = families.get(key, 0) + 1
    return families, total


def open_ledger(
    out_dir: Path,
    *,
    run_id: str,
    planned_families: Sequence[str],
    allow_repeat_test: bool,
) -> AttemptLedger:
    """Open the append-only ledger, refusing a second look unless it was authorised."""

    prior, prior_rows = read_prior_ledger(Path(out_dir) / "attempts.log")
    clashes = sorted(set(planned_families) & set(prior))
    if clashes and not allow_repeat_test:
        raise RepeatTestEvaluation(
            f"{Path(out_dir) / 'attempts.log'} already records a held-out test evaluation "
            f"for {len(clashes)} planned configuration famil{'y' if len(clashes) == 1 else 'ies'} "
            f"(first: {clashes[0]}). Plan section 5.3 allows one test evaluation per "
            "configuration family; running again here would be a second look. Write to a "
            "fresh --out-dir, or pass --allow-repeat-test to record the repeat explicitly "
            "-- which surfaces the cumulative count in the artifact and fails the gate."
        )
    return AttemptLedger(run_id=run_id, prior_test_families=prior, prior_rows=prior_rows)


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

    A configuration is selection-eligible only if **every** planned seed produced a finite
    validation MSE. Averaging over the seeds that happened to survive would let an unstable
    configuration win by discarding its own failures -- the mean of the good runs of a
    method that diverges one time in three is not that method's score.
    """

    scored: list[dict[str, Any]] = []
    for config_index, config in enumerate(entrant.grid):
        per_seed: list[float] = []
        for seed in seeds:
            started = time.perf_counter()
            with capture_training_logs() as logs:
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
                epoch_budget=int(config.get("epochs", 0)),
                seed=int(seed),
                val_mse=value,
                wall_clock_seconds=round(elapsed, 4),
                training=summarise_training_logs(logs),
                error=error,
            )
        all_finite = bool(per_seed) and all(np.isfinite(value) for value in per_seed)
        scored.append(
            {
                "config_index": config_index,
                "config": dict(config),
                "val_mse": float(np.mean(per_seed)) if all_finite else float("inf"),
                "val_mse_by_seed": [float(value) for value in per_seed],
                "seed_spread": float(np.std(per_seed)) if all_finite else float("nan"),
                "selection_eligible": all_finite,
            }
        )

    eligible = [row for row in scored if row["selection_eligible"]]
    disqualified = [
        {
            "config_index": row["config_index"],
            "config": row["config"],
            "val_mse_by_seed": row["val_mse_by_seed"],
            "reason": "at least one planned seed produced a non-finite validation MSE",
        }
        for row in scored
        if not row["selection_eligible"]
    ]
    if disqualified:
        ledger.record(
            phase="selection_disqualification",
            entrant_id=entrant.entrant_id,
            horizon=int(horizon),
            n_disqualified=len(disqualified),
            disqualified=disqualified,
        )
    if not eligible:
        raise EntrantFailed(
            f"none of the {entrant.configs_tried} configurations of {entrant.entrant_id!r} "
            f"produced a finite validation MSE on all {len(list(seeds))} planned seeds at "
            f"horizon {horizon}; refusing to run a test evaluation, because there is no "
            "selection-eligible configuration to evaluate"
        )

    best = min(eligible, key=lambda row: (row["val_mse"], row["config_index"]))
    edges = axis_edge_report(entrant.axes, best["config"])
    return {
        "config": best["config"],
        "config_index": best["config_index"],
        "val_mse": best["val_mse"],
        "val_mse_by_seed": best["val_mse_by_seed"],
        "val_seed_spread": best["seed_spread"],
        "grid_edge": edges,
        "grid_edge_selected": bool(edges["any_axis_at_edge"]),
        "axes_without_edge_detection": edges["axes_without_edge_detection"],
        "configs_tried": len(entrant.grid),
        "configs_selection_eligible": len(eligible),
        "disqualified_configs": disqualified,
        "val_mse_by_config": [row["val_mse"] for row in scored],
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
    load_reports: list[Any] = []
    for seed in seeds:
        started = time.perf_counter()
        with capture_training_logs() as logs:
            runner = entrant.factory(selection["config"])
            prediction = np.asarray(runner(task, int(seed), int(train_steps)), dtype=np.float32)
        load_reports.extend(checkpoint_reports_from_logs(logs))
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
            training=summarise_training_logs(logs),
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
        "n_test_subjects": len(subjects),
        "checkpoint_load_reports": load_reports,
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
        "method": "percentile bootstrap over test subjects, unpaired (marginal)",
    }


# ---------------------------------------------------------------------------
# Paired statistics (plan sections 3.1 and 5.1)
#
# The marginal per-entrant CI above answers "how precisely do we know this entrant's mean
# subject error"; it does NOT answer "is entrant A better than entrant B", because the two
# entrants are scored on the *same* 27 subjects and their errors are strongly correlated.
# Two overlapping marginal CIs are routinely compatible with a paired difference that
# excludes zero by a wide margin. Every ranking claim therefore keys off the paired
# statistics computed here, and the report says so under every table.
# ---------------------------------------------------------------------------


CI_METHOD_NOTE = (
    "95% intervals are percentile bootstrap over the 27 held-out test **subjects** "
    "(the resampling unit is a subject, not a window). The per-entrant column is "
    "*marginal and unpaired*; the delta column is *paired*, resampling the same subject "
    "indices for both entrants (common random numbers) so the between-entrant correlation "
    "is preserved. Ranks and significance are read off the paired column only."
)

SEED_VARIANCE_NOTE = (
    "Each subject's error is averaged over the seeds before resampling, so the bootstrap "
    "CI describes subject-to-subject variability at a fixed seed set and **excludes seed "
    "variance**. The seed spread is reported beside every mean; a delta smaller than the "
    "seed spread is not a result."
)


def _paired_summary(comparison: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The ranking table's view of one paired comparison (or ``None`` for the reference)."""

    if comparison is None:
        return None
    return {
        "reference_entrant_id": comparison["reference_entrant_id"],
        "mean_delta": comparison["mean_delta"],
        "ci95_low": comparison["ci95_low"],
        "ci95_high": comparison["ci95_high"],
        "ci_width": comparison["ci_width"],
        "wilcoxon_p": comparison["wilcoxon_p"],
        "holm_p": comparison["holm_p"],
        "significant_after_holm": comparison["significant_after_holm"],
        "statistical_tie": comparison["statistical_tie"],
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks of ``values`` starting at 1, ties receiving their average rank."""

    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    index = 0
    while index < values.size:
        stop = index + 1
        while stop < values.size and sorted_values[stop] == sorted_values[index]:
            stop += 1
        ranked[order[index:stop]] = 0.5 * (index + stop + 1)
        index = stop
    return ranked


def _normal_sf(z: float) -> float:
    """Upper-tail probability of the standard normal, via ``math.erfc``."""

    return 0.5 * math.erfc(float(z) / math.sqrt(2.0))


def _exact_signed_rank_sf(statistic: float, n: int) -> float:
    """``P(W+ <= statistic)`` under the null, by exact enumeration over sign patterns.

    Dynamic programme over the achievable rank sums: ``2**n`` equally likely sign
    assignments of ranks ``1..n``, so the distribution of ``W+`` is the coefficient vector
    of ``prod_k (1 + x**k)``. Exact, and cheap for the ``n <= 20`` this is used for.
    """

    total = n * (n + 1) // 2
    counts = np.zeros(total + 1, dtype=np.float64)
    counts[0] = 1.0
    for rank in range(1, n + 1):
        # Out-of-place: ``counts[rank:] += counts[:-rank]`` aliases two overlapping views
        # of the same buffer, which numpy is not obliged to buffer correctly.
        shifted = np.zeros_like(counts)
        shifted[rank:] = counts[:-rank]
        counts = counts + shifted
    cutoff = int(math.floor(statistic + 1e-9))
    cutoff = max(0, min(total, cutoff))
    return float(counts[: cutoff + 1].sum() / (2.0**n))


#: Above this many non-zero differences the exact enumeration is replaced by the normal
#: approximation. 20 keeps the DP trivial; with 27 test subjects the approximation is the
#: one that actually runs, and at n>=20 its error on the two-sided p-value is well under
#: the Holm-corrected thresholds this benchmark uses.
WILCOXON_EXACT_MAX_N = 20


def wilcoxon_signed_rank(deltas: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired differences, without SciPy.

    SciPy is not a dependency of this repo (``pyproject.toml`` does not pin it and the
    A100 environment files are quarantined against additions), so the test plan section
    3.1 names as the headline is implemented here rather than imported.

    Zero differences are dropped (Wilcoxon's original handling) and the reduced ``n`` is
    reported, because a subject on which two entrants tie exactly carries no sign
    information. Ties among the absolute differences receive average ranks and the tie
    correction is applied to the variance of the normal approximation. Exact enumeration
    is used when ``n <= WILCOXON_EXACT_MAX_N`` and there are no ties; otherwise the
    normal approximation with a continuity correction is used and named in ``method``.
    """

    values = np.asarray(list(deltas), dtype=np.float64)
    values = values[np.isfinite(values)]
    n_total = int(values.size)
    nonzero = values[values != 0.0]
    n = int(nonzero.size)
    n_zeros = n_total - n
    base = {
        "n_pairs": n_total,
        "n_nonzero_pairs": n,
        "n_zero_pairs": n_zeros,
        "n_favouring_reference": int((values > 0).sum()),
    }
    if n == 0:
        return {
            **base,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "degenerate: every paired difference is exactly zero",
        }

    magnitudes = np.abs(nonzero)
    ranks = _average_ranks(magnitudes)
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    statistic = float(min(w_plus, w_minus))
    _, tie_counts = np.unique(magnitudes, return_counts=True)
    has_ties = bool((tie_counts > 1).any())

    if n <= WILCOXON_EXACT_MAX_N and not has_ties:
        p_value = min(1.0, 2.0 * _exact_signed_rank_sf(statistic, n))
        method = f"exact enumeration over 2**{n} sign patterns"
    else:
        mean = n * (n + 1) / 4.0
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - float((tie_counts**3 - tie_counts).sum()) / 48.0
        if variance <= 0.0:
            return {
                **base,
                "statistic": statistic,
                "p_value": 1.0,
                "method": "degenerate: zero variance under the null after tie correction",
            }
        z = (abs(w_plus - mean) - 0.5) / math.sqrt(variance)
        p_value = min(1.0, 2.0 * _normal_sf(max(z, 0.0)))
        method = (
            f"normal approximation with continuity and tie correction (n={n}"
            f"{', ties present' if has_ties else ''}"
            f"{f', {n_zeros} zero pairs dropped' if n_zeros else ''})"
        )
    return {**base, "statistic": statistic, "p_value": float(p_value), "method": method}


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, monotone and capped at 1 (plan section 5.1.3)."""

    values = [float(p) for p in p_values]
    m = len(values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda index: values[index])
    adjusted = [1.0] * m
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (m - position) * values[index]))
        adjusted[index] = running
    return adjusted


def paired_bootstrap(deltas: Sequence[float], resample_index: np.ndarray) -> dict[str, Any]:
    """Percentile bootstrap of the mean paired delta, on shared resample indices."""

    values = np.asarray(list(deltas), dtype=np.float64)
    if values.size == 0:
        nan = float("nan")
        return {"mean_delta": nan, "median_delta": nan, "ci95_low": nan, "ci95_high": nan, "ci_width": nan}
    draws = values[resample_index].mean(axis=1)
    low = float(np.percentile(draws, 2.5))
    high = float(np.percentile(draws, 97.5))
    return {
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "ci95_low": low,
        "ci95_high": high,
        "ci_width": float(high - low),
        "n_boot": int(resample_index.shape[0]),
    }


def build_paired_statistics(
    results: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    *,
    n_boot: int,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Per horizon, the paired incumbent-minus-external deltas and their tests.

    The reference at each horizon is the **best incumbent** by test MSE; the delta is
    ``external - incumbent`` per subject, so a positive mean delta means the incumbent has
    the lower error. Holm correction runs *per entrant pair, across the claim-eligible
    horizons* (plan section 5.1.3): the family is the set of horizons at which a claim
    about that pair could be made, which is exactly the set of tests a reader would
    cherry-pick from. ``h = 1`` is protocol-ineligible and is excluded from every family.
    """

    by_horizon: list[dict[str, Any]] = []
    comparison_index: dict[tuple[int, str], dict[str, Any]] = {}
    for horizon in horizons:
        rows = [row for row in results if int(row["horizon"]) == int(horizon) and row.get("ranked")]
        incumbents = [row for row in rows if row.get("family") == FAMILY_INCUMBENT]
        externals = [row for row in rows if row.get("family") == FAMILY_EXTERNAL]
        block: dict[str, Any] = {
            "horizon": int(horizon),
            "claim_eligible": bool(rows and all(row.get("claim_eligible_protocol") for row in rows)),
            "reference_entrant_id": None,
            "n_subjects": 0,
            "comparisons": [],
        }
        if not incumbents or not externals:
            block["note"] = (
                "no paired comparison: this horizon has "
                f"{len(incumbents)} ranked incumbent row(s) and {len(externals)} ranked "
                "external row(s), and a paired delta needs at least one of each"
            )
            by_horizon.append(block)
            continue

        reference = min(incumbents, key=lambda row: (float(row["test_mse_mean"]), row["entrant_id"]))
        reference_subjects = dict(reference.get("per_subject_test_mse") or {})
        subjects = sorted(reference_subjects)
        block["reference_entrant_id"] = reference["entrant_id"]
        block["reference_test_mse_mean"] = float(reference["test_mse_mean"])
        block["n_subjects"] = len(subjects)
        if not subjects:
            block["note"] = "no paired comparison: the reference row carries no per-subject errors"
            by_horizon.append(block)
            continue

        # Common random numbers: one resample-index matrix per horizon, shared by every
        # entrant pair, so two entrants' CIs are never compared across different resamples.
        rng = np.random.default_rng(int(seed) + int(horizon))
        resample_index = rng.integers(0, len(subjects), size=(int(n_boot), len(subjects)))

        for row in sorted(externals, key=lambda item: item["entrant_id"]):
            challenger = dict(row.get("per_subject_test_mse") or {})
            shared = [subject for subject in subjects if subject in challenger]
            if len(shared) != len(subjects):
                block.setdefault("warnings", []).append(
                    f"{row['entrant_id']} is scored on {len(shared)} of the reference's "
                    f"{len(subjects)} subjects; the pair is dropped rather than partially compared"
                )
                continue
            deltas = np.asarray(
                [float(challenger[subject]) - float(reference_subjects[subject]) for subject in subjects],
                dtype=np.float64,
            )
            interval = paired_bootstrap(deltas, resample_index)
            test = wilcoxon_signed_rank(deltas)
            comparison = {
                "entrant_id": row["entrant_id"],
                "model_id": row.get("model_id"),
                "condition": row.get("condition"),
                "reference_entrant_id": reference["entrant_id"],
                "test_mse_mean": float(row["test_mse_mean"]),
                **interval,
                "wilcoxon_statistic": test["statistic"],
                "wilcoxon_p": test["p_value"],
                "wilcoxon_method": test["method"],
                "n_subjects_favouring_reference": test["n_favouring_reference"],
                "n_zero_pairs": test["n_zero_pairs"],
                "statistical_tie": bool(interval["ci95_low"] <= 0.0 <= interval["ci95_high"]),
                "reference_lower_error": bool(interval["mean_delta"] > 0.0),
                # Filled in below, once the Holm family for this pair is known.
                "holm_p": None,
                "holm_family_size": 0,
                "significant_after_holm": False,
            }
            block["comparisons"].append(comparison)
            comparison_index[(int(horizon), str(row["entrant_id"]))] = comparison
        by_horizon.append(block)

    # Holm, per entrant pair, over the claim-eligible horizons only.
    families: dict[str, list[dict[str, Any]]] = {}
    for block in by_horizon:
        if not block.get("claim_eligible"):
            continue
        for comparison in block["comparisons"]:
            families.setdefault(str(comparison["entrant_id"]), []).append(comparison)
    for entrant_id, family in sorted(families.items()):
        adjusted = holm_adjust([float(item["wilcoxon_p"]) for item in family])
        for item, value in zip(family, adjusted):
            item["holm_p"] = float(value)
            item["holm_family_size"] = len(family)
            item["significant_after_holm"] = bool(value < ALPHA_LEVEL and item["ci95_low"] > 0.0)

    return {
        "method": {
            "delta": "per-test-subject (external entrant MSE) - (best incumbent MSE)",
            "sign_convention": "positive mean delta means the incumbent has the lower error",
            "bootstrap": CI_METHOD_NOTE,
            "seed_variance": SEED_VARIANCE_NOTE,
            "wilcoxon": (
                "two-sided Wilcoxon signed-rank over test subjects, implemented in-driver "
                "(SciPy is not a repo dependency): exact enumeration for n <= "
                f"{WILCOXON_EXACT_MAX_N} without ties, otherwise the tie- and "
                "continuity-corrected normal approximation, which is what the 27-subject "
                "F8 split actually uses"
            ),
            "multiplicity": (
                "Holm step-down per entrant pair across the claim-eligible horizons; "
                "h = 1 is protocol-ineligible and is not in any family"
            ),
            "alpha": ALPHA_LEVEL,
            "n_boot": int(n_boot),
        },
        "by_horizon": by_horizon,
        "index": {f"h{horizon}|{entrant}": value for (horizon, entrant), value in comparison_index.items()},
    }


def paired_lookup(paired: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    """``(horizon, entrant_id) -> comparison`` view over :func:`build_paired_statistics`."""

    lookup: dict[tuple[int, str], Mapping[str, Any]] = {}
    for block in paired.get("by_horizon", []):
        for comparison in block.get("comparisons", []):
            lookup[(int(block["horizon"]), str(comparison["entrant_id"]))] = comparison
    return lookup


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def _honest_reference_mse(selection: Mapping[str, Any], seed_position: int = 0) -> float:
    """The selected configuration's honest val MSE at the seed the control will re-use.

    Comparing a single-seed corrupted run against a seed-*averaged* honest number would
    fold seed noise into the control's margin, so the same seed's honest value is used.
    """

    by_seed = list(selection.get("val_mse_by_seed") or [])
    if by_seed and seed_position < len(by_seed):
        return float(by_seed[seed_position])
    return float(selection["val_mse"])


def _corruption_control(
    entrant: Entrant,
    bundle: ForecastTaskBundle,
    selection: Mapping[str, Any],
    *,
    control: str,
    corrupt: Callable[[np.ndarray], np.ndarray],
    min_degradation: float,
    description: str,
    ledger: AttemptLedger,
    seed: int,
) -> dict[str, Any]:
    """Refit the selected configuration on corrupted training targets and score honestly.

    The corruption touches ``y_train`` only. Validation inputs and validation targets stay
    honest, so a passing control means "the fitted parameters were carrying real
    information about the true targets", which is the property the control is for.
    """

    honest_mse = _honest_reference_mse(selection)
    threshold = honest_mse * (1.0 + float(min_degradation))
    row: dict[str, Any] = {
        "control": control,
        "entrant_id": entrant.entrant_id,
        "model_id": entrant.model_id,
        "condition": entrant.condition,
        "horizon": bundle.horizon,
        "val_mse_honest": honest_mse,
        "min_degradation": float(min_degradation),
        "falsification_threshold": (
            f"{description}; the corrupted validation MSE must exceed the honest one by at "
            f"least {min_degradation:.0%} (>= {threshold:.6g}) or the control fails"
        ),
    }
    if not entrant.fits_targets:
        # Nothing is fitted to the targets, so there is nothing for the corruption to
        # damage. Recording this as "passed" would be a lie; recording it as "failed"
        # would block the gate on a control that does not apply.
        row.update(
            {
                "applicable": False,
                "passed": True,
                "val_mse_corrupted": float("nan"),
                "note": (
                    "parameter-free entrant: it never reads the training targets, so a "
                    "corrupted fit is undefined rather than uncollapsed"
                ),
            }
        )
        return row

    task = bundle.selection_task
    corrupted_targets = corrupt(np.asarray(task.y_train, dtype=np.float32).copy())
    poisoned = replace(task, y_train=corrupted_targets)
    started = time.perf_counter()
    with capture_training_logs() as logs:
        try:
            runner = entrant.factory(selection["config"])
            prediction = runner(poisoned, int(seed), int(selection["config"].get("epochs", 1) or 1))
            corrupted_mse = mse(task.y_test, prediction)
            error = ""
        except Exception as exc:  # noqa: BLE001 - a control that crashes is a failed control
            corrupted_mse = float("nan")
            error = f"{type(exc).__name__}: {exc}"
    ledger.record(
        phase="control",
        control=control,
        entrant_id=entrant.entrant_id,
        horizon=int(bundle.horizon),
        config=dict(selection["config"]),
        seed=int(seed),
        val_mse_corrupted=corrupted_mse,
        val_mse_honest=honest_mse,
        wall_clock_seconds=round(time.perf_counter() - started, 4),
        training=summarise_training_logs(logs),
        error=error,
    )
    row.update(
        {
            "applicable": True,
            "val_mse_corrupted": float(corrupted_mse),
            "degradation_ratio": (
                float(corrupted_mse / honest_mse) if honest_mse > 0 and np.isfinite(corrupted_mse) else float("nan")
            ),
            "passed": bool(np.isfinite(corrupted_mse) and corrupted_mse >= threshold),
            "error": error,
        }
    )
    return row


def label_shuffle_control(
    entrant: Entrant,
    bundle: ForecastTaskBundle,
    selection: Mapping[str, Any],
    *,
    ledger: AttemptLedger,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Permute the training targets; the entrant's validation skill must collapse."""

    rng = np.random.default_rng(int(seed))
    return _corruption_control(
        entrant,
        bundle,
        selection,
        control="label_shuffle",
        corrupt=lambda targets: targets[rng.permutation(targets.shape[0])],
        min_degradation=LABEL_SHUFFLE_MIN_DEGRADATION,
        description="training targets permuted across windows",
        ledger=ledger,
        seed=seed,
    )


def time_shift_control(
    entrant: Entrant,
    bundle: ForecastTaskBundle,
    selection: Mapping[str, Any],
    *,
    ledger: AttemptLedger,
    seed: int = SPLIT_SEED,
) -> dict[str, Any]:
    """Roll the training targets off their true lag; skill must not survive it.

    Distinct from the label shuffle: rolling preserves the *temporal ordering* and the
    marginal distribution of the targets and changes only their alignment to the contexts.
    A model that survives a shuffle because it learned the target distribution will still
    fail this one, and vice versa, which is why plan section 4 asks for both.
    """

    shift = int(TIME_SHIFT_WINDOWS)
    return _corruption_control(
        entrant,
        bundle,
        selection,
        control="time_shift",
        corrupt=lambda targets: np.roll(targets, shift, axis=0),
        min_degradation=TIME_SHIFT_MIN_DEGRADATION,
        description=(
            f"training targets rolled {shift} windows off their true lag "
            f"({shift * STRIDE_SAMPLES / SAMPLING_RATE_HZ:.2f} s at {SAMPLING_RATE_HZ:g} Hz)"
        ),
        ledger=ledger,
        seed=seed,
    )


def _ridge_sanity_scores(contexts: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """``(tuned ridge MSE, mean-predictor MSE)`` on a held-out tail of a synthetic task.

    Three-way 60/20/20 split: alpha is chosen on the *middle* slice and scored on the
    last one. Picking alpha on the scored slice -- which is what a two-way split would
    do -- is the exact selection sin this driver polices everywhere else, and it would
    flatter the ridge in the one control whose job is to catch a flattering harness.
    """

    from _forecast_common import ridge_fit, ridge_predict

    n = contexts.shape[0]
    fit_stop = max(1, int(n * 0.6))
    select_stop = max(fit_stop + 1, int(n * 0.8))
    x_fit, y_fit = contexts[:fit_stop], targets[:fit_stop]
    x_select, y_select = contexts[fit_stop:select_stop], targets[fit_stop:select_stop]
    x_eval, y_eval = contexts[select_stop:], targets[select_stop:]

    best_alpha, best_val = float(RIDGE_ALPHAS[0]), float("inf")
    for alpha in RIDGE_ALPHAS:
        weights = ridge_fit(x_fit, y_fit, float(alpha))
        value = float(np.mean((y_select - ridge_predict(weights, x_select)) ** 2))
        if value < best_val:
            best_alpha, best_val = float(alpha), value
    weights = ridge_fit(x_fit, y_fit, best_alpha)
    ridge_mse = float(np.mean((y_eval - ridge_predict(weights, x_eval)) ** 2))
    mean_mse = float(np.mean((y_eval - float(y_fit.mean())) ** 2))
    return ridge_mse, mean_mse


def harness_sanity_controls(*, ledger: AttemptLedger, seed: int = SPLIT_SEED) -> list[dict[str, Any]]:
    """Plan section 4's two harness self-tests, run before any ranking is claimable.

    * **AR(2) sanity (F8 T5).** On a stationary AR(2) process the minimum-MSE predictor is
      linear, so a tuned ridge must beat the mean predictor by a wide margin. If it does
      not, the windowing, the target index rule or the scoring is broken, and every number
      downstream is meaningless.
    * **Synthetic NULL.** On white noise nothing is predictable, so the same tuned ridge
      must *not* beat the mean predictor by more than a small out-of-sample tolerance. A
      harness that scores on noise is manufacturing skill.
    """

    rng = np.random.default_rng(int(seed))
    n_windows, context = 2_000, CONTEXT_SAMPLES
    phi1, phi2 = AR2_COEFFICIENTS

    # Burn in from the zero start, otherwise the first samples of every window are a
    # deterministic transient rather than draws from the stationary process, and the
    # control would be measuring how well ridge predicts a transient.
    burn_in = 500
    length = context + 1
    series = np.zeros((n_windows, burn_in + length), dtype=np.float64)
    noise = rng.normal(size=(n_windows, burn_in + length))
    for step in range(2, burn_in + length):
        series[:, step] = phi1 * series[:, step - 1] + phi2 * series[:, step - 2] + noise[:, step]
    series = series[:, burn_in:]
    ar2_ridge, ar2_mean = _ridge_sanity_scores(series[:, :context], series[:, context])

    null = rng.normal(size=(n_windows, length))
    null_ridge, null_mean = _ridge_sanity_scores(null[:, :context], null[:, context])
    ceiling = (1.0 - phi2) / ((1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1**2))

    rows = [
        {
            "control": "ar2_sanity",
            "process": (
                f"AR(2) phi = {list(AR2_COEFFICIENTS)}, {n_windows} windows, context "
                f"{context}, {burn_in}-sample burn-in"
            ),
            "theoretical_max_improvement": float(1.0 - 1.0 / ceiling),
            "tuned_ridge_mse": ar2_ridge,
            "mean_predictor_mse": ar2_mean,
            "improvement": float(1.0 - ar2_ridge / ar2_mean) if ar2_mean > 0 else float("nan"),
            "min_improvement": AR2_SANITY_MIN_IMPROVEMENT,
            "passed": bool(
                np.isfinite(ar2_ridge)
                and ar2_mean > 0
                and ar2_ridge <= ar2_mean * (1.0 - AR2_SANITY_MIN_IMPROVEMENT)
            ),
            "falsification_threshold": (
                "on a stationary AR(2) process a tuned linear model must beat the mean "
                f"predictor by at least {AR2_SANITY_MIN_IMPROVEMENT:.0%}; if it does not, the "
                "harness cannot detect linear structure and no ranking it produces is meaningful"
            ),
        },
        {
            "control": "synthetic_null",
            "process": f"i.i.d. standard normal, {n_windows} windows, context {context}",
            "tuned_ridge_mse": null_ridge,
            "mean_predictor_mse": null_mean,
            "improvement": float(1.0 - null_ridge / null_mean) if null_mean > 0 else float("nan"),
            "max_improvement": SYNTHETIC_NULL_MAX_IMPROVEMENT,
            "passed": bool(
                np.isfinite(null_ridge)
                and null_mean > 0
                and null_ridge >= null_mean * (1.0 - SYNTHETIC_NULL_MAX_IMPROVEMENT)
            ),
            "falsification_threshold": (
                "on a signal with no predictable structure the tuned ridge must not beat the "
                f"mean predictor by more than {SYNTHETIC_NULL_MAX_IMPROVEMENT:.0%}; scoring on "
                "noise means the harness is broken"
            ),
        },
    ]
    for row in rows:
        ledger.record(phase="control", control=row["control"], passed=row["passed"], detail=row)
    return rows


def unimplemented_controls() -> list[dict[str, Any]]:
    """Plan section 4 controls this driver does not run, as explicit failing rows.

    Fail closed: an absent control is reported as a named failure rather than as silence,
    so nobody can read a controls section and conclude that everything listed in the plan
    was checked.
    """

    return [
        {
            "control": name,
            "implemented": False,
            "passed": False,
            "reason": UNIMPLEMENTED_CONTROL_REASONS.get(name, "not implemented"),
        }
        for name in PLAN_SECTION_4_CONTROLS
        if name not in IMPLEMENTED_CONTROLS
    ]


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
                "control": "random_init_twin",
                "entrant_id": pretrained_id,
                "model_id": pretrained.get("model_id"),
                "condition": pretrained.get("condition"),
                "horizon": horizon,
                "pretrained_test_mse": float(pretrained["test_mse_mean"]),
                "random_twin_test_mse": float(row["test_mse_mean"]),
                "gap_random_minus_pretrained": gap,
                "pretraining_helped": bool(gap > 0.0),
                # Plan section 5 makes this a kill criterion, not a diagnostic: an FM whose
                # pretraining bought nothing turns every "foundation model" sentence in the
                # write-up into a sentence about architectures. So the control's pass/fail
                # is conjoined into the gate exactly like every other control.
                "passed": bool(gap > 0.0),
                "falsification_threshold": (
                    "a pretrained model that does not beat its own random-init twin shows "
                    "pretraining bought nothing on this task (plan section 4: 'the most "
                    "important control in the design'; plan section 5 kill criterion)"
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


RANKING_UNIT_NOTE = (
    "Each (model x condition) is its own entrant row. There is deliberately no "
    "cross-condition winner picked on validation before the test look: under that scheme "
    "a model's two conditions would share one 14-configuration budget while an incumbent "
    "spent a whole budget on one row, and the losing condition would still have been "
    "test-evaluated to produce the number that was then discarded -- a test look that "
    "bought nothing and was never reported. Frozen and full_ft therefore each receive the "
    "standard budget, each is test-evaluated exactly once, and both appear in the table."
)


def build_ranking(
    results: Sequence[Mapping[str, Any]],
    horizons: Sequence[int],
    paired: Mapping[tuple[int, str], Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per horizon, rank every ranked entrant row -- one row per (model, condition).

    Random-init twins are controls and never appear here. Tie flags come from the paired
    statistics (CI containing zero), never from marginal-CI overlap.
    """

    lookup = dict(paired or {})
    ranking: list[dict[str, Any]] = []
    for horizon in horizons:
        rows = [row for row in results if row["horizon"] == int(horizon) and row.get("ranked")]
        ordered = sorted(rows, key=lambda row: (float(row["test_mse_mean"]), row["entrant_id"]))
        reference_id = next(
            (
                str(lookup[(int(horizon), str(row["entrant_id"]))]["reference_entrant_id"])
                for row in ordered
                if (int(horizon), str(row["entrant_id"])) in lookup
            ),
            None,
        )
        ranking.append(
            {
                "horizon": int(horizon),
                "ranking_unit": "one row per (model, condition); no cross-condition selection",
                "paired_reference_entrant_id": reference_id,
                "rows": [
                    {
                        "rank": index + 1,
                        "entrant_id": row["entrant_id"],
                        "model_id": row.get("model_id"),
                        "family": row.get("family"),
                        "condition": row.get("condition"),
                        "condition_selected_on": "not_selected__both_conditions_ranked_separately",
                        "val_mse": float(row["val_mse"]),
                        "test_mse_mean": float(row["test_mse_mean"]),
                        "test_mse_ci95_low": float(row["test_mse_ci95_low"]),
                        "test_mse_ci95_high": float(row["test_mse_ci95_high"]),
                        "seed_spread": float(row.get("seed_spread", float("nan"))),
                        "contamination_status": row["contamination_status"],
                        "model_selection_exposure": row["model_selection_exposure"],
                        "grid_edge_selected": row["grid_edge_selected"],
                        "configs_tried": row["configs_tried"],
                        "paired_vs_reference": _paired_summary(
                            lookup.get((int(horizon), str(row["entrant_id"])))
                        ),
                    }
                    for index, row in enumerate(ordered)
                ],
            }
        )
    return ranking


def claim_eligible_horizons(results: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> list[int]:
    """Horizons whose v2 spec is claim-eligible, i.e. every horizon except ``h = 1``."""

    eligible: list[int] = []
    for horizon in horizons:
        rows = [row for row in results if int(row["horizon"]) == int(horizon)]
        if rows and all(row.get("claim_eligible_protocol") for row in rows):
            eligible.append(int(horizon))
    return eligible


def checkpoint_observations(
    model_ids: Sequence[str],
    weights_dir: Path,
    load_reports: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Per ranked pretrained model, the digest of the checkpoint that *actually ran*.

    The manifest in ``weights_manifest`` records what the benchmark *expects*; this records
    what it *observed*. They are separate blocks on purpose: a checkpoints artifact that
    only restates the frozen manifest proves nothing about the bytes that were loaded, and
    a run whose weights directory was swapped between the manifest check and the forward
    pass would look identical. The load report is the one the adapter attached to the
    encoder it built, captured through the condition log sink.
    """

    from neurotwin.benchmarks.fm_forecast.weights_manifest import (
        checkpoint_spec,
        resolve_checkpoint,
        sha256_file,
    )

    rows: list[dict[str, Any]] = []
    for model_id in sorted(set(model_ids)):
        row: dict[str, Any] = {
            "model_id": model_id,
            "load_report": load_reports.get(model_id),
            "load_report_captured": model_id in load_reports,
        }
        try:
            spec = checkpoint_spec(model_id)
            path = resolve_checkpoint(model_id, Path(weights_dir))
            observed = sha256_file(path)
            row.update(
                {
                    "path": str(path),
                    "filename": spec.filename,
                    "observed_sha256": observed,
                    "expected_sha256": spec.sha256,
                    "digest_matches": bool(observed == spec.sha256),
                    "license_note": spec.license_note,
                    "verified": bool(observed == spec.sha256),
                }
            )
        except Exception as exc:  # noqa: BLE001 - an unverifiable checkpoint is a gate failure
            row.update(
                {
                    "verified": False,
                    "digest_matches": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)
    return rows


def budget_symmetry(ledger_rows: Sequence[Mapping[str, Any]], entrants: Sequence[Entrant]) -> dict[str, Any]:
    """Plan section 3.4 rule 5, computed from the ledger rather than asserted.

    Symmetry is over the **gradient-trained ranked entrants** -- the incumbent GRU and
    every foundation-model row. Persistence is parameter-free, and the ridge and GBM rungs
    are not trained by epochs, so folding them in would make the check meaningless. Grid
    edges are a *separate* property (:func:`axis_edge_report`): a symmetric budget spent on
    a grid that was too narrow is still a symmetric budget, and conflating the two hides
    both.
    """

    tracked = {entrant.entrant_id for entrant in entrants if entrant.gradient_trained and entrant.ranked}
    per_entrant: dict[str, dict[str, Any]] = {}
    for row in ledger_rows:
        entrant_id = str(row.get("entrant_id", ""))
        if entrant_id not in tracked or row.get("phase") != "validation":
            continue
        record = per_entrant.setdefault(
            entrant_id, {"config_indices": set(), "seeds": set(), "epoch_budgets": set()}
        )
        record["config_indices"].add(int(row.get("config_index", -1)))
        record["seeds"].add(int(row.get("seed", -1)))
        record["epoch_budgets"].add(int(row.get("epoch_budget", 0)))

    summary = {
        entrant_id: {
            "configs_tried": len(record["config_indices"]),
            "seeds": sorted(record["seeds"]),
            "epoch_budgets": sorted(record["epoch_budgets"]),
        }
        for entrant_id, record in sorted(per_entrant.items())
    }
    configs = {row["configs_tried"] for row in summary.values()}
    seeds = {tuple(row["seeds"]) for row in summary.values()}
    epochs = {tuple(row["epoch_budgets"]) for row in summary.values()}
    missing = sorted(tracked - set(summary))
    return {
        "tracked_entrants": sorted(tracked),
        "entrants_with_no_ledger_rows": missing,
        "per_entrant": summary,
        "equal_configs_tried": len(configs) <= 1,
        "equal_seeds": len(seeds) <= 1,
        "equal_epoch_budget": len(epochs) <= 1,
        "symmetric": bool(
            summary and not missing and len(configs) == 1 and len(seeds) == 1 and len(epochs) == 1
        ),
    }


def protocol_completeness(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Did the *declared* protocol actually run, or only some reduced fragment of it?

    Without this, a two-subject smoke run producing three numbers would satisfy every
    other gate term and emerge claim-allowed. Each violation is its own check so the
    failure reasons name what was missing rather than saying "protocol incomplete".
    """

    config = dict(payload.get("config") or {})
    results = list(payload.get("results") or [])
    split = dict(payload.get("split") or {})
    ranked = [row for row in results if row.get("ranked")]
    ranked_models = {row.get("model_id") for row in ranked if row.get("model_id")}
    conditions_by_model: dict[str, set[str]] = {}
    for row in ranked:
        if row.get("model_id"):
            conditions_by_model.setdefault(str(row["model_id"]), set()).add(str(row.get("condition")))

    # Per-entrant horizon coverage. ``full_horizon_grid`` only checks what the run
    # *declared*; in a merged dossier the declared grid can be complete while a shard
    # contributed one entrant at three of the nine horizons, which would leave holes in
    # every table without failing a single other check.
    declared_horizons = [int(horizon) for horizon in (config.get("horizons") or [])]
    present_pairs = {(str(row["entrant_id"]), int(row["horizon"])) for row in ranked}
    missing_entrant_horizons = sorted(
        f"{entrant_id}@h{horizon}"
        for entrant_id in {str(row["entrant_id"]) for row in ranked}
        for horizon in declared_horizons
        if (entrant_id, horizon) not in present_pairs
    )

    split_sizes = tuple(len(split.get(name, [])) for name in ("train", "val", "test"))
    missing_models = sorted(set(FM_MODEL_IDS) - ranked_models)
    missing_conditions = sorted(
        f"{model_id}:{condition}"
        for model_id in FM_MODEL_IDS
        for condition in CONDITIONS
        if condition not in conditions_by_model.get(model_id, set())
    )
    claim_horizons = claim_eligible_horizons(results, config.get("horizons") or [])

    return {
        "non_smoke": not bool(config.get("smoke")),
        "split_sizes": list(split_sizes),
        "split_sizes_match_f8": split_sizes == F8_SPLIT_SIZES,
        "all_fm_models_ranked": not missing_models,
        "missing_fm_models": missing_models,
        "both_conditions_per_fm": not missing_conditions,
        "missing_model_conditions": missing_conditions,
        "full_horizon_grid": tuple(int(h) for h in (config.get("horizons") or [])) == HORIZONS,
        "full_horizon_grid_per_entrant": not missing_entrant_horizons,
        "missing_entrant_horizons": missing_entrant_horizons,
        "seeds_sufficient": len(config.get("seeds") or []) >= MIN_SEEDS,
        "n_boot_sufficient": int(config.get("n_boot") or 0) >= MIN_CLAIMABLE_N_BOOT,
        "claim_eligible_horizons": claim_horizons,
        "claimable_horizons_exclude_h1": 1 not in claim_horizons,
        "closed_form_grid_truncated": bool(config.get("closed_form_grid_truncated")),
    }


def control_coverage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Which ranked entrants actually carry a passing corruption control, and at which horizon.

    "Every row in the list passed" is not "every entrant was checked": a controls list that
    happens to be non-empty satisfies an all-passed test even when half the ranked slate is
    absent from it. In a single run the two coincide, because the controls are produced in
    the same loop that produces the rows. In a merged dossier they do not: each shard
    contributed controls for its own entrants only, and a shard that skipped them would
    otherwise disappear behind another shard's passing rows.

    A row also has to sit at a *claim-eligible* horizon. A label shuffle demonstrated only
    at the protocol-ineligible ``h = 1`` says nothing about the horizons being claimed.
    """

    results = list(payload.get("results") or [])
    config = dict(payload.get("config") or {})
    controls = dict(payload.get("controls") or {})
    eligible = set(claim_eligible_horizons(results, config.get("horizons") or []))
    ranked_ids = sorted({str(row["entrant_id"]) for row in results if row.get("ranked")})

    report: dict[str, Any] = {
        "claim_eligible_horizons": sorted(eligible),
        "ranked_entrants": ranked_ids,
    }
    for name in ("label_shuffle", "time_shift"):
        covered: set[str] = set()
        for row in controls.get(name) or []:
            horizon = row.get("horizon")
            if not row.get("passed") or horizon is None:
                continue
            if int(horizon) in eligible:
                covered.add(str(row.get("entrant_id")))
        report[f"{name}_missing"] = sorted(set(ranked_ids) - covered)
    report["complete"] = not (report["label_shuffle_missing"] or report["time_shift_missing"])
    return report


def _powered(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The ``powered`` gate term: enough subjects, and a CI narrow enough to mean anything.

    "Excludes zero" is not power. A study with three subjects can exclude zero by luck and
    a study with 27 can exclude zero with an interval so wide that it is consistent with
    both a negligible and an enormous effect. ``powered`` therefore requires the full F8
    test split *and* a paired-delta interval no wider than the declared MDE, at every
    claim-eligible horizon, against the best external entrant.
    """

    results = list(payload.get("results") or [])
    config = dict(payload.get("config") or {})
    paired = dict(payload.get("paired_statistics") or {})
    n_test_subjects = max(
        (int(row.get("n_test_subjects", 0)) for row in results),
        default=0,
    )
    horizons = claim_eligible_horizons(results, config.get("horizons") or [])
    widths: list[dict[str, Any]] = []
    for block in paired.get("by_horizon", []):
        if int(block["horizon"]) not in horizons:
            continue
        comparisons = list(block.get("comparisons") or [])
        if not comparisons:
            continue
        best = min(comparisons, key=lambda item: float(item["test_mse_mean"]))
        widths.append(
            {
                "horizon": int(block["horizon"]),
                "entrant_id": best["entrant_id"],
                "ci_width": float(best["ci_width"]),
                "within_mde": bool(float(best["ci_width"]) <= MDE_ZSCORED_MSE),
            }
        )
    return {
        "n_test_subjects": n_test_subjects,
        "full_test_split": bool(n_test_subjects == F8_TEST_SUBJECTS),
        "mde_zscored_mse": MDE_ZSCORED_MSE,
        "mde_source": MDE_SOURCE,
        "per_horizon_ci_width": widths,
        "all_within_mde": bool(widths) and all(row["within_mde"] for row in widths),
        "powered": bool(
            n_test_subjects == F8_TEST_SUBJECTS and widths and all(row["within_mde"] for row in widths)
        ),
    }


def _beat_best_baseline(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Do the incumbents beat the **best** external row, at every claim-eligible horizon?

    Conservative by construction: the comparator is the single strongest external entrant
    at that horizon, not the mean external, not the model's weaker condition. A win must be
    a win on the paired delta after Holm correction -- never on marginal-CI overlap and
    never on point estimates alone.
    """

    results = list(payload.get("results") or [])
    config = dict(payload.get("config") or {})
    paired = dict(payload.get("paired_statistics") or {})
    horizons = claim_eligible_horizons(results, config.get("horizons") or [])
    per_horizon: list[dict[str, Any]] = []
    for block in paired.get("by_horizon", []):
        if int(block["horizon"]) not in horizons:
            continue
        comparisons = list(block.get("comparisons") or [])
        if not comparisons:
            per_horizon.append({"horizon": int(block["horizon"]), "beaten": False, "reason": "no external comparator"})
            continue
        best = min(comparisons, key=lambda item: float(item["test_mse_mean"]))
        per_horizon.append(
            {
                "horizon": int(block["horizon"]),
                "best_external_entrant_id": best["entrant_id"],
                "reference_entrant_id": best["reference_entrant_id"],
                "mean_delta": float(best["mean_delta"]),
                "ci95_low": float(best["ci95_low"]),
                "holm_p": best["holm_p"],
                "beaten": bool(best["significant_after_holm"]),
                "statistical_tie": bool(best["statistical_tie"]),
            }
        )
    return {
        "per_horizon": per_horizon,
        "n_claim_eligible_horizons": len(horizons),
        "beat_best_baseline": bool(horizons)
        and len(per_horizon) == len(horizons)
        and all(row["beaten"] for row in per_horizon),
    }


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
    def _all_passed(name: str) -> bool:
        rows = controls.get(name) or []
        return bool(rows) and all(bool(row.get("passed")) for row in rows)

    coverage = control_coverage(payload)
    index_audit_passed = _all_passed("index_audit")
    normalization_passed = not bool(payload.get("normalization", {}).get("transductive", True))
    label_shuffle_passed = _all_passed("label_shuffle") and not coverage["label_shuffle_missing"]
    time_shift_passed = _all_passed("time_shift") and not coverage["time_shift_missing"]
    harness_sanity_passed = _all_passed("harness_sanity")
    random_twin_present = bool(controls.get("random_init_twin"))
    random_twin_passed = _all_passed("random_init_twin")
    missing_controls = sorted(row["control"] for row in controls.get("not_implemented") or [])
    contamination_passed = all(
        row["contamination_status"] == "clean"
        for row in results
        if row.get("model_id") and row.get("ranked")
    ) and not payload.get("skipped_models")
    ladder = payload.get("incumbent_ladder") or ladder_report(results)
    incumbents_present = bool(ladder.get("complete"))
    no_grid_edge = all(not row["grid_edge_selected"] for row in results)
    undetectable_axes = sorted(
        {
            f"{row['entrant_id']}:{axis}"
            for row in results
            for axis in (row.get("axes_without_edge_detection") or [])
        }
    )
    symmetry = dict(payload.get("tuning_budget_symmetry") or {})
    budget_symmetric = bool(symmetry.get("symmetric"))
    protocol = dict(payload.get("protocol_completeness") or {})
    powered_block = dict(payload.get("powered") or {})
    baseline_block = dict(payload.get("beat_best_baseline") or {})
    powered = bool(powered_block.get("powered"))
    beat_best_baseline = bool(baseline_block.get("beat_best_baseline"))
    determinism = dict(payload.get("determinism") or {})
    checkpoint_rows = list((payload.get("checkpoints") or {}).get("verified") or [])
    ranked_pretrained = {
        str(row["model_id"]) for row in results if row.get("ranked") and row.get("model_id")
    }
    verified_models = {str(row["model_id"]) for row in checkpoint_rows if row.get("verified")}
    unverified_checkpoints = sorted(ranked_pretrained - verified_models)
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
        missing = coverage["label_shuffle_missing"]
        reasons.append(
            "label-shuffle control did not collapse for every ranked entrant by the "
            f"preregistered {LABEL_SHUFFLE_MIN_DEGRADATION:.0%} margin"
            + (
                "; no passing row at a claim-eligible horizon for: " + ", ".join(missing)
                if missing
                else ""
            )
        )
    if not time_shift_passed:
        missing = coverage["time_shift_missing"]
        reasons.append(
            "time-shift control did not collapse for every ranked entrant by the "
            f"preregistered {TIME_SHIFT_MIN_DEGRADATION:.0%} margin"
            + (
                "; no passing row at a claim-eligible horizon for: " + ", ".join(missing)
                if missing
                else ""
            )
        )
    if not harness_sanity_passed:
        reasons.append(
            "harness sanity control failed: a tuned linear model must beat the mean "
            "predictor on AR(2) and must not beat it on white noise"
        )
    if not random_twin_present:
        reasons.append("random-init architectural twin control is missing")
    elif not random_twin_passed:
        failed = sorted(
            f"{row.get('model_id')}:{row.get('condition')}@h{row.get('horizon')}"
            for row in controls.get("random_init_twin", [])
            if not row.get("passed")
        )
        reasons.append(
            "pretraining bought nothing on this task for "
            + ", ".join(failed)
            + " (random-init twin at least as good as the pretrained model): plan section 5 "
            "kill criterion, so the ranking claim is blocked, not merely annotated"
        )
    for name in missing_controls:
        reasons.append(f"control not implemented: {name}")
    if not contamination_passed:
        reasons.append("not every ranked external entrant is 'clean' on Sleep-EDF")
    if not incumbents_present:
        reasons.append(
            "incumbent ladder incomplete; unrealised rungs: "
            + ", ".join(ladder.get("missing_rungs") or [])
        )
    if not no_grid_edge:
        reasons.append("a selected hyperparameter sits on a grid edge; extend the grid and re-run")
    if undetectable_axes:
        reasons.append(
            "grid-edge detection is impossible on a two-point axis for "
            + ", ".join(undetectable_axes)
            + "; a 2-point axis cannot show whether the optimum is interior"
        )
    if not budget_symmetric:
        reasons.append(
            "tuning budget is not demonstrably symmetric across the gradient-trained ranked "
            "entrants (equal configurations, seeds and epoch budget, computed from the ledger)"
        )
    if not beat_best_baseline:
        reasons.append(
            "incumbents do not beat the best external entrant at every claim-eligible "
            "horizon on the Holm-corrected paired delta"
        )
    if not powered:
        reasons.append(
            f"run is not powered: needs the full {F8_TEST_SUBJECTS}-subject F8 test split and "
            f"a paired-delta 95% CI no wider than the declared MDE of {MDE_ZSCORED_MSE} "
            "z-scored MSE at every claim-eligible horizon"
        )
    if not protocol.get("non_smoke", False):
        reasons.append("smoke run: a reduced-scale rehearsal is never claim-bearing")
    if not protocol.get("split_sizes_match_f8", False):
        reasons.append(
            f"split sizes {protocol.get('split_sizes')} are not F8's {list(F8_SPLIT_SIZES)}"
        )
    if not protocol.get("all_fm_models_ranked", False):
        reasons.append(
            "not every declared foundation model produced a ranked row; missing: "
            + ", ".join(protocol.get("missing_fm_models") or [])
        )
    if not protocol.get("both_conditions_per_fm", False):
        reasons.append(
            "not every foundation model was ranked under both admissible conditions; missing: "
            + ", ".join(protocol.get("missing_model_conditions") or [])
        )
    if not protocol.get("full_horizon_grid", False):
        reasons.append(f"horizon grid is not the declared {list(HORIZONS)}")
    if not protocol.get("full_horizon_grid_per_entrant", False):
        gaps = list(protocol.get("missing_entrant_horizons") or [])
        reasons.append(
            "not every ranked entrant produced a row at every declared horizon: "
            + ", ".join(gaps[:5])
            + (f" (and {len(gaps) - 5} more)" if len(gaps) > 5 else "")
        )
    if not protocol.get("seeds_sufficient", False):
        reasons.append(f"fewer than {MIN_SEEDS} seeds per selected configuration (plan section 3.4)")
    if not protocol.get("n_boot_sufficient", False):
        reasons.append(
            f"bootstrap draws below the declared {MIN_CLAIMABLE_N_BOOT} (plan section 3.1)"
        )
    if not protocol.get("claimable_horizons_exclude_h1", True):
        reasons.append("h = 1 is protocol-ineligible and must not feed a claimable ranking")
    if protocol.get("closed_form_grid_truncated"):
        reasons.append(
            "the ridge alpha and GBM grids were truncated below their full 14 points; a "
            "truncated baseline grid is an under-tuned baseline"
        )
    if not determinism.get("fully_deterministic", False):
        reasons.append(
            "full determinism could not be enabled: "
            + str(determinism.get("reason") or "torch determinism flags were not all set")
        )
    if unverified_checkpoints:
        reasons.append(
            "no verified checkpoint provenance for ranked pretrained entrant(s): "
            + ", ".join(unverified_checkpoints)
        )
    repeated = list((payload.get("attempts") or {}).get("repeated_test_families") or [])
    if repeated:
        reasons.append(
            "config family test-evaluated more than once: "
            + ", ".join(repeated[:5])
            + (f" (and {len(repeated) - 5} more)" if len(repeated) > 5 else "")
        )
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
    # Reasons a *caller* has to contribute because they are properties of how the payload
    # was assembled rather than of anything inside it -- currently only the merge path's
    # smoke-shard admission (see SMOKE_MERGE_FAILURE_REASON).
    reasons.extend(
        str(reason) for reason in payload.get("additional_failure_reasons") or [] if str(reason).strip()
    )

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
            "time_shift_collapsed": time_shift_passed,
            "control_coverage": coverage,
            "harness_sanity_passed": harness_sanity_passed,
            "random_twin_control_present": random_twin_present,
            "pretraining_helped_every_fm": random_twin_passed,
            "controls_not_implemented": missing_controls,
            "contamination_audit_passed": contamination_passed,
            "incumbent_ladder_complete": incumbents_present,
            "no_grid_edge_selected": no_grid_edge,
            "grid_edge_undetectable_axes": undetectable_axes,
            "tuning_budget_symmetric": budget_symmetric,
            "beat_best_baseline": beat_best_baseline,
            "powered": powered,
            "protocol_ran_as_declared": bool(
                protocol.get("non_smoke")
                and protocol.get("split_sizes_match_f8")
                and protocol.get("all_fm_models_ranked")
                and protocol.get("both_conditions_per_fm")
                and protocol.get("full_horizon_grid")
                and protocol.get("full_horizon_grid_per_entrant")
                and protocol.get("seeds_sufficient")
                and protocol.get("n_boot_sufficient")
                and protocol.get("claimable_horizons_exclude_h1")
                and not protocol.get("closed_form_grid_truncated")
            ),
            "determinism_enabled": bool(determinism.get("fully_deterministic")),
            "checkpoint_provenance_verified": not unverified_checkpoints,
            "single_test_look_per_family": not repeated,
            "controls_pass": bool(
                index_audit_passed
                and label_shuffle_passed
                and time_shift_passed
                and harness_sanity_passed
                and random_twin_present
                and random_twin_passed
                and normalization_passed
                and not missing_controls
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


def report_caveats(payload: Mapping[str, Any]) -> list[str]:
    """Every known handicap and asymmetry in this benchmark, in prose, in one place.

    These are properties of the *design*, not of the run, so they are stated whether or not
    the gate passed: a reader who sees only the numbers would otherwise have to reconstruct
    them from column headers, and several of them (EEGPT's model-selection exposure, BIOT's
    single-token context) can change how a row should be read.
    """

    config = dict(payload.get("config") or {})
    contexts = ", ".join(
        f"{model_id} {CONTEXT_SECONDS_BY_MODEL[model_id]:.2f} s"
        for model_id in FM_MODEL_IDS
        if model_id in CONTEXT_SECONDS_BY_MODEL
    )
    return [
        "**EEGPT model-selection exposure.** Sleep-EDFx was in EEGPT's published downstream "
        "suite, so its architecture and hyperparameters were chosen with Sleep-EDF results "
        "visible. Its weights are clean of Sleep-EDF, and the `selection exposure` column "
        "says True, but the column understates it: this is not a contamination-free row, it "
        "is a row whose *design* saw the corpus. Read an EEGPT win as an upper bound.",
        "**BIOT context handicap.** BIOT PREST-16 consumes a 1.0 s window as a single STFT "
        "token, so it sees one spectral snapshot rather than a sequence and has strictly "
        "less temporal resolution here than any other entrant. It is also mapped onto "
        "**slot 0 of its 16-channel bipolar montage, i.e. the FP1-F7 channel identity**: "
        "Sleep-EDF's Fpz-Cz derivation is not in that montage, so the single available "
        "channel is presented under another channel's learned identity. Both are structural "
        "disadvantages of running it off-task, not evidence about BIOT.",
        "**EEGPT channel mapping.** Fpz-Cz is presented to EEGPT as **FPZ**, following the "
        "upstream Sleep-EDF precedent in `external/vendor/eegpt/downstream`. A bipolar "
        "derivation is not the same signal as its anode, and the alternative -- a learned "
        "projection from the derivation into EEGPT's channel space -- was **not** used, "
        "because a trained projection would add capacity that no other entrant receives.",
        "**LaBraM `fc_norm` is untrained under the frozen probe.** LaBraM-Base's final "
        "normalisation layer is fitted during upstream fine-tuning, so in the frozen "
        "condition it runs at its pretrained-initialised statistics. The frozen LaBraM row "
        "is therefore a slightly pessimistic reading of the frozen encoder; its full "
        "fine-tune row does not carry this caveat.",
        "**`kahlus_gru` was re-tuned, not copied.** The incumbent GRU here is F8's "
        "architecture under *this* benchmark's symmetric budget (same lr axis, epoch "
        "budget, patience and validation-selection rule as every FM head), not F8's "
        "fixed-gradient-step recipe. Its numbers are therefore **not byte-identical to the "
        "published F8 row** and must not be quoted as if they were.",
        "**Per-model context lengths differ.** Plan section 3.2 gives each FM the largest "
        f"patch-aligned context not exceeding F8's {INCUMBENT_CONTEXT_SECONDS:.2f} s: "
        f"{contexts}, against the incumbents' {INCUMBENT_CONTEXT_SECONDS:.2f} s. Every FM "
        "therefore sees *less* history than the incumbents -- a conservative handicap that "
        "flatters the incumbents and must be stated beside any incumbent win.",
        "**Input scaling is a swept hyperparameter for CBraMod and LaBraM.** Both upstreams "
        "assume microvolts divided by 100; the harness feeds unitless z-scored windows, so "
        f"the scale is swept over {list(INPUT_SCALES)} on validation. It occupies the sweep "
        "slot the other two models spend on head width, so the *axes* differ across FM rows "
        "even though the configuration count does not.",
        "**Unfreezing depth was not swept.** " + str(UNFREEZE_DEPTH_DECISION["reason"]),
        "**The external models are evaluated off-task.** They are bidirectional "
        "masked-infilling encoders and classifiers, not forecasters. A poor number here is "
        "evidence about transfer to strictly-future single-sample forecasting, and is not "
        "evidence about the model's published capabilities.",
        f"**Declared MDE.** The `powered` gate term uses {MDE_ZSCORED_MSE} z-scored MSE "
        f"(`{MDE_SOURCE}`). The design document names no MDE; this value was declared in "
        "the driver before the data and ratified by Krish Garg on 2026-08-21, before any "
        "full-scale test evaluation had been merged or read.",
    ] + (
        [
            "**This was a smoke run.** Reduced subjects, horizons, configurations and seeds. "
            "It exercises the pipeline and can never be claim-bearing."
        ]
        if config.get("smoke")
        else []
    )


def write_markdown_report(path: Path, payload: Mapping[str, Any]) -> Path:
    """Human-readable report generated from the JSON; never hand-edited."""

    gate = payload.get("gate", {})
    lines: list[str] = [
        "# External Pretrained EEG Models vs Kahlus Incumbents — Sleep-EDF Forecasting",
        "",
        f"- Schema: `{payload.get('schema')}`",
        f"- Created: {payload.get('created_utc')}",
    ]
    if payload.get("mode") == MERGE_MODE:
        lines.append(
            f"- Mode: `{MERGE_MODE}` from {len(payload.get('shards') or [])} shard run(s) "
            "— see the Shards section"
        )
    lines += [
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

    paired = payload.get("paired_statistics") or {}
    lookup = paired_lookup(paired)
    multiplicity = (
        f"Multiplicity: {(paired.get('method') or {}).get('multiplicity', 'Holm per entrant pair')}. "
        f"alpha = {(paired.get('method') or {}).get('alpha', ALPHA_LEVEL)}."
    )

    lines += [
        "## How to read these tables",
        "",
        f"- {RANKING_UNIT_NOTE}",
        f"- {CI_METHOD_NOTE}",
        f"- {SEED_VARIANCE_NOTE}",
        "- **Do not read a ranking off overlapping per-entrant CIs.** The `95% CI` column is "
        "marginal and unpaired; two entrants whose marginal intervals overlap can still "
        "differ with a paired interval far from zero, and two whose intervals barely "
        "overlap can be a statistical tie. The `paired Δ` column is the one that decides.",
        "",
        "## Ranking by horizon",
        "",
    ]
    for block in payload.get("ranking", []):
        horizon = int(block["horizon"])
        lines += [
            f"### h = {horizon}",
            "",
            f"Paired reference: `{block.get('paired_reference_entrant_id') or 'n/a'}` "
            "(best incumbent at this horizon). Δ = entrant − reference, per test subject; "
            "positive Δ means the incumbent has the lower error.",
            "",
            "| rank | entrant | family | condition | val MSE | test MSE | 95% CI (marginal) | "
            "seed spread | paired Δ [95% CI] | Holm p | tie | contamination | selection exposure | grid edge |",
            "|---:|---|---|---|---:|---:|---|---:|---|---:|---|---|---|---|",
        ]
        for row in block["rows"]:
            comparison = lookup.get((horizon, str(row["entrant_id"])))
            if comparison is None:
                # Incumbents other than the reference are not paired against it: the
                # claim under test is "external vs best incumbent", and manufacturing an
                # incumbent-vs-incumbent column would invite exactly the intra-incumbent
                # comparisons this scope does not license.
                delta = (
                    "**reference**"
                    if str(row["entrant_id"]) == str(block.get("paired_reference_entrant_id"))
                    else "not paired (incumbent)"
                )
                holm = "—"
                tie = "—"
            else:
                delta = (
                    f"{comparison['mean_delta']:+.6f} "
                    f"[{comparison['ci95_low']:+.6f}, {comparison['ci95_high']:+.6f}]"
                )
                holm = "—" if comparison["holm_p"] is None else f"{comparison['holm_p']:.4g}"
                tie = "**tie**" if comparison["statistical_tie"] else "no"
            spread = row.get("seed_spread")
            spread_text = "—" if spread is None or not np.isfinite(float(spread)) else f"{float(spread):.6f}"
            lines.append(
                f"| {row['rank']} | `{row['entrant_id']}` | {row.get('family') or '-'} | "
                f"{row['condition'] or '-'} | {row['val_mse']:.6f} | {row['test_mse_mean']:.6f} | "
                f"[{row['test_mse_ci95_low']:.6f}, {row['test_mse_ci95_high']:.6f}] | {spread_text} | "
                f"{delta} | {holm} | {tie} | {row['contamination_status']} | "
                f"{row['model_selection_exposure']} | {row['grid_edge_selected']} |"
            )
        lines += ["", multiplicity, ""]

    lines += ["## Controls", "", "| control | subject | outcome | threshold |", "|---|---|---|---|"]
    for name, rows in sorted((payload.get("controls") or {}).items()):
        for row in rows:
            parts = [str(row[key]) for key in ("entrant_id", "process") if row.get(key) is not None]
            if row.get("horizon") is not None:
                parts.append(f"@ h{row['horizon']}")
            subject = " ".join(parts) or "—"
            if row.get("implemented") is False:
                outcome = "**NOT IMPLEMENTED**"
            elif row.get("applicable") is False:
                outcome = "n/a"
            else:
                outcome = "PASS" if row.get("passed") else "**FAIL**"
            threshold = str(row.get("falsification_threshold") or row.get("reason") or "")
            threshold = threshold.replace("\n", " ")
            lines.append(f"| `{name}` | {subject} | {outcome} | {threshold[:220]} |")
    lines.append("")

    budget = payload.get("tuning_budget", {})
    symmetry = payload.get("tuning_budget_symmetry", {})
    attempts = payload.get("attempts", {})
    lines += [
        "## Tuning budget",
        "",
        f"- configurations per gradient-trained entrant: {budget.get('n_configs_per_entrant')}",
        f"- seeds: {budget.get('seeds')}",
        f"- epoch budget: {budget.get('max_train_epochs')}",
        f"- budget symmetric across gradient entrants (from the ledger): {symmetry.get('symmetric')}",
        f"- grid subset rule (fixed before the run): {budget.get('grid_subset_rule') or GRID_SUBSET_RULE}",
        f"- attempts logged this run: {attempts.get('n_attempts')} "
        f"(including prior runs into this directory: {attempts.get('n_attempts_including_prior_runs')})",
        f"- test-set evaluations this run: {attempts.get('n_test_evaluations')}; "
        f"cumulative in this directory: {attempts.get('n_test_evaluations_cumulative')}",
        f"- unfreeze-depth axis: swept = {UNFREEZE_DEPTH_DECISION['swept']} "
        f"({UNFREEZE_DEPTH_DECISION['realised_value']})",
        "",
        "## Checkpoints actually loaded",
        "",
    ]
    for row in (payload.get("checkpoints") or {}).get("verified", []):
        if row.get("verified"):
            lines.append(
                f"- `{row['model_id']}` — `{row.get('path')}` observed sha256 "
                f"`{row.get('observed_sha256')}` (matches manifest: {row.get('digest_matches')}); "
                f"load report: `{json.dumps(row.get('load_report'), sort_keys=True, default=str)[:160]}`"
            )
        else:
            lines.append(f"- `{row['model_id']}` — **unverified**: {row.get('error')}")
    lines.append("")

    shards = list(payload.get("shards") or [])
    if shards:
        lines += [
            "## Shards",
            "",
            f"Merged from {len(shards)} shard run(s). Every table above was recomputed from the "
            "union of the shards' per-subject test errors: no shard's own ranking, paired "
            "statistics or gate verdict was carried over, and the shards were refused unless "
            "they agreed on protocol, split, normalisation, code revision and checkpoint bytes.",
            "",
            "| shard dir | run id | commit | entrants carried | horizons | result rows |",
            "|---|---|---|---|---:|---:|",
        ]
        for shard in shards:
            entrants = ", ".join(f"`{name}`" for name in shard.get("entrant_ids") or []) or "—"
            horizons = ", ".join(str(horizon) for horizon in shard.get("horizons") or []) or "—"
            lines.append(
                f"| `{shard.get('shard_dir')}` | `{shard.get('run_id')}` | "
                f"`{str((shard.get('git') or {}).get('commit') or '')[:12]}` | {entrants} | "
                f"{horizons} | {shard.get('n_result_rows')} |"
            )
        lines.append("")

    lines += ["## Caveats", ""] + [f"- {caveat}" for caveat in report_caveats(payload)] + [""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_artifacts(
    out_dir: Path, payload: Mapping[str, Any], ledger: "AttemptLedger | ConcatenatedLedger"
) -> dict[str, Path]:
    """Write the JSON dossier, the gate, the ledger and the Markdown report."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "benchmark_json": write_json(out / "eeg_fm_benchmark.json", payload),
        "gate_json": write_json(out / "eeg_fm_benchmark_gate.json", payload["gate"]),
        "contamination_json": write_json(out / "contamination.json", payload["contamination_table"]),
        "checkpoints_json": write_json(out / "checkpoints.json", payload["checkpoints"]),
        "controls_json": write_json(out / "controls.json", payload["controls"]),
        "paired_statistics_json": write_json(
            out / "paired_statistics.json", payload.get("paired_statistics", {})
        ),
        "environment_json": write_json(out / "environment.json", payload["environment"]),
        "git_state_json": write_json(out / "git_state.json", payload["git"]),
        "attempts_log": ledger.write(out / "attempts.log"),
        "report_md": write_markdown_report(out / "EEG_FM_BENCHMARK_REPORT.md", payload),
    }
    return written


# ---------------------------------------------------------------------------
# Shard merge
#
# A full-scale run is one process on one GPU (see "Parallelism and devices"), so using
# four GPUs means four invocations with disjoint ``--models`` and four out-dirs. Each of
# those dossiers is a fragment of one experiment, and none of its summaries is a partial
# answer to the merged question: a per-horizon ranking over three entrants is not a
# sub-ranking of the nine-entrant table, the Holm family over one pair is not the family
# over eight, and a shard's ``beat_best_baseline`` compared against whichever externals
# that shard happened to carry.
#
# Merging therefore recomputes everything downstream of the per-subject test errors from
# the union, and asserts the things that must hold for the union to *be* one experiment --
# same protocol, same split, same normalisation, same code revision, same checkpoint
# bytes, and no ``(entrant, horizon)`` scored twice. Every one of those is a hard error
# raised before a single byte is written, because a merged dossier that silently spans two
# protocols is worse than no dossier at all.
# ---------------------------------------------------------------------------


MERGE_MODE = "merged"

#: Top-level artifact blocks that must be identical across shards for the union to be one
#: experiment. ``normalization`` and ``split`` are the two that decide whether the numbers
#: are even commensurable; ``lane`` and ``dataset_id`` decide what they are numbers *about*.
MERGE_IDENTICAL_TOP_KEYS: tuple[str, ...] = ("lane", "dataset_id", "normalization", "split")

#: ``config`` keys that define the protocol rather than the shard. Anything not listed here
#: (``models``, ``device``, ``out_dir``, ``include_incumbents``) is expected to differ --
#: that difference is the whole point of a shard.
MERGE_IDENTICAL_CONFIG_KEYS: tuple[str, ...] = (
    "subjects",
    "stride",
    "max_windows_per_recording",
    "seeds",
    "n_boot",
    "batch_size",
    "max_train_epochs",
    "patience",
    "horizons",
)

SMOKE_MERGE_FAILURE_REASON = (
    "merged from smoke shards: a merged dossier is full-scale by definition, and "
    "--allow-smoke-merge was passed, so this artifact is a rehearsal of the merge path "
    "and can never be claim-bearing"
)


class ShardMergeError(RuntimeError):
    """The given shard dossiers do not describe one experiment, so they cannot be merged.

    Always raised before anything is written. A merge is not a best-effort combination:
    either the shards agree on the protocol that makes their numbers comparable, or the
    union is not a benchmark run and no amount of downstream statistics can repair it.
    """


@dataclass(frozen=True)
class ConcatenatedLedger:
    """The shard ledgers, appended verbatim into the merged directory.

    Deliberately not an :class:`AttemptLedger`: re-serialising the rows through
    ``json.dumps`` would rewrite the very evidence the ledger exists to preserve -- key
    order, float formatting, and any field a future driver version writes that this one
    does not know to keep. ``write`` is the whole interface :func:`write_artifacts` needs.
    """

    lines: tuple[str, ...]

    def write(self, path: Path) -> Path:
        """Append the shard rows, in shard order; never truncate an existing ledger."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for line in self.lines:
                handle.write(line + "\n")
        return path


@dataclass(frozen=True)
class ShardDossier:
    """One shard out-dir: its artifact, and its ledger both raw and parsed."""

    shard_dir: Path
    payload: dict[str, Any]
    ledger_lines: tuple[str, ...]
    ledger_rows: tuple[dict[str, Any], ...]

    @property
    def config(self) -> dict[str, Any]:
        return dict(self.payload.get("config") or {})

    @property
    def run_id(self) -> str:
        return str((self.payload.get("attempts") or {}).get("run_id") or "unknown-run")

    @property
    def results(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.payload.get("results") or []]

    @property
    def is_smoke(self) -> bool:
        return bool(self.config.get("smoke"))


def read_ledger_file(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Raw lines and parsed rows of one shard's append-only ledger.

    A malformed line is carried through verbatim and simply not parsed, matching
    :func:`read_prior_ledger`'s tolerance of a truncated final line from a killed run:
    dropping the row from the *parsed* view can only undercount attempts, which every
    consumer treats conservatively, whereas dropping it from the merged file would destroy
    evidence.
    """

    if not Path(path).is_file():
        return [], []
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return lines, rows


def load_shard(shard_dir: Path) -> ShardDossier:
    """Read one shard out-dir, or refuse it as un-mergeable."""

    shard_dir = Path(shard_dir)
    artifact = shard_dir / "eeg_fm_benchmark.json"
    if not artifact.is_file():
        raise ShardMergeError(
            f"{shard_dir} is not a benchmark out-dir: {artifact.name} is missing. --merge takes "
            "the --out-dir of a completed shard run, not a parent directory."
        )
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ShardMergeError(f"{artifact} is not readable JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ShardMergeError(f"{artifact} is not a JSON object")
    lines, rows = read_ledger_file(shard_dir / "attempts.log")
    return ShardDossier(
        shard_dir=shard_dir,
        payload=payload,
        ledger_lines=tuple(lines),
        ledger_rows=tuple(rows),
    )


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _require_identical(
    shards: Sequence[ShardDossier],
    getter: Callable[[ShardDossier], Any],
    label: str,
    problems: list[str],
) -> None:
    """Record a problem when ``getter`` is not the same value in every shard."""

    grouped: dict[str, list[str]] = {}
    for shard in shards:
        grouped.setdefault(_encode(getter(shard)), []).append(str(shard.shard_dir))
    if len(grouped) <= 1:
        return
    detail = "; ".join(
        f"{', '.join(dirs)} -> {value[:160]}" for value, dirs in sorted(grouped.items())
    )
    problems.append(f"{label} differs across shards, so they are not one experiment: {detail}")


def validate_shards(shards: Sequence[ShardDossier], *, allow_smoke_merge: bool = False) -> None:
    """Refuse a merge that would fabricate an experiment. Raises before any write.

    Every violation is collected and reported together rather than one per invocation:
    an operator who mis-sharded a run has usually mis-sharded it in more than one way, and
    discovering that one error at a time costs a GPU-hour each round.
    """

    if not shards:
        raise ShardMergeError("--merge needs at least one shard out-dir")

    problems: list[str] = []

    schemas = {str(shard.payload.get("schema")) for shard in shards}
    if len(schemas) > 1:
        problems.append(f"shards carry different artifact schema versions: {sorted(schemas)}")
    elif schemas != {ARTIFACT_SCHEMA}:
        problems.append(
            f"shard artifact schema {sorted(schemas)} is not {ARTIFACT_SCHEMA!r}; this merge "
            "recomputes v1 statistics and refuses to guess at another layout"
        )

    for key in MERGE_IDENTICAL_TOP_KEYS:
        _require_identical(
            shards, lambda shard, key=key: shard.payload.get(key), f"artifact block {key!r}", problems
        )
    for key in MERGE_IDENTICAL_CONFIG_KEYS:
        _require_identical(
            shards, lambda shard, key=key: shard.config.get(key), f"config key {key!r}", problems
        )

    commits = {
        str(shard.shard_dir): str((shard.payload.get("git") or {}).get("commit") or "") for shard in shards
    }
    if len(set(commits.values())) > 1:
        problems.append(
            "shards were produced at different git commits, so the merged dossier could not "
            "name the code that produced it: "
            + ", ".join(f"{path}={commit[:12] or 'unknown'}" for path, commit in sorted(commits.items()))
        )
    dirty = sorted(
        str(shard.shard_dir)
        for shard in shards
        if not bool((shard.payload.get("git") or {}).get("clean_worktree"))
    )
    if dirty:
        problems.append(
            "shard(s) produced from a dirty worktree, so their commit does not identify the "
            "code that ran: " + ", ".join(dirty)
        )

    seen: dict[tuple[str, int], str] = {}
    duplicates: list[str] = []
    for shard in shards:
        for row in shard.results:
            try:
                key = (str(row["entrant_id"]), int(row["horizon"]))
            except (KeyError, TypeError, ValueError):
                problems.append(f"{shard.shard_dir}: a result row has no usable (entrant_id, horizon)")
                continue
            if key in seen:
                duplicates.append(f"{key[0]}@h{key[1]} ({seen[key]} and {shard.shard_dir})")
            else:
                seen[key] = str(shard.shard_dir)
    if duplicates:
        listed = sorted(duplicates)
        problems.append(
            "the same (entrant, horizon) was scored in more than one shard, which is a second "
            "held-out test evaluation of one configuration family (plan section 5.3), not a "
            "merge: "
            + ", ".join(listed[:10])
            + (f" (and {len(listed) - 10} more)" if len(listed) > 10 else "")
            + ". Give exactly one shard each entrant -- e.g. --incumbents off on every shard "
            "but one -- or merge a different set of directories."
        )

    smoke = sorted(str(shard.shard_dir) for shard in shards if shard.is_smoke)
    if smoke and not allow_smoke_merge:
        problems.append(
            "smoke shard(s) " + ", ".join(smoke) + ": a merged dossier is full-scale by "
            "definition, so a reduced-scale rehearsal is refused at the merge rather than "
            "recorded as a gate reason. --allow-smoke-merge merges them anyway and marks the "
            "result non-claim-bearing."
        )

    digests: dict[str, dict[str, list[str]]] = {}
    for shard in shards:
        for row in (shard.payload.get("checkpoints") or {}).get("verified") or []:
            digest = row.get("observed_sha256")
            if not digest:
                # An unverifiable checkpoint is already a gate failure in the merged
                # dossier; it carries no bytes to disagree about, so it is not a merge error.
                continue
            digests.setdefault(str(row.get("model_id")), {}).setdefault(str(digest), []).append(
                str(shard.shard_dir)
            )
    for model_id, by_digest in sorted(digests.items()):
        if len(by_digest) > 1:
            problems.append(
                f"shards loaded different checkpoint bytes for {model_id!r}: "
                + "; ".join(
                    f"{digest[:16]} in {', '.join(sorted(dirs))}" for digest, dirs in sorted(by_digest.items())
                )
            )

    if problems:
        raise ShardMergeError(
            f"refusing to merge {len(shards)} shard(s):\n- " + "\n- ".join(problems)
        )


def _entrants_from_logs(rows: Sequence[Mapping[str, Any]]) -> list[Entrant]:
    """Rebuild the view :func:`budget_symmetry` reads, from artifact entrant logs.

    Only the identity and the ``ranked`` / ``gradient_trained`` flags are load-bearing
    there. The grid and the factory are not reachable from a JSON artifact, so they are
    left empty and unreachable rather than guessed at: a merged dossier recomputes
    statistics, it never re-runs a model.
    """

    def _unavailable(_config: Mapping[str, Any]) -> EntrantRunner:
        raise ShardMergeError("a merged entrant carries no runnable factory")

    return [
        Entrant(
            entrant_id=str(row.get("entrant_id")),
            family=str(row.get("family")),
            model_id=row.get("model_id"),
            condition=row.get("condition"),
            pretrained=bool(row.get("pretrained")),
            grid=(),
            factory=_unavailable,
            ranked=bool(row.get("ranked")),
            gradient_trained=bool(row.get("gradient_trained")),
            fits_targets=bool(row.get("fits_targets", True)),
        )
        for row in rows
    ]


def _dedupe(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Concatenation with exact duplicates removed, in first-seen order."""

    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        key = _encode(row)
        if key in seen:
            continue
        seen.add(key)
        kept.append(dict(row))
    return kept


def _merged_checkpoints(shards: Sequence[ShardDossier]) -> dict[str, Any]:
    """One row per model: the verified observation if any shard produced one.

    :func:`validate_shards` has already refused a set in which two shards observed
    different bytes for the same model, so preferring the verified row cannot hide a
    disagreement -- it can only avoid letting a shard that never loaded the checkpoint
    mask a shard that did.
    """

    by_model: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for row in (shard.payload.get("checkpoints") or {}).get("verified") or []:
            model_id = str(row.get("model_id"))
            existing = by_model.get(model_id)
            if existing is None or (not existing.get("verified") and row.get("verified")):
                by_model[model_id] = {**dict(row), "observed_in_shard": str(shard.shard_dir)}
    return {
        "verified": [by_model[model_id] for model_id in sorted(by_model)],
        "manifest_reference": list(
            (shards[0].payload.get("checkpoints") or {}).get("manifest_reference") or []
        ),
        "note": (
            "'verified' is the union over shards, one row per model, carrying the shard that "
            "observed it. Two shards observing different bytes for one model is a merge error, "
            "not a row here."
        ),
    }


def _merged_attempts(shards: Sequence[ShardDossier], *, run_id: str) -> dict[str, Any]:
    """The merged ledger summary, including any family two shards both test-evaluated."""

    def _test_families(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        return {
            _family_key(str(row["entrant_id"]), int(row["horizon"]))
            for row in rows
            if row.get("phase") == "test" and row.get("entrant_id") and row.get("horizon") is not None
        }

    counts: dict[str, int] = {}
    repeated: set[str] = set()
    prior: dict[str, int] = {}
    for shard in shards:
        for family in _test_families(shard.ledger_rows):
            counts[family] = counts.get(family, 0) + 1
        attempts = dict(shard.payload.get("attempts") or {})
        repeated |= {str(family) for family in attempts.get("repeated_test_families") or []}
        for family, count in dict(attempts.get("prior_test_families") or {}).items():
            prior[str(family)] = prior.get(str(family), 0) + int(count)
    # A family whose test rows appear in two shards' ledgers is a second look even when
    # neither shard emitted a result row for it (a shard that died after the evaluation).
    repeated |= {family for family, count in counts.items() if count > 1}

    return {
        "run_id": run_id,
        "mode": MERGE_MODE,
        "n_attempts": sum(len(shard.ledger_lines) for shard in shards),
        "n_attempts_including_prior_runs": sum(
            int(dict(shard.payload.get("attempts") or {}).get("n_attempts_including_prior_runs") or 0)
            for shard in shards
        ),
        "n_test_evaluations": sum(
            1 for shard in shards for row in shard.ledger_rows if row.get("phase") == "test"
        ),
        "n_test_evaluations_cumulative": sum(
            int(dict(shard.payload.get("attempts") or {}).get("n_test_evaluations_cumulative") or 0)
            for shard in shards
        ),
        "prior_test_families": dict(sorted(prior.items())),
        "repeated_test_families": sorted(repeated),
        "log_file": "attempts.log",
        "append_only": True,
        "per_shard": [
            {
                "shard_dir": str(shard.shard_dir),
                "run_id": shard.run_id,
                "n_ledger_rows": len(shard.ledger_lines),
                "n_test_evaluations": sum(1 for row in shard.ledger_rows if row.get("phase") == "test"),
            }
            for shard in shards
        ],
    }


def build_merged_payload(
    shards: Sequence[ShardDossier], *, out_dir: Path, allow_smoke_merge: bool = False
) -> dict[str, Any]:
    """Recompute one dossier from validated shard dossiers.

    Everything downstream of the per-subject test errors is recomputed over the union --
    ranking, paired deltas, Holm families, power, beat-best-baseline, protocol
    completeness, budget symmetry from the concatenated ledger, and the gate -- because
    each shard's own version of those was computed over a fragment of the slate and is not
    a partial answer to the merged question.
    """

    reference = shards[0].payload
    config = dict(reference.get("config") or {})
    horizons = [int(horizon) for horizon in (config.get("horizons") or [])]
    smoke_shards = sorted(str(shard.shard_dir) for shard in shards if shard.is_smoke)

    results = [row for shard in shards for row in shard.results]
    ledger_rows = [dict(row) for shard in shards for row in shard.ledger_rows]

    entrant_logs: dict[str, dict[str, Any]] = {}
    for shard in shards:
        for row in shard.payload.get("entrants") or []:
            entrant_logs.setdefault(str(row.get("entrant_id")), dict(row))

    # Controls are concatenated, with the two exceptions that are functions of the union
    # rather than of a shard: the random-init twin control is a comparison between merged
    # rows, and the not-implemented list is a constant property of this driver which would
    # otherwise be repeated once per shard and inflate the gate's reason list.
    controls: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        for name, rows in (shard.payload.get("controls") or {}).items():
            controls.setdefault(str(name), []).extend(dict(row) for row in rows or [])
    controls["random_init_twin"] = random_twin_controls(results)
    controls["not_implemented"] = unimplemented_controls()

    determinism_reasons = []
    for shard in shards:
        reason = str((shard.payload.get("determinism") or {}).get("reason") or "").strip()
        if reason:
            determinism_reasons.append(f"{shard.shard_dir}: {reason}")
    determinism = {
        "fully_deterministic": all(
            bool((shard.payload.get("determinism") or {}).get("fully_deterministic")) for shard in shards
        ),
        "reason": "; ".join(determinism_reasons),
        "merged_over_shards": len(shards),
        "note": (
            "the merged verdict is the conjunction over shards: one shard that could not "
            "enable determinism makes the union non-reproducible, however clean the others were"
        ),
    }

    merged_config = {
        **config,
        "out_dir": str(out_dir),
        "mode": MERGE_MODE,
        "merged_shard_count": len(shards),
        "models": sorted({str(model) for shard in shards for model in shard.config.get("models") or []}),
        "smoke": any(shard.is_smoke for shard in shards),
        "include_incumbents": any(bool(shard.config.get("include_incumbents", True)) for shard in shards),
        "include_random_twin": all(
            bool(shard.config.get("include_random_twin", False)) for shard in shards
        ),
        "allow_repeat_test": any(bool(shard.config.get("allow_repeat_test")) for shard in shards),
        "closed_form_grid_truncated": any(
            bool(shard.config.get("closed_form_grid_truncated")) for shard in shards
        ),
        "note": (
            "protocol-defining keys are identical across shards by construction (the merge "
            "refuses otherwise); shard-specific keys such as device, data_root and weights_dir "
            "are the reference shard's, with every shard's full config under 'shards'"
        ),
    }

    tuning_budget = dict(reference.get("tuning_budget") or {})
    notes: list[str] = []
    for shard in shards:
        for note in (shard.payload.get("tuning_budget") or {}).get("notes") or []:
            if note not in notes:
                notes.append(str(note))
    tuning_budget["notes"] = notes
    tuning_budget["closed_form_grid_truncated"] = bool(merged_config["closed_form_grid_truncated"])

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-pid{os.getpid()}-merge{len(shards)}"
    )

    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "mode": MERGE_MODE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": reference.get("dataset_id"),
        "lane": reference.get("lane"),
        "config": merged_config,
        "environment": _environment_payload(),
        "determinism": determinism,
        "git": dict(reference.get("git") or {}),
        "recordings": _dedupe([row for shard in shards for row in shard.payload.get("recordings") or []]),
        "protocol": dict(reference.get("protocol") or {}),
        "normalization": dict(reference.get("normalization") or {}),
        "split": dict(reference.get("split") or {}),
        "tuning_budget": tuning_budget,
        "entrants": [entrant_logs[key] for key in sorted(entrant_logs)],
        "incumbent_ladder": ladder_report(results),
        "results": results,
        "contamination_table": list(reference.get("contamination_table") or []),
        "checkpoints": _merged_checkpoints(shards),
        "controls": controls,
        "skipped_models": _dedupe([row for shard in shards for row in shard.payload.get("skipped_models") or []]),
        "unavailable_models": _dedupe(
            [row for shard in shards for row in shard.payload.get("unavailable_models") or []]
        ),
        "failed_entrants": _dedupe(
            [row for shard in shards for row in shard.payload.get("failed_entrants") or []]
        ),
        "attempts": _merged_attempts(shards, run_id=run_id),
        "merge": {
            "shard_dirs": [str(shard.shard_dir) for shard in shards],
            "n_shards": len(shards),
            "allow_smoke_merge": bool(allow_smoke_merge),
            "smoke_shards": smoke_shards,
            "merge_process_git": _git_state(),
            "identical_top_keys_checked": list(MERGE_IDENTICAL_TOP_KEYS),
            "identical_config_keys_checked": list(MERGE_IDENTICAL_CONFIG_KEYS),
            "note": (
                "the shards' own ranking, paired statistics, powered, beat_best_baseline and "
                "gate blocks were discarded and recomputed from the union of their per-subject "
                "test errors; only the per-(entrant, horizon) rows and the controls were carried"
            ),
        },
        "shards": [
            {
                "shard_dir": str(shard.shard_dir),
                "run_id": shard.run_id,
                "created_utc": shard.payload.get("created_utc"),
                "git": dict(shard.payload.get("git") or {}),
                "environment": dict(shard.payload.get("environment") or {}),
                "determinism": dict(shard.payload.get("determinism") or {}),
                "recordings": list(shard.payload.get("recordings") or []),
                "config": shard.config,
                "entrant_ids": sorted({str(row["entrant_id"]) for row in shard.results}),
                "model_ids": sorted(
                    {str(row["model_id"]) for row in shard.results if row.get("model_id")}
                ),
                "horizons": sorted({int(row["horizon"]) for row in shard.results}),
                "n_result_rows": len(shard.results),
                "n_ledger_rows": len(shard.ledger_lines),
            }
            for shard in shards
        ],
    }
    if smoke_shards:
        payload["additional_failure_reasons"] = [SMOKE_MERGE_FAILURE_REASON]

    payload["tuning_budget_symmetry"] = budget_symmetry(
        ledger_rows, _entrants_from_logs(payload["entrants"])
    )
    payload["paired_statistics"] = build_paired_statistics(
        results, horizons, n_boot=int(merged_config.get("n_boot") or DEFAULT_N_BOOT), seed=SPLIT_SEED
    )
    payload["ranking"] = build_ranking(results, horizons, paired_lookup(payload["paired_statistics"]))
    payload["protocol_completeness"] = protocol_completeness(payload)
    payload["powered"] = _powered(payload)
    payload["beat_best_baseline"] = _beat_best_baseline(payload)
    payload["caveats"] = report_caveats(payload)
    payload["gate"] = build_gate_dossier(payload)
    return payload


@dataclass(frozen=True)
class MergeConfig:
    """The resolved ``--merge`` invocation."""

    shard_dirs: tuple[Path, ...]
    out_dir: Path
    allow_smoke_merge: bool = False


def resolve_merge_config(args: argparse.Namespace) -> MergeConfig:
    """Turn parsed arguments into a merge configuration."""

    shard_dirs = tuple(Path(shard_dir) for shard_dir in (args.merge or ()))
    if not shard_dirs:
        raise ValueError("--merge needs at least one shard out-dir")
    return MergeConfig(
        shard_dirs=shard_dirs,
        out_dir=Path(args.out_dir),
        allow_smoke_merge=bool(getattr(args, "allow_smoke_merge", False)),
    )


def run_merge(config: MergeConfig) -> dict[str, Any]:
    """Validate, merge and re-gate a set of shard dossiers into one merged dossier."""

    shards = [load_shard(shard_dir) for shard_dir in config.shard_dirs]
    validate_shards(shards, allow_smoke_merge=config.allow_smoke_merge)
    payload = build_merged_payload(
        shards, out_dir=config.out_dir, allow_smoke_merge=config.allow_smoke_merge
    )
    ledger = ConcatenatedLedger(
        lines=tuple(line for shard in shards for line in shard.ledger_lines)
    )
    written = write_artifacts(config.out_dir, payload, ledger)
    for shard in shards:
        print(
            f"shard {shard.shard_dir}: run_id {shard.run_id}, "
            f"{len(shard.results)} result row(s), entrants "
            f"{sorted({str(row['entrant_id']) for row in shard.results})}",
            flush=True,
        )
    for name, path in sorted(written.items()):
        print(f"wrote {name}: {path}", flush=True)
    print(
        f"merged {len(shards)} shard(s) into {len(payload['results'])} result row(s); "
        f"scientific_claim_allowed = {payload['gate']['scientific_claim_allowed']}; "
        f"failure_reasons = {payload['gate']['failure_reasons']}",
        flush=True,
    )
    return payload


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
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Sleep-EDF sleep-cassette directory (required unless --merge)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="directory of hand-staged checkpoints (required unless --merge)",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="artifact output directory")
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        default=None,
        metavar="SHARD_DIR",
        help=(
            "MERGE MODE: instead of running the benchmark, recompute one dossier from these "
            "shard out-dirs (each a previous --out-dir). The union is re-ranked, re-tested and "
            "re-gated; --data-root and --weights-dir are neither required nor read. Shards must "
            "agree on protocol, split, normalisation, git commit and checkpoint bytes, and no "
            "(entrant, horizon) may appear twice, or the merge is refused"
        ),
    )
    parser.add_argument(
        "--allow-smoke-merge",
        action="store_true",
        help=(
            "permit --merge over smoke shards. A merged dossier is full-scale by definition, so "
            "this is a rehearsal switch: it injects a gate failure reason and the result can "
            "never be claim-bearing"
        ),
    )
    parser.add_argument(
        "--incumbents",
        default="on",
        choices=("on", "off"),
        help=(
            "run the incumbent ladder (persistence, tuned_ridge_ar, gbm, kahlus_gru) in this "
            "invocation. Turn it off on every shard but one: the ladder is cheap, but running "
            "it twice is two held-out test looks at one configuration family and the merge "
            "refuses it"
        ),
    )
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
    parser.add_argument(
        "--allow-repeat-test",
        action="store_true",
        help=(
            "permit a second held-out test evaluation of a configuration family this "
            "--out-dir has already evaluated. Records the cumulative count in the artifact "
            "and fails the gate; it exists so a repeat is visible, not so it is allowed"
        ),
    )
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
    include_incumbents: bool
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
    allow_repeat_test: bool = False

    @property
    def closed_form_truncation(self) -> int | None:
        """Cap on the ridge/GBM grids. ``None`` outside smoke: they are never truncated."""

        return self.n_configs if self.smoke else None

    def as_log(self) -> dict[str, Any]:
        return {
            "data_root": str(self.data_root),
            "weights_dir": str(self.weights_dir),
            "out_dir": str(self.out_dir),
            "models": list(self.models),
            "horizons": list(self.horizons),
            "conditions": list(self.conditions),
            "include_random_twin": self.include_random_twin,
            "include_incumbents": self.include_incumbents,
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
            "allow_repeat_test": self.allow_repeat_test,
            "closed_form_grid_truncated": self.closed_form_truncation is not None,
        }


def _csv_ints(raw: str, name: str) -> tuple[int, ...]:
    values = tuple(int(token) for token in str(raw).split(",") if token.strip())
    if not values:
        raise ValueError(f"--{name} must list at least one value")
    return values


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """Turn parsed arguments into the resolved, smoke-adjusted run configuration."""

    missing = [
        flag
        for flag, value in (("--data-root", args.data_root), ("--weights-dir", args.weights_dir))
        if value is None
    ]
    if missing:
        raise ValueError(
            f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} required to run the "
            "benchmark; they are optional only in --merge mode, which reads no data and loads "
            "no checkpoint"
        )

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
    n_boot = int(args.n_boot)
    if smoke:
        subjects = min(subjects, 6)
        horizons = horizons[:1]
        max_windows = min(max_windows, 32)
        n_configs = min(n_configs, 2)
        max_train_epochs = min(max_train_epochs, 2)
        seeds = seeds[:1]
        n_boot = min(n_boot, SMOKE_N_BOOT)
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
        include_incumbents=str(getattr(args, "incumbents", "on")) == "on",
        seeds=seeds,
        subjects=subjects,
        max_windows_per_recording=max_windows,
        stride=int(args.stride),
        n_configs=n_configs,
        max_train_epochs=max_train_epochs,
        batch_size=int(args.batch_size),
        patience=int(args.patience),
        n_boot=n_boot,
        device=device,
        normalization_prefix=int(args.normalization_prefix),
        legacy_normalization=bool(args.legacy_normalization),
        smoke=smoke,
        allow_repeat_test=bool(getattr(args, "allow_repeat_test", False)),
    )


#: cuBLAS needs a workspace configuration for deterministic GEMMs, and it is read when the
#: CUDA context is created -- setting it after the first CUDA call is too late, which is
#: why it is set here at process scope rather than requested of the operator in prose.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def set_determinism(seed: int) -> dict[str, Any]:
    """Seed every generator this driver can reach and return the *realised* flags.

    Returns rather than asserts: "we asked for determinism" and "determinism is in force"
    are different claims, and only the second one belongs in an artifact. Determinism is
    requested with ``warn_only=True`` so a kernel without a deterministic implementation
    warns instead of aborting a multi-hour run; the flag state is recorded, and the gate
    fails if anything could not be enabled.
    """

    import random

    random.seed(int(seed))
    np.random.seed(int(seed))

    previous_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if previous_cublas is None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    flags: dict[str, Any] = {
        "seed": int(seed),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cublas_workspace_config_preset_by_operator": previous_cublas is not None,
        "torch_available": False,
        "deterministic_algorithms": False,
        "deterministic_algorithms_warn_only": True,
        "cudnn_benchmark": None,
        "cudnn_deterministic": None,
        "cuda_visible": False,
    }
    try:
        import torch

        flags["torch_available"] = True
        torch.manual_seed(int(seed))
        cuda = bool(torch.cuda.is_available())
        flags["cuda_visible"] = cuda
        if cuda:
            torch.cuda.manual_seed_all(int(seed))
        torch.use_deterministic_algorithms(True, warn_only=True)
        flags["deterministic_algorithms"] = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        flags["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
        flags["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
    except Exception as exc:  # noqa: BLE001 - torch absence is reported, never fatal here
        flags["error"] = f"{type(exc).__name__}: {exc}"

    problems: list[str] = []
    if not flags["torch_available"]:
        problems.append("torch is not importable, so no torch determinism flag could be set")
    else:
        if not flags["deterministic_algorithms"]:
            problems.append("torch.use_deterministic_algorithms could not be enabled")
        if flags["cudnn_benchmark"] is not False:
            problems.append("cudnn.benchmark is not False")
        if flags["cudnn_deterministic"] is not True:
            problems.append("cudnn.deterministic is not True")
        if flags["cuda_visible"] and not flags["cublas_workspace_config"]:
            problems.append("CUBLAS_WORKSPACE_CONFIG is unset with CUDA visible")
    flags["fully_deterministic"] = not problems
    flags["reason"] = "; ".join(problems)
    flags["note"] = (
        "warn_only=True: a kernel with no deterministic implementation warns rather than "
        "aborting the run, so a warning in the log is a real caveat on bitwise "
        "reproducibility even when this block reports fully_deterministic"
    )
    return flags


def run_benchmark(config: RunConfig) -> dict[str, Any]:
    """Load, tune on validation, evaluate once on test, and write the dossier."""

    from neurotwin.benchmarks.fm_forecast.weights_manifest import manifest_rows

    determinism = set_determinism(SPLIT_SEED)
    if not determinism["fully_deterministic"]:
        print(f"determinism: NOT fully enabled ({determinism['reason']})", flush=True)
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
        truncate_closed_form=config.closed_form_truncation,
        include_incumbents=config.include_incumbents,
    )
    if not config.include_incumbents:
        print(
            "incumbents: OFF -- this invocation is a shard; its ladder and protocol "
            "completeness checks will fail until it is merged with the shard that carries them",
            flush=True,
        )

    git_state = _git_state()
    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-pid{os.getpid()}-{str(git_state.get('commit') or 'nogit')[:12]}"
    )
    ledger = open_ledger(
        config.out_dir,
        run_id=run_id,
        planned_families=[
            _family_key(entrant.entrant_id, horizon)
            for entrant in entrants
            for horizon in config.horizons
        ],
        allow_repeat_test=config.allow_repeat_test,
    )
    print(f"run_id: {run_id}", flush=True)

    results: list[dict[str, Any]] = []
    unavailable: list[dict[str, str]] = []
    failed: list[dict[str, Any]] = []
    controls: dict[str, list[dict[str, Any]]] = {
        "index_audit": [],
        "label_shuffle": [],
        "time_shift": [],
    }
    checkpoint_load_reports: dict[str, Any] = {}
    blocked_entrants: set[str] = set()

    # Corruption controls are run at one designated horizon per plan section 4, and the
    # designated horizon is claim-eligible whenever the run has one: a control demonstrated
    # only at the protocol-ineligible h=1 would not support the horizons being claimed.
    eligible_horizons = [horizon for horizon in config.horizons if horizon >= 2]
    control_horizon = eligible_horizons[0] if eligible_horizons else config.horizons[0]

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
            if entrant.model_id and entrant.pretrained and evaluation.get("checkpoint_load_reports"):
                checkpoint_load_reports.setdefault(
                    entrant.model_id, evaluation["checkpoint_load_reports"][0]
                )
            row = {
                **entrant.as_log(),
                "horizon": int(horizon),
                "claim_eligible_protocol": bundle.claim_eligible,
                "selected_config": selection["config"],
                "grid_edge": selection["grid_edge"],
                "grid_edge_selected": selection["grid_edge_selected"],
                "axes_without_edge_detection": selection["axes_without_edge_detection"],
                "configs_selection_eligible": selection["configs_selection_eligible"],
                "disqualified_configs": selection["disqualified_configs"],
                "val_mse": selection["val_mse"],
                "val_seed_spread": selection["val_seed_spread"],
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

            # Plan section 4: every control, for every ranked entrant, not just the ridge.
            if entrant.ranked and int(horizon) == int(control_horizon):
                controls["label_shuffle"].append(
                    label_shuffle_control(entrant, bundle, selection, ledger=ledger)
                )
                controls["time_shift"].append(
                    time_shift_control(entrant, bundle, selection, ledger=ledger)
                )

    controls["random_init_twin"] = random_twin_controls(results)
    controls["harness_sanity"] = harness_sanity_controls(ledger=ledger)
    controls["not_implemented"] = unimplemented_controls()

    payload: dict[str, Any] = {
        "schema": ARTIFACT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "lane": "lane_a_single_channel_fpz_cz",
        "config": config.as_log(),
        "environment": _environment_payload(),
        "determinism": determinism,
        "git": git_state,
        "recordings": recording_provenance(corpus),
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
            "grid_subset_rule": GRID_SUBSET_RULE,
            "ridge_alpha_grid": list(RIDGE_ALPHAS),
            "gbm_grid": {
                "learning_rate": list(GBM_LEARNING_RATES),
                "max_leaf_nodes": list(GBM_MAX_LEAF_NODES),
                "max_iter": GBM_MAX_ITER,
            },
            "input_scale_axis": {
                "values": list(INPUT_SCALES),
                "applies_to": list(INPUT_SCALE_MODEL_IDS),
                "replaces": "head width, so the configuration count is unchanged",
            },
            "unfreeze_depth": UNFREEZE_DEPTH_DECISION,
            "closed_form_grid_truncated": config.closed_form_truncation is not None,
            "notes": [entrant.budget_note for entrant in entrants if entrant.budget_note],
        },
        "entrants": [entrant.as_log() for entrant in entrants],
        "incumbent_ladder": ladder_report(results),
        "results": results,
        "contamination_table": contamination_dump(),
        "checkpoints": {
            "verified": checkpoint_observations(
                sorted({row["model_id"] for row in results if row.get("ranked") and row.get("model_id")}),
                config.weights_dir,
                checkpoint_load_reports,
            ),
            "manifest_reference": manifest_rows(),
            "note": (
                "'verified' records the checkpoint that actually ran -- observed digest, "
                "resolved path and the adapter's load report. 'manifest_reference' is the "
                "frozen expectation. Restating the manifest alone would prove nothing about "
                "the bytes that were loaded."
            ),
        },
        "controls": controls,
        "skipped_models": skipped,
        "unavailable_models": unavailable,
        "failed_entrants": failed,
        "attempts": ledger.as_log(),
    }
    payload["tuning_budget_symmetry"] = budget_symmetry(ledger.rows, entrants)
    payload["paired_statistics"] = build_paired_statistics(
        results, config.horizons, n_boot=config.n_boot, seed=SPLIT_SEED
    )
    payload["ranking"] = build_ranking(
        results, config.horizons, paired_lookup(payload["paired_statistics"])
    )
    payload["protocol_completeness"] = protocol_completeness(payload)
    payload["powered"] = _powered(payload)
    payload["beat_best_baseline"] = _beat_best_baseline(payload)
    payload["caveats"] = report_caveats(payload)
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
    if getattr(args, "merge", None):
        try:
            merge_config = resolve_merge_config(args)
        except ValueError as exc:
            parser.error(str(exc))
            return 2
        run_merge(merge_config)
        return 0
    try:
        config = resolve_config(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    run_benchmark(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

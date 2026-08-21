"""Contract tests for the EEG-FM benchmark driver, ``scripts/eeg_fm_benchmark.py``.

These run on CPU, on synthetic arrays, with stub entrants. Nothing here imports a
vendored upstream, touches a staged checkpoint, reads Sleep-EDF, or trains a real model:
the properties under test are the ones that decide whether a produced number is
admissible, not whether a model is good.

Covered:

* the causal normalisation statistics window never overlaps any context or target sample;
* CLI parsing and the configuration echo, including the smoke shrink and the refusal of
  the acausal infilling condition;
* selection cannot see test -- a NaN-poisoned ``y_test`` provably cannot reach the
  selection path, and test predictions are requested exactly once per seed afterwards;
* the artifact writer emits the schema fields, including the model-selection-exposure
  column the plan requires;
* the gate dossier passes under the new narrow claim scope only when every check is
  True, and fails closed otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIVER_PATH = _REPO_ROOT / "scripts" / "eeg_fm_benchmark.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("eeg_fm_benchmark_driver_under_test", _DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before executing: the module defines dataclasses, and dataclasses
    # resolves annotations through ``sys.modules[cls.__module__]``.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


# ---------------------------------------------------------------------------
# Normalisation: the transductive z-scoring fix
# ---------------------------------------------------------------------------


class NormalizationDisjointnessTests(unittest.TestCase):
    def test_causal_prefix_statistics_precede_every_evaluated_sample(self):
        plan = driver.plan_normalization(skip=1000, take=600, prefix=200, legacy=False)

        self.assertEqual(plan.mode, driver.NORMALIZATION_CAUSAL_PREFIX)
        self.assertFalse(plan.is_transductive)
        self.assertEqual((plan.stats_start, plan.stats_stop), (800, 1000))
        self.assertEqual((plan.usable_start, plan.usable_stop), (1000, 1600))
        self.assertLessEqual(plan.stats_stop, plan.usable_start)

    def test_no_context_or_target_index_ever_touches_the_statistics_window(self):
        plan = driver.plan_normalization(skip=1000, take=4000, prefix=300, legacy=False)
        horizons = (1, 2, 4, 8, 16, 32)
        starts = driver.window_starts(
            plan, context=127, max_horizon=max(horizons), stride=13, max_windows=10_000, seed=0
        )

        self.assertGreater(starts.size, 0)
        stats_indices = set(range(plan.stats_start, plan.stats_stop))
        for start in starts.tolist():
            for horizon in horizons:
                ctx_start, ctx_stop, target_start, target_stop = driver.window_span(
                    start, context=127, horizon=horizon
                )
                used = set(range(ctx_start, ctx_stop)) | set(range(target_start, target_stop))
                self.assertEqual(used & stats_indices, set())
                self.assertTrue(plan.is_disjoint_from(ctx_start, target_stop))

        # The property is also asserted by the driver itself, not just by this test.
        driver.assert_normalization_disjoint(plan, starts, context=127, horizons=horizons)

    def test_head_fallback_burns_the_prefix_when_there_is_no_history(self):
        plan = driver.plan_normalization(skip=0, take=5000, prefix=500, legacy=False)

        self.assertEqual(plan.mode, driver.NORMALIZATION_CAUSAL_HEAD)
        self.assertEqual((plan.stats_start, plan.stats_stop), (0, 500))
        self.assertEqual(plan.usable_start, 500)
        starts = driver.window_starts(plan, context=127, max_horizon=16, stride=64, max_windows=1000)
        for start in starts.tolist():
            ctx_start, _, _, target_stop = driver.window_span(start, context=127, horizon=16)
            self.assertTrue(plan.is_disjoint_from(ctx_start, target_stop))

    def test_head_fallback_refuses_an_impossible_prefix(self):
        with self.assertRaises(ValueError):
            driver.plan_normalization(skip=0, take=100, prefix=500, legacy=False)

    def test_legacy_mode_is_transductive_and_says_so(self):
        plan = driver.plan_normalization(skip=1000, take=600, prefix=200, legacy=True)

        self.assertEqual(plan.mode, driver.NORMALIZATION_LEGACY)
        self.assertTrue(plan.is_transductive)
        # F8's statistics window IS the evaluated window: that is the caveat being ablated.
        self.assertEqual((plan.stats_start, plan.stats_stop), (plan.usable_start, plan.usable_stop))
        self.assertTrue(plan.as_log()["transductive"])
        # The driver does not raise in legacy mode; it records the mode and lets the
        # gate refuse the claim.
        driver.assert_normalization_disjoint(plan, [1000, 1256], context=127, horizons=(1, 8))

    def test_window_span_leaves_zero_shared_samples_at_every_horizon(self):
        for horizon in driver.HORIZONS:
            ctx_start, ctx_stop, target_start, target_stop = driver.window_span(0, context=127, horizon=horizon)
            self.assertEqual((ctx_start, ctx_stop), (0, 127))
            self.assertEqual(target_start, 127 + horizon - 1)
            self.assertEqual(target_stop - target_start, 1)
            self.assertGreaterEqual(target_start, ctx_stop)

    def test_normalised_series_uses_prefix_statistics_only(self):
        # A synthetic series whose evaluated segment has a different mean/scale from its
        # prefix: a causal normaliser must NOT centre the evaluated segment on itself.
        plan = driver.plan_normalization(skip=200, take=400, prefix=200, legacy=False)
        series = np.concatenate([np.zeros(200) + 1.0, np.zeros(400) + 9.0])
        stats = series[plan.stats_start : plan.stats_stop]
        usable = series[plan.usable_start : plan.usable_stop]
        normalised = (usable - stats.mean()) / (stats.std() + 1e-8)

        self.assertNotAlmostEqual(float(normalised.mean()), 0.0)
        self.assertAlmostEqual(float(stats.mean()), 1.0)


# ---------------------------------------------------------------------------
# CLI and configuration echo
# ---------------------------------------------------------------------------


def _cli(*extra: str):
    return driver.build_parser().parse_args(
        ["--data-root", "/tmp/data", "--weights-dir", "/tmp/weights", "--out-dir", "/tmp/out", *extra]
    )


class CliTests(unittest.TestCase):
    def test_defaults_resolve_to_the_full_design_grid(self):
        config = driver.resolve_config(_cli())

        self.assertEqual(config.models, driver.FM_MODEL_IDS)
        self.assertEqual(config.horizons, driver.HORIZONS)
        self.assertEqual(config.conditions, driver.CONDITIONS)
        self.assertEqual(config.seeds, driver.DEFAULT_SEEDS)
        self.assertTrue(config.include_random_twin)
        self.assertFalse(config.legacy_normalization)
        self.assertEqual(config.n_configs, driver.TUNING_GRID_SIZE)
        self.assertGreaterEqual(config.n_boot, 1000)

    def test_config_echo_round_trips_through_json(self):
        config = driver.resolve_config(_cli("--models", "cbramod,biot_prest16", "--horizons", "1,8"))
        echo = json.loads(json.dumps(config.as_log()))

        self.assertEqual(echo["models"], ["cbramod", "biot_prest16"])
        self.assertEqual(echo["horizons"], [1, 8])
        self.assertEqual(echo["out_dir"], "/tmp/out")
        for key in ("n_configs", "max_train_epochs", "seeds", "device", "legacy_normalization", "smoke"):
            self.assertIn(key, echo)

    def test_smoke_shrinks_every_expensive_axis(self):
        config = driver.resolve_config(_cli("--smoke"))

        self.assertTrue(config.smoke)
        self.assertEqual(config.subjects, 6)
        self.assertEqual(len(config.horizons), 1)
        self.assertEqual(config.max_train_epochs, 2)
        self.assertEqual(len(config.seeds), 1)
        self.assertEqual(config.device, "cpu")

    def test_smoke_off_is_the_default_and_is_expressible_as_a_job_parameter(self):
        self.assertFalse(driver.resolve_config(_cli()).smoke)
        self.assertFalse(driver.resolve_config(_cli("--smoke", "off")).smoke)
        self.assertTrue(driver.resolve_config(_cli("--smoke", "on")).smoke)

    def test_random_twin_control_can_be_disabled_explicitly(self):
        self.assertFalse(driver.resolve_config(_cli("--no-include-random-twin")).include_random_twin)

    def test_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            driver.resolve_config(_cli("--models", "luna"))

    def test_acausal_infilling_condition_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            driver.resolve_config(_cli("--conditions", "acausal_infill"))
        self.assertIn("infilling", str(caught.exception))

    def test_legacy_normalization_is_opt_in(self):
        self.assertTrue(driver.resolve_config(_cli("--legacy-normalization")).legacy_normalization)


# ---------------------------------------------------------------------------
# Synthetic corpus and stub entrants
# ---------------------------------------------------------------------------


def _synthetic_corpus(*, n_subjects: int = 8, n_windows: int = 6, horizons=(1, 4), seed: int = 0):
    """A tiny corpus in the driver's own ``RecordingWindows`` shape."""

    plan = driver.plan_normalization(skip=1000, take=5000, prefix=400, legacy=False)
    rng = np.random.default_rng(seed)
    corpus: dict[str, list] = {}
    for index in range(n_subjects):
        subject = f"SC{index:03d}"
        starts = driver.window_starts(
            plan, context=driver.CONTEXT_SAMPLES, max_horizon=max(horizons), stride=64,
            max_windows=n_windows, seed=index,
        )[:n_windows]
        context = rng.normal(size=(starts.size, driver.CONTEXT_SAMPLES)).astype(np.float32)
        targets = {
            int(h): (context[:, -1] * 0.5 + rng.normal(scale=0.1, size=starts.size)).astype(np.float32)
            for h in horizons
        }
        corpus[subject] = [
            driver.RecordingWindows(
                subject=subject,
                record_id=f"{subject}E0-PSG",
                starts=starts,
                context=context,
                targets=targets,
                stats_mean=0.0,
                stats_std=1.0,
            )
        ]
    return corpus


def _bundle(horizon: int = 4):
    corpus = _synthetic_corpus(horizons=(1, horizon))
    subjects = sorted(corpus)
    return driver.build_task_bundle(
        corpus,
        horizon=horizon,
        train_subjects=subjects[:4],
        val_subjects=subjects[4:6],
        test_subjects=subjects[6:],
    )


class _StubEntrantRecorder:
    """Records every task a stub runner is handed, so leakage is observable."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def factory(self, config):
        bias = float(config["bias"])

        def runner(task, seed, train_steps):
            self.calls.append(
                {
                    "bias": bias,
                    "seed": int(seed),
                    "n_test_rows": int(task.y_test.shape[0]),
                    "saw_non_finite_test_targets": bool(not np.isfinite(np.asarray(task.y_test)).all()),
                }
            )
            return np.full(task.y_test.shape, bias, dtype=np.float32)

        return runner

    def entrant(self, biases=(0.0, 5.0), *, entrant_id="stub"):
        return driver.Entrant(
            entrant_id=entrant_id,
            family=driver.FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=tuple({"bias": bias} for bias in biases),
            factory=self.factory,
        )


class TaskConstructionTests(unittest.TestCase):
    def test_bundle_shapes_and_index_audit(self):
        bundle = _bundle(horizon=4)

        self.assertEqual(bundle.task.x_train.shape[1:], (driver.CONTEXT_SAMPLES, 1))
        self.assertEqual(bundle.task.y_train.shape[1:], (1, 1))
        self.assertGreater(bundle.n_train, 0)
        self.assertGreater(bundle.n_val, 0)
        self.assertGreater(bundle.n_test, 0)

        audit = driver.index_audit(bundle)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["windows_with_shared_samples"], 0)
        self.assertGreater(audit["windows_checked"], 0)

    def test_every_design_horizon_resolves_against_the_v2_contract(self):
        for horizon in driver.HORIZONS:
            with self.subTest(horizon=horizon):
                spec, claim_eligible = driver._forecast_spec(horizon)
                resolved = spec.resolve(driver.SAMPLING_RATE_HZ)
                self.assertEqual(resolved.context_samples, driver.CONTEXT_SAMPLES)
                self.assertEqual(resolved.gap_samples, horizon - 1)
                self.assertEqual(claim_eligible, horizon >= 2)

    def test_selection_view_replaces_test_arrays_with_validation_arrays(self):
        bundle = _bundle(horizon=4)
        selection = bundle.selection_task

        np.testing.assert_array_equal(selection.y_test, bundle.task.y_val)
        np.testing.assert_array_equal(selection.x_test, bundle.task.x_val)
        self.assertNotEqual(selection.y_test.shape[0], 0)


class SelectionDisciplineTests(unittest.TestCase):
    def test_nan_poisoned_test_targets_cannot_reach_selection(self):
        bundle = _bundle(horizon=4)
        poisoned = replace(bundle.task, y_test=np.full(bundle.task.y_test.shape, np.nan, dtype=np.float32))
        selection_view = driver.make_selection_task(poisoned)

        # The selection view is built from validation arrays, so the poison is unreachable.
        self.assertTrue(np.isfinite(np.asarray(selection_view.y_test)).all())

        recorder = _StubEntrantRecorder()
        entrant = recorder.entrant(biases=(0.0, 500.0))
        ledger = driver.AttemptLedger()
        selection = driver.select_configuration(
            entrant, selection_view, horizon=4, seeds=(0,), train_steps=1, ledger=ledger
        )

        self.assertEqual(selection["config"], {"bias": 0.0})
        self.assertTrue(np.isfinite(selection["val_mse"]))
        self.assertFalse(any(call["saw_non_finite_test_targets"] for call in recorder.calls))
        self.assertTrue(all(row["phase"] == "validation" for row in ledger.rows))
        self.assertEqual(ledger.n_test_evaluations, 0)

    def test_selection_is_identical_whether_or_not_test_is_poisoned(self):
        bundle = _bundle(horizon=4)
        poisoned = replace(bundle.task, y_test=np.full(bundle.task.y_test.shape, np.nan, dtype=np.float32))

        chosen = []
        for task in (bundle.task, poisoned):
            recorder = _StubEntrantRecorder()
            chosen.append(
                driver.select_configuration(
                    recorder.entrant(biases=(0.0, 3.0, 9.0)),
                    driver.make_selection_task(task),
                    horizon=4,
                    seeds=(0, 1),
                    train_steps=1,
                    ledger=driver.AttemptLedger(),
                )["config"]
            )
        self.assertEqual(chosen[0], chosen[1])

    def test_test_predictions_are_requested_exactly_once_per_seed(self):
        bundle = _bundle(horizon=4)
        recorder = _StubEntrantRecorder()
        entrant = recorder.entrant(biases=(0.0, 5.0))
        ledger = driver.AttemptLedger()
        selection = driver.select_configuration(
            entrant, bundle.selection_task, horizon=4, seeds=(0,), train_steps=1, ledger=ledger
        )
        n_selection_calls = len(recorder.calls)

        evaluation = driver.evaluate_selected(
            entrant,
            bundle.task,
            selection,
            horizon=4,
            seeds=(0, 1, 2),
            train_steps=1,
            test_subject_bounds=bundle.test_subject_bounds,
            ledger=ledger,
        )

        self.assertEqual(len(recorder.calls) - n_selection_calls, 3)
        self.assertEqual(evaluation["n_test_evaluations"], 3)
        self.assertEqual(ledger.n_test_evaluations, 3)
        self.assertEqual(sorted(evaluation["per_subject_test_mse"]), sorted(bundle.test_subject_bounds))
        self.assertTrue(np.isfinite(evaluation["test_mse_mean"]))

    def test_a_failing_configuration_is_logged_rather_than_crashing_the_run(self):
        bundle = _bundle(horizon=4)

        def exploding_factory(config):
            def runner(task, seed, train_steps):
                if config["bias"] == 0.0:
                    raise RuntimeError("synthetic optimiser divergence")
                return np.full(task.y_test.shape, config["bias"], dtype=np.float32)

            return runner

        entrant = driver.Entrant(
            entrant_id="flaky",
            family=driver.FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=({"bias": 0.0}, {"bias": 1.0}),
            factory=exploding_factory,
        )
        ledger = driver.AttemptLedger()
        selection = driver.select_configuration(
            entrant, bundle.selection_task, horizon=4, seeds=(0,), train_steps=1, ledger=ledger
        )

        self.assertEqual(selection["config"], {"bias": 1.0})
        self.assertTrue(any(row.get("error") for row in ledger.rows))

    def test_an_entrant_that_never_produces_a_finite_val_mse_is_refused_not_scored(self):
        bundle = _bundle(horizon=4)

        def always_explodes(config):
            def runner(task, seed, train_steps):
                raise RuntimeError("synthetic total failure")

            return runner

        entrant = driver.Entrant(
            entrant_id="doomed",
            family=driver.FAMILY_INCUMBENT,
            model_id=None,
            condition=None,
            pretrained=False,
            grid=({"bias": 0.0}, {"bias": 1.0}),
            factory=always_explodes,
        )
        ledger = driver.AttemptLedger()
        with self.assertRaises(driver.EntrantFailed):
            driver.select_configuration(
                entrant, bundle.selection_task, horizon=4, seeds=(0,), train_steps=1, ledger=ledger
            )
        # No test evaluation may follow a failed selection.
        self.assertEqual(ledger.n_test_evaluations, 0)

    def test_missing_staged_checkpoint_blocks_only_that_model(self):
        # A missing weights directory is a staging problem, not a large MSE: the factory
        # must surface AdapterUnavailable rather than let the entrant be scored badly.
        factory = driver._fm_factory(
            "cbramod", "frozen", Path("/nonexistent-weights-dir"), pretrained=True, device="cpu"
        )
        with self.assertRaises(driver.AdapterUnavailable):
            factory(
                {"lr": 1e-3, "head_hidden": 0, "epochs": 1, "batch_size": 4, "patience": 1}
            )

    def test_bootstrap_ci_is_seeded_and_brackets_the_mean(self):
        per_subject = {f"S{i}": 0.5 + 0.01 * i for i in range(27)}
        first = driver.bootstrap_subject_ci(per_subject, n_boot=1000, seed=0)
        second = driver.bootstrap_subject_ci(per_subject, n_boot=1000, seed=0)

        self.assertEqual(first, second)
        self.assertEqual(first["n_boot"], 1000)
        self.assertEqual(first["n_subjects"], 27)
        self.assertLessEqual(first["ci95_low"], first["mean"])
        self.assertLessEqual(first["mean"], first["ci95_high"])


class EntrantSlateTests(unittest.TestCase):
    def _entrants(self, **overrides):
        kwargs = dict(
            models=("cbramod",),
            conditions=("frozen", "full_ft"),
            weights_dir=Path("/tmp/weights"),
            include_random_twin=True,
            n_configs=driver.TUNING_GRID_SIZE,
            max_train_epochs=3,
            batch_size=8,
            patience=2,
            device="cpu",
        )
        kwargs.update(overrides)
        return driver.build_entrants(**kwargs)

    def test_incumbents_are_always_present_and_ranked(self):
        entrants, _ = self._entrants()
        by_id = {entrant.entrant_id: entrant for entrant in entrants}

        for incumbent in driver.INCUMBENT_IDS:
            self.assertIn(incumbent, by_id)
            self.assertTrue(by_id[incumbent].ranked)
            self.assertIsNone(by_id[incumbent].model_id)

    def test_random_twins_are_controls_not_competitors(self):
        entrants, _ = self._entrants()
        twins = [entrant for entrant in entrants if entrant.family == driver.FAMILY_RANDOM_TWIN]

        self.assertTrue(twins)
        for twin in twins:
            self.assertFalse(twin.pretrained)
            self.assertFalse(twin.ranked)
            self.assertTrue(twin.entrant_id.endswith("__random_twin"))

    def test_gradient_trained_entrants_share_the_configuration_count(self):
        entrants, _ = self._entrants()
        gradient = [
            entrant
            for entrant in entrants
            if entrant.entrant_id != "persistence"
        ]

        self.assertTrue(gradient)
        for entrant in gradient:
            self.assertEqual(entrant.configs_tried, driver.TUNING_GRID_SIZE)
        # Persistence is parameter free; the artifact says so instead of padding it.
        persistence = next(entrant for entrant in entrants if entrant.entrant_id == "persistence")
        self.assertEqual(persistence.configs_tried, 1)
        self.assertIn("parameter-free", persistence.budget_note)

    def test_no_tuned_entrant_has_an_axis_too_short_to_detect_an_edge(self):
        # A two-point axis makes every selection an endpoint, so plan section 3.4's
        # no-edge rule could neither be satisfied nor checked. Every tuned axis is >= 3.
        entrants, _ = self._entrants(models=driver.FM_MODEL_IDS)

        tuned = [entrant for entrant in entrants if entrant.axes]
        self.assertTrue(tuned)
        for entrant in tuned:
            for name, values in entrant.axes.items():
                self.assertGreaterEqual(len(values), 3, f"{entrant.entrant_id}:{name}")

    def test_the_dropped_cell_is_the_fixed_corner_not_a_per_entrant_choice(self):
        entrants, _ = self._entrants(models=driver.FM_MODEL_IDS)

        for entrant in entrants:
            if len(entrant.axes) < 2:
                continue
            with self.subTest(entrant=entrant.entrant_id):
                corner = {name: max(values) for name, values in entrant.axes.items()}
                self.assertEqual(entrant.configs_tried, driver.TUNING_GRID_SIZE)
                # Exactly the max-on-every-axis cell is missing...
                self.assertFalse(
                    [
                        point
                        for point in entrant.grid
                        if all(point[name] == value for name, value in corner.items())
                    ]
                )
                # ...and dropping a corner leaves every axis value present elsewhere, so
                # per-axis edge detection is untouched by the truncation.
                for name, values in entrant.axes.items():
                    self.assertEqual({point[name] for point in entrant.grid}, set(values))

    def test_a_grid_shape_that_misses_the_budget_is_refused_not_padded(self):
        with self.assertRaises(ValueError):
            driver._budget_grid({"lr": (1e-4, 1e-3), "hidden": (32, 128)})

    def test_entrant_log_carries_contamination_and_exposure_columns(self):
        entrants, _ = self._entrants(models=("eegpt",))
        eegpt = next(entrant for entrant in entrants if entrant.entrant_id == "eegpt__frozen")

        self.assertEqual(eegpt.as_log()["contamination_status"], "clean")
        self.assertTrue(eegpt.as_log()["model_selection_exposure"])

    def test_slate_excluded_checkpoint_is_not_addressable(self):
        with self.assertRaises(driver.AdapterUnavailable):
            driver.load_adapter("biot_six_datasets")


# ---------------------------------------------------------------------------
# Artifacts and gate
# ---------------------------------------------------------------------------


TEST_SUBJECTS = [f"T{index:03d}" for index in range(driver.F8_TEST_SUBJECTS)]
TRAIN_SUBJECTS = [f"R{index:03d}" for index in range(39)]
VAL_SUBJECTS = [f"V{index:03d}" for index in range(12)]

#: Per-subject errors are built as ``base + 0.001 * index`` for every entrant, so the
#: paired delta is a constant and the paired bootstrap CI is tight and excludes zero --
#: which is what a *powered* passing run looks like. The marginal per-entrant CIs of two
#: entrants overlap heavily under this construction, which is exactly why the gate keys off
#: the paired statistics and not off marginal overlap.
def _per_subject(base: float) -> dict[str, float]:
    return {subject: base + 0.001 * index for index, subject in enumerate(TEST_SUBJECTS)}


def _result_row(entrant_id, *, family, model_id, condition, base, horizon, ranked=True, configs=None):
    per_subject = _per_subject(base)
    values = sorted(per_subject.values())
    return {
        "entrant_id": entrant_id,
        "family": family,
        "model_id": model_id,
        "condition": condition,
        "ranked": ranked,
        "gradient_trained": entrant_id == "kahlus_gru" or model_id is not None,
        "configs_tried": driver.TUNING_GRID_SIZE if configs is None else configs,
        "contamination_status": "clean" if model_id else "not_applicable_no_pretraining",
        "model_selection_exposure": bool(model_id == "eegpt"),
        "horizon": int(horizon),
        "claim_eligible_protocol": bool(horizon >= 2),
        "val_mse": base - 0.01,
        "test_mse_mean": float(np.mean(values)),
        "test_mse_ci95_low": float(values[0]),
        "test_mse_ci95_high": float(values[-1]),
        "per_subject_test_mse": per_subject,
        "n_test_subjects": len(TEST_SUBJECTS),
        "seed_spread": 0.001,
        "grid_edge_selected": False,
        "axes_without_edge_detection": [],
    }


def _passing_results(horizons):
    """One row per (entrant, horizon) for a slate that satisfies every protocol check."""

    rows = []
    for horizon in horizons:
        rows.append(_result_row("persistence", family=driver.FAMILY_INCUMBENT, model_id=None, condition=None, base=0.60, horizon=horizon, configs=1))
        rows.append(_result_row("tuned_ridge_ar", family=driver.FAMILY_INCUMBENT, model_id=None, condition=None, base=0.30, horizon=horizon))
        rows.append(_result_row("gbm", family=driver.FAMILY_INCUMBENT, model_id=None, condition=None, base=0.35, horizon=horizon))
        rows.append(_result_row("kahlus_gru", family=driver.FAMILY_INCUMBENT, model_id=None, condition=None, base=0.20, horizon=horizon))
        for offset, model_id in enumerate(driver.FM_MODEL_IDS):
            for condition_offset, condition in enumerate(driver.CONDITIONS):
                base = 0.26 + 0.01 * offset + 0.005 * condition_offset
                rows.append(
                    _result_row(
                        f"{model_id}__{condition}",
                        family=driver.FAMILY_EXTERNAL,
                        model_id=model_id,
                        condition=condition,
                        base=base,
                        horizon=horizon,
                    )
                )
                rows.append(
                    _result_row(
                        f"{model_id}__{condition}__random_twin",
                        family=driver.FAMILY_RANDOM_TWIN,
                        model_id=model_id,
                        condition=condition,
                        base=base + 0.10,
                        horizon=horizon,
                        ranked=False,
                    )
                )
    return rows


def _passing_payload(*, horizons=None, n_boot=driver.MIN_CLAIMABLE_N_BOOT, **overrides):
    from neurotwin.benchmarks.fm_forecast.weights_manifest import manifest_rows

    horizons = list(driver.HORIZONS if horizons is None else horizons)
    results = _passing_results(horizons)
    ranked_ids = sorted({row["entrant_id"] for row in results if row["ranked"] and row["model_id"] is None})

    payload = {
        "schema": driver.ARTIFACT_SCHEMA,
        "created_utc": "2026-08-20T00:00:00+00:00",
        "dataset_id": driver.DATASET_ID,
        "lane": "lane_a_single_channel_fpz_cz",
        "config": {
            "smoke": False,
            "horizons": horizons,
            "seeds": [0, 1, 2],
            "n_boot": n_boot,
            "n_configs": driver.TUNING_GRID_SIZE,
            "closed_form_grid_truncated": False,
        },
        "environment": {"python": "3.11"},
        "determinism": {"fully_deterministic": True, "reason": ""},
        "git": {"commit": "deadbeef", "clean_worktree": True, "dirty_paths": []},
        "recordings": [
            {"record_id": "SC4001E0-PSG", "resolved_channel": driver.CHANNEL, "sampling_rate_hz": 100.0}
        ],
        "protocol": {"horizons": horizons},
        "normalization": driver.plan_normalization().as_log(),
        "split": {
            "seed": 0,
            "train": list(TRAIN_SUBJECTS),
            "val": list(VAL_SUBJECTS),
            "test": list(TEST_SUBJECTS),
        },
        "tuning_budget": {
            "n_configs_per_entrant": driver.TUNING_GRID_SIZE,
            "seeds": [0, 1, 2],
            "max_train_epochs": 50,
        },
        "tuning_budget_symmetry": {"symmetric": True, "per_entrant": {}},
        "entrants": [],
        "incumbent_ladder": driver.ladder_report(results),
        "results": results,
        "contamination_table": driver.contamination_dump(),
        "checkpoints": {
            "verified": [
                {
                    "model_id": model_id,
                    "verified": True,
                    "digest_matches": True,
                    "observed_sha256": "0" * 64,
                    "path": f"/tmp/weights/{model_id}.ckpt",
                    "load_report": {"missing_keys": []},
                }
                for model_id in driver.FM_MODEL_IDS
            ],
            "manifest_reference": manifest_rows(),
        },
        "controls": {
            "index_audit": [{"horizon": h, "passed": True, "windows_with_shared_samples": 0} for h in horizons],
            "label_shuffle": [
                {"control": "label_shuffle", "entrant_id": entrant_id, "horizon": horizons[-1], "passed": True}
                for entrant_id in ranked_ids
            ],
            "time_shift": [
                {"control": "time_shift", "entrant_id": entrant_id, "horizon": horizons[-1], "passed": True}
                for entrant_id in ranked_ids
            ],
            "harness_sanity": [
                {"control": "ar2_sanity", "passed": True},
                {"control": "synthetic_null", "passed": True},
            ],
            "random_init_twin": driver.random_twin_controls(results),
            "not_implemented": [],
        },
        "skipped_models": [],
        "unavailable_models": [],
        "failed_entrants": [],
        "attempts": {
            "run_id": "test-run",
            "n_attempts": 10,
            "n_test_evaluations": 5,
            "n_test_evaluations_cumulative": 5,
            "repeated_test_families": [],
            "log_file": "attempts.log",
        },
    }
    payload["paired_statistics"] = driver.build_paired_statistics(results, horizons, n_boot=n_boot)
    payload["ranking"] = driver.build_ranking(
        results, horizons, driver.paired_lookup(payload["paired_statistics"])
    )
    payload["protocol_completeness"] = driver.protocol_completeness(payload)
    payload["powered"] = driver._powered(payload)
    payload["beat_best_baseline"] = driver._beat_best_baseline(payload)
    payload["caveats"] = driver.report_caveats(payload)
    payload.update(overrides)
    return payload


class ContaminationDumpTests(unittest.TestCase):
    def test_dump_adds_the_model_selection_exposure_column(self):
        rows = driver.contamination_dump()

        self.assertTrue(rows)
        for row in rows:
            self.assertIn("model_selection_exposure", row)
            for key in ("model_id", "dataset_id", "status", "reason"):
                self.assertIn(key, row)
        exposed = {
            (row["model_id"], row["dataset_id"]) for row in rows if row["model_selection_exposure"]
        }
        self.assertIn(("eegpt", driver.DATASET_ID), exposed)


class RankingTests(unittest.TestCase):
    def test_every_model_condition_is_its_own_entrant_row(self):
        # The single-look rule (plan section 5.3) is enforced by never letting validation
        # pick *between* conditions: frozen and full_ft each spend their own 14-config
        # budget and each get exactly one test look, so both must appear in the table.
        payload = _passing_payload(horizons=[2])
        ranking = driver.build_ranking(payload["results"], [2])
        entrant_ids = [row["entrant_id"] for row in ranking[0]["rows"]]

        for model_id in driver.FM_MODEL_IDS:
            for condition in driver.CONDITIONS:
                self.assertIn(f"{model_id}__{condition}", entrant_ids)
        self.assertNotIn("eegpt__frozen__random_twin", entrant_ids)
        for row in ranking[0]["rows"]:
            self.assertEqual(
                row["condition_selected_on"],
                "not_selected__both_conditions_ranked_separately",
            )
            self.assertIn("model_selection_exposure", row)
            self.assertIn("contamination_status", row)

    def test_ranking_is_ordered_by_test_mse_and_carries_the_paired_verdict(self):
        payload = _passing_payload(horizons=[2], n_boot=200)
        rows = payload["ranking"][0]["rows"]

        means = [row["test_mse_mean"] for row in rows]
        self.assertEqual(means, sorted(means))
        self.assertEqual(rows[0]["entrant_id"], "kahlus_gru")
        external = next(row for row in rows if row["entrant_id"] == "eegpt__frozen")
        self.assertIsNotNone(external["paired_vs_reference"])
        self.assertTrue(external["paired_vs_reference"]["significant_after_holm"])
        self.assertFalse(external["paired_vs_reference"]["statistical_tie"])

    def test_random_twin_control_reports_the_pretraining_gap(self):
        results = _passing_payload(horizons=[2], n_boot=200)["results"]
        controls = driver.random_twin_controls(results)

        self.assertEqual(len(controls), len(driver.FM_MODEL_IDS) * len(driver.CONDITIONS))
        for control in controls:
            self.assertTrue(control["pretraining_helped"])
            self.assertTrue(control["passed"])
            self.assertAlmostEqual(control["gap_random_minus_pretrained"], 0.10, places=6)


class PairedStatisticsTests(unittest.TestCase):
    def test_wilcoxon_matches_known_exact_and_approximate_values(self):
        # Cross-checked against scipy.stats.wilcoxon; the driver keeps a scipy-free
        # implementation so the gate never depends on an optional dependency.
        exact = driver.wilcoxon_signed_rank([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, -0.5])
        self.assertIn("exact", exact["method"])
        self.assertAlmostEqual(exact["p_value"], 0.015625, places=9)
        self.assertAlmostEqual(exact["statistic"], 1.0, places=9)

        approx = driver.wilcoxon_signed_rank([0.1] * 27)
        self.assertIn("normal approximation", approx["method"])
        self.assertAlmostEqual(approx["statistic"], 0.0, places=9)
        self.assertLess(approx["p_value"], 1e-5)

    def test_zero_deltas_are_reported_not_silently_dropped(self):
        result = driver.wilcoxon_signed_rank([0.0, 0.0, 1.0, 2.0])

        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["n_zero_pairs"], 2)
        self.assertEqual(result["n_nonzero_pairs"], 2)

    def test_all_zero_deltas_are_an_undefined_test_not_a_pass(self):
        result = driver.wilcoxon_signed_rank([0.0, 0.0, 0.0])

        self.assertEqual(result["p_value"], 1.0)
        self.assertEqual(result["n_nonzero_pairs"], 0)

    def test_holm_is_monotone_and_never_shrinks_a_p_value(self):
        adjusted = driver.holm_adjust([0.01, 0.02, 0.03])

        self.assertEqual(adjusted, sorted(adjusted))
        for raw, corrected in zip([0.01, 0.02, 0.03], adjusted):
            self.assertGreaterEqual(corrected, raw)
        self.assertAlmostEqual(adjusted[0], 0.03, places=9)

    def test_paired_bootstrap_uses_common_random_numbers(self):
        rng = np.random.default_rng(0)
        index = rng.integers(0, 27, size=(500, 27))
        deltas = [0.05 + 0.001 * i for i in range(27)]

        first = driver.paired_bootstrap(deltas, index)
        second = driver.paired_bootstrap(deltas, index)

        self.assertEqual(first, second)
        self.assertGreater(first["ci95_low"], 0.0)

    def test_paired_statistics_only_cover_claim_eligible_horizons(self):
        payload = _passing_payload(horizons=[1, 2], n_boot=200)
        blocks = payload["paired_statistics"]["by_horizon"]

        self.assertEqual({block["horizon"] for block in blocks}, {1, 2})
        by_horizon = {block["horizon"]: block for block in blocks}
        self.assertFalse(by_horizon[1]["claim_eligible"])
        self.assertTrue(by_horizon[2]["claim_eligible"])
        for comparison in by_horizon[1]["comparisons"]:
            self.assertFalse(comparison["significant_after_holm"])


class HarnessSanityTests(unittest.TestCase):
    def test_the_harness_finds_ar2_structure_and_finds_nothing_in_noise(self):
        ledger = driver.AttemptLedger()
        rows = {row["control"]: row for row in driver.harness_sanity_controls(ledger=ledger)}

        self.assertTrue(rows["ar2_sanity"]["passed"])
        self.assertTrue(rows["synthetic_null"]["passed"])
        # The preregistered margin must sit well inside the theoretical ceiling, or a
        # healthy harness would fail the control for arithmetic reasons.
        self.assertLess(
            driver.AR2_SANITY_MIN_IMPROVEMENT,
            rows["ar2_sanity"]["theoretical_max_improvement"] - 0.2,
        )
        self.assertLess(rows["synthetic_null"]["improvement"], driver.SYNTHETIC_NULL_MAX_IMPROVEMENT)
        self.assertTrue(any(row["phase"] == "control" for row in ledger.rows))

    def test_every_unimplemented_plan_control_is_named(self):
        rows = driver.unimplemented_controls()
        named = {row["control"] for row in rows}

        self.assertEqual(named, set(driver.PLAN_SECTION_4_CONTROLS) - set(driver.IMPLEMENTED_CONTROLS))
        for row in rows:
            self.assertTrue(row["reason"])


class AxisEdgeTests(unittest.TestCase):
    def test_two_point_axis_is_reported_as_undetectable_not_as_a_pass(self):
        report = driver.axis_edge_report(
            {"lr": (1e-4, 1e-3, 1e-2), "head_hidden": (0, 64)},
            {"lr": 1e-3, "head_hidden": 0},
        )

        self.assertEqual(report["axes_without_edge_detection"], ["head_hidden"])
        self.assertIn(driver.AXIS_TOO_SMALL_NOTE, report["per_axis"]["head_hidden"]["note"])
        self.assertFalse(report["per_axis"]["lr"]["at_edge"])
        self.assertFalse(report["any_axis_at_edge"])

    def test_interior_and_edge_selections_are_distinguished(self):
        axes = {"lr": (1e-4, 1e-3, 1e-2)}

        self.assertFalse(driver.axis_edge_report(axes, {"lr": 1e-3})["any_axis_at_edge"])
        self.assertTrue(driver.axis_edge_report(axes, {"lr": 1e-4})["any_axis_at_edge"])
        self.assertTrue(driver.axis_edge_report(axes, {"lr": 1e-2})["any_axis_at_edge"])


class DataStrictnessTests(unittest.TestCase):
    def test_exact_alias_resolves(self):
        self.assertEqual(driver.resolve_channel(["EEG Fpz-Cz", "EEG Pz-Oz"]), "EEG Fpz-Cz")

    def test_a_near_miss_channel_is_a_hard_error_not_a_fallback(self):
        with self.assertRaises(driver.DataStrictnessError):
            driver.resolve_channel(["EEG Pz-Oz", "EOG horizontal"])

    def test_ambiguous_aliases_are_refused(self):
        with self.assertRaises(driver.DataStrictnessError):
            driver.resolve_channel(["EEG Fpz-Cz", "Fpz-Cz"])


class LedgerTests(unittest.TestCase):
    def test_ledger_appends_across_invocations_and_carries_a_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = driver.open_ledger(out, run_id="run-a", planned_families=set(), allow_repeat_test=False)
            first.record(phase="test", entrant_id="stub", horizon=2)
            first.write(out / "attempts.log")

            second = driver.open_ledger(out, run_id="run-b", planned_families=set(), allow_repeat_test=False)
            second.record(phase="validation", entrant_id="stub", horizon=2)
            second.write(out / "attempts.log")

            lines = [
                json.loads(line)
                for line in (out / "attempts.log").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(lines), 2)
        self.assertEqual([line["run_id"] for line in lines], ["run-a", "run-b"])

    def test_a_second_test_look_at_the_same_family_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = driver.open_ledger(out, run_id="run-a", planned_families=set(), allow_repeat_test=False)
            first.record(phase="test", entrant_id="stub", horizon=2)
            first.write(out / "attempts.log")

            planned = {driver._family_key("stub", 2)}
            with self.assertRaises(driver.RepeatTestEvaluation):
                driver.open_ledger(out, run_id="run-b", planned_families=planned, allow_repeat_test=False)

            allowed = driver.open_ledger(
                out, run_id="run-b", planned_families=planned, allow_repeat_test=True
            )
            allowed.record(phase="test", entrant_id="stub", horizon=2)

        self.assertEqual(allowed.n_test_evaluations_cumulative, 2)
        self.assertEqual(allowed.repeated_test_families(), [driver._family_key("stub", 2)])


class GateDossierTests(unittest.TestCase):
    def test_dossier_passes_under_the_new_scope_when_every_check_is_true(self):
        gate = driver.build_gate_dossier(_passing_payload())

        self.assertEqual(gate["claim_scope"], driver.CLAIM_SCOPE)
        self.assertEqual(gate["failure_reasons"], [])
        self.assertTrue(gate["scientific_claim_allowed"])
        for key in (
            "index_audit_passed",
            "normalization_causal",
            "label_shuffle_collapsed",
            "random_twin_control_present",
            "contamination_audit_passed",
            "incumbent_ladder_complete",
            "tuning_budget_symmetric",
            "controls_pass",
            "human_sign_off",
            "stop_reason",
        ):
            self.assertIn(key, gate)

    def test_dossier_fails_closed_on_each_precondition(self):
        controls = _passing_payload(horizons=[2], n_boot=200)["controls"]
        cases = {
            "legacy normalisation": (
                {"normalization": driver.plan_normalization(legacy=True).as_log()},
                "transductive",
            ),
            "index audit": (
                {"controls": {**controls, "index_audit": [{"horizon": 2, "passed": False}]}},
                "index audit",
            ),
            "label shuffle": (
                {
                    "controls": {
                        **controls,
                        "label_shuffle": [{"entrant_id": "tuned_ridge_ar", "horizon": 2, "passed": False}],
                    }
                },
                "label-shuffle",
            ),
            "time shift": (
                {
                    "controls": {
                        **controls,
                        "time_shift": [{"entrant_id": "tuned_ridge_ar", "horizon": 2, "passed": False}],
                    }
                },
                "time-shift",
            ),
            "harness sanity": (
                {"controls": {**controls, "harness_sanity": [{"control": "ar2_sanity", "passed": False}]}},
                "harness sanity",
            ),
            "missing random twin": ({"controls": {**controls, "random_init_twin": []}}, "random-init"),
            "unimplemented control": (
                {
                    "controls": {
                        **controls,
                        "not_implemented": [{"control": "subject_identity_probe", "reason": "x"}],
                    }
                },
                "control not implemented: subject_identity_probe",
            ),
            "unavailable model": (
                {"unavailable_models": [{"model_id": "labram_base", "reason": "no vendor"}]},
                "unavailable",
            ),
            "failed entrant": (
                {"failed_entrants": [{"entrant_id": "labram_base__full_ft", "horizon": 2, "reason": "diverged"}]},
                "no finite validation MSE",
            ),
            "dirty worktree": (
                {"git": {"commit": "x", "clean_worktree": False, "dirty_paths": ["a"]}},
                "dirty worktree",
            ),
            "overlapping split": (
                {"split": {"seed": 0, "train": ["a"], "val": ["a"], "test": ["b"]}},
                None,
            ),
            "asymmetric budget": (
                {"tuning_budget_symmetry": {"symmetric": False, "per_entrant": {}}},
                "symmetric",
            ),
            "non-deterministic": (
                {"determinism": {"fully_deterministic": False, "reason": "cuda unavailable"}},
                "determinism",
            ),
            "unverified checkpoint": (
                {"checkpoints": {"verified": [], "manifest_reference": []}},
                "checkpoint provenance",
            ),
            "repeated test look": (
                {
                    "attempts": {
                        "run_id": "r",
                        "n_test_evaluations_cumulative": 2,
                        "repeated_test_families": ["eegpt__frozen@h2"],
                    }
                },
                "test-evaluated more than once",
            ),
            "underpowered": (
                {"powered": {"powered": False, "reasons": ["too few subjects"]}},
                "not powered",
            ),
            "does not beat the best baseline": (
                {"beat_best_baseline": {"beat_best_baseline": False, "reasons": []}},
                "best external entrant",
            ),
            "smoke run": (
                {"protocol_completeness": {"non_smoke": False}},
                "smoke run",
            ),
        }
        for name, (override, expected) in cases.items():
            with self.subTest(case=name):
                gate = driver.build_gate_dossier(
                    _passing_payload(horizons=[2], n_boot=200, **override)
                )
                self.assertFalse(gate["scientific_claim_allowed"], name)
                self.assertTrue(gate["failure_reasons"], name)
                if expected is not None:
                    self.assertTrue(
                        any(expected in reason for reason in gate["failure_reasons"]),
                        f"{name}: {gate['failure_reasons']}",
                    )

    def test_a_two_point_axis_is_flagged_rather_than_quietly_accepted(self):
        payload = _passing_payload(horizons=[2], n_boot=200)
        for row in payload["results"]:
            if row["entrant_id"] == "eegpt__frozen":
                row["axes_without_edge_detection"] = ["head_hidden"]

        gate = driver.build_gate_dossier(payload)
        self.assertEqual(gate["grid_edge_undetectable_axes"], ["eegpt__frozen:head_hidden"])
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(
            any("two-point axis" in reason for reason in gate["failure_reasons"])
        )

    def test_a_random_twin_that_matches_its_pretrained_model_blocks_the_claim(self):
        payload = _passing_payload(horizons=[2], n_boot=200)
        controls = list(payload["controls"]["random_init_twin"])
        controls[0] = {**controls[0], "pretraining_helped": False, "passed": False}
        payload["controls"] = {**payload["controls"], "random_init_twin": controls}

        gate = driver.build_gate_dossier(payload)
        self.assertFalse(gate["pretraining_helped_every_fm"])
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(
            any("pretraining bought nothing" in reason for reason in gate["failure_reasons"])
        )

    def test_a_missing_model_or_condition_is_a_protocol_failure(self):
        payload = _passing_payload(horizons=[2], n_boot=200)
        payload["results"] = [
            row for row in payload["results"] if row["entrant_id"] != "biot_prest16__full_ft"
        ]
        payload["protocol_completeness"] = driver.protocol_completeness(payload)

        gate = driver.build_gate_dossier(payload)
        self.assertFalse(gate["protocol_ran_as_declared"])
        self.assertTrue(
            any("both admissible conditions" in reason for reason in gate["failure_reasons"])
        )

    def test_incomplete_incumbent_ladder_blocks_the_claim(self):
        payload = _passing_payload(horizons=[2], n_boot=200)
        payload["results"] = [row for row in payload["results"] if row["entrant_id"] != "gbm"]
        payload["incumbent_ladder"] = driver.ladder_report(payload["results"])

        gate = driver.build_gate_dossier(payload)
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("incumbent ladder" in reason for reason in gate["failure_reasons"]))

    def test_contaminated_entrant_blocks_the_claim(self):
        payload = _passing_payload(horizons=[2], n_boot=200)
        for row in payload["results"]:
            if row["entrant_id"] == "eegpt__frozen":
                row["contamination_status"] = "unverified"

        gate = driver.build_gate_dossier(payload)
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("clean" in reason for reason in gate["failure_reasons"]))

    def test_a_smoke_run_can_never_be_claim_bearing(self):
        payload = _passing_payload(horizons=[1, 2], n_boot=driver.SMOKE_N_BOOT)
        payload["config"] = {**payload["config"], "smoke": True}
        payload["protocol_completeness"] = driver.protocol_completeness(payload)

        gate = driver.build_gate_dossier(payload)
        reasons = " | ".join(gate["failure_reasons"])
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn("smoke run", reasons)
        self.assertIn("bootstrap draws below", reasons)
        self.assertIn("horizon grid", reasons)


class ArtifactWriterTests(unittest.TestCase):
    def test_writer_emits_the_full_artifact_set(self):
        payload = _passing_payload(horizons=[1, 2], n_boot=200)
        payload["gate"] = driver.build_gate_dossier(payload)
        ledger = driver.AttemptLedger()
        ledger.record(phase="validation", entrant_id="stub", horizon=1, val_mse=0.1)
        ledger.record(phase="test", entrant_id="stub", horizon=1, test_mse_mean=0.1)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            written = driver.write_artifacts(out, payload, ledger)

            for name in (
                "benchmark_json",
                "gate_json",
                "contamination_json",
                "checkpoints_json",
                "controls_json",
                "environment_json",
                "git_state_json",
                "attempts_log",
                "paired_statistics_json",
                "report_md",
            ):
                self.assertIn(name, written)
                self.assertTrue(Path(written[name]).is_file(), name)

            document = json.loads((out / "eeg_fm_benchmark.json").read_text(encoding="utf-8"))
            gate = json.loads((out / "eeg_fm_benchmark_gate.json").read_text(encoding="utf-8"))
            report = (out / "EEG_FM_BENCHMARK_REPORT.md").read_text(encoding="utf-8")
            attempts = [
                json.loads(line)
                for line in (out / "attempts.log").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        for key in (
            "schema",
            "created_utc",
            "dataset_id",
            "config",
            "environment",
            "git",
            "protocol",
            "normalization",
            "split",
            "tuning_budget",
            "results",
            "ranking",
            "contamination_table",
            "checkpoints",
            "controls",
            "attempts",
            "gate",
            "determinism",
            "paired_statistics",
            "protocol_completeness",
            "powered",
            "beat_best_baseline",
            "incumbent_ladder",
            "tuning_budget_symmetry",
            "recordings",
            "caveats",
        ):
            self.assertIn(key, document)

        self.assertEqual(document["schema"], driver.ARTIFACT_SCHEMA)
        self.assertIn("model_selection_exposure", document["contamination_table"][0])
        self.assertIn("model_selection_exposure", document["ranking"][0]["rows"][0])
        # Expected-manifest rows stay available, but the claim-bearing block is what was
        # actually loaded off disk during this run.
        self.assertIn("sha256", document["checkpoints"]["manifest_reference"][0])
        self.assertIn("observed_sha256", document["checkpoints"]["verified"][0])
        self.assertEqual(gate["claim_scope"], driver.CLAIM_SCOPE)
        self.assertEqual(len(attempts), 2)
        self.assertIn("Claim scope", report)
        self.assertIn(driver.CLAIM_SCOPE, report)
        self.assertIn("nothing clinical", report)

    def test_report_documents_the_statistics_and_the_caveats(self):
        payload = _passing_payload(horizons=[1, 2], n_boot=200)
        payload["gate"] = driver.build_gate_dossier(payload)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            driver.write_artifacts(out, payload, driver.AttemptLedger())
            report = (out / "EEG_FM_BENCHMARK_REPORT.md").read_text(encoding="utf-8")

        # A reader must not be able to eyeball ranks off the overlapping marginal CIs.
        self.assertIn("Caveats", report)
        self.assertIn("percentile", report)
        self.assertIn("Holm", report)
        self.assertIn("paired", report.lower())
        self.assertIn("Controls", report)
        for fragment in (
            "model-selection exposure",
            "FP1-F7",
            "fc_norm",
            "seed variance",
        ):
            self.assertIn(fragment, report)


if __name__ == "__main__":
    unittest.main()

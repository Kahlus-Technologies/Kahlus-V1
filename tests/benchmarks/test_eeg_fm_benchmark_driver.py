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


def _passing_payload(**overrides):
    from neurotwin.benchmarks.fm_forecast.weights_manifest import manifest_rows

    results = [
        {
            "entrant_id": "kahlus_gru",
            "family": driver.FAMILY_INCUMBENT,
            "model_id": None,
            "condition": None,
            "ranked": True,
            "configs_tried": driver.TUNING_GRID_SIZE,
            "contamination_status": "not_applicable_no_pretraining",
            "model_selection_exposure": False,
            "horizon": 1,
            "val_mse": 0.10,
            "test_mse_mean": 0.11,
            "test_mse_ci95_low": 0.10,
            "test_mse_ci95_high": 0.12,
            "grid_edge_selected": False,
        },
        {
            "entrant_id": "tuned_ridge_ar",
            "family": driver.FAMILY_INCUMBENT,
            "model_id": None,
            "condition": None,
            "ranked": True,
            "configs_tried": driver.TUNING_GRID_SIZE,
            "contamination_status": "not_applicable_no_pretraining",
            "model_selection_exposure": False,
            "horizon": 1,
            "val_mse": 0.12,
            "test_mse_mean": 0.13,
            "test_mse_ci95_low": 0.12,
            "test_mse_ci95_high": 0.14,
            "grid_edge_selected": False,
        },
        {
            "entrant_id": "persistence",
            "family": driver.FAMILY_INCUMBENT,
            "model_id": None,
            "condition": None,
            "ranked": True,
            "configs_tried": 1,
            "contamination_status": "not_applicable_no_pretraining",
            "model_selection_exposure": False,
            "horizon": 1,
            "val_mse": 0.30,
            "test_mse_mean": 0.31,
            "test_mse_ci95_low": 0.30,
            "test_mse_ci95_high": 0.32,
            "grid_edge_selected": False,
        },
        {
            "entrant_id": "eegpt__frozen",
            "family": driver.FAMILY_EXTERNAL,
            "model_id": "eegpt",
            "condition": "frozen",
            "ranked": True,
            "configs_tried": driver.TUNING_GRID_SIZE,
            "contamination_status": "clean",
            "model_selection_exposure": True,
            "horizon": 1,
            "val_mse": 0.20,
            "test_mse_mean": 0.21,
            "test_mse_ci95_low": 0.20,
            "test_mse_ci95_high": 0.22,
            "grid_edge_selected": False,
        },
        {
            "entrant_id": "eegpt__frozen__random_twin",
            "family": driver.FAMILY_RANDOM_TWIN,
            "model_id": "eegpt",
            "condition": "frozen",
            "ranked": False,
            "configs_tried": driver.TUNING_GRID_SIZE,
            "contamination_status": "clean",
            "model_selection_exposure": True,
            "horizon": 1,
            "val_mse": 0.40,
            "test_mse_mean": 0.41,
            "test_mse_ci95_low": 0.40,
            "test_mse_ci95_high": 0.42,
            "grid_edge_selected": False,
        },
    ]
    payload = {
        "schema": driver.ARTIFACT_SCHEMA,
        "created_utc": "2026-08-20T00:00:00+00:00",
        "dataset_id": driver.DATASET_ID,
        "lane": "lane_a_single_channel_fpz_cz",
        "config": {"smoke": False},
        "environment": {"python": "3.11"},
        "git": {"commit": "deadbeef", "clean_worktree": True, "dirty_paths": []},
        "protocol": {"horizons": [1]},
        "normalization": driver.plan_normalization().as_log(),
        "split": {"seed": 0, "train": ["a", "b"], "val": ["c"], "test": ["d", "e"]},
        "tuning_budget": {"n_configs_per_entrant": driver.TUNING_GRID_SIZE, "seeds": [0], "max_train_epochs": 2},
        "entrants": [],
        "results": results,
        "contamination_table": driver.contamination_dump(),
        "checkpoints": manifest_rows(),
        "controls": {
            "index_audit": [{"horizon": 1, "passed": True, "windows_with_shared_samples": 0}],
            "label_shuffle": [{"entrant_id": "tuned_ridge_ar", "horizon": 1, "passed": True}],
            "random_init_twin": driver.random_twin_controls(results),
        },
        "skipped_models": [],
        "unavailable_models": [],
        "failed_entrants": [],
        "attempts": {"n_attempts": 10, "n_test_evaluations": 5, "log_file": "attempts.log"},
    }
    payload["ranking"] = driver.build_ranking(results, [1])
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
    def test_ranking_uses_validation_to_pick_a_model_condition(self):
        results = _passing_payload()["results"] + [
            {
                "entrant_id": "eegpt__full_ft",
                "family": driver.FAMILY_EXTERNAL,
                "model_id": "eegpt",
                "condition": "full_ft",
                "ranked": True,
                "configs_tried": driver.TUNING_GRID_SIZE,
                "contamination_status": "clean",
                "model_selection_exposure": True,
                "horizon": 1,
                # Worse on validation but better on test: must NOT be the selected row,
                # because selecting on test would be tuning on test.
                "val_mse": 0.25,
                "test_mse_mean": 0.05,
                "test_mse_ci95_low": 0.04,
                "test_mse_ci95_high": 0.06,
                "grid_edge_selected": False,
            }
        ]
        ranking = driver.build_ranking(results, [1])
        entrant_ids = [row["entrant_id"] for row in ranking[0]["rows"]]

        self.assertIn("eegpt__frozen", entrant_ids)
        self.assertNotIn("eegpt__full_ft", entrant_ids)
        self.assertNotIn("eegpt__frozen__random_twin", entrant_ids)
        for row in ranking[0]["rows"]:
            self.assertEqual(row["condition_selected_on"], "validation_mse")
            self.assertIn("model_selection_exposure", row)
            self.assertIn("contamination_status", row)

    def test_random_twin_control_reports_the_pretraining_gap(self):
        controls = driver.random_twin_controls(_passing_payload()["results"])

        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["model_id"], "eegpt")
        self.assertTrue(controls[0]["pretraining_helped"])
        self.assertAlmostEqual(controls[0]["gap_random_minus_pretrained"], 0.20, places=6)


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
        cases = {
            "legacy normalisation": {
                "normalization": driver.plan_normalization(legacy=True).as_log()
            },
            "index audit": {
                "controls": {
                    **_passing_payload()["controls"],
                    "index_audit": [{"horizon": 1, "passed": False}],
                }
            },
            "label shuffle": {
                "controls": {
                    **_passing_payload()["controls"],
                    "label_shuffle": [{"entrant_id": "tuned_ridge_ar", "horizon": 1, "passed": False}],
                }
            },
            "missing random twin": {
                "controls": {**_passing_payload()["controls"], "random_init_twin": []}
            },
            "unavailable model": {"unavailable_models": [{"model_id": "labram_base", "reason": "no vendor"}]},
            "failed entrant": {
                "failed_entrants": [{"entrant_id": "labram_base__full_ft", "horizon": 1, "reason": "diverged"}]
            },
            "dirty worktree": {"git": {"commit": "x", "clean_worktree": False, "dirty_paths": ["a"]}},
            "overlapping split": {
                "split": {"seed": 0, "train": ["a"], "val": ["a"], "test": ["b"]}
            },
        }
        for name, override in cases.items():
            with self.subTest(case=name):
                gate = driver.build_gate_dossier(_passing_payload(**override))
                self.assertFalse(gate["scientific_claim_allowed"], name)
                self.assertTrue(gate["failure_reasons"], name)

    def test_incomplete_incumbent_ladder_blocks_the_claim(self):
        payload = _passing_payload()
        payload["results"] = [row for row in payload["results"] if row["entrant_id"] != "persistence"]
        payload["ranking"] = driver.build_ranking(payload["results"], [1])

        gate = driver.build_gate_dossier(payload)
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("incumbent ladder" in reason for reason in gate["failure_reasons"]))

    def test_contaminated_entrant_blocks_the_claim(self):
        payload = _passing_payload()
        for row in payload["results"]:
            if row["entrant_id"] == "eegpt__frozen":
                row["contamination_status"] = "unverified"
        payload["ranking"] = driver.build_ranking(payload["results"], [1])

        gate = driver.build_gate_dossier(payload)
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("clean" in reason for reason in gate["failure_reasons"]))


class ArtifactWriterTests(unittest.TestCase):
    def test_writer_emits_the_full_artifact_set(self):
        payload = _passing_payload()
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
        ):
            self.assertIn(key, document)

        self.assertEqual(document["schema"], driver.ARTIFACT_SCHEMA)
        self.assertIn("model_selection_exposure", document["contamination_table"][0])
        self.assertIn("model_selection_exposure", document["ranking"][0]["rows"][0])
        self.assertIn("sha256", document["checkpoints"][0])
        self.assertEqual(gate["claim_scope"], driver.CLAIM_SCOPE)
        self.assertEqual(len(attempts), 2)
        self.assertIn("Claim scope", report)
        self.assertIn(driver.CLAIM_SCOPE, report)
        self.assertIn("nothing clinical", report)


if __name__ == "__main__":
    unittest.main()

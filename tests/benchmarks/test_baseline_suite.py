import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from neurotwin.benchmarks.baseline_catalog import baseline_catalog_rows
from neurotwin.benchmarks.baseline_suite import (
    BASELINE_CATALOG,
    BrainVistaStyleConfig,
    EXECUTABLE_BASELINE_RUNNERS,
    SupervisedWindowTask,
    _causal_stimulus_features,
    run_supervised_window_tasks,
    run_synthetic_baseline_suite,
)
from neurotwin.eval.paper_gate import (
    effective_scientific_claim_allowed,
    effective_scientific_claim_allowed_for_run,
    load_run_summary,
    paper_mode_gate_allows_claim,
    validate_paper_mode_payload,
)
from neurotwin.eval.claim_contracts import claim_contract_sha256, collect_task_claim_contracts
from neurotwin.eval.forecast_eligibility import forecast_eligibility_sha256
from neurotwin.models.baselines import NumpyRidgeBaseline
from neurotwin.models.tribe_style import TribeStyleModel, TribeStyleStimulusInput
from tests.forecast_eligibility_fixtures import build_bound_forecast_eligibility


def _valid_forecast_eligibility_artifact() -> dict[str, object]:
    with tempfile.TemporaryDirectory() as tmp:
        return build_bound_forecast_eligibility(Path(tmp))


class BaselineSuiteTests(unittest.TestCase):
    def test_serialized_paper_gate_cannot_bypass_bound_task_and_eligibility_artifacts(self):
        task_payload = {"tasks": {"future_state_forecasting": {}}}
        contracts, unknown = collect_task_claim_contracts(task_payload)
        self.assertFalse(unknown)
        eligibility = _valid_forecast_eligibility_artifact()
        gate = {
            "passed": True,
            "require_ci": True,
            "violations": [],
            "required_seeds": [0, 1, 2],
            "observed_seeds": [0, 1, 2],
            "claim_contract_sha256": claim_contract_sha256(contracts),
            "forecast_eligibility_required": True,
            "forecast_eligibility_passed": True,
            "forecast_eligibility_sha256": forecast_eligibility_sha256(eligibility),
        }

        self.assertTrue(
            paper_mode_gate_allows_claim(
                gate,
                task_payload=task_payload,
                forecast_eligibility=eligibility,
            )
        )
        self.assertFalse(paper_mode_gate_allows_claim(gate, task_payload=task_payload))
        self.assertFalse(
            paper_mode_gate_allows_claim(
                {**gate, "forecast_eligibility_required": False},
                task_payload={"tasks": {"oracle_conditional_stable_sleep_transition": {}}},
            )
        )

    def test_synthetic_baseline_suite_runs_all_required_models(self):
        payload = run_synthetic_baseline_suite(seed=4, train_steps=1)

        self.assertEqual(payload["scope"]["status"], "synthetic-only")
        future = payload["tasks"]["future_state_forecasting"]
        self.assertEqual(future["status"], "completed")
        self.assertIn("persistence", future["metrics_by_model"])
        self.assertIn("train_mean", future["metrics_by_model"])
        self.assertIn("random_permutation", future["metrics_by_model"])
        self.assertIn("linear_ridge", future["metrics_by_model"])
        self.assertIn("autoregressive_ridge", future["metrics_by_model"])
        self.assertIn("neurotwin", future["metrics_by_model"])
        self.assertIn("mse_ci_low", future["metrics_by_model"]["linear_ridge"])
        self.assertIn("mse_ci_high", future["metrics_by_model"]["linear_ridge"])
        self.assertTrue(payload["aggregate"]["aggregate_rank"])
        self.assertIn("baseline_catalog", payload)
        catalog_ids = {row["model_id"] for row in payload["baseline_catalog"]}
        self.assertIn("persistence", catalog_ids)
        self.assertIn("train_mean", catalog_ids)
        self.assertIn("random_permutation", catalog_ids)
        self.assertIn("autoregressive_ridge", catalog_ids)
        self.assertEqual(payload["seeds"], [4])
        self.assertEqual(payload["benchmark_contract"]["required_seeds"], [0, 1, 2])

    def test_ridge_baseline_is_finite_for_ill_conditioned_windows(self):
        x = np.ones((12, 3), dtype=np.float64)
        x[:, 1] = np.linspace(0.0, 1e12, 12)
        x[:, 2] = x[:, 1] + 1e-6
        y = np.stack([x[:, 1] * 0.5, x[:, 2] * -0.25], axis=1)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = NumpyRidgeBaseline(alpha=1e-2).fit(x, y)
            pred = model.predict(x)

        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(pred.shape, y.shape)

    def test_failed_baseline_is_recorded_and_excluded_from_ranking(self):
        x = np.random.default_rng(0).normal(size=(6, 4, 2)).astype(np.float32)
        task = SupervisedWindowTask(
            task_id="future_state_forecasting",
            source_modality="eeg",
            target_modality="eeg",
            x_train=x[:4],
            y_train=x[:4],
            x_test=x[4:],
            y_test=x[4:],
        )

        with mock.patch(
            "neurotwin.benchmarks.baseline_suite._fit_ridge",
            return_value=np.full_like(task.y_test, np.nan),
        ):
            payload = run_supervised_window_tasks((task,), seed=0, train_steps=1)

        failures = payload["baseline_failures"]
        ranked = {row["model_id"] for row in payload["tasks"]["future_state_forecasting"]["ranking"]}
        self.assertTrue(any(row["model_id"] == "linear_ridge" for row in failures))
        self.assertNotIn("linear_ridge", ranked)

    def test_filtered_baselines_fail_for_unknown_model_id(self):
        x = np.random.default_rng(0).normal(size=(6, 4, 2)).astype(np.float32)
        task = SupervisedWindowTask(
            task_id="future_state_forecasting",
            source_modality="eeg",
            target_modality="eeg",
            x_train=x[:4],
            y_train=x[:4],
            x_test=x[4:],
            y_test=x[4:],
        )

        payload = run_supervised_window_tasks((task,), seed=0, train_steps=1, model_ids=("not_a_model",))
        result = payload["tasks"]["future_state_forecasting"]

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["metrics_by_model"])
        self.assertTrue(any(row["model_id"] == "not_a_model" and "unknown" in row["reason"] for row in result["failures"]))
        self.assertTrue(any(row["model_id"] == "not_a_model" for row in payload["baseline_failures"]))

    def test_filtered_baselines_record_unavailable_model_id(self):
        x = np.random.default_rng(0).normal(size=(6, 4, 2)).astype(np.float32)
        task = SupervisedWindowTask(
            task_id="future_state_forecasting",
            source_modality="eeg",
            target_modality="eeg",
            x_train=x[:4],
            y_train=x[:4],
            x_test=x[4:],
            y_test=x[4:],
        )

        payload = run_supervised_window_tasks((task,), seed=0, train_steps=1, model_ids=("tribe_style",))
        result = payload["tasks"]["future_state_forecasting"]

        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["metrics_by_model"])
        self.assertTrue(any(row["model_id"] == "tribe_style" and "unavailable" in row["reason"] for row in result["failures"]))

    def test_tribe_style_runs_on_stimulus_fmri_task(self):
        rng = np.random.default_rng(7)
        x_train = rng.normal(size=(10, 4, 3)).astype(np.float32)
        y_train = rng.normal(size=(10, 4, 5)).astype(np.float32)
        x_test = rng.normal(size=(4, 4, 3)).astype(np.float32)
        y_test = rng.normal(size=(4, 4, 5)).astype(np.float32)
        task = SupervisedWindowTask(
            task_id="stimulus_to_fmri_response",
            source_modality="stimulus",
            target_modality="fmri",
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            notes=("tribe_style is a clean_room_approximation",),
        )

        payload = run_supervised_window_tasks((task,), seed=0, train_steps=1)
        result = payload["tasks"]["stimulus_to_fmri_response"]
        metrics = result["metrics_by_model"]["tribe_style"]
        catalog = {row["model_id"]: row for row in payload["baseline_catalog"]}

        self.assertIn("mse_ci_low", metrics)
        self.assertIn("mse_ci_high", metrics)
        self.assertTrue(np.isfinite(metrics["mse"]))
        self.assertIn("tribe_style", {row["model_id"] for row in result["ranking"]})
        self.assertEqual(catalog["tribe_style"]["status"], "clean_room_approximation")
        self.assertIn("not an exact TRIBE v2 reproduction", catalog["tribe_style"]["notes"])
        self.assertIn("brainvista_style", result["metrics_by_model"])
        self.assertIn("pair_operator", result["metrics_by_model"])
        self.assertIn("tribe_style_clean_room", result["metrics_by_model"])
        self.assertIn("pair_operator_no_pair_state", result["metrics_by_model"])

    def test_baseline_catalog_is_separate_from_executable_runners(self):
        catalog_ids = {entry.model_id for entry in BASELINE_CATALOG}
        runner_ids = {runner.model_id for runner in EXECUTABLE_BASELINE_RUNNERS}

        self.assertIn("brainvista_style", catalog_ids)
        self.assertIn("brain_of_style", catalog_ids)
        self.assertIn("tribe_style_clean_room", catalog_ids)
        self.assertIn("pair_operator_no_pair_state", catalog_ids)
        self.assertIn("brainvista_style", runner_ids)
        self.assertNotIn("brain_of_style", runner_ids)
        self.assertIn("tribe_style", runner_ids)
        self.assertIn("tribe_style_clean_room", runner_ids)
        self.assertIn("pair_operator", runner_ids)
        self.assertIn("pair_operator_no_pair_state", runner_ids)

    def test_every_executable_runner_has_a_catalog_entry(self):
        catalog_ids = {entry.model_id for entry in BASELINE_CATALOG}
        runner_ids = {runner.model_id for runner in EXECUTABLE_BASELINE_RUNNERS}

        self.assertIn("gbm", catalog_ids)
        self.assertEqual(sorted(runner_ids - catalog_ids), [])

    def test_gbm_catalog_entry_describes_the_local_implementation(self):
        catalog = {entry.model_id: entry for entry in BASELINE_CATALOG}
        gbm = catalog["gbm"]

        self.assertEqual(gbm.status, "local_baseline")
        self.assertFalse(gbm.uses_upstream_code)
        self.assertFalse(gbm.uses_upstream_weights)
        self.assertIsNone(gbm.upstream_reference)
        self.assertIn("gradient-boosting", gbm.notes)

    def test_external_pretrained_eeg_models_are_catalogued_as_unavailable(self):
        catalog = {entry.model_id: entry for entry in BASELINE_CATALOG}
        runner_ids = {runner.model_id for runner in EXECUTABLE_BASELINE_RUNNERS}

        for model_id in ("eegpt", "labram_base", "cbramod", "biot_prest16", "biot_six_datasets"):
            with self.subTest(model_id=model_id):
                self.assertIn(model_id, catalog)
                entry = catalog[model_id]
                self.assertEqual(entry.status, "unavailable")
                self.assertFalse(entry.exact_reproduction)
                self.assertFalse(entry.uses_upstream_code)
                self.assertFalse(entry.uses_upstream_weights)
                self.assertIn("external pretrained weights not vendored", entry.notes.lower())
                self.assertIn("contamination audit", entry.notes)
                self.assertNotIn(model_id, runner_ids)

    def test_external_pretrained_models_are_excluded_from_catalog_rows_as_available(self):
        rows = {
            row["model_id"]: row
            for row in baseline_catalog_rows({"future_state_forecasting"}, {"eeg"})
        }

        self.assertEqual(rows["gbm"]["status"], "local_baseline")
        for model_id in ("eegpt", "labram_base", "cbramod", "biot_prest16", "biot_six_datasets"):
            with self.subTest(model_id=model_id):
                self.assertEqual(rows[model_id]["status"], "unavailable")

    def test_catalog_model_ids_are_unique(self):
        model_ids = [entry.model_id for entry in BASELINE_CATALOG]

        self.assertEqual(len(model_ids), len(set(model_ids)))

    def test_executable_baseline_ids_are_unique_and_have_no_model_alias(self):
        runner_ids = [runner.model_id for runner in EXECUTABLE_BASELINE_RUNNERS]

        self.assertEqual(len(runner_ids), len(set(runner_ids)))
        self.assertNotIn("model", runner_ids)
        self.assertIn("ssm_fallback", runner_ids)
        self.assertIn("tiny_ssm", runner_ids)

    def test_brainvista_style_stimulus_features_are_history_only_by_default(self):
        x = np.arange(1 * 5 * 3, dtype=np.float32).reshape(1, 5, 3)
        config = BrainVistaStyleConfig(stimulus_lag_steps=1, include_current_stimulus=False, hrf_lag_steps=2)
        baseline_features = _causal_stimulus_features(x, config=config)

        future_changed = x.copy()
        future_changed[:, 3, :] += 1000.0
        self.assertTrue(np.array_equal(baseline_features[:, 2, :], _causal_stimulus_features(future_changed, config=config)[:, 2, :]))

        current_changed = x.copy()
        current_changed[:, 2, :] += 1000.0
        self.assertTrue(np.array_equal(baseline_features[:, 2, :], _causal_stimulus_features(current_changed, config=config)[:, 2, :]))

        past_changed = x.copy()
        past_changed[:, 1, :] += 1000.0
        self.assertFalse(np.array_equal(baseline_features[:, 2, :], _causal_stimulus_features(past_changed, config=config)[:, 2, :]))

    def test_tribe_style_facade_predicts_without_external_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "stimulus.txt"
            text_path.write_text("visual auditory language stimulus", encoding="utf-8")
            model = TribeStyleModel.from_checkpoint(
                "local",
                config_update={"stimulus_dim": 6, "output_dim": 4, "hidden_dim": 8},
            )

            events = model.build_events(TribeStyleStimulusInput(path=text_path, modality="text"))
            preds, segments = model.predict(events, verbose=False)

        self.assertEqual(model.model_id, "tribe_style")
        self.assertEqual(model.implementation_status, "clean_room_approximation")
        self.assertEqual(preds.shape, (4, 4))
        self.assertEqual(len(segments), 4)
        self.assertTrue(np.isfinite(preds).all())

    def test_tribe_style_compatibility_shims_delegate_to_preferred_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "tribe_style_config.json"
            text_path = Path(tmp) / "stimulus.txt"
            config_path.write_text(
                json.dumps({"stimulus_dim": 5, "output_dim": 3, "hidden_dim": 7, "seed": 11}),
                encoding="utf-8",
            )
            text_path.write_text("compatibility shim", encoding="utf-8")

            model = TribeStyleModel.from_pretrained(tmp)
            preferred_events = model.build_events(text_path, modality="text")
            row_events = model.get_event_rows(text_path, modality="text")
            shim_events = model.get_events_dataframe(text_path=str(text_path))
            preds, _ = model.predict(shim_events, verbose=False)

        self.assertEqual(model.stimulus_dim, 5)
        self.assertEqual(model.output_dim, 3)
        self.assertEqual(preferred_events, row_events)
        self.assertEqual(preferred_events, shim_events)
        self.assertEqual(preds.shape, (2, 3))
        self.assertTrue(np.isfinite(preds).all())
        self.assertIn("Sunset this alias", TribeStyleModel.from_pretrained.__doc__)
        self.assertIn("row-oriented name", TribeStyleModel.get_event_rows.__doc__)
        self.assertIn("temporary compatibility alias", TribeStyleModel.get_events_dataframe.__doc__)

    def test_tribe_style_event_inputs_reject_ambiguous_legacy_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "stimulus.txt"
            audio_path = Path(tmp) / "stimulus.wav"
            text_path.write_text("ambiguous event input", encoding="utf-8")
            audio_path.write_bytes(b"RIFF")

            model = TribeStyleModel.from_checkpoint("local")

            with self.assertRaisesRegex(ValueError, "Exactly one stimulus input"):
                model.build_events(text_path=str(text_path), audio_path=str(audio_path))

    def test_paper_mode_gate_enforces_audit_seeds_ranking_and_ci_contract(self):
        def seed_record(seed: int, with_ci: bool = True) -> dict[str, object]:
            mse = 0.30 + seed * 0.01
            metrics = {"mse": mse, "mae": mse + 0.1}
            if with_ci:
                metrics.update({"mse_ci_low": mse - 0.01, "mse_ci_high": mse + 0.01})
            return {
                "seed": seed,
                "tasks": {
                    "future_state_forecasting": {
                        "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": mse, "rank": 1.0}],
                        "metrics_by_model": {"linear_ridge": metrics},
                    }
                },
            }

        def strict_payload(*records: dict[str, object]) -> dict[str, object]:
            return {
                "forecast_eligibility": _valid_forecast_eligibility_artifact(),
                "aggregate": {
                    "selection_metric": "mse",
                    "higher_is_better": False,
                    "aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0, "tasks_ranked": len(records)}],
                },
                "seed_results": list(records),
                "seed_aggregate": [
                    {
                        "task_id": "future_state_forecasting",
                        "model_id": "linear_ridge",
                        "metric": "mse",
                        "mean": 0.31,
                        "std": 0.01,
                        "ci_low": 0.29,
                        "ci_high": 0.33,
                        "n_seeds": len(records),
                    }
                ],
            }

        payload = strict_payload(seed_record(0))
        missing_seed_report = validate_paper_mode_payload(payload, audit_report={"passed": True})
        self.assertFalse(missing_seed_report.passed)
        self.assertTrue(any("missing 1,2" in violation for violation in missing_seed_report.violations))
        self.assertEqual(missing_seed_report.required_seeds, (0, 1, 2))
        with self.assertRaises(TypeError):
            validate_paper_mode_payload(payload, audit_report={"passed": True}, **{"required_seeds": (0,)})

        metadata_only_report = validate_paper_mode_payload(
            {
                "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]},
                "paper_mode_contract": {"observed_seeds": [0, 1, 2]},
                "seed_results": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
                "seed_aggregate": [],
            },
            audit_report={"passed": True},
        )
        self.assertFalse(metadata_only_report.passed)
        self.assertEqual(metadata_only_report.observed_seeds, ())
        self.assertTrue(any("missing 0,1,2" in violation for violation in metadata_only_report.violations))

        invalid_seed_payload = {
            "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]},
            "seed_results": [
                seed_record(0),
                {"seed": 1, "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]}},
                {
                    "seed": 2,
                    "tasks": {
                        "future_state_forecasting": {
                            "status": "completed",
                            "metrics": {"mse": 0.42},
                        }
                    },
                },
            ],
            "seed_aggregate": strict_payload(seed_record(0), seed_record(1), seed_record(2))["seed_aggregate"],
        }
        invalid_seed_report = validate_paper_mode_payload(invalid_seed_payload, audit_report={"passed": True})
        self.assertFalse(invalid_seed_report.passed)
        self.assertIn(0, invalid_seed_report.observed_seeds)
        self.assertNotIn(1, invalid_seed_report.observed_seeds)
        self.assertNotIn(2, invalid_seed_report.observed_seeds)
        self.assertTrue(any("seed 1 lacks task payloads with ranked baselines" in violation for violation in invalid_seed_report.violations))
        self.assertTrue(any("seed 2 lacks ranked baseline evidence" in violation for violation in invalid_seed_report.violations))

        payload = strict_payload(seed_record(0), seed_record(1), seed_record(2))
        passed_report = validate_paper_mode_payload(payload, audit_report={"passed": True})
        self.assertTrue(passed_report.passed, passed_report.violations)
        self.assertEqual(passed_report.observed_seeds, (0, 1, 2))
        self.assertIn("ci_summaries", passed_report.checked)
        self.assertNotIn("metric_evidence_without_ci", passed_report.checked)

        top_level_audit_payload = json.loads(json.dumps(payload))
        top_level_audit_payload["eval_audit"] = {"passed": True}
        top_level_audit_report = validate_paper_mode_payload(top_level_audit_payload)
        self.assertTrue(top_level_audit_report.passed, top_level_audit_report.violations)

        failed_top_level_audit_payload = json.loads(json.dumps(payload))
        failed_top_level_audit_payload["eval_audit"] = {"passed": False, "violations": ["leakage"]}
        failed_top_level_audit_report = validate_paper_mode_payload(failed_top_level_audit_payload)
        self.assertFalse(failed_top_level_audit_report.passed)
        self.assertTrue(any("audit did not pass" in violation for violation in failed_top_level_audit_report.violations))

        failed_audit_report = validate_paper_mode_payload(payload, audit_report={"passed": False, "violations": ["leakage"]})
        self.assertFalse(failed_audit_report.passed)
        self.assertTrue(any("audit did not pass" in violation for violation in failed_audit_report.violations))

        no_ranking = dict(payload)
        no_ranking["aggregate"] = {"aggregate_rank": []}
        no_ranking_report = validate_paper_mode_payload(no_ranking, audit_report={"passed": True})
        self.assertFalse(no_ranking_report.passed)
        self.assertTrue(any("aggregate_rank is empty" in violation for violation in no_ranking_report.violations))

        aggregate_mismatch = json.loads(json.dumps(payload))
        aggregate_mismatch["aggregate"]["aggregate_rank"][0]["mean_rank"] = 2.0
        aggregate_mismatch_report = validate_paper_mode_payload(aggregate_mismatch, audit_report={"passed": True})
        self.assertFalse(aggregate_mismatch_report.passed)
        self.assertTrue(any("does not match per-seed" in violation for violation in aggregate_mismatch_report.violations))

        no_ci = strict_payload(seed_record(0), seed_record(1, with_ci=False), seed_record(2))
        no_ci_report = validate_paper_mode_payload(no_ci, audit_report={"passed": True})
        self.assertFalse(no_ci_report.passed)
        self.assertNotIn(1, no_ci_report.observed_seeds)
        self.assertTrue(any("seed 1:future_state_forecasting:linear_ridge lacks finite mse CI summary" in violation for violation in no_ci_report.violations))

        no_seed_aggregate_ci = strict_payload(seed_record(0), seed_record(1), seed_record(2))
        no_seed_aggregate_ci["seed_aggregate"] = [
            {
                "task_id": "future_state_forecasting",
                "model_id": "linear_ridge",
                "metric": "mse",
                "mean": 0.31,
                "std": 0.01,
                "n_seeds": 3,
            }
        ]
        no_seed_aggregate_ci_report = validate_paper_mode_payload(no_seed_aggregate_ci, audit_report={"passed": True})
        self.assertFalse(no_seed_aggregate_ci_report.passed)
        self.assertTrue(any("seed_aggregate:linear_ridge:mse lacks finite cross-seed CI" in violation for violation in no_seed_aggregate_ci_report.violations))

    def test_paper_mode_gate_can_count_seed_metrics_without_ci_when_ci_is_optional(self):
        def seed_record(seed: int) -> dict[str, object]:
            mse = 0.30 + seed * 0.01
            return {
                "seed": seed,
                "tasks": {
                    "future_state_forecasting": {
                        "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": mse, "rank": 1.0}],
                        "metrics_by_model": {"linear_ridge": {"mse": mse}},
                    }
                },
            }

        payload = {
            "forecast_eligibility": _valid_forecast_eligibility_artifact(),
            "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]},
            "seed_results": [
                seed_record(0),
                seed_record(1),
                seed_record(2),
            ],
        }

        strict_report = validate_paper_mode_payload(payload, audit_report={"passed": True})
        self.assertFalse(strict_report.passed)
        self.assertEqual(strict_report.observed_seeds, ())
        self.assertTrue(any("missing 0,1,2" in violation for violation in strict_report.violations))
        self.assertTrue(any("lacks finite" in violation for violation in strict_report.violations))

        relaxed_report = validate_paper_mode_payload(payload, audit_report={"passed": True}, require_ci=False)
        self.assertTrue(relaxed_report.passed)
        self.assertEqual(relaxed_report.observed_seeds, (0, 1, 2))
        self.assertIn("metric_evidence_without_ci", relaxed_report.checked)
        self.assertNotIn("ci_summaries", relaxed_report.checked)
        self.assertFalse(relaxed_report.violations)

        loose_metric_payload = {
            "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]},
            "seed_results": [
                seed_record(0),
                {"seed": 1, "tasks": {"future_state_forecasting": {"metrics": {"mse": 0.40}}}},
                {"seed": 2, "task_results": [{"task_id": "future_state_forecasting", "test_mse": 0.50}]},
            ],
        }
        loose_metric_report = validate_paper_mode_payload(loose_metric_payload, audit_report={"passed": True}, require_ci=False)
        self.assertFalse(loose_metric_report.passed)
        self.assertEqual(loose_metric_report.observed_seeds, (0,))

        metadata_only = {
            "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]},
            "seed_results": [{"seed": 0}, {"seed": 1}, {"seed": 2}],
        }
        metadata_only_report = validate_paper_mode_payload(
            metadata_only,
            audit_report={"passed": True},
            require_ci=False,
        )
        self.assertFalse(metadata_only_report.passed)
        self.assertEqual(metadata_only_report.observed_seeds, ())
        self.assertTrue(any("missing 0,1,2" in violation for violation in metadata_only_report.violations))

        ranking_only = {
            "aggregate": {"aggregate_rank": [{"model_id": "linear_ridge", "mean_rank": 1.0}]},
            "seed_results": [
                {
                    "seed": 0,
                    "tasks": {
                        "future_state_forecasting": {
                            "ranking": [{"model_id": "linear_ridge", "rank": 1}],
                        }
                    },
                }
            ],
        }
        ranking_only_report = validate_paper_mode_payload(ranking_only, audit_report={"passed": True}, require_ci=False)
        self.assertFalse(ranking_only_report.passed)
        self.assertEqual(ranking_only_report.observed_seeds, ())

    def test_effective_scientific_claim_allowed_requires_real_run_and_canonical_gate(self):
        task_payload = {"tasks": {"masked_neural_reconstruction": {}}}
        contracts, unknown = collect_task_claim_contracts(task_payload)
        self.assertFalse(unknown)
        valid_gate = {
            "passed": True,
            "require_ci": True,
            "violations": [],
            "required_seeds": [0, 1, 2],
            "observed_seeds": [0, 1, 2],
            "claim_contract_sha256": claim_contract_sha256(contracts),
            "forecast_eligibility_required": False,
            "forecast_eligibility_passed": True,
            "forecast_eligibility_sha256": None,
        }
        self.assertTrue(paper_mode_gate_allows_claim(valid_gate, task_payload=task_payload))
        self.assertTrue(
            effective_scientific_claim_allowed(
                {"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True},
                valid_gate,
                task_payload=task_payload,
            )
        )
        self.assertFalse(
            effective_scientific_claim_allowed(
                {"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": False},
                valid_gate,
                task_payload=task_payload,
            )
        )

        self.assertFalse(
            effective_scientific_claim_allowed(
                {"synthetic_only": True, "real_data_smoke": False, "scientific_claim_allowed": True},
                valid_gate,
                task_payload=task_payload,
            )
        )
        self.assertFalse(
            effective_scientific_claim_allowed(
                {"synthetic_only": False, "real_data_smoke": True, "scientific_claim_allowed": True},
                valid_gate,
                task_payload=task_payload,
            )
        )
        self.assertFalse(
            effective_scientific_claim_allowed(
                {"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True},
                {
                    "passed": True,
                    "require_ci": False,
                    "violations": [],
                    "required_seeds": [0, 1, 2],
                    "observed_seeds": [0, 1, 2],
                },
                task_payload=task_payload,
            )
        )
        self.assertFalse(
            effective_scientific_claim_allowed(
                {"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True},
                {
                    "passed": True,
                    "require_ci": True,
                    "violations": [],
                    "required_seeds": [0, 1],
                    "observed_seeds": [0, 1, 2],
                },
                task_payload=task_payload,
            )
        )
        self.assertFalse(
            effective_scientific_claim_allowed(
                {"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True},
                None,
                task_payload=task_payload,
            )
        )

    def test_paper_mode_evidence_object_cannot_bypass_forecast_eligibility(self):
        def record(seed: int) -> dict[str, object]:
            mse = 0.3 + 0.01 * seed
            return {
                "seed": seed,
                "tasks": {
                    "future_state_forecasting": {
                        "ranking": [{"model_id": "linear_ridge", "metric": "mse", "value": mse, "rank": 1.0}],
                        "metrics_by_model": {
                            "linear_ridge": {"mse": mse, "mse_ci_low": mse - 0.01, "mse_ci_high": mse + 0.01}
                        },
                    }
                },
            }

        from neurotwin.eval.paper_contracts import build_paper_mode_evidence

        evidence = build_paper_mode_evidence([record(0), record(1), record(2)])
        gate = validate_paper_mode_payload(evidence, audit_report={"passed": True})
        self.assertFalse(gate.passed)
        self.assertIn("forecast eligibility evidence is missing", gate.violations)

    def test_effective_scientific_claim_allowed_for_run_uses_tolerant_summary_and_gate_loaders(self):
        task_payload = {"tasks": {"masked_neural_reconstruction": {}}}
        contracts, unknown = collect_task_claim_contracts(task_payload)
        self.assertFalse(unknown)
        valid_gate = {
            "passed": True,
            "require_ci": True,
            "violations": [],
            "required_seeds": [0, 1, 2],
            "observed_seeds": [0, 1, 2],
            "claim_contract_sha256": claim_contract_sha256(contracts),
            "forecast_eligibility_required": False,
            "forecast_eligibility_passed": True,
            "forecast_eligibility_sha256": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "summary.json").write_text(
                json.dumps({"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True}),
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text(json.dumps(valid_gate), encoding="utf-8")
            (run_dir / "prepared_baseline_suite.json").write_text(json.dumps(task_payload), encoding="utf-8")
            self.assertEqual(
                load_run_summary(run_dir),
                {"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True},
            )
            self.assertTrue(effective_scientific_claim_allowed_for_run(run_dir))

            (run_dir / "summary.json").write_text("{broken\n", encoding="utf-8")
            self.assertEqual(load_run_summary(run_dir)["error"], "invalid_json")
            self.assertFalse(effective_scientific_claim_allowed_for_run(run_dir))

            (run_dir / "summary.json").unlink()
            self.assertEqual(load_run_summary(run_dir), {})
            self.assertFalse(effective_scientific_claim_allowed_for_run(run_dir))

            (run_dir / "summary.json").write_text(
                json.dumps({"synthetic_only": False, "real_data_smoke": False, "scientific_claim_allowed": True}),
                encoding="utf-8",
            )
            (run_dir / "paper_mode_gate.json").write_text("{broken\n", encoding="utf-8")
            self.assertFalse(effective_scientific_claim_allowed_for_run(run_dir))

    def test_eval_suite_writes_baseline_artifact(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "neurotwin.cli",
                    "eval",
                    "--suite",
                    "neural_translation_v1",
                    "--out-dir",
                    str(out_dir),
                ],
                check=True,
                text=True,
                capture_output=True,
                env=env,
            )

            baseline_path = out_dir / "baseline_suite.json"
            self.assertTrue(baseline_path.exists())
            payload = json.loads(baseline_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["scope"]["status"], "synthetic-only")
            self.assertIn("local_baseline_suite", result.stdout)


if __name__ == "__main__":
    unittest.main()

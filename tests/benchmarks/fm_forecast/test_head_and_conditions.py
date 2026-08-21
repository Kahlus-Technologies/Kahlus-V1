import json
import unittest

import numpy as np
import torch
from torch import nn

from neurotwin.benchmarks.fm_forecast.conditions import (
    ADMISSIBLE_CONDITIONS,
    Condition,
    ConditionConfig,
    run_condition,
    select_best_condition,
)
from neurotwin.benchmarks.fm_forecast.data_prep import prepare_forecast_arrays
from neurotwin.benchmarks.fm_forecast.head import (
    ForecastHead,
    HeadConfig,
    TrainConfig,
    fit_head,
    predict,
    resolve_device,
)


def _learnable_problem(n: int, embed_dim: int = 8, output_dim: int = 1, seed: int = 0):
    """A strictly linear embedding -> target mapping, so a linear head must learn it.

    ``seed`` draws the samples only; the mapping itself is fixed, so train and validation
    are draws from one problem rather than two different problems.
    """

    weights = np.random.default_rng(9_001).normal(size=(embed_dim, output_dim)).astype(np.float32)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, embed_dim)).astype(np.float32)
    y = (x @ weights + 0.05 * rng.normal(size=(n, output_dim))).astype(np.float32)
    return x, y


class _ProjectionEncoder(nn.Module):
    """Tiny stand-in for a pretrained encoder: (N, C, T) -> (N, embed_dim)."""

    def __init__(self, n_channels: int, n_samples: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Linear(n_channels * n_samples, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.reshape(x.shape[0], -1))


def _forecast_arrays(n_train=48, n_val=16, n_test=12, length=64, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    def make(n):
        x = rng.normal(size=(n, length, channels)).astype(np.float32)
        # Target is a deterministic function of the context, so a real head can learn it.
        y = (x[:, -4:, :].mean(axis=1, keepdims=True) * 1.5).astype(np.float32)
        return x, y

    x_train, y_train = make(n_train)
    x_val, y_val = make(n_val)
    x_test, y_test = make(n_test)
    return prepare_forecast_arrays(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        sr_in=100,
        sr_out=100,
    )


class ForecastHeadTests(unittest.TestCase):
    def test_linear_head_shapes(self):
        head = ForecastHead(8, 3)
        out = head(torch.zeros(5, 8))
        self.assertEqual(tuple(out.shape), (5, 3))

    def test_mlp_head_shapes(self):
        head = ForecastHead(8, 2, HeadConfig(hidden_dim=16, dropout=0.1))
        self.assertEqual(tuple(head(torch.zeros(5, 8)).shape), (5, 2))
        self.assertGreater(sum(p.numel() for p in head.parameters()), 8 * 2 + 2)

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaises(ValueError):
            ForecastHead(0, 1)
        with self.assertRaises(ValueError):
            ForecastHead(4, 0)


class FitHeadTests(unittest.TestCase):
    def test_training_reduces_validation_mse_on_a_learnable_mapping(self):
        x_train, y_train = _learnable_problem(160, seed=1)
        x_val, y_val = _learnable_problem(64, seed=2)
        head = ForecastHead(8, 1)
        config = TrainConfig(lr=5e-2, epochs=40, batch_size=32, patience=40, seed=0)

        baseline = float(np.mean((predict(head, x_val) - y_val) ** 2))
        result = fit_head(head, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, config=config)

        self.assertLess(result.best_val_mse, baseline)
        self.assertLess(result.best_val_mse, 0.1 * baseline)
        self.assertGreaterEqual(result.best_epoch, 1)
        self.assertEqual(len(result.history), result.epochs_ran)
        self.assertGreater(result.gradient_steps, 0)

    def test_selection_restores_the_best_epoch_weights(self):
        x_train, y_train = _learnable_problem(120, seed=3)
        x_val, y_val = _learnable_problem(48, seed=4)
        head = ForecastHead(8, 1)
        result = fit_head(
            head,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=TrainConfig(lr=5e-2, epochs=25, batch_size=16, patience=25, seed=0),
        )
        restored = float(np.mean((predict(head, x_val) - y_val) ** 2))
        self.assertAlmostEqual(restored, result.best_val_mse, places=5)
        self.assertEqual(
            result.best_val_mse,
            min(record.val_mse for record in result.history),
        )

    def test_same_seed_is_deterministic_and_a_different_seed_is_not(self):
        x_train, y_train = _learnable_problem(96, seed=5)
        x_val, y_val = _learnable_problem(32, seed=6)

        def run(seed: int) -> np.ndarray:
            # Seed the head's own initialisation too: fit_head owns the training loop's
            # determinism, not the caller's parameter init.
            torch.manual_seed(4242)
            head = ForecastHead(8, 1)
            fit_head(
                head,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                config=TrainConfig(lr=1e-2, epochs=8, batch_size=16, patience=8, seed=seed),
            )
            return predict(head, x_val)

        np.testing.assert_array_equal(run(0), run(0))
        self.assertFalse(np.array_equal(run(0), run(1)))

    def test_early_stopping_fires_when_validation_stops_improving(self):
        x_train, y_train = _learnable_problem(64, seed=7)
        x_val, y_val = _learnable_problem(32, seed=8)
        head = ForecastHead(8, 1)
        result = fit_head(
            head,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            # A huge min_delta makes every epoch after the first count as no improvement.
            config=TrainConfig(lr=1e-3, epochs=50, batch_size=32, patience=2, min_delta=1e3, seed=0),
        )
        self.assertTrue(result.stopped_early)
        self.assertLess(result.epochs_ran, 50)

    def test_invalid_budgets_are_rejected(self):
        x_train, y_train = _learnable_problem(16, seed=9)
        head = ForecastHead(8, 1)
        for bad in (TrainConfig(epochs=0), TrainConfig(batch_size=0), TrainConfig(patience=0)):
            with self.subTest(config=bad):
                with self.assertRaises(ValueError):
                    fit_head(head, x_train=x_train, y_train=y_train, x_val=x_train, y_val=y_train, config=bad)

    def test_resolve_device_runs_on_cpu_without_a_gpu(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertIn(resolve_device("auto").type, {"cpu", "cuda"})


class RunConditionTests(unittest.TestCase):
    def _config(self, condition: Condition, **overrides) -> ConditionConfig:
        payload = dict(
            condition=condition,
            lr=5e-2,
            epochs=12,
            batch_size=16,
            patience=12,
            seed=0,
            device="cpu",
        )
        payload.update(overrides)
        return ConditionConfig(**payload)

    def test_frozen_condition_predicts_the_harness_target_shape(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        result = run_condition(encoder, arrays, self._config(Condition.FROZEN))
        self.assertEqual(result.predictions.shape, arrays.test_target_shape)
        self.assertEqual(result.predictions.dtype, np.float32)
        self.assertTrue(np.isfinite(result.predictions).all())
        self.assertTrue(result.is_admissible)

    def test_frozen_condition_leaves_the_encoder_weights_untouched(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        before = [p.detach().clone() for p in encoder.parameters()]
        run_condition(encoder, arrays, self._config(Condition.FROZEN))
        for original, current in zip(before, encoder.parameters()):
            torch.testing.assert_close(original, current)

    def test_full_ft_condition_updates_the_encoder_weights(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        before = [p.detach().clone() for p in encoder.parameters()]
        result = run_condition(encoder, arrays, self._config(Condition.FULL_FT))
        self.assertEqual(result.predictions.shape, arrays.test_target_shape)
        changed = any(
            not torch.allclose(original, current)
            for original, current in zip(before, encoder.parameters())
        )
        self.assertTrue(changed)

    def test_full_ft_tolerates_a_non_float_encoder_parameter(self):
        # BIOT registers an integer channel-index tensor as an nn.Parameter with
        # requires_grad=False; unfreezing must skip it, not crash on it.
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        encoder.index = torch.nn.Parameter(
            torch.arange(arrays.n_channels), requires_grad=False
        )
        result = run_condition(encoder, arrays, self._config(Condition.FULL_FT))
        self.assertEqual(result.predictions.shape, arrays.test_target_shape)
        self.assertTrue(np.isfinite(result.predictions).all())
        self.assertFalse(encoder.index.requires_grad)

    def test_full_ft_uses_a_lower_encoder_learning_rate_by_default(self):
        config = self._config(Condition.FULL_FT, lr=1e-2)
        self.assertLess(config.resolved_encoder_lr(), config.lr)
        self.assertAlmostEqual(config.resolved_encoder_lr(), 1e-3)
        self.assertAlmostEqual(
            ConditionConfig(encoder_lr=7e-4).resolved_encoder_lr(), 7e-4
        )

    def test_full_ft_requires_a_module_not_a_plain_callable(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        with self.assertRaises(TypeError):
            run_condition(lambda x: encoder(x), arrays, self._config(Condition.FULL_FT))

    def test_frozen_condition_accepts_a_plain_embedding_callable(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        result = run_condition(lambda x: encoder(x), arrays, self._config(Condition.FROZEN))
        self.assertEqual(result.predictions.shape, arrays.test_target_shape)

    def test_conditions_learn_something_on_a_learnable_target(self):
        arrays = _forecast_arrays()
        mean_baseline = float(np.mean((arrays.y_val - arrays.y_train.mean(axis=0)) ** 2))
        for condition in ADMISSIBLE_CONDITIONS:
            with self.subTest(condition=condition):
                encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
                result = run_condition(encoder, arrays, self._config(condition))
                self.assertLess(result.best_val_mse, mean_baseline)

    def test_run_condition_is_deterministic_for_a_fixed_seed(self):
        arrays = _forecast_arrays()
        for condition in ADMISSIBLE_CONDITIONS:
            with self.subTest(condition=condition):
                torch.manual_seed(123)
                encoder_a = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
                torch.manual_seed(123)
                encoder_b = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
                first = run_condition(encoder_a, arrays, self._config(condition))
                second = run_condition(encoder_b, arrays, self._config(condition))
                np.testing.assert_array_equal(first.predictions, second.predictions)

    def test_test_targets_are_never_read(self):
        # Replacing y_test with NaNs must not perturb a single prediction: training and
        # selection see train and validation only (plan sections 3.3 / 5.3).
        arrays = _forecast_arrays()
        poisoned = type(arrays)(
            x_train=arrays.x_train,
            y_train=arrays.y_train,
            x_val=arrays.x_val,
            y_val=arrays.y_val,
            x_test=arrays.x_test,
            test_target_shape=arrays.test_target_shape,
            y_test=np.full_like(arrays.y_test, np.nan),
            sr_in=arrays.sr_in,
            sr_out=arrays.sr_out,
            n_samples_out=arrays.n_samples_out,
        )
        for condition in ADMISSIBLE_CONDITIONS:
            with self.subTest(condition=condition):
                torch.manual_seed(7)
                encoder_a = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
                torch.manual_seed(7)
                encoder_b = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
                clean = run_condition(encoder_a, arrays, self._config(condition))
                dirty = run_condition(encoder_b, poisoned, self._config(condition))
                np.testing.assert_array_equal(clean.predictions, dirty.predictions)

    def test_acausal_infill_is_a_documented_slot_not_an_implementation(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        with self.assertRaises(NotImplementedError) as caught:
            run_condition(encoder, arrays, self._config(Condition.ACAUSAL_INFILL))
        message = str(caught.exception)
        self.assertIn("infilling", message)
        self.assertIn("forecasting", message)
        self.assertFalse(Condition.ACAUSAL_INFILL.is_admissible)
        self.assertEqual(Condition.ACAUSAL_INFILL.label, "infilling")
        self.assertNotIn(Condition.ACAUSAL_INFILL, ADMISSIBLE_CONDITIONS)

    def test_training_log_is_json_serialisable_and_records_the_budget(self):
        arrays = _forecast_arrays()
        encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
        result = run_condition(encoder, arrays, self._config(Condition.FROZEN))
        payload = json.loads(json.dumps(result.log))
        for key in (
            "condition",
            "condition_label",
            "admissible_for_ranking",
            "config",
            "embed_dim",
            "output_dim",
            "n_train_windows",
            "n_val_windows",
            "n_test_windows",
            "wall_clock_seconds",
            "encoder_parameters",
            "trainable_parameters",
            "fit",
        ):
            with self.subTest(key=key):
                self.assertIn(key, payload)
        self.assertEqual(payload["condition"], "frozen")
        self.assertEqual(payload["condition_label"], "forecasting")
        self.assertGreater(payload["fit"]["gradient_steps"], 0)
        self.assertEqual(len(payload["fit"]["history"]), payload["fit"]["epochs_ran"])
        self.assertEqual(payload["config"]["seed"], 0)


class SelectBestConditionTests(unittest.TestCase):
    def test_selection_is_argmin_validation_mse_over_admissible_conditions(self):
        arrays = _forecast_arrays()
        results = []
        for condition in ADMISSIBLE_CONDITIONS:
            torch.manual_seed(11)
            encoder = _ProjectionEncoder(arrays.n_channels, arrays.n_samples)
            results.append(
                run_condition(
                    encoder,
                    arrays,
                    ConditionConfig(condition=condition, lr=5e-2, epochs=10, batch_size=16, patience=10, device="cpu"),
                )
            )
        best = select_best_condition(results)
        self.assertEqual(best.best_val_mse, min(result.best_val_mse for result in results))
        self.assertIn(best.condition, ADMISSIBLE_CONDITIONS)

    def test_selection_refuses_an_empty_or_inadmissible_set(self):
        with self.assertRaises(ValueError):
            select_best_condition([])

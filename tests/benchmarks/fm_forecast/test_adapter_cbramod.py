import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from neurotwin.benchmarks.fm_forecast.adapters import cbramod_adapter
from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
from neurotwin.benchmarks.fm_forecast.vendor_import import is_vendored
from neurotwin.data.prepared_tasks import SupervisedWindowTask


VENDORED = is_vendored("cbramod")
WEIGHTS_DIR = Path("/tmp/kahlus-weights")
REAL_CHECKPOINT = WEIGHTS_DIR / "cbramod_pretrained_weights.pth"


def _tiny_forecast_task(seed: int = 0) -> SupervisedWindowTask:
    rng = np.random.default_rng(seed)

    def split(n_windows: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n_windows, 127, 1)).astype(np.float32)
        y = x[:, -3:, :].mean(axis=1, keepdims=True).astype(np.float32)
        return x, y

    x_train, y_train = split(6)
    x_val, y_val = split(3)
    x_test, y_test = split(2)
    return SupervisedWindowTask(
        task_id="future_state_forecasting",
        source_modality="eeg",
        target_modality="eeg",
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )


class CBraModAdapterSurfaceTests(unittest.TestCase):
    def test_module_surface_and_lazy_imports(self):
        self.assertEqual(cbramod_adapter.MODEL_ID, "cbramod")
        self.assertEqual(cbramod_adapter.SR_OUT, 200.0)
        self.assertEqual(cbramod_adapter.N_SAMPLES_OUT, 200)
        self.assertEqual(cbramod_adapter.EMBED_DIM, 200)
        self.assertFalse(hasattr(cbramod_adapter, "torch"))

    def test_make_runner_refuses_when_contamination_gate_refuses(self):
        with mock.patch.object(
            cbramod_adapter, "is_evaluation_permitted", return_value=False
        ) as gate:
            with self.assertRaisesRegex(RuntimeError, "contamination"):
                cbramod_adapter.make_runner(
                    Path("/checkpoint-resolution-must-not-run"), Condition.FROZEN
                )
        gate.assert_called_once_with(
            cbramod_adapter.MODEL_ID, cbramod_adapter.SLEEP_EDF_EVAL_DATASET_ID
        )

    def test_random_twin_runner_records_standard_metadata(self):
        runner = cbramod_adapter.make_runner(
            Path("/checkpoint-resolution-must-not-run"),
            Condition.FROZEN,
            pretrained=False,
        )

        self.assertEqual(runner.model_id, "cbramod")
        self.assertIs(runner.condition, Condition.FROZEN)
        self.assertFalse(runner.pretrained)
        self.assertIsNone(runner.checkpoint_path)
        self.assertFalse(runner.model_selection_exposure)


@unittest.skipUnless(VENDORED, "external/vendor/cbramod is not populated (gitignored)")
class CBraModVendoredTests(unittest.TestCase):
    def test_random_init_forward_is_finite_and_seed_deterministic(self):
        import torch

        x = torch.linspace(-1.0, 1.0, 8 * 200).reshape(8, 1, 200)

        torch.manual_seed(91)
        first_encoder = cbramod_adapter.load_encoder(None)
        torch.manual_seed(91)
        second_encoder = cbramod_adapter.load_encoder(None)

        with torch.no_grad():
            first = first_encoder(x)
            second = second_encoder(x)

        self.assertEqual(tuple(first.shape), (8, cbramod_adapter.EMBED_DIM))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_loading_vendor_leaves_no_bare_models_modules(self):
        sentinel = object()
        previous_models = sys.modules.pop("models", sentinel)
        previous_child = sys.modules.pop("models.criss_cross_transformer", sentinel)
        sys.modules.pop(cbramod_adapter._VENDOR_MODEL_KEY, None)
        try:
            module = cbramod_adapter._load_vendor_cbramod_module()

            self.assertTrue(hasattr(module, "CBraMod"))
            self.assertNotIn("models", sys.modules)
            self.assertNotIn("models.criss_cross_transformer", sys.modules)
            self.assertIn(cbramod_adapter._VENDOR_MODEL_KEY, sys.modules)
            self.assertIn(cbramod_adapter._VENDOR_TRANSFORMER_KEY, sys.modules)
        finally:
            if previous_models is not sentinel:
                sys.modules["models"] = previous_models
            if previous_child is not sentinel:
                sys.modules["models.criss_cross_transformer"] = previous_child

    @unittest.skipUnless(
        REAL_CHECKPOINT.exists(),
        "/tmp/kahlus-weights/cbramod_pretrained_weights.pth is not staged",
    )
    def test_real_checkpoint_loads_strictly_and_changes_the_embedding(self):
        import torch

        torch.manual_seed(17)
        pretrained = cbramod_adapter.load_encoder(REAL_CHECKPOINT)
        torch.manual_seed(17)
        random_init = cbramod_adapter.load_encoder(None)

        self.assertEqual(
            pretrained.checkpoint_load_stats,
            {
                "matched_keys": 211,
                "missing_keys": (),
                "unexpected_keys": (),
            },
        )
        x = torch.linspace(-0.5, 0.5, 4 * 200).reshape(4, 1, 200)
        with torch.no_grad():
            pretrained_embedding = pretrained(x)
            random_embedding = random_init(x)

        self.assertEqual(tuple(pretrained_embedding.shape), (4, cbramod_adapter.EMBED_DIM))
        self.assertTrue(torch.isfinite(pretrained_embedding).all())
        self.assertFalse(torch.equal(pretrained_embedding, random_embedding))

    @unittest.skipUnless(
        REAL_CHECKPOINT.exists(),
        "/tmp/kahlus-weights/cbramod_pretrained_weights.pth is not staged",
    )
    def test_frozen_runner_with_real_weights_returns_finite_harness_shape(self):
        task = _tiny_forecast_task()
        runner = cbramod_adapter.make_runner(
            WEIGHTS_DIR,
            Condition.FROZEN,
            config=ConditionConfig(
                condition=Condition.FROZEN,
                lr=1e-2,
                epochs=99,
                batch_size=3,
                patience=1,
                device="cpu",
            ),
        )

        prediction = runner(task, seed=23, train_steps=1)

        self.assertEqual(prediction.shape, task.y_test.shape)
        self.assertEqual(prediction.dtype, np.float32)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()

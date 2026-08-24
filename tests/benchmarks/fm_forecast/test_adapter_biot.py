import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from neurotwin.benchmarks.fm_forecast.adapters import biot_adapter
from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig
from neurotwin.benchmarks.fm_forecast.vendor_import import is_vendored
from neurotwin.benchmarks.pretraining_contamination import (
    SLEEP_EDF_EVAL_DATASET_ID,
    is_evaluation_permitted,
)
from neurotwin.data.prepared_tasks import SupervisedWindowTask


BIOT_CHECKPOINT = Path("/tmp/kahlus-weights/biot-EEG-PREST-16-channels.ckpt")


def _tiny_forecast_task(seed: int = 0) -> SupervisedWindowTask:
    rng = np.random.default_rng(seed)

    def split(n_windows: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n_windows, 127, 1)).astype(np.float32)
        y = x[:, -4:, :].mean(axis=1, keepdims=True).astype(np.float32)
        return x, y

    x_train, y_train = split(8)
    x_val, y_val = split(4)
    x_test, y_test = split(3)
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


class BIOTAdapterPolicyTests(unittest.TestCase):
    def test_six_datasets_checkpoint_is_refused_even_for_sleep_edf(self):
        self.assertFalse(
            is_evaluation_permitted("biot_six_datasets", SLEEP_EDF_EVAL_DATASET_ID)
        )

    def test_make_runner_fails_closed_when_contamination_gate_refuses(self):
        with mock.patch.object(
            biot_adapter, "is_evaluation_permitted", return_value=False
        ):
            with self.assertRaisesRegex(RuntimeError, "contamination"):
                biot_adapter.make_runner(
                    Path("/checkpoint-resolution-must-not-run"), Condition.FROZEN
                )


@unittest.skipUnless(
    is_vendored("biot"), "external/vendor/biot is not populated (gitignored)"
)
class BIOTAdapterVendoredTests(unittest.TestCase):
    def test_random_init_forward_is_finite_and_seed_deterministic(self):
        x = torch.linspace(-1.0, 1.0, 8 * 200).reshape(8, 1, 200)

        torch.manual_seed(91)
        first_encoder = biot_adapter.load_encoder(None)
        torch.manual_seed(91)
        second_encoder = biot_adapter.load_encoder(None)
        first_encoder.eval()
        second_encoder.eval()

        with torch.no_grad():
            first = first_encoder(x)
            second = second_encoder(x)

        self.assertEqual(tuple(first.shape), (8, 256))
        self.assertTrue(torch.isfinite(first).all())
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(first_encoder.encoder.channel_tokens.num_embeddings, 16)

    def test_loading_vendor_never_registers_bare_model_module(self):
        sentinel = object()
        previous = sys.modules.pop("model", sentinel)
        try:
            biot_adapter.load_encoder(None)
            self.assertNotIn("model", sys.modules)
        finally:
            if previous is not sentinel:
                sys.modules["model"] = previous

    @unittest.skipUnless(
        BIOT_CHECKPOINT.exists(),
        "/tmp/kahlus-weights/biot-EEG-PREST-16-channels.ckpt is not staged",
    )
    def test_real_prest16_checkpoint_loads_all_encoder_keys(self):
        encoder = biot_adapter.load_encoder(BIOT_CHECKPOINT)

        self.assertEqual(
            encoder.checkpoint_load_stats,
            {
                "matched_keys": 57,
                "missing_keys": (),
                "unexpected_keys": (),
            },
        )

        x = torch.linspace(-0.5, 0.5, 4 * 200).reshape(4, 1, 200)
        torch.manual_seed(17)
        random_encoder = biot_adapter.load_encoder(None)
        encoder.eval()
        random_encoder.eval()
        with torch.no_grad():
            pretrained_embedding = encoder(x)
            random_embedding = random_encoder(x)
        self.assertFalse(torch.equal(pretrained_embedding, random_embedding))

    @unittest.skipUnless(
        BIOT_CHECKPOINT.exists(),
        "/tmp/kahlus-weights/biot-EEG-PREST-16-channels.ckpt is not staged",
    )
    def test_frozen_runner_returns_finite_predictions_in_harness_shape(self):
        task = _tiny_forecast_task()
        runner = biot_adapter.make_runner(
            BIOT_CHECKPOINT.parent,
            Condition.FROZEN,
            config=ConditionConfig(
                condition=Condition.FROZEN,
                lr=1e-2,
                epochs=99,
                batch_size=4,
                patience=2,
                device="cpu",
            ),
        )

        prediction = runner(task, seed=23, train_steps=2)

        self.assertEqual(prediction.shape, task.y_test.shape)
        self.assertEqual(prediction.dtype, np.float32)
        self.assertTrue(np.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()

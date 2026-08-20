import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from neurotwin.benchmarks.fm_forecast.adapters import eegpt_adapter
from neurotwin.benchmarks.fm_forecast.vendor_import import is_vendored, vendored_root
from neurotwin.benchmarks.fm_forecast.weights_manifest import checkpoint_spec
from neurotwin.data.prepared_tasks import SupervisedWindowTask


VENDORED = is_vendored("eegpt")

#: Weights are staged by hand and are not in the repo; tests that need the real bytes skip
#: when they are absent rather than failing on an environment fact.
WEIGHTS_DIR = Path(os.environ.get("NEUROTWIN_WEIGHTS_DIR", "/tmp/kahlus-weights"))
CHECKPOINT_PATH = WEIGHTS_DIR / checkpoint_spec("eegpt").filename
HAVE_CHECKPOINT = CHECKPOINT_PATH.exists()

#: Bare top-level names the vendored EEG upstreams collide on. Importing this adapter or
#: building its encoder must never register any of them: four upstreams get imported into
#: one pytest process and, e.g., eegpt's ``downstream/utils.py`` and labram's ``utils.py``
#: are different modules with the same name.
COLLIDING_MODULE_NAMES = (
    "utils",
    "utils_eval",
    "Modules",
    "Modules.models",
    "Modules.models.EEGPT_mcae",
    "Modules.models.EEGPT_mcae_finetune",
    "EEGPT_mcae_finetune",
    "model",
    "models",
    "modeling_finetune",
)


def _forecast_task(
    n_train: int = 12,
    n_val: int = 6,
    n_test: int = 5,
    context: int = 127,
    channels: int = 1,
    seed: int = 0,
) -> SupervisedWindowTask:
    """A tiny task shaped like the F8 Sleep-EDF forecasting lane.

    ``(N, 127, 1)`` context at 100 Hz and a single-sample ``(N, 1, 1)`` target, matching the
    shapes ``data_prep`` documents for this lane. The target is a deterministic function of
    the context so a real head has something to fit.
    """

    rng = np.random.default_rng(seed)

    def make(n: int) -> tuple[np.ndarray, np.ndarray]:
        x = rng.normal(size=(n, context, channels)).astype(np.float32)
        y = (0.8 * x[:, -1:, :] + 0.2 * x[:, -2:-1, :]).astype(np.float32)
        return x, y

    x_train, y_train = make(n_train)
    x_val, y_val = make(n_val)
    x_test, y_test = make(n_test)
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


def _tiny_config():
    from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig

    return ConditionConfig(
        condition=Condition.FROZEN,
        lr=1e-2,
        epochs=2,
        batch_size=4,
        patience=2,
        device="cpu",
    )


class EEGPTAdapterSurfaceTests(unittest.TestCase):
    """Facts that hold with no vendor tree and no weights on disk."""

    def test_module_constants_match_the_documented_window(self):
        self.assertEqual(eegpt_adapter.MODEL_ID, "eegpt")
        self.assertEqual(eegpt_adapter.SR_OUT, 256.0)
        self.assertEqual(eegpt_adapter.N_SAMPLES_OUT, 320)
        # 320 samples at 256 Hz is 1.25 s: five whole 64-sample patches, and <= the 1.27 s
        # incumbent context (design doc section 3.2).
        self.assertEqual(eegpt_adapter.N_SAMPLES_OUT % eegpt_adapter.PATCH_SIZE, 0)
        self.assertEqual(eegpt_adapter.N_SAMPLES_OUT // eegpt_adapter.PATCH_SIZE, 5)
        self.assertLessEqual(eegpt_adapter.N_SAMPLES_OUT / eegpt_adapter.SR_OUT, 1.27)
        self.assertEqual(eegpt_adapter.EMBED_DIM, 4 * 512)
        self.assertEqual(eegpt_adapter.LANE_A_CHANNEL_NAMES, ("FPZ",))

    def test_importing_the_adapter_pulls_in_no_torch_and_no_vendored_code(self):
        for name in ("torch", "nn", "np", "numpy", "EEGTransformer", "CHANNEL_DICT"):
            with self.subTest(name=name):
                self.assertFalse(
                    hasattr(eegpt_adapter, name),
                    f"{name!r} is bound at adapter module scope; torch and vendored code "
                    "must only be imported inside functions",
                )

    def test_importing_the_adapter_registers_no_colliding_module_names(self):
        for name in COLLIDING_MODULE_NAMES:
            with self.subTest(name=name):
                self.assertNotIn(name, sys.modules)


class EEGPTContaminationGateTests(unittest.TestCase):
    def test_make_runner_refuses_when_evaluation_is_not_permitted(self):
        from neurotwin.benchmarks import pretraining_contamination
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        # A directory that cannot possibly hold a staged checkpoint: if the gate did not
        # fire first we would see CheckpointNotStagedError from resolve_checkpoint instead.
        missing_weights = Path("/nonexistent/kahlus-weights")
        with mock.patch.object(
            pretraining_contamination, "is_evaluation_permitted", return_value=False
        ) as gate:
            with self.assertRaises(RuntimeError) as caught:
                eegpt_adapter.make_runner(missing_weights, Condition.FROZEN)
        gate.assert_called_once_with(
            "eegpt", pretraining_contamination.SLEEP_EDF_EVAL_DATASET_ID
        )
        message = str(caught.exception)
        self.assertIn("not permitted", message)
        self.assertIn("eegpt", message)

    def test_permitted_runner_records_the_model_selection_exposure(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        runner = eegpt_adapter.make_runner(WEIGHTS_DIR, Condition.FROZEN, pretrained=False)
        self.assertEqual(runner.model_id, "eegpt")
        self.assertIs(runner.condition, Condition.FROZEN)
        self.assertFalse(runner.pretrained)
        self.assertIsNone(runner.checkpoint_path)
        # EEGPT's authors evaluated on Sleep-EDFx: CLEAN corpus, reportable exposure.
        self.assertTrue(runner.model_selection_exposure)


@unittest.skipUnless(VENDORED, "vendored eegpt checkout is required")
class EEGPTRandomInitEncoderTests(unittest.TestCase):
    def test_forward_shape_and_finiteness(self):
        import torch

        torch.manual_seed(0)
        encoder = eegpt_adapter.load_encoder(None)
        x = torch.randn(8, 1, eegpt_adapter.N_SAMPLES_OUT)
        with torch.no_grad():
            embedding = encoder(x)
        self.assertEqual(tuple(embedding.shape), (8, eegpt_adapter.EMBED_DIM))
        self.assertEqual(embedding.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(embedding).all()))
        self.assertFalse(encoder.pretrained)

    def test_random_init_is_deterministic_under_a_fixed_seed(self):
        import torch

        x = torch.randn(4, 1, eegpt_adapter.N_SAMPLES_OUT)

        def embed() -> torch.Tensor:
            torch.manual_seed(1234)
            encoder = eegpt_adapter.load_encoder(None)
            with torch.no_grad():
                return encoder(x)

        first = embed()
        second = embed()
        self.assertTrue(bool(torch.equal(first, second)))

    def test_window_must_be_whole_patches_and_montage_must_be_lane_a(self):
        with self.assertRaises(ValueError):
            eegpt_adapter.load_encoder(None, n_samples=300)
        with self.assertRaises(ValueError):
            eegpt_adapter.load_encoder(None, n_channels=2)

    def test_building_the_encoder_leaves_no_colliding_names_or_sys_path_entries(self):
        eegpt_adapter.load_encoder(None)
        for name in COLLIDING_MODULE_NAMES:
            with self.subTest(name=name):
                self.assertNotIn(name, sys.modules)
        self.assertIn(eegpt_adapter._VENDOR_MODULE_KEY, sys.modules)
        self.assertNotIn(str(vendored_root("eegpt")), sys.path)
        self.assertNotIn(str(vendored_root("eegpt") / "downstream"), sys.path)


@unittest.skipUnless(VENDORED, "vendored eegpt checkout is required")
@unittest.skipUnless(HAVE_CHECKPOINT, f"staged EEGPT checkpoint is required at {CHECKPOINT_PATH}")
class EEGPTPretrainedCheckpointTests(unittest.TestCase):
    def test_checkpoint_loads_strictly_into_the_published_geometry(self):
        import torch

        from neurotwin.benchmarks.fm_forecast.weights_manifest import resolve_checkpoint

        path = resolve_checkpoint("eegpt", WEIGHTS_DIR)
        torch.manual_seed(0)
        transformer = eegpt_adapter._build_transformer(1, eegpt_adapter.N_SAMPLES_OUT)
        total_tensors = len(transformer.state_dict())
        loaded_tensors = eegpt_adapter.load_pretrained_state(transformer, path)
        # strict=True inside load_pretrained_state already rules out missing/unexpected
        # keys; assert the count so a silently narrower checkpoint cannot slip through.
        self.assertEqual(loaded_tensors, total_tensors)
        self.assertEqual(
            sum(p.numel() for p in transformer.parameters()),
            25_287_168,
            "published eegpt_mcae_58chs_4s_large4E encoder parameter count",
        )

    def test_pretrained_embedding_differs_from_the_random_init_twin(self):
        import torch

        from neurotwin.benchmarks.fm_forecast.weights_manifest import resolve_checkpoint

        path = resolve_checkpoint("eegpt", WEIGHTS_DIR)
        x = torch.randn(4, 1, eegpt_adapter.N_SAMPLES_OUT)

        torch.manual_seed(7)
        random_init = eegpt_adapter.load_encoder(None)
        pretrained = eegpt_adapter.load_encoder(path)
        self.assertTrue(pretrained.pretrained)
        with torch.no_grad():
            random_embedding = random_init(x)
            pretrained_embedding = pretrained(x)

        self.assertEqual(tuple(pretrained_embedding.shape), (4, eegpt_adapter.EMBED_DIM))
        self.assertTrue(bool(torch.isfinite(pretrained_embedding).all()))
        # If the checkpoint had silently not been applied these would be identical.
        self.assertFalse(bool(torch.allclose(random_embedding, pretrained_embedding)))
        self.assertGreater(
            float((random_embedding - pretrained_embedding).abs().max()), 1e-3
        )

    def test_end_to_end_runner_with_real_weights(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        task = _forecast_task(seed=3)
        runner = eegpt_adapter.make_runner(
            WEIGHTS_DIR, Condition.FROZEN, config=_tiny_config(), pretrained=True
        )
        self.assertEqual(runner.checkpoint_path, CHECKPOINT_PATH)
        predictions = runner(task, 0, 2)
        self.assertEqual(predictions.shape, task.y_test.shape)
        self.assertTrue(bool(np.isfinite(predictions).all()))


@unittest.skipUnless(VENDORED, "vendored eegpt checkout is required")
class EEGPTRunnerTests(unittest.TestCase):
    def test_frozen_runner_matches_the_harness_target_shape(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        task = _forecast_task(seed=1)
        runner = eegpt_adapter.make_runner(
            WEIGHTS_DIR, Condition.FROZEN, config=_tiny_config(), pretrained=False
        )
        predictions = runner(task, 0, 2)
        self.assertEqual(predictions.shape, task.y_test.shape)
        self.assertEqual(predictions.dtype, np.float32)
        self.assertTrue(bool(np.isfinite(predictions).all()))

    def test_runner_threads_the_seed_and_is_reproducible(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        task = _forecast_task(seed=2)
        runner = eegpt_adapter.make_runner(
            WEIGHTS_DIR, Condition.FROZEN, config=_tiny_config(), pretrained=False
        )
        first = runner(task, 0, 2)
        again = runner(task, 0, 2)
        other_seed = runner(task, 5, 2)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, other_seed))

    def test_runner_never_reads_the_test_targets(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        task = _forecast_task(seed=4)
        runner = eegpt_adapter.make_runner(
            WEIGHTS_DIR, Condition.FROZEN, config=_tiny_config(), pretrained=False
        )
        baseline = runner(task, 0, 2)

        poisoned = _forecast_task(seed=4)
        poisoned.y_test[...] = np.nan
        # y_test only fixes the output shape; nothing downstream may use its values.
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(poisoned.y_test, task.y_test)
        np.testing.assert_array_equal(runner(poisoned, 0, 2), baseline)


if __name__ == "__main__":
    unittest.main()

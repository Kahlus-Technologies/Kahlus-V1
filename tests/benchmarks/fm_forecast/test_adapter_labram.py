"""Tests for the LaBraM-Base forecast adapter.

Two skip gates, both environment facts rather than failures: ``external/vendor/labram``
is gitignored and populated on demand, and the pinned checkpoint is staged by hand
out of band. When both are present these tests exercise the real 12-layer encoder and
the real key surgery.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from neurotwin.benchmarks.fm_forecast.adapters import labram_adapter
from neurotwin.benchmarks.fm_forecast.vendor_import import is_vendored
from neurotwin.data.prepared_tasks import SupervisedWindowTask


VENDORED = is_vendored("labram")
WEIGHTS_DIR = Path("/tmp/kahlus-weights")
REAL_CHECKPOINT = WEIGHTS_DIR / "labram-base.pth"

_VENDOR_REASON = "external/vendor/labram is not populated (gitignored)"
_WEIGHTS_REASON = f"{REAL_CHECKPOINT} is not staged"


def _synthetic_forecast_task(n_train=12, n_val=8, n_test=6, context=127, channels=1, seed=0):
    """A tiny F8-shaped forecast task: 127 samples at 100 Hz in, one sample out."""

    rng = np.random.default_rng(seed)

    def make(n):
        x = rng.normal(size=(n, context, channels)).astype(np.float32)
        y = (x[:, -3:, :].mean(axis=1, keepdims=True) * 0.8).astype(np.float32)
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


class AdapterSurfaceTests(unittest.TestCase):
    """Constants are part of the contract shared by all four FM adapters."""

    def test_module_surface_constants(self):
        self.assertEqual(labram_adapter.MODEL_ID, "labram_base")
        self.assertEqual(labram_adapter.UPSTREAM_ID, "labram")
        self.assertEqual(labram_adapter.SR_OUT, 200.0)
        self.assertEqual(labram_adapter.N_SAMPLES_OUT, 200)
        self.assertEqual(labram_adapter.EMBED_DIM, 200)
        # One complete patch: the largest patch-aligned context not exceeding F8's 1.27 s.
        self.assertEqual(labram_adapter.N_SAMPLES_OUT % labram_adapter.PATCH_SAMPLES, 0)
        self.assertEqual(labram_adapter.N_SAMPLES_OUT // labram_adapter.PATCH_SAMPLES, 1)

    def test_model_id_is_registered_in_the_manifest_and_contamination_audit(self):
        from neurotwin.benchmarks.pretraining_contamination import (
            AUDITED_MODEL_IDS,
            SLEEP_EDF_EVAL_DATASET_ID,
            is_evaluation_permitted,
        )
        from neurotwin.benchmarks.fm_forecast.weights_manifest import checkpoint_spec

        self.assertIn(labram_adapter.MODEL_ID, AUDITED_MODEL_IDS)
        self.assertEqual(checkpoint_spec(labram_adapter.MODEL_ID).filename, "labram-base.pth")
        self.assertTrue(is_evaluation_permitted(labram_adapter.MODEL_ID, SLEEP_EDF_EVAL_DATASET_ID))


class HermeticImportTests(unittest.TestCase):
    """The vendor trees collide on top-level names; one pytest process runs all four."""

    def test_importing_the_adapter_does_not_import_torch_at_module_level(self):
        self.assertFalse(hasattr(labram_adapter, "torch"))
        self.assertFalse(hasattr(labram_adapter, "nn"))

    @unittest.skipUnless(VENDORED, _VENDOR_REASON)
    def test_vendored_module_lands_under_a_namespaced_key_only(self):
        # Re-executed from scratch so the assertion is about *this* loader's effect on
        # sys.modules, not about whichever adapter happened to run first.
        before = set(sys.modules)
        sys.modules.pop(labram_adapter.VENDOR_MODULE_NAME, None)
        module = labram_adapter._load_vendored_module()
        added = set(sys.modules) - before

        self.assertIn(labram_adapter.VENDOR_MODULE_NAME, sys.modules)
        self.assertTrue(hasattr(module, "labram_base_patch200_200"))
        # A bare `import modeling_finetune` / `import utils` would make the winner depend
        # on test ordering across the four vendored repos.
        self.assertNotIn("modeling_finetune", added)
        self.assertNotIn("utils", added)
        self.assertNotIn("modeling_finetune", sys.modules)

    @unittest.skipUnless(VENDORED, _VENDOR_REASON)
    def test_vendor_root_is_not_placed_on_sys_path(self):
        labram_adapter._load_vendored_module()
        from neurotwin.benchmarks.fm_forecast.vendor_import import require_vendored

        self.assertNotIn(str(require_vendored("labram")), sys.path)


@unittest.skipUnless(VENDORED, _VENDOR_REASON)
class LoadEncoderTests(unittest.TestCase):
    def test_random_init_forward_shape_is_finite_and_seed_deterministic(self):
        import torch

        def embed(seed: int) -> torch.Tensor:
            torch.manual_seed(seed)
            encoder = labram_adapter.load_encoder(None, n_channels=1, n_samples=200)
            x = torch.as_tensor(
                np.random.default_rng(0).normal(size=(8, 1, 200)).astype(np.float32)
            )
            with torch.no_grad():
                return encoder(x)

        first = embed(1234)
        self.assertEqual(tuple(first.shape), (8, labram_adapter.EMBED_DIM))
        self.assertTrue(torch.isfinite(first).all())

        repeat = embed(1234)
        torch.testing.assert_close(first, repeat)

        different = embed(4321)
        self.assertFalse(torch.allclose(first, different))

    def test_random_init_encoder_reports_no_pretraining(self):
        encoder = labram_adapter.load_encoder(None)
        self.assertFalse(encoder.checkpoint_report["pretrained"])
        self.assertEqual(encoder.input_chans, [0, 1])
        self.assertEqual(encoder.input_scale, 1.0)

    def test_channel_count_beyond_pos_embed_slots_is_refused(self):
        # pos_embed has 128 electrode slots + CLS; standard_1020 lists 136 names, and
        # anything past slot 128 is the upstream out-of-bounds bug.
        with self.assertRaises(ValueError) as caught:
            labram_adapter.load_encoder(None, n_channels=labram_adapter.MAX_CHANNEL_INDEX + 1)
        message = str(caught.exception)
        self.assertIn("128", message)
        self.assertIn("standard_1020", message)

        with self.assertRaises(ValueError):
            labram_adapter.load_encoder(None, n_channels=0)

    def test_window_must_be_a_whole_number_of_patches_within_time_embed(self):
        with self.assertRaises(ValueError):
            labram_adapter.load_encoder(None, n_samples=254)
        with self.assertRaises(ValueError):
            labram_adapter.load_encoder(None, n_samples=100)
        with self.assertRaises(ValueError):
            labram_adapter.load_encoder(
                None,
                n_samples=(labram_adapter.MAX_TIME_PATCHES + 1) * labram_adapter.PATCH_SAMPLES,
            )

    def test_forward_rejects_a_window_it_was_not_built_for(self):
        import torch

        encoder = labram_adapter.load_encoder(None, n_channels=1, n_samples=200)
        with self.assertRaises(ValueError):
            encoder(torch.zeros(2, 1, 400))
        with self.assertRaises(ValueError):
            encoder(torch.zeros(2, 3, 200))

    def test_input_scale_is_applied_and_defaults_to_no_rescaling(self):
        import torch

        x = torch.as_tensor(np.random.default_rng(7).normal(size=(4, 1, 200)).astype(np.float32))
        torch.manual_seed(11)
        plain = labram_adapter.load_encoder(None)
        torch.manual_seed(11)
        scaled = labram_adapter.load_encoder(None, input_scale=0.01)
        with torch.no_grad():
            self.assertFalse(torch.allclose(plain(x), scaled(x)))
            # Default is a literal no-op, not a hidden unit conversion.
            torch.testing.assert_close(plain(x), plain(x * 1.0))


@unittest.skipUnless(VENDORED, _VENDOR_REASON)
@unittest.skipUnless(REAL_CHECKPOINT.exists(), _WEIGHTS_REASON)
class RealCheckpointTests(unittest.TestCase):
    def test_pretrained_weights_load_with_the_documented_key_surgery(self):
        from neurotwin.benchmarks.fm_forecast.weights_manifest import resolve_checkpoint

        path = resolve_checkpoint(labram_adapter.MODEL_ID, WEIGHTS_DIR)
        encoder = labram_adapter.load_encoder(path, n_channels=1, n_samples=200)
        report = encoder.checkpoint_report

        self.assertTrue(report["pretrained"])
        self.assertEqual(report["checkpoint_key"], "model")
        # Frozen counts for the SHA-pinned checkpoint: 229 entries in the bundle, 224 of
        # them under 'student.', 219 of those loaded into the 245-entry finetune model.
        self.assertEqual(report["checkpoint_entries"], 229)
        self.assertEqual(report["student_entries"], 224)
        self.assertEqual(report["model_entries"], 245)
        self.assertEqual(report["loaded_entries"], 219)
        # Dropped: pretraining-only heads and the pre-pooling norm the finetune model
        # replaces with Identity.
        self.assertEqual(
            report["dropped_keys"],
            ["lm_head.bias", "lm_head.weight", "mask_token", "norm.bias", "norm.weight"],
        )
        self.assertEqual(report["unexpected_keys"], [])
        # Missing keys are only the identity-initialised fc_norm and the zero q/v biases.
        self.assertEqual(len(report["missing_keys"]), 26)
        for key in report["missing_keys"]:
            self.assertTrue(
                key in {"fc_norm.weight", "fc_norm.bias"}
                or key.endswith((".attn.q_bias", ".attn.v_bias")),
                msg=f"unexplained missing key {key}",
            )

    def test_pretrained_embedding_differs_from_the_random_init_twin(self):
        import torch

        x = torch.as_tensor(np.random.default_rng(3).normal(size=(8, 1, 200)).astype(np.float32))
        torch.manual_seed(0)
        pretrained = labram_adapter.load_encoder(REAL_CHECKPOINT, n_channels=1, n_samples=200)
        torch.manual_seed(0)
        random_init = labram_adapter.load_encoder(None, n_channels=1, n_samples=200)

        with torch.no_grad():
            hot = pretrained(x)
            cold = random_init(x)

        self.assertEqual(tuple(hot.shape), (8, labram_adapter.EMBED_DIM))
        self.assertTrue(torch.isfinite(hot).all())
        self.assertFalse(torch.allclose(hot, cold, atol=1e-3))
        self.assertGreater(float((hot - cold).abs().mean()), 1e-3)

    def test_frozen_runner_matches_the_harness_prediction_shape(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig

        task = _synthetic_forecast_task()
        runner = labram_adapter.make_runner(
            WEIGHTS_DIR,
            Condition.FROZEN,
            config=ConditionConfig(
                condition=Condition.FROZEN,
                lr=1e-2,
                epochs=2,
                batch_size=4,
                patience=2,
                device="cpu",
            ),
        )
        predictions = runner(task, 0, 2)

        self.assertEqual(predictions.shape, task.y_test.shape)
        self.assertTrue(np.isfinite(predictions).all())


@unittest.skipUnless(VENDORED, _VENDOR_REASON)
class RunnerTests(unittest.TestCase):
    def test_frozen_runner_on_the_random_init_twin_never_touches_the_weights_dir(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig

        task = _synthetic_forecast_task(seed=5)
        runner = labram_adapter.make_runner(
            Path("/nonexistent-weights-dir"),
            Condition.FROZEN,
            pretrained=False,
            config=ConditionConfig(
                condition=Condition.FROZEN,
                lr=1e-2,
                epochs=2,
                batch_size=4,
                patience=2,
                device="cpu",
            ),
        )
        predictions = runner(task, 1, 2)

        self.assertEqual(predictions.shape, task.y_test.shape)
        self.assertTrue(np.isfinite(predictions).all())

    def test_runner_is_seed_deterministic(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition, ConditionConfig

        task = _synthetic_forecast_task(seed=6)
        config = ConditionConfig(
            condition=Condition.FROZEN,
            lr=1e-2,
            epochs=2,
            batch_size=4,
            patience=2,
            device="cpu",
        )
        runner = labram_adapter.make_runner(
            Path("/nonexistent-weights-dir"),
            Condition.FROZEN,
            pretrained=False,
            config=config,
        )
        np.testing.assert_allclose(runner(task, 3, 2), runner(task, 3, 2), rtol=0, atol=0)

    def test_acausal_infill_is_refused_by_the_condition_framework(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        task = _synthetic_forecast_task(seed=7)
        runner = labram_adapter.make_runner(
            Path("/nonexistent-weights-dir"),
            Condition.ACAUSAL_INFILL,
            pretrained=False,
        )
        with self.assertRaises(NotImplementedError):
            runner(task, 0, 1)


class ContaminationGateTests(unittest.TestCase):
    def test_make_runner_refuses_when_the_pair_is_not_audited_clean(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        with mock.patch.object(labram_adapter, "is_evaluation_permitted", return_value=False):
            with self.assertRaises(labram_adapter.LaBraMAdapterError) as caught:
                labram_adapter.make_runner(WEIGHTS_DIR, Condition.FROZEN)
        self.assertIn("labram_base", str(caught.exception))
        self.assertIn("sleep-edf", str(caught.exception))

    @unittest.skipUnless(VENDORED, _VENDOR_REASON)
    def test_an_already_built_runner_re_checks_the_gate_at_call_time(self):
        from neurotwin.benchmarks.fm_forecast.conditions import Condition

        runner = labram_adapter.make_runner(
            Path("/nonexistent-weights-dir"), Condition.FROZEN, pretrained=False
        )
        task = _synthetic_forecast_task(seed=8)
        with mock.patch.object(labram_adapter, "is_evaluation_permitted", return_value=False):
            with self.assertRaises(labram_adapter.LaBraMAdapterError):
                runner(task, 0, 1)


if __name__ == "__main__":
    unittest.main()

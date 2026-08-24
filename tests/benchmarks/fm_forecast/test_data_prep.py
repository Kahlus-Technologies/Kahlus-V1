import unittest

import numpy as np

from neurotwin.benchmarks.fm_forecast.data_prep import (
    ForecastArrays,
    flatten_targets,
    prepare_forecast_arrays,
    prepare_windows,
    resample_windows,
    to_channels_first,
    unflatten_targets,
)


def _windows(n: int = 6, length: int = 127, channels: int = 1, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, length, channels)).astype(np.float32)


class ResampleWindowsTests(unittest.TestCase):
    def test_shape_and_dtype_for_the_f8_lane_a_context(self):
        # F8: L=127 at 100 Hz (1.27 s). EEGPT runs at 256 Hz and wants a multiple of 64
        # samples, so the largest patch-aligned context not exceeding 1.27 s is 320
        # samples (1.25 s) - plan section 3.2.
        x = _windows()
        out = resample_windows(x, sr_in=100, sr_out=256, n_samples_out=320)
        self.assertEqual(out.shape, (6, 320, 1))
        self.assertEqual(out.dtype, np.float32)

    def test_two_dimensional_input_stays_two_dimensional(self):
        x = _windows(channels=1)[:, :, 0]
        out = resample_windows(x, sr_in=100, sr_out=200, n_samples_out=200)
        self.assertEqual(out.shape, (6, 200))
        self.assertEqual(out.dtype, np.float32)

    def test_natural_length_is_used_when_no_explicit_length_is_given(self):
        x = _windows(length=100)
        self.assertEqual(resample_windows(x, sr_in=100, sr_out=200).shape, (6, 200, 1))
        self.assertEqual(resample_windows(x, sr_in=100, sr_out=50).shape, (6, 50, 1))

    def test_constant_signal_is_preserved_exactly(self):
        # A DC-only window has no other frequency content, so any correct resampler
        # returns the same constant. The zero-padded polyphase form does not.
        for value in (0.0, 3.5, -2.25):
            for sr_out, n_out in ((256, 320), (200, 254), (50, 60)):
                with self.subTest(value=value, sr_out=sr_out):
                    x = np.full((3, 127, 2), value, dtype=np.float32)
                    out = resample_windows(x, sr_in=100, sr_out=sr_out, n_samples_out=n_out)
                    self.assertEqual(out.shape, (3, n_out, 2))
                    np.testing.assert_allclose(out, value, atol=1e-5)

    def test_low_frequency_sine_survives_upsampling(self):
        # 8 whole cycles across the 2.56 s window, so the periodic extension the FFT
        # method assumes is exact and the only error left is float32 rounding.
        frequency = 8.0 / 2.56
        x = np.sin(2 * np.pi * frequency * np.arange(256) / 100.0).astype(np.float32)[None, :, None]
        out = resample_windows(x, sr_in=100, sr_out=200)
        expected = np.sin(2 * np.pi * frequency * np.arange(512) / 200.0)
        self.assertEqual(out.shape, (1, 512, 1))
        np.testing.assert_allclose(out[0, :, 0], expected, atol=1e-5)

    def test_resampling_is_per_window_with_no_cross_window_coupling(self):
        # Leakage hygiene: batching windows together must not change any single window's
        # output, or train/test statistics would bleed into each other.
        x = _windows(n=8, seed=3)
        batched = resample_windows(x, sr_in=100, sr_out=256, n_samples_out=320)
        for index in range(x.shape[0]):
            with self.subTest(index=index):
                single = resample_windows(x[index : index + 1], sr_in=100, sr_out=256, n_samples_out=320)
                np.testing.assert_array_equal(batched[index], single[0])

    def test_channels_are_resampled_independently(self):
        x = _windows(n=4, channels=3, seed=5)
        multi = resample_windows(x, sr_in=100, sr_out=200, n_samples_out=254)
        for channel in range(3):
            with self.subTest(channel=channel):
                single = resample_windows(
                    x[:, :, channel : channel + 1], sr_in=100, sr_out=200, n_samples_out=254
                )
                np.testing.assert_array_equal(multi[:, :, channel], single[:, :, 0])

    def test_resampling_is_deterministic(self):
        x = _windows(seed=7)
        first = resample_windows(x, sr_in=100, sr_out=256, n_samples_out=320)
        second = resample_windows(x, sr_in=100, sr_out=256, n_samples_out=320)
        np.testing.assert_array_equal(first, second)

    def test_trimming_keeps_the_tail_of_the_window(self):
        # The window must keep ending at the same instant, so no entrant sees samples
        # closer to the target than another does.
        x = _windows(n=2, seed=11)
        natural = resample_windows(x, sr_in=100, sr_out=256)
        trimmed = resample_windows(x, sr_in=100, sr_out=256, n_samples_out=320)
        self.assertEqual(natural.shape[1], 325)
        np.testing.assert_array_equal(trimmed, natural[:, natural.shape[1] - 320 :, :])

    def test_short_output_is_edge_padded_at_the_front(self):
        x = _windows(n=2, length=64, seed=13)
        natural = resample_windows(x, sr_in=100, sr_out=100)
        padded = resample_windows(x, sr_in=100, sr_out=100, n_samples_out=70)
        self.assertEqual(padded.shape, (2, 70, 1))
        np.testing.assert_allclose(padded[:, 6:, :], natural, atol=1e-6)
        np.testing.assert_allclose(padded[:, :6, :], np.repeat(natural[:, :1, :], 6, axis=1), atol=1e-6)

    def test_identity_rate_returns_the_same_values(self):
        x = _windows(seed=17)
        np.testing.assert_allclose(resample_windows(x, sr_in=100, sr_out=100), x, atol=1e-6)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            resample_windows(np.zeros((4,), dtype=np.float32), sr_out=200)
        with self.assertRaises(ValueError):
            resample_windows(_windows(), sr_in=0, sr_out=200)
        with self.assertRaises(ValueError):
            resample_windows(_windows(), sr_out=200, n_samples_out=0)


class ShapeHelperTests(unittest.TestCase):
    def test_to_channels_first_matches_the_encoder_layout(self):
        x = _windows(n=5, length=32, channels=2)
        out = to_channels_first(x)
        self.assertEqual(out.shape, (5, 2, 32))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out[3, 1], x[3, :, 1])

    def test_to_channels_first_promotes_two_dimensional_windows(self):
        self.assertEqual(to_channels_first(np.zeros((5, 32), dtype=np.float32)).shape, (5, 1, 32))

    def test_prepare_windows_resamples_then_reorders(self):
        x = _windows(n=3, channels=2)
        out = prepare_windows(x, sr_in=100, sr_out=256, n_samples_out=320)
        self.assertEqual(out.shape, (3, 2, 320))
        self.assertEqual(out.dtype, np.float32)

    def test_flatten_and_unflatten_targets_round_trip(self):
        # F8 lane A targets are a single sample on a single channel: (N, 1, 1).
        y = np.arange(12, dtype=np.float32).reshape(4, 3, 1)
        flat = flatten_targets(y)
        self.assertEqual(flat.shape, (4, 3))
        np.testing.assert_array_equal(unflatten_targets(flat, y.shape), y)

    def test_unflatten_rejects_a_shape_that_does_not_match_the_harness_target(self):
        with self.assertRaises(ValueError):
            unflatten_targets(np.zeros((4, 3), dtype=np.float32), (4, 2, 1))
        with self.assertRaises(ValueError):
            unflatten_targets(np.zeros((3, 1), dtype=np.float32), (4, 1, 1))


class PrepareForecastArraysTests(unittest.TestCase):
    def _task_arrays(self, n_train=10, n_val=4, n_test=6, length=127, channels=1):
        rng = np.random.default_rng(0)
        make_x = lambda n: rng.normal(size=(n, length, channels)).astype(np.float32)  # noqa: E731
        make_y = lambda n: rng.normal(size=(n, 1, channels)).astype(np.float32)  # noqa: E731
        return {
            "x_train": make_x(n_train),
            "y_train": make_y(n_train),
            "x_val": make_x(n_val),
            "y_val": make_y(n_val),
            "x_test": make_x(n_test),
            "y_test": make_y(n_test),
        }

    def test_shapes_follow_the_harness_contract(self):
        arrays = prepare_forecast_arrays(**self._task_arrays(), sr_in=100, sr_out=256, n_samples_out=320)
        self.assertIsInstance(arrays, ForecastArrays)
        self.assertEqual(arrays.x_train.shape, (10, 1, 320))
        self.assertEqual(arrays.y_train.shape, (10, 1))
        self.assertEqual(arrays.x_val.shape, (4, 1, 320))
        self.assertEqual(arrays.x_test.shape, (6, 1, 320))
        self.assertEqual(arrays.test_target_shape, (6, 1, 1))
        self.assertEqual(arrays.n_channels, 1)
        self.assertEqual(arrays.n_samples, 320)
        self.assertEqual(arrays.output_dim, 1)

    def test_multi_channel_lane_b_shapes(self):
        arrays = prepare_forecast_arrays(**self._task_arrays(channels=2), sr_in=100, sr_out=200, n_samples_out=200)
        self.assertEqual(arrays.x_train.shape, (10, 2, 200))
        self.assertEqual(arrays.y_train.shape, (10, 2))
        self.assertEqual(arrays.test_target_shape, (6, 1, 2))
        self.assertEqual(arrays.output_dim, 2)

    def test_missing_validation_split_is_refused(self):
        payload = self._task_arrays()
        payload["x_val"] = None
        with self.assertRaises(ValueError) as caught:
            prepare_forecast_arrays(**payload)
        self.assertIn("validation", str(caught.exception))

    def test_mismatched_test_window_count_is_refused(self):
        payload = self._task_arrays()
        payload["y_test"] = payload["y_test"][:3]
        with self.assertRaises(ValueError):
            prepare_forecast_arrays(**payload)

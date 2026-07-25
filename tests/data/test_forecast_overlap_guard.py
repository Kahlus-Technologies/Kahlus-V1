"""The overlap guard must fail construction of a copy-scorable forecast task.

Finding 4 of docs/research/eval_leakage_audit_2026-07-21.md: nothing in the repo
forbade `forecast_horizon < window_length` without a strictly-future mask, which
is the exact shape that produced the invalidated 128->127/127 result.
"""
from __future__ import annotations

import unittest

import numpy as np

from neurotwin.data.forecast_contract import (
    ForecastProtocolError,
    assert_forecast_scoring_is_leakage_safe,
    strictly_future_metric_mask,
)
from neurotwin.eeg_v1.dataset import build_future_forecasting_task, make_synthetic_eeg_v1_dataset


class ForecastOverlapGuardTests(unittest.TestCase):
    def test_overlapping_windows_without_mask_are_rejected(self) -> None:
        with self.assertRaises(ForecastProtocolError) as ctx:
            assert_forecast_scoring_is_leakage_safe(
                window_length=8, forecast_horizon=1, metric_mask=None
            )
        message = str(ctx.exception)
        self.assertIn("forecast_horizon=1", message)
        self.assertIn("window_length=8", message)

    def test_empty_mask_is_rejected(self) -> None:
        # A mask that selects nothing is as unsafe as no mask at all.
        with self.assertRaises(ForecastProtocolError):
            assert_forecast_scoring_is_leakage_safe(
                window_length=8,
                forecast_horizon=1,
                metric_mask=np.zeros((4, 8, 2), dtype=bool),
            )

    def test_disjoint_windows_pass_without_a_mask(self) -> None:
        # horizon >= window_length: the target starts at/after the input ends.
        assert_forecast_scoring_is_leakage_safe(
            window_length=8, forecast_horizon=8, metric_mask=None
        )
        assert_forecast_scoring_is_leakage_safe(
            window_length=8, forecast_horizon=16, metric_mask=None
        )

    def test_overlapping_windows_pass_with_a_strictly_future_mask(self) -> None:
        y = np.zeros((4, 8, 2), dtype=np.float32)
        mask = strictly_future_metric_mask(y, forecast_horizon=1, input_length=8)
        assert_forecast_scoring_is_leakage_safe(
            window_length=8, forecast_horizon=1, metric_mask=mask
        )

    def test_real_builder_still_satisfies_the_guard(self) -> None:
        # The shipped (8, 1) default is overlapping but masked, so it must build.
        dataset = make_synthetic_eeg_v1_dataset(seed=0, n_subjects=6, n_time=48, n_channels=4)
        task = build_future_forecasting_task(
            dataset, window_length=8, forecast_horizon=1, stride=2
        )
        self.assertIsNotNone(task.metric_mask)
        self.assertTrue(bool(np.asarray(task.metric_mask).any()))
        self.assertTrue(task.metadata["strictly_future_metric_mask"])


if __name__ == "__main__":
    unittest.main()

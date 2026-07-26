"""Arousal-class split: respiratory and non-respiratory must not be pooled.

The autonomic-precedence hypothesis is supported for spontaneous/movement
arousals and contradicted for respiratory-event arousals in apnea cohorts. slpdb
is an apnea cohort, so pooling averages a plausible effect against a documented
null. These tests pin the token classification and the invariant that the pooled
mask is exactly the union of the two classes.
"""
from __future__ import annotations

import unittest

from neurotwin.adapters.slpdb import (
    AROUSAL_TOKENS,
    NON_RESPIRATORY_AROUSAL_TOKENS,
    RESPIRATORY_AROUSAL_TOKENS,
    parse_stage_and_arousal,
    parse_stage_and_arousal_classes,
)


class ArousalClassTests(unittest.TestCase):
    def test_classes_partition_the_arousal_tokens(self) -> None:
        self.assertEqual(
            RESPIRATORY_AROUSAL_TOKENS | NON_RESPIRATORY_AROUSAL_TOKENS, AROUSAL_TOKENS
        )
        self.assertEqual(RESPIRATORY_AROUSAL_TOKENS & NON_RESPIRATORY_AROUSAL_TOKENS, set())

    def test_respiratory_tokens(self) -> None:
        for token in ("HA", "CAA", "X"):
            stage, respiratory, non_respiratory = parse_stage_and_arousal_classes(f"2 {token}")
            self.assertEqual(stage, "2")
            self.assertTrue(respiratory, token)
            self.assertFalse(non_respiratory, token)

    def test_non_respiratory_tokens(self) -> None:
        for token in ("A", "LA"):
            stage, respiratory, non_respiratory = parse_stage_and_arousal_classes(f"2 {token}")
            self.assertFalse(respiratory, token)
            self.assertTrue(non_respiratory, token)

    def test_non_arousal_events_are_neither_class(self) -> None:
        # L is a leg movement without arousal; H a hypopnea without one.
        for note in ("4 L L", "2 H", "W", ""):
            _, respiratory, non_respiratory = parse_stage_and_arousal_classes(note)
            self.assertFalse(respiratory, note)
            self.assertFalse(non_respiratory, note)

    def test_one_epoch_can_carry_both_classes(self) -> None:
        _, respiratory, non_respiratory = parse_stage_and_arousal_classes("2 LA HA")
        self.assertTrue(respiratory)
        self.assertTrue(non_respiratory)

    def test_pooled_view_is_the_union(self) -> None:
        for note in ("2 LA", "2 HA", "2 LA HA", "4 L L", "2 CAA", "1 X", "3 A"):
            _, respiratory, non_respiratory = parse_stage_and_arousal_classes(note)
            _, pooled = parse_stage_and_arousal(note)
            self.assertEqual(pooled, respiratory or non_respiratory, note)

    def test_nul_bytes_are_stripped_before_classification(self) -> None:
        _, respiratory, non_respiratory = parse_stage_and_arousal_classes("2 HA\x00\x00")
        self.assertTrue(respiratory)
        self.assertFalse(non_respiratory)


if __name__ == "__main__":
    unittest.main()

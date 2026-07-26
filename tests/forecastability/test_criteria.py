"""GateCriteria must reproduce each module's current rule exactly.

The point of the extraction is that it is behaviour-preserving, so these tests
pin the CURRENT constants -- including the ones that disagree with each other.
Reconciling them is a separate change; a test here failing means the extraction
changed a decision, which is the one thing it must not do.
"""
from __future__ import annotations

import unittest

from neurotwin.forecastability.criteria import (
    REGISTRY,
    GateCriteria,
    GateInputs,
    evaluate,
    passes,
)


def inputs(**kw) -> GateInputs:
    base = dict(rfs_bits=0.10, rfs_ci_low=0.01, control_bits={"shuffle": 0.0, "shift": 0.0},
                positive_events=1000, event_patients=20, n_subjects=20)
    base.update(kw)
    return GateInputs(**base)


class RegistryPinsCurrentConstants(unittest.TestCase):
    def test_effect_floors_as_they_stand(self) -> None:
        self.assertEqual(REGISTRY["m5_integrated"].effect_floor_bits, 0.001)
        self.assertEqual(REGISTRY["rfs_horizon_real"].effect_floor_bits, 0.02)
        self.assertEqual(REGISTRY["passive_pci_real"].effect_floor_bits, 0.02)
        self.assertEqual(REGISTRY["m1_known"].effect_floor_bits, 0.03)
        self.assertIsNone(REGISTRY["m3_seizure"].effect_floor_bits)
        self.assertIsNone(REGISTRY["m2_real_sleep_edf"].effect_floor_bits)

    def test_control_ratios_as_they_stand(self) -> None:
        self.assertEqual(REGISTRY["rfs_horizon_real"].control_ratio, 0.40)
        self.assertEqual(REGISTRY["m1_known"].control_ratio, 0.40)
        self.assertEqual(REGISTRY["passive_pci_real"].control_ratio, 0.40)
        self.assertEqual(REGISTRY["m3_seizure"].control_ratio, 0.50)
        self.assertEqual(REGISTRY["m5_integrated"].control_ratio, 0.50)
        self.assertEqual(REGISTRY["m2_real_sleep_edf"].control_ratio, 0.50)


class EvaluateMatchesTheOldPredicates(unittest.TestCase):
    def test_horizon_real_accepts_what_it_accepted(self) -> None:
        c = REGISTRY["rfs_horizon_real"]
        self.assertTrue(passes(c, inputs(rfs_bits=0.10, rfs_ci_low=0.01,
                                         control_bits={"a": 0.03, "b": 0.01})))

    def test_horizon_real_rejects_effect_at_the_floor(self) -> None:
        c = REGISTRY["rfs_horizon_real"]
        self.assertIn("effect_below_floor", evaluate(c, inputs(rfs_bits=0.02)))
        self.assertNotIn("effect_below_floor", evaluate(c, inputs(rfs_bits=0.021)))

    def test_horizon_real_rejects_control_at_forty_percent(self) -> None:
        c = REGISTRY["rfs_horizon_real"]
        # limit is 0.4 * 0.10. Exact-boundary behaviour is float-dependent
        # (0.4 * 0.10 == 0.04000000000000001), so assert either side of it --
        # the original code has the same property.
        self.assertIn("control_too_close_a", evaluate(c, inputs(control_bits={"a": 0.05})))
        self.assertNotIn("control_too_close_a", evaluate(c, inputs(control_bits={"a": 0.03})))

    def test_m3_certifies_without_any_effect_floor(self) -> None:
        """Pins the audit finding: m3 has no floor, so a tiny effect clears it."""
        c = REGISTRY["m3_seizure"]
        tiny = inputs(rfs_bits=0.0001, rfs_ci_low=0.00001,
                      control_bits={"a": 0.0}, positive_events=1000, event_patients=20)
        self.assertNotIn("effect_below_floor", evaluate(c, tiny))
        self.assertTrue(passes(c, tiny), "m3 certifies a 0.0001-bit effect today")

    def test_probe_failures_block(self) -> None:
        c = REGISTRY["rfs_horizon_real"]
        self.assertIn("nuisance_probe_contamination",
                      evaluate(c, inputs(probe_failures=("patient",))))


class SignGuard(unittest.TestCase):
    """The control check is undefined for a non-positive effect.

    Written as `control >= full * ratio` with full negative, the bar rises ABOVE
    the effect and the comparison inverts. Several modules carry that form
    unguarded. Here it must never silently pass.
    """

    def test_negative_effect_does_not_silently_pass_the_control_check(self) -> None:
        c = GateCriteria(name="t", anchor="test", effect_floor_bits=None,
                         require_ci_low_above_zero=False)
        # old form: -0.20 >= -0.10 * 0.4 == -0.04  ->  False  ->  no failure
        failures = evaluate(c, inputs(rfs_bits=-0.10, rfs_ci_low=-0.20,
                                      control_bits={"a": -0.20}))
        self.assertIn("control_check_undefined_for_nonpositive_effect", failures)
        self.assertFalse(passes(c, inputs(rfs_bits=-0.10, rfs_ci_low=-0.20,
                                          control_bits={"a": -0.20})))

    def test_zero_effect_also_guarded(self) -> None:
        c = GateCriteria(name="t", anchor="test", effect_floor_bits=None,
                         require_ci_low_above_zero=False)
        self.assertIn("control_check_undefined_for_nonpositive_effect",
                      evaluate(c, inputs(rfs_bits=0.0, control_bits={"a": 0.0})))


class CriteriaSerialise(unittest.TestCase):
    def test_describe_round_trips_into_a_report(self) -> None:
        d = REGISTRY["m2_real_sleep_edf"].describe()
        self.assertEqual(d["control_ratio"], 0.50)
        self.assertIsNone(d["effect_floor_bits"])
        self.assertIn("m2.py", d["anchor"])


if __name__ == "__main__":
    unittest.main()

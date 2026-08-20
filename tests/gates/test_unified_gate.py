import tempfile
import unittest
from pathlib import Path

from neurotwin.gates import (
    NARROW_CLAIM_SCOPES,
    evaluate_gate,
    read_evidence_gate,
    write_evidence_gate,
)


def _passing_kwargs(**overrides):
    base = dict(
        branch="v2",
        dataset="dual_field_synthetic",
        split_audit_passed=True,
        baseline_table_present=True,
        finite_metrics=True,
        calibration_checked=True,
        claim_scope="synthetic_dual_field_recovery",
    )
    base.update(overrides)
    return base


class UnifiedGateTests(unittest.TestCase):
    def test_schema_fields_present(self):
        gate = evaluate_gate(**_passing_kwargs())
        for key in (
            "branch",
            "dataset",
            "split_audit_passed",
            "baseline_table_present",
            "finite_metrics",
            "calibration_checked",
            "claim_scope",
            "scientific_claim_allowed",
            "failure_reasons",
        ):
            self.assertIn(key, gate)

    def test_gate_fails_when_baselines_missing(self):
        gate = evaluate_gate(**_passing_kwargs(baseline_table_present=False))
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("baseline table missing" in reason for reason in gate["failure_reasons"]))

    def test_gate_fails_on_non_finite_metrics(self):
        gate = evaluate_gate(**_passing_kwargs(finite_metrics=False))
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("non-finite metrics" in reason for reason in gate["failure_reasons"]))

    def test_gate_fails_when_claim_scope_too_broad(self):
        gate = evaluate_gate(**_passing_kwargs(claim_scope="beats_all_baselines_sota"))
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("claim scope too broad" in reason for reason in gate["failure_reasons"]))

    def test_gate_passes_only_for_narrow_synthetic_claim(self):
        gate = evaluate_gate(**_passing_kwargs())
        self.assertTrue(gate["scientific_claim_allowed"])
        self.assertEqual(gate["failure_reasons"], [])
        self.assertIn(gate["claim_scope"], NARROW_CLAIM_SCOPES)

    def test_unknown_branch_blocks_claim(self):
        gate = evaluate_gate(**_passing_kwargs(branch="moonshot"))
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("unknown branch" in reason for reason in gate["failure_reasons"]))

    def test_extra_failure_reasons_block_claim(self):
        gate = evaluate_gate(**_passing_kwargs(extra_failure_reasons=["required task quarantined"]))
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertIn("required task quarantined", gate["failure_reasons"])

    def test_ktm_training_harness_scope_allowed_when_checks_pass(self):
        gate = evaluate_gate(
            **_passing_kwargs(branch="v3", dataset="ktm_training_synthetic",
                              claim_scope="synthetic_ktm_training_harness")
        )
        self.assertTrue(gate["scientific_claim_allowed"])
        self.assertIn("synthetic_ktm_training_harness", NARROW_CLAIM_SCOPES)

    def test_ktm_recovery_scope_blocked_until_earned(self):
        # The harness scope passing must NOT imply the stronger recovery scope: a
        # not-yet-earned recovery claim is blocked by attaching an explicit reason.
        gate = evaluate_gate(
            **_passing_kwargs(
                branch="v3", dataset="ktm_training_synthetic",
                claim_scope="synthetic_ktm_recovery",
                extra_failure_reasons=["KTM did not beat baselines on locked metrics"],
            )
        )
        self.assertFalse(gate["scientific_claim_allowed"])
        self.assertTrue(any("did not beat baselines" in r for r in gate["failure_reasons"]))

    def test_eeg_fm_ranking_scope_allowed_when_every_check_passes(self):
        gate = evaluate_gate(
            **_passing_kwargs(
                branch="v1",
                dataset="sleep-edf-expanded-sleep-cassette",
                claim_scope="external_pretrained_eeg_baseline_ranking_sleep_edf_forecast",
            )
        )
        self.assertTrue(gate["scientific_claim_allowed"])
        self.assertIn("external_pretrained_eeg_baseline_ranking_sleep_edf_forecast", NARROW_CLAIM_SCOPES)

    def test_eeg_fm_ranking_scope_still_fails_closed_on_a_failed_check(self):
        # Being on the allowlist only makes the scope eligible; the conjunctive gate
        # still has to pass, and a benchmark-specific failure reason still blocks it.
        for override in (
            {"split_audit_passed": False},
            {"finite_metrics": False},
            {"extra_failure_reasons": ["random-init architectural twin control is missing"]},
        ):
            with self.subTest(override=sorted(override)):
                gate = evaluate_gate(
                    **_passing_kwargs(
                        branch="v1",
                        dataset="sleep-edf-expanded-sleep-cassette",
                        claim_scope="external_pretrained_eeg_baseline_ranking_sleep_edf_forecast",
                        **override,
                    )
                )
                self.assertFalse(gate["scientific_claim_allowed"])
                self.assertTrue(gate["failure_reasons"])

    def test_no_sota_or_superiority_scope_is_reachable(self):
        for scope in (
            "beats_all_baselines_sota",
            "kahlus_beats_foundation_models",
            "external_pretrained_eeg_baseline_ranking_any_dataset",
            "clinical_seizure_prediction",
        ):
            with self.subTest(scope=scope):
                self.assertNotIn(scope, NARROW_CLAIM_SCOPES)

    def test_round_trip_write_read(self):
        gate = evaluate_gate(**_passing_kwargs())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence_gate.json"
            write_evidence_gate(path, gate)
            loaded = read_evidence_gate(path)
        self.assertEqual(loaded["branch"], "v2")
        self.assertEqual(loaded["scientific_claim_allowed"], gate["scientific_claim_allowed"])


if __name__ == "__main__":
    unittest.main()

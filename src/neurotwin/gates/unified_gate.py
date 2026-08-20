"""Branch-aware evidence gate that blocks unsupported scientific claims.

The gate writes/reads the unified-dossier JSON schema::

    {
      "branch": "v1 | v2 | v3 | em | researchdock | stf",
      "dataset": "...",
      "split_audit_passed": true,
      "baseline_table_present": true,
      "finite_metrics": true,
      "calibration_checked": true,
      "claim_scope": "...",
      "scientific_claim_allowed": false,
      "failure_reasons": []
    }

``scientific_claim_allowed`` is True only when every required boolean is True, the
``claim_scope`` is one of the narrow synthetic/audit scopes in :data:`NARROW_CLAIM_SCOPES`,
and no extra failure reasons were supplied. Anything broader is blocked with a
human-readable reason. This is a falsification gate, not a results generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from neurotwin.repro import write_json

GATE_SCHEMA = "kahlus.unified_evidence_gate.v1"

ALLOWED_BRANCHES: frozenset[str] = frozenset({"v1", "v2", "v3", "em", "researchdock", "stf"})

#: Claim scopes narrow enough to be eligible (only on a fully passing gate). Anything
#: outside this allowlist is treated as "too broad" and blocks the claim. There is no
#: scope here that permits SOTA / baseline-superiority / clinical claims by design.
#: ``synthetic_ktm_training_harness`` covers infrastructure readiness only (training runs,
#: loss decreases, checkpoint/resume, bundle writes); it never implies the model recovered
#: anything. ``synthetic_ktm_recovery`` is the stronger claim and is earned only when a trained
#: KTM actually beats strong baselines on locked held-out metrics.
#: ``external_pretrained_eeg_baseline_ranking_sleep_edf_forecast`` is the single scope that
#: touches a cross-model ranking. It is pinned to one dataset, one protocol and a
#: contamination-audited entrant slate, and it still permits no SOTA, generic-superiority,
#: or clinical language; see the inline comment on its entry below.
NARROW_CLAIM_SCOPES: frozenset[str] = frozenset(
    {
        "none",
        "eeg_fewshot_adaptation_benchmark_ready",
        "eeg_future_forecasting_benchmark_ready",
        # Permits reporting a *ranking* of named external pretrained EEG models against
        # the internal baseline ladder, on the Sleep-EDF ``kahlus.forecast.v2_nonoverlap``
        # forecasting task only, at declared horizons, under a declared symmetric tuning
        # budget, with every ranked entrant ``clean`` in ``pretraining_contamination``.
        # It does NOT permit: SOTA claims, generic model-superiority claims, claims about
        # the external models' published tasks, claims on any other dataset, claims from
        # the acausal masked-infilling condition, or any clinical claim. Being in this
        # allowlist only makes the scope *eligible*: ``scientific_claim_allowed`` still
        # requires every other gate boolean to pass and no extra failure reason.
        # Governance: added on explicit human sign-off (Krish Garg, 2026-08-19), per
        # ``docs/research/eeg_fm_benchmark_plan_2026-08-18.md`` section 6, which forbids
        # an agent from adding it unilaterally.
        "external_pretrained_eeg_baseline_ranking_sleep_edf_forecast",
        "stf_epilepsy_benchmark_definition_ready",
        "synthetic_plumbing_only",
        "synthetic_dual_field_recovery",
        "synthetic_transition_gym",
        "synthetic_transition_operator_recovery",
        "synthetic_ktm_training_harness",
        "synthetic_ktm_recovery",
        "em_artifact_audit_no_human",
        "em_stage0_artifact_audit",
        "researchdock_synthetic_response_profile",
    }
)


def evaluate_gate(
    *,
    branch: str,
    dataset: str,
    split_audit_passed: bool,
    baseline_table_present: bool,
    finite_metrics: bool,
    calibration_checked: bool,
    claim_scope: str,
    extra_failure_reasons: Iterable[str] = (),
) -> dict[str, Any]:
    """Evaluate the gate and return the dossier-schema payload.

    All checks are conjunctive: a single failed check forces
    ``scientific_claim_allowed=False`` and records why.
    """

    branch = str(branch)
    claim_scope = str(claim_scope)
    failure_reasons: list[str] = []

    if branch not in ALLOWED_BRANCHES:
        failure_reasons.append(
            f"unknown branch {branch!r}; expected one of {sorted(ALLOWED_BRANCHES)}"
        )
    if not str(dataset).strip():
        failure_reasons.append("dataset identifier is empty")
    if not bool(split_audit_passed):
        failure_reasons.append("split audit did not pass")
    if not bool(baseline_table_present):
        failure_reasons.append("baseline table missing; cannot rank against baselines")
    if not bool(finite_metrics):
        failure_reasons.append("non-finite metrics present")
    if not bool(calibration_checked):
        failure_reasons.append("calibration not checked")
    if claim_scope not in NARROW_CLAIM_SCOPES:
        failure_reasons.append(
            f"claim scope too broad: {claim_scope!r} is not in the narrow synthetic allowlist"
        )
    failure_reasons.extend(str(reason) for reason in extra_failure_reasons if str(reason).strip())

    scientific_claim_allowed = not failure_reasons

    return {
        "schema": GATE_SCHEMA,
        "branch": branch,
        "dataset": str(dataset),
        "split_audit_passed": bool(split_audit_passed),
        "baseline_table_present": bool(baseline_table_present),
        "finite_metrics": bool(finite_metrics),
        "calibration_checked": bool(calibration_checked),
        "claim_scope": claim_scope,
        "scientific_claim_allowed": scientific_claim_allowed,
        "failure_reasons": failure_reasons,
    }


def write_evidence_gate(path: str | Path, gate: dict[str, Any]) -> Path:
    """Persist a gate payload as pretty, sorted JSON (reuses repro.write_json)."""

    return write_json(path, gate)


def read_evidence_gate(path: str | Path) -> dict[str, Any]:
    """Load a gate payload written by :func:`write_evidence_gate`."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence gate at {path} is not a JSON object")
    return payload

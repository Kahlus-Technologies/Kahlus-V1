"""One definition of what a gate decides, and the per-module constants as they stand.

Ten predicates across this package ask the same six questions -- minimum events,
minimum subjects, effect-size floor, CI-low above zero, control ratio, probe
failures -- and differ only in the dict key path into the payload and in the
constant used. That is why the constants drifted: there was no place a reviewer
could see them side by side.

The registry below records what each module does TODAY, transcribed from source
with file:line anchors, not what it should do. Reconciling the divergence is a
separate, reviewable change: folding a threshold move into a refactor is exactly
how a published result's meaning changes without anyone noticing.

Known divergence at the time of writing:

  effect floor   0.001 (m5) / 0.02 (_rfs_eval, passive_pci) / 0.03 (m1, m4)
                 / none (m2, m3)
  control ratio  0.40 (_rfs_eval, m1, m2-synthetic, passive_pci)
                 / 0.50 (m2-real, m3, m5)

`evaluate` returns the failure reasons rather than a bool, so a caller wanting a
boolean uses `not evaluate(...)` and a caller wanting an EvidenceDecision feeds
them straight into `failed_requirements`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Sentinel for "this module applies no effect-size floor". Distinct from 0.0,
#: which would still reject a negative effect.
NO_FLOOR = None


@dataclass(frozen=True)
class GateCriteria:
    """The six questions a gate asks, with the constants that answer them."""

    name: str
    #: file:line where these constants live today, so a reviewer can check them.
    anchor: str
    effect_floor_bits: float | None = NO_FLOOR
    control_ratio: float = 0.40
    require_ci_low_above_zero: bool = True
    min_positive_events: int = 0
    min_event_patients: int = 0
    min_subjects: int = 0
    require_no_probe_failures: bool = True

    def describe(self) -> dict[str, Any]:
        """Serialisable form, so the rule can travel in the report it produced."""
        return {
            "name": self.name,
            "anchor": self.anchor,
            "effect_floor_bits": self.effect_floor_bits,
            "control_ratio": self.control_ratio,
            "require_ci_low_above_zero": self.require_ci_low_above_zero,
            "min_positive_events": self.min_positive_events,
            "min_event_patients": self.min_event_patients,
            "min_subjects": self.min_subjects,
            "require_no_probe_failures": self.require_no_probe_failures,
        }


@dataclass(frozen=True)
class GateInputs:
    """A gate decision's inputs, already extracted from whatever payload shape.

    Extraction is per payload family (horizon / state / flat metrics); the
    decision itself is not.
    """

    rfs_bits: float
    rfs_ci_low: float
    control_bits: dict[str, float] = field(default_factory=dict)
    positive_events: int = 0
    event_patients: int = 0
    n_subjects: int = 0
    probe_failures: tuple[str, ...] = ()


def evaluate(criteria: GateCriteria, inputs: GateInputs) -> tuple[str, ...]:
    """Return failure reasons. Empty means the gate passes."""
    failures: list[str] = []

    if inputs.positive_events < criteria.min_positive_events:
        failures.append("underpowered_positive_events")
    if inputs.event_patients < criteria.min_event_patients:
        failures.append("underpowered_event_patients")
    if inputs.n_subjects < criteria.min_subjects:
        failures.append("underpowered_subjects")
    if criteria.effect_floor_bits is not None and inputs.rfs_bits <= criteria.effect_floor_bits:
        failures.append("effect_below_floor")
    if criteria.require_ci_low_above_zero and inputs.rfs_ci_low <= 0.0:
        failures.append("rfs_ci_includes_zero")

    # The control comparison is only meaningful for a positive effect. Written as
    # `control >= full * ratio` with a negative `full`, the bar moves ABOVE the
    # effect and the test inverts -- the worse the primary performs, the easier
    # its own control check becomes. Several modules carry that form without a
    # guard; it is currently masked by the CI-low check firing first, which is
    # incidental. Guard it once, here.
    if inputs.rfs_bits > 0.0:
        limit = inputs.rfs_bits * criteria.control_ratio
        for control_name, bits in sorted(inputs.control_bits.items()):
            if bits >= limit:
                failures.append(f"control_too_close_{control_name}")
    elif inputs.control_bits:
        failures.append("control_check_undefined_for_nonpositive_effect")

    if criteria.require_no_probe_failures and inputs.probe_failures:
        failures.append("nuisance_probe_contamination")

    return tuple(failures)


def passes(criteria: GateCriteria, inputs: GateInputs) -> bool:
    """Boolean form, for callers that want the bool rather than the reasons."""
    return not evaluate(criteria, inputs)


#: What each module does TODAY. Verbatim, not reconciled. Any change here is a
#: scientific decision and belongs in its own commit with a justification.
REGISTRY: dict[str, GateCriteria] = {
    "rfs_horizon_real": GateCriteria(
        name="rfs_horizon_real",
        anchor="_rfs_eval.py:167-191",
        effect_floor_bits=0.02,
        control_ratio=0.40,
    ),
    "rfs_horizon_synthetic_known": GateCriteria(
        name="rfs_horizon_synthetic_known",
        anchor="_rfs_eval.py:137-151",
        effect_floor_bits=0.02,
        control_ratio=0.40,
        require_ci_low_above_zero=False,
        min_positive_events=40,
    ),
    "passive_pci_real": GateCriteria(
        name="passive_pci_real",
        anchor="passive_pci.py:387-409",
        effect_floor_bits=0.02,
        control_ratio=0.40,
        min_positive_events=100,
        min_subjects=8,
    ),
    "m1_known": GateCriteria(
        name="m1_known",
        anchor="m1.py:437-448",
        effect_floor_bits=0.03,
        control_ratio=0.40,
        min_positive_events=40,
        min_event_patients=8,
        require_no_probe_failures=False,
    ),
    "m5_integrated": GateCriteria(
        name="m5_integrated",
        anchor="m5.py:345-349",
        effect_floor_bits=0.001,
        control_ratio=0.50,
        min_event_patients=8,
        require_no_probe_failures=False,
    ),
    "m3_seizure": GateCriteria(
        name="m3_seizure",
        anchor="m3.py:392-417",
        effect_floor_bits=NO_FLOOR,  # m3 has no floor; see docs/audit
        control_ratio=0.50,
        min_positive_events=100,
        min_event_patients=8,
    ),
    "m2_real_sleep_edf": GateCriteria(
        name="m2_real_sleep_edf",
        anchor="m2.py:583-600",
        effect_floor_bits=NO_FLOOR,  # m2 has no floor
        control_ratio=0.50,  # looser than its own synthetic arm at 0.40
    ),
}

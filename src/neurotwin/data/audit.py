from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations

from neurotwin.data.leakage import check_manifest_leakage
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest


POLICY_KEYS = {
    "subject": ("subject_id",),
    "session": ("session_id",),
    "site": ("site_id",),
    "dataset": ("dataset",),
    "run": ("metadata.run_id",),
}


@dataclass(frozen=True)
class AuditReport:
    passed: bool
    violations: tuple[str, ...]
    checked: tuple[str, ...]


def audit_split_manifest(
    manifest: SplitManifest,
    policy: str,
    forbidden_metadata_fields: tuple[str, ...] = ("label", "target", "target_label", "task_label", "diagnosis"),
) -> AuditReport:
    violations: list[str] = []
    checked: list[str] = ["record_reuse", "group_overlap", "metadata_labels", "window_overlap"]
    keys = tuple(key for key in POLICY_KEYS.get(policy, ()) if not key.startswith("metadata."))
    if keys:
        leakage = check_manifest_leakage(manifest, keys=keys)
        violations.extend(leakage.violations)
    if policy == "run":
        violations.extend(_metadata_group_overlap(manifest, "run_id"))
    violations.extend(_forbidden_metadata(manifest.all_records, forbidden_metadata_fields))
    violations.extend(_window_overlap(manifest))
    return AuditReport(not violations, tuple(violations), tuple(checked))


def audit_person_level_split(
    manifest: SplitManifest,
    *,
    person_key: Callable[[RecordingRecord], str] | None = None,
) -> AuditReport:
    """Catch nights-as-identities leakage that subject_id auditing cannot see.

    ``audit_split_manifest`` verifies that declared ``subject_id`` values do not
    span splits, but it trusts those labels. When two nights of one person are
    labelled as two subjects, that check passes while the person sits on both
    sides of the split -- the failure behind the invalidated 26-person-overlap
    run.

    ``person_key`` maps a record to a stable person identifier. Supply the
    dataset's convention (e.g. Sleep-EDF ``SC4nnN`` -> ``nn``, slpdb
    ``slp01a``/``slp01b`` -> ``01``). Without it this falls back to
    ``metadata['person_id']`` and, failing that, reports the check as unverified
    rather than silently passing.
    """

    violations: list[str] = []
    checked: list[str] = ["person_group_overlap", "nights_as_identities"]

    def default_key(record: RecordingRecord) -> str | None:
        value = record.metadata.get("person_id")
        return str(value) if value not in (None, "") else None

    resolve = person_key or default_key
    splits = {"train": manifest.train, "val": manifest.val, "test": manifest.test}
    per_split: dict[str, set[str]] = {}
    unresolved = 0
    for split, records in splits.items():
        keys: set[str] = set()
        for record in records:
            key = resolve(record)
            if key is None:
                unresolved += 1
                continue
            keys.add(str(key))
        per_split[split] = keys

    if unresolved:
        violations.append(
            f"person identity unverified for {unresolved} record(s): no person_key supplied and "
            "no metadata['person_id'] present, so nights-as-identities leakage cannot be ruled out"
        )
        return AuditReport(False, tuple(violations), tuple(checked))

    for left, right in combinations(per_split, 2):
        overlap = per_split[left] & per_split[right]
        if overlap:
            violations.append(
                f"person leakage between {left} and {right}: {', '.join(sorted(overlap))}"
            )

    all_records = manifest.all_records
    people = set().union(*per_split.values()) if per_split else set()
    declared = {record.subject_id for record in all_records}
    if len(people) < len(declared):
        # Multi-night data: more declared subject_ids than real people. Legal, but
        # only because the person-level overlap check above passed.
        checked.append(f"collapsed {len(declared)} subject_ids to {len(people)} people")

    return AuditReport(not violations, tuple(violations), tuple(checked))


def _forbidden_metadata(records: list[RecordingRecord], fields: tuple[str, ...]) -> list[str]:
    violations = []
    lowered = tuple(field.lower() for field in fields)
    for record in records:
        for key in record.metadata:
            if key.lower() in lowered:
                violations.append(f"forbidden metadata field {key!r} in record {record.record_id!r}")
    return violations


def _metadata_group_overlap(manifest: SplitManifest, metadata_key: str) -> list[str]:
    split_values = {
        "train": {str(record.metadata.get(metadata_key)) for record in manifest.train if metadata_key in record.metadata},
        "val": {str(record.metadata.get(metadata_key)) for record in manifest.val if metadata_key in record.metadata},
        "test": {str(record.metadata.get(metadata_key)) for record in manifest.test if metadata_key in record.metadata},
    }
    violations = []
    for left, right in combinations(split_values, 2):
        overlap = split_values[left] & split_values[right]
        if overlap:
            violations.append(f"metadata.{metadata_key} leakage between {left} and {right}: {', '.join(sorted(overlap))}")
    return violations


def _window_overlap(manifest: SplitManifest) -> list[str]:
    by_source: dict[str, list[tuple[str, float, float, str]]] = {}
    for split, records in (("train", manifest.train), ("val", manifest.val), ("test", manifest.test)):
        for record in records:
            source = str(record.metadata.get("source_record_id", record.record_id))
            start = float(record.metadata.get("window_start", record.start_time))
            end = float(record.metadata.get("window_end", record.end_time))
            by_source.setdefault(source, []).append((split, start, end, record.record_id))

    violations: list[str] = []
    for source, windows in by_source.items():
        for left, right in combinations(windows, 2):
            left_split, left_start, left_end, left_id = left
            right_split, right_start, right_end, right_id = right
            if left_split == right_split:
                continue
            if max(left_start, right_start) < min(left_end, right_end):
                violations.append(f"window overlap for source {source!r}: {left_id} vs {right_id}")
    return violations

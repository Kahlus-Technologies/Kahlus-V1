"""Finding 5: subject_id auditing cannot see nights-as-identities leakage.

The historical failure labelled two nights of one person as two subjects. The
declared-subject_id check passes on that manifest -- the ids genuinely do not
overlap -- while the person is on both sides of the split. These tests pin the
distinction using the two real conventions in the repo:

  Sleep-EDF  SC4nnN  -> subject nn, night N   (SC4001/SC4002 are one person)
  slpdb      slpNNx  -> subject NN            (slp01a/slp01b are one person)
"""
from __future__ import annotations

import unittest

from neurotwin.data.audit import audit_person_level_split, audit_split_manifest
from neurotwin.data.split_manifest import RecordingRecord, SplitManifest


def _record(record_id: str, subject_id: str, **metadata) -> RecordingRecord:
    return RecordingRecord(
        record_id=record_id,
        modality="eeg",
        dataset="sleep_edf",
        subject_id=subject_id,
        session_id="ses-01",
        site_id="site-a",
        start_time=0.0,
        end_time=100.0,
        metadata=metadata,
    )


def _sleep_edf_person(record: RecordingRecord) -> str:
    """SC4nnN -> nn. SC4001 and SC4002 are nights 1 and 2 of subject 00."""
    return record.record_id[3:5]


def _manifest(train, val, test) -> SplitManifest:
    records = train + val + test
    return SplitManifest(
        policy="subject",
        seed=0,
        train=train,
        val=val,
        test=test,
        record_hashes={r.record_id: r.stable_hash() for r in records},
    )


class PersonLevelSplitAuditTests(unittest.TestCase):
    def test_nights_as_identities_passes_subject_audit_but_fails_person_audit(self) -> None:
        # SC4001 (subject 00 night 1) in train, SC4002 (subject 00 night 2) in test.
        # Declared subject_ids differ, so the subject-level audit sees no overlap.
        train = [_record("SC4001E0", "SC4001")]
        test = [_record("SC4002E0", "SC4002")]
        manifest = _manifest(train, [], test)

        subject_report = audit_split_manifest(manifest, policy="subject")
        self.assertTrue(
            subject_report.passed,
            "declared subject_ids do not overlap, so this check cannot catch the bug",
        )

        person_report = audit_person_level_split(manifest, person_key=_sleep_edf_person)
        self.assertFalse(person_report.passed)
        self.assertTrue(
            any("person leakage between train and test" in v for v in person_report.violations),
            person_report.violations,
        )
        self.assertTrue(any("00" in v for v in person_report.violations))

    def test_person_held_out_split_passes(self) -> None:
        # Both nights of subject 00 in train; both nights of subject 01 in test.
        train = [_record("SC4001E0", "SC4001"), _record("SC4002E0", "SC4002")]
        test = [_record("SC4011E0", "SC4011"), _record("SC4012E0", "SC4012")]
        report = audit_person_level_split(_manifest(train, [], test), person_key=_sleep_edf_person)
        self.assertTrue(report.passed, report.violations)
        self.assertTrue(
            any("collapsed 4 subject_ids to 2 people" in c for c in report.checked),
            report.checked,
        )

    def test_missing_person_identity_fails_closed(self) -> None:
        # No person_key and no metadata['person_id']: the audit must refuse to
        # certify rather than silently passing.
        manifest = _manifest([_record("SC4001E0", "SC4001")], [], [_record("SC4011E0", "SC4011")])
        report = audit_person_level_split(manifest)
        self.assertFalse(report.passed)
        self.assertTrue(any("unverified" in v for v in report.violations), report.violations)

    def test_person_id_metadata_is_used_when_present(self) -> None:
        train = [_record("SC4001E0", "SC4001", person_id="00")]
        test = [_record("SC4002E0", "SC4002", person_id="00")]
        report = audit_person_level_split(_manifest(train, [], test))
        self.assertFalse(report.passed)
        self.assertTrue(any("person leakage" in v for v in report.violations))

    def test_slpdb_two_part_recordings_are_one_person(self) -> None:
        # slp01a/slp01b are the same person; splitting them across train/test leaks.
        def slpdb_person(record: RecordingRecord) -> str:
            return record.record_id[3:5]

        train = [_record("slp01a", "slp01a")]
        test = [_record("slp01b", "slp01b")]
        report = audit_person_level_split(_manifest(train, [], test), person_key=slpdb_person)
        self.assertFalse(report.passed)
        self.assertTrue(any("01" in v for v in report.violations), report.violations)


if __name__ == "__main__":
    unittest.main()

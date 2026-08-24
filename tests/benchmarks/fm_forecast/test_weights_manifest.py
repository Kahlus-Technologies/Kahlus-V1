import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

from neurotwin.benchmarks.fm_forecast.vendor_import import is_vendored, vendored_root
from neurotwin.benchmarks.fm_forecast.weights_manifest import (
    WEIGHTS_MANIFEST,
    CheckpointChecksumError,
    CheckpointNotStagedError,
    CheckpointSpec,
    WeightsManifestError,
    checkpoint_spec,
    manifest_model_ids,
    manifest_rows,
    resolve_checkpoint,
    sha256_file,
    verify_checkpoint,
)
from neurotwin.benchmarks.pretraining_contamination import (
    AUDITED_MODEL_IDS,
    SLATE_EXCLUDED_MODEL_IDS,
    SLEEP_EDF_EVAL_DATASET_ID,
    is_evaluation_permitted,
)


_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class WeightsManifestTests(unittest.TestCase):
    def test_manifest_holds_the_four_slate_checkpoints_with_unique_identity(self):
        self.assertEqual(len(WEIGHTS_MANIFEST), 4)
        self.assertEqual(len(set(manifest_model_ids())), 4)
        self.assertEqual(len({spec.filename for spec in WEIGHTS_MANIFEST}), 4)
        self.assertEqual(len({spec.sha256 for spec in WEIGHTS_MANIFEST}), 4)
        self.assertEqual(len({spec.upstream_id for spec in WEIGHTS_MANIFEST}), 4)

    def test_every_row_has_a_well_formed_digest_source_and_licence(self):
        for spec in WEIGHTS_MANIFEST:
            with self.subTest(model_id=spec.model_id):
                self.assertRegex(spec.sha256, _HEX64)
                self.assertTrue(spec.source_url.startswith("https://"))
                self.assertTrue(spec.license_note.strip())
                self.assertTrue(spec.filename.strip())

    def test_model_ids_join_against_the_contamination_registry(self):
        for spec in WEIGHTS_MANIFEST:
            with self.subTest(model_id=spec.model_id):
                self.assertIn(spec.model_id, AUDITED_MODEL_IDS)
                # Sleep-EDF is the only arena clean for every entrant (plan section 2).
                self.assertTrue(is_evaluation_permitted(spec.model_id, SLEEP_EDF_EVAL_DATASET_ID))

    def test_slate_excluded_checkpoints_are_absent_from_the_manifest(self):
        for model_id in SLATE_EXCLUDED_MODEL_IDS:
            with self.subTest(model_id=model_id):
                self.assertNotIn(model_id, manifest_model_ids())
                with self.assertRaises(WeightsManifestError):
                    checkpoint_spec(model_id)

    def test_lookup_accepts_the_upstream_alias(self):
        self.assertEqual(checkpoint_spec("labram").model_id, "labram_base")
        self.assertEqual(checkpoint_spec("biot").model_id, "biot_prest16")
        self.assertEqual(checkpoint_spec("eegpt").model_id, "eegpt")
        self.assertEqual(checkpoint_spec("cbramod").model_id, "cbramod")

    def test_unknown_model_id_raises_and_names_the_registered_ids(self):
        with self.assertRaises(WeightsManifestError) as caught:
            checkpoint_spec("not_a_model")
        for model_id in manifest_model_ids():
            self.assertIn(model_id, str(caught.exception))

    def test_manifest_rows_are_json_serialisable_artifact_rows(self):
        rows = manifest_rows()
        self.assertEqual(len(rows), len(WEIGHTS_MANIFEST))
        payload = json.loads(json.dumps(rows))
        for row in payload:
            with self.subTest(row=row):
                self.assertEqual(
                    set(row),
                    {
                        "model_id",
                        "upstream_id",
                        "filename",
                        "sha256",
                        "source_url",
                        "license_note",
                        "upstream_relpath",
                    },
                )

    def test_sha256_file_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob.bin"
            payload = os.urandom(2048)
            path.write_bytes(payload)
            self.assertEqual(sha256_file(path), hashlib.sha256(payload).hexdigest())

    def test_resolve_checkpoint_fails_closed_when_the_weights_dir_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope"
            with self.assertRaises(CheckpointNotStagedError) as caught:
                resolve_checkpoint("eegpt", missing)
            self.assertIn(str(missing), str(caught.exception))

    def test_resolve_checkpoint_fails_closed_when_the_file_is_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = checkpoint_spec("cbramod")
            with self.assertRaises(CheckpointNotStagedError) as caught:
                resolve_checkpoint("cbramod", tmp)
            message = str(caught.exception)
            self.assertIn(spec.filename, message)
            self.assertIn(spec.source_url, message)

    def test_resolve_checkpoint_fails_closed_on_a_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            spec = checkpoint_spec("eegpt")
            staged = Path(tmp) / spec.filename
            staged.write_bytes(b"not the real checkpoint")
            with self.assertRaises(CheckpointChecksumError) as caught:
                resolve_checkpoint("eegpt", tmp)
            message = str(caught.exception)
            self.assertIn(spec.sha256, message)
            self.assertIn(hashlib.sha256(b"not the real checkpoint").hexdigest(), message)

    def test_truncated_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = os.urandom(4096)
            path = Path(tmp) / "staged.bin"
            path.write_bytes(payload[:-1])
            spec = CheckpointSpec(
                model_id="fixture",
                upstream_id="fixture",
                filename="staged.bin",
                sha256=hashlib.sha256(payload).hexdigest(),
                source_url="https://example.invalid/fixture",
                license_note="fixture",
            )
            with self.assertRaises(CheckpointChecksumError):
                verify_checkpoint(path, spec)

    def test_verify_checkpoint_returns_the_path_on_an_exact_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = os.urandom(4096)
            path = Path(tmp) / "staged.bin"
            path.write_bytes(payload)
            spec = CheckpointSpec(
                model_id="fixture",
                upstream_id="fixture",
                filename="staged.bin",
                sha256=hashlib.sha256(payload).hexdigest(),
                source_url="https://example.invalid/fixture",
                license_note="fixture",
            )
            self.assertEqual(verify_checkpoint(path, spec), path)

    @unittest.skipUnless(is_vendored("biot"), "external/vendor/biot is not populated (gitignored)")
    def test_frozen_digest_matches_the_real_biot_checkpoint(self):
        # Skips rather than fails when the vendor tree is absent. When it is present this
        # is the only place the frozen digest is checked against real upstream bytes.
        spec = checkpoint_spec("biot_prest16")
        root = vendored_root("biot")
        assert root is not None and spec.upstream_relpath is not None
        upstream = root / spec.upstream_relpath
        if not upstream.is_file():
            self.skipTest(f"{upstream} is not present in the vendored checkout")
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / spec.filename
            staged.symlink_to(upstream)
            self.assertEqual(resolve_checkpoint("biot_prest16", tmp), staged)
            self.assertEqual(sha256_file(upstream), spec.sha256)

import sys
import tempfile
import unittest
from pathlib import Path

from neurotwin.benchmarks.fm_forecast.vendor_import import (
    VENDORED_UPSTREAM_IDS,
    VendoredUpstreamMissing,
    ensure_on_sys_path,
    is_vendored,
    require_vendored,
    vendor_command,
    vendor_dir,
    vendored_root,
)


class VendorImportTests(unittest.TestCase):
    def test_vendored_upstream_ids_match_the_lock_file_keys(self):
        import json

        repo_root = Path(__file__).resolve().parents[3]
        lock = json.loads((repo_root / "external" / "upstreams.lock.json").read_text())
        for upstream_id in VENDORED_UPSTREAM_IDS:
            with self.subTest(upstream_id=upstream_id):
                self.assertIn(upstream_id, lock["upstreams"])

    def test_missing_upstream_returns_none_and_raises_with_the_vendor_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertIsNone(vendored_root("eegpt", vendor_dir_path=root))
            self.assertFalse(is_vendored("eegpt", vendor_dir_path=root))
            with self.assertRaises(VendoredUpstreamMissing) as caught:
                require_vendored("eegpt", vendor_dir_path=root)
            message = str(caught.exception)
            self.assertIn("python3 scripts/vendor_upstreams.py --ids eegpt --include-restricted", message)
            self.assertIn("eegpt", message)
            self.assertIn(str(root / "eegpt"), message)

    def test_empty_checkout_counts_as_missing(self):
        # A partial or interrupted clone leaves an empty directory behind; treating it
        # as present would produce an ImportError three frames deep instead of the
        # actionable message.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "labram").mkdir()
            self.assertIsNone(vendored_root("labram", vendor_dir_path=root))
            with self.assertRaises(VendoredUpstreamMissing):
                require_vendored("labram", vendor_dir_path=root)

    def test_present_checkout_resolves_and_is_added_to_sys_path_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "biot"
            target.mkdir()
            (target / "model.py").write_text("")
            self.assertEqual(vendored_root("biot", vendor_dir_path=root), target)
            self.assertTrue(is_vendored("biot", vendor_dir_path=root))
            original = list(sys.path)
            try:
                ensure_on_sys_path("biot", vendor_dir_path=root)
                ensure_on_sys_path("biot", vendor_dir_path=root)
                self.assertEqual(sys.path.count(str(target)), 1)
            finally:
                sys.path[:] = original

    def test_vendor_command_names_the_include_restricted_flag(self):
        # labram and biot are reuse_status="mixed" in the lock file, so vendoring them
        # without --include-restricted silently skips them.
        for upstream_id in VENDORED_UPSTREAM_IDS:
            with self.subTest(upstream_id=upstream_id):
                self.assertIn("--include-restricted", vendor_command(upstream_id))

    def test_default_vendor_dir_points_at_external_vendor(self):
        repo_root = Path(__file__).resolve().parents[3]
        self.assertEqual(vendor_dir(), repo_root / "external" / "vendor")

    @unittest.skipUnless(is_vendored("eegpt"), "external/vendor/eegpt is not populated (gitignored)")
    def test_real_vendored_checkout_is_usable_when_present(self):
        # Skips rather than fails: external/vendor/ is gitignored and absent in CI.
        root = require_vendored("eegpt")
        self.assertTrue(root.is_dir())
        self.assertTrue(any(root.iterdir()))

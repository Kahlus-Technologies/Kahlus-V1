import json
import subprocess
import sys
import unittest
from pathlib import Path

from neurotwin.adapters.registry import dataset_registry
from neurotwin.upstreams import permissive_upstreams, upstream_registry


class RegistryTests(unittest.TestCase):
    def test_dataset_registry_names_core_data_sources(self):
        dataset_ids = {dataset.dataset_id for dataset in dataset_registry()}

        self.assertIn("openneuro_bids", dataset_ids)
        self.assertIn("dandi_nwb", dataset_ids)
        self.assertIn("moabb_eeg", dataset_ids)
        self.assertIn("tuh_eeg", dataset_ids)
        self.assertIn("paired_eeg_fmri", dataset_ids)

    def test_cli_implemented_dataset_adapters_are_not_marked_planned(self):
        registry = {dataset.dataset_id: dataset for dataset in dataset_registry()}

        for dataset_id in ("openneuro_bids", "moabb_eeg"):
            self.assertIn("implemented", registry[dataset_id].adapter_status)
            self.assertNotIn("planned", registry[dataset_id].adapter_status)

    def test_upstream_registry_tracks_reuse_and_quarantine(self):
        upstreams = {upstream.upstream_id: upstream for upstream in upstream_registry()}

        self.assertEqual(upstreams["mamba"].reuse_status, "permissive")
        self.assertEqual(upstreams["brainlm"].reuse_status, "restricted")
        self.assertIn("mamba", {upstream.upstream_id for upstream in permissive_upstreams()})
        self.assertNotIn("brainlm", {upstream.upstream_id for upstream in permissive_upstreams()})

    def test_upstream_registry_matches_lockfile_for_external_eeg_pretrained_repos(self):
        upstreams = {upstream.upstream_id: upstream for upstream in upstream_registry()}
        lock = json.loads(Path("external/upstreams.lock.json").read_text())["upstreams"]
        permissive_ids = {upstream.upstream_id for upstream in permissive_upstreams()}

        for upstream_id in ("eegpt", "labram", "cbramod", "biot"):
            with self.subTest(upstream_id=upstream_id):
                self.assertIn(upstream_id, upstreams)
                self.assertEqual(upstreams[upstream_id].repo, lock[upstream_id]["repo"])
                self.assertEqual(upstreams[upstream_id].commit, lock[upstream_id]["commit"])
                self.assertEqual(upstreams[upstream_id].reuse_status, lock[upstream_id]["reuse_status"])

        # Checkpoint-bearing repos must never be default-vendorable.
        self.assertNotIn("labram", permissive_ids)
        self.assertNotIn("biot", permissive_ids)

    def test_lockfile_contains_verified_commits(self):
        lock = json.loads(Path("external/upstreams.lock.json").read_text())

        self.assertEqual(lock["upstreams"]["mamba"]["commit"], "a14b1dff0454a3bc27d9eb31355dc01e4b2490ec")
        self.assertEqual(lock["upstreams"]["braindecode"]["repo"], "https://github.com/braindecode/braindecode")

    def test_lockfile_pins_external_pretrained_eeg_upstreams(self):
        lock = json.loads(Path("external/upstreams.lock.json").read_text())
        upstreams = lock["upstreams"]

        # Repos that commit checkpoint weights in-tree (labram, biot) are "mixed" so
        # default vendoring never clones pretrained weights; weight-free repos are
        # "permissive".
        expected = {
            "eegpt": ("https://github.com/BINE022/EEGPT", "permissive"),
            "labram": ("https://github.com/935963004/LaBraM", "mixed"),
            "cbramod": ("https://github.com/wjq-learning/CBraMod", "permissive"),
            "biot": ("https://github.com/ycq091044/BIOT", "mixed"),
        }
        for upstream_id, (repo, reuse_status) in expected.items():
            with self.subTest(upstream_id=upstream_id):
                entry = upstreams[upstream_id]
                self.assertEqual(set(entry), {"repo", "commit", "reuse_status", "license_status"})
                self.assertEqual(entry["repo"], repo)
                self.assertEqual(entry["reuse_status"], reuse_status)
                self.assertRegex(entry["commit"], r"^[0-9a-f]{40}$")

        self.assertIn("figshare", upstreams["eegpt"]["license_status"])
        self.assertIn("OUTSIDE the repo", upstreams["eegpt"]["license_status"])
        self.assertIn("OUTSIDE the repo", upstreams["cbramod"]["license_status"])
        self.assertIn("Hugging Face", upstreams["cbramod"]["license_status"])
        self.assertIn("in-repo", upstreams["labram"]["license_status"])
        self.assertIn("in-repo", upstreams["biot"]["license_status"])

    def test_eeg_fm_baseline_requirements_stay_quarantined(self):
        quarantined = Path("requirements/eeg-fm-baselines.txt").read_text(encoding="utf-8")

        self.assertIn("braindecode[hub]>=1.7", quarantined)
        self.assertIn("huggingface_hub", quarantined)
        self.assertIn("MUST NEVER be merged", quarantined)

        for path in (
            "pyproject.toml",
            "deploy/a100/environment-a100.yml",
            "requirements/cluster-a100.txt",
        ):
            with self.subTest(path=path):
                text = Path(path).read_text(encoding="utf-8").lower()
                self.assertNotIn("huggingface_hub", text)
                self.assertNotIn("eeg-fm-baselines", text)

    def test_vendor_script_dry_run(self):
        result = subprocess.run(
            [sys.executable, "scripts/vendor_upstreams.py", "--dry-run", "--ids", "mamba"],
            check=True,
            text=True,
            capture_output=True,
        )

        self.assertIn("would clone mamba", result.stdout)

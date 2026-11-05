from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurotwin.forecastability.propofol_pci import (
    _macrostate_from_task,
    load_ds005620_fixture,
    make_propofol_pci_fixture,
    run_propofol_pci_gate,
)


class PropofolPciTests(unittest.TestCase):
    def test_macrostate_mapping(self) -> None:
        self.assertEqual(_macrostate_from_task("awake"), 0)
        self.assertEqual(_macrostate_from_task("sed"), 1)
        self.assertEqual(_macrostate_from_task("sed2"), 1)
        self.assertIsNone(_macrostate_from_task("unknown"))

    def test_synthetic_known_and_null_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_propofol_pci_gate(
                Path(tmp),
                seed=5,
                ds_root=None,
                bootstrap_mode="smoke",
            )
            known = gate["synthetic_known"]["states"][0]["residual_model"]
            null = gate["synthetic_null"]["states"][0]["residual_model"]
            self.assertGreater(known["rfs_bits"], 0.02)
            self.assertLess(abs(null["rfs_bits"]), 0.03)
            self.assertLessEqual(null["rfs_ci_high"], 0.05)

    def test_gate_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gate = run_propofol_pci_gate(Path(tmp), seed=1, ds_root=None, bootstrap_mode="smoke")
            self.assertEqual(gate["schema"], "kahlus.forecastability.propofol_pci.v1")
            self.assertIn("claim_scope", gate)
            self.assertIn("stop_reason", gate)

    def test_loader_on_synthetic_bids_stub(self) -> None:
        try:
            import pybv  # noqa: F401  (writes the BrainVision stub)
        except ImportError:
            self.skipTest("pybv not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "participants.tsv").write_text(
                "participant_id\tage\nsub-01\t20\nsub-02\t21\nsub-03\t22\n"
                "sub-04\t23\nsub-05\t24\nsub-06\t25\nsub-07\t26\nsub-08\t27\n",
                encoding="utf-8",
            )
            for subj, task, macro in [
                ("sub-01", "awake", 0),
                ("sub-02", "sed", 1),
                ("sub-03", "sed2", 1),
                ("sub-04", "awake", 0),
                ("sub-05", "sed", 1),
                ("sub-06", "awake", 0),
                ("sub-07", "sed", 1),
                ("sub-08", "sed2", 1),
            ]:
                eeg_dir = root / subj / "ses-1" / "eeg"
                eeg_dir.mkdir(parents=True, exist_ok=True)
                self._write_stub_brainvision(
                    eeg_dir / f"{subj}_ses-1_task-{task}_eeg",
                    macro=macro,
                    subject=int(subj.split("-")[1]),
                )
            try:
                fixture = load_ds005620_fixture(root)
            except RuntimeError as exc:
                if "mne is required" in str(exc):
                    self.skipTest("mne not installed")
                raise
            self.assertGreaterEqual(fixture.eeg_windows.shape[0], 16)
            self.assertGreaterEqual(len(set(fixture.subject.tolist())), 8)
            self.assertEqual(set(fixture.macrostate.tolist()), {0, 1})

    def _write_stub_brainvision(self, stem: Path, *, macro: int, subject: int) -> None:
        import mne

        sfreq = 100.0
        n_times = 2000
        t = np.arange(n_times) / sfreq
        rng = np.random.default_rng(subject + macro * 17)
        data = rng.normal(0.0, 0.1, size=(4, n_times))
        data[0] += (0.5 + 0.3 * macro) * np.sin(2 * np.pi * 3 * t)
        data[1] += (0.4 + 0.2 * macro) * np.cos(2 * np.pi * 5 * t)
        info = mne.create_info(["Fz", "Cz", "Pz", "Oz"], sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, verbose=False)
        mne.export.export_raw(stem.with_suffix(".vhdr"), raw, fmt="brainvision", overwrite=True, verbose=False)

    def test_fixture_has_two_states(self) -> None:
        fixture = make_propofol_pci_fixture(seed=2, residual_complexity=True)
        self.assertEqual(set(fixture.macrostate.tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()

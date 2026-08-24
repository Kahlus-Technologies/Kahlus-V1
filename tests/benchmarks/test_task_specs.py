import unittest

from neurotwin.benchmarks.registry import competitor_registry
from neurotwin.benchmarks.task_specs import default_translation_tasks


class TaskSpecTests(unittest.TestCase):
    def test_required_v1_translation_tasks_are_present(self):
        task_ids = {task.task_id for task in default_translation_tasks()}

        self.assertIn("stimulus_past_fmri_to_future_fmri", task_ids)
        self.assertIn("eeg_meg_to_shared_latent_state", task_ids)
        self.assertIn("fmri_to_eeg_meg_spectral_proxy", task_ids)
        self.assertIn("anatomy_fmri_to_subject_conditioned_state", task_ids)
        self.assertIn("missing_modality_reconstruction", task_ids)
        self.assertIn("few_shot_subject_adaptation", task_ids)

    def test_competitor_registry_names_the_crowded_lanes(self):
        competitors = {competitor.competitor_id: competitor for competitor in competitor_registry()}

        self.assertIn("tribe_v2", competitors)
        self.assertIn("brainvista", competitors)
        self.assertIn("brain_of", competitors)
        self.assertIn("brainomni", competitors)
        self.assertIn("brain_harmony", competitors)
        self.assertEqual(competitors["tribe_v2"].implementation_status, "clean_room_approximation")
        self.assertIn("not exact TRIBE v2", competitors["tribe_v2"].notes)

    def test_external_pretrained_eeg_models_are_registered_with_executable_adapters(self):
        competitors = {competitor.competitor_id: competitor for competitor in competitor_registry()}

        for competitor_id in ("eegpt", "labram", "cbramod", "biot"):
            with self.subTest(competitor_id=competitor_id):
                self.assertIn(competitor_id, competitors)
                spec = competitors[competitor_id]
                self.assertEqual(spec.implementation_status, "executable_adapter")
                self.assertTrue(spec.upstream_url.startswith("https://github.com/"))
                # An executable adapter is not a vendored checkpoint: the adapter still
                # resolves a hand-staged, hash-pinned file and fails closed without it.
                self.assertIn("No weights vendored.", spec.notes)
                self.assertIn("fm_forecast.adapters", spec.notes)
                self.assertIn("fails closed", spec.notes)

    def test_external_eeg_competitor_notes_record_weight_hosting_and_task_mismatch(self):
        competitors = {competitor.competitor_id: competitor for competitor in competitor_registry()}

        self.assertIn("figshare", competitors["eegpt"].notes)
        self.assertIn("mirrored", competitors["eegpt"].notes)
        self.assertIn("not a forecaster", competitors["labram"].notes)
        self.assertIn("withheld", competitors["labram"].notes)
        self.assertIn("Hugging Face", competitors["cbramod"].license_status)
        self.assertIn("not forecasting", competitors["cbramod"].notes)
        self.assertIn("TUEG", competitors["cbramod"].notes)
        self.assertIn("FORBIDDEN for CHB-MIT", competitors["biot"].notes)

    def test_competitor_ids_are_unique(self):
        competitor_ids = [competitor.competitor_id for competitor in competitor_registry()]

        self.assertEqual(len(competitor_ids), len(set(competitor_ids)))

import unittest

from neurotwin.benchmarks.pretraining_contamination import (
    AUDITED_MODEL_IDS,
    CHB_MIT_EVAL_DATASET_ID,
    CLEAN,
    CONTAMINATED,
    CONTAMINATION_STATUSES,
    EVAL_DATASET_IDS,
    SLEEP_EDF_EVAL_DATASET_ID,
    TUSZ_EVAL_DATASET_ID,
    UNVERIFIED,
    contaminated_pairs,
    contamination_reason,
    contamination_status,
    contamination_table,
    is_evaluation_permitted,
)


class PretrainingContaminationTests(unittest.TestCase):
    def test_every_audited_model_and_eval_dataset_pair_has_a_status(self):
        for model_id in AUDITED_MODEL_IDS:
            for dataset_id in EVAL_DATASET_IDS:
                with self.subTest(model_id=model_id, dataset_id=dataset_id):
                    self.assertIn(contamination_status(model_id, dataset_id), CONTAMINATION_STATUSES)

    def test_unknown_pairs_fail_closed_to_unverified(self):
        self.assertEqual(contamination_status("not_a_model", SLEEP_EDF_EVAL_DATASET_ID), UNVERIFIED)
        self.assertEqual(contamination_status("eegpt", "not_a_dataset"), UNVERIFIED)
        self.assertEqual(contamination_status("", ""), UNVERIFIED)
        self.assertNotEqual(contamination_status("not_a_model", "not_a_dataset"), CLEAN)
        self.assertFalse(is_evaluation_permitted("not_a_model", SLEEP_EDF_EVAL_DATASET_ID))
        self.assertIn("fail closed", contamination_reason("not_a_model", "not_a_dataset"))

    def test_eval_dataset_ids_are_the_canonical_repo_ids(self):
        from neurotwin.adapters.sleep_edf import SLEEP_EDF_DATASET_ID
        from neurotwin.stf.public_data import CHB_MIT_DATASET_ID

        self.assertEqual(SLEEP_EDF_EVAL_DATASET_ID, SLEEP_EDF_DATASET_ID)
        self.assertEqual(CHB_MIT_EVAL_DATASET_ID, CHB_MIT_DATASET_ID)
        self.assertEqual(TUSZ_EVAL_DATASET_ID, "tuh_eeg")

    def test_eegpt_is_clean_on_all_three_eval_datasets(self):
        for dataset_id in EVAL_DATASET_IDS:
            with self.subTest(dataset_id=dataset_id):
                self.assertEqual(contamination_status("eegpt", dataset_id), CLEAN)
                self.assertTrue(is_evaluation_permitted("eegpt", dataset_id))

    def test_cbramod_is_contaminated_on_tusz_only(self):
        self.assertEqual(contamination_status("cbramod", TUSZ_EVAL_DATASET_ID), CONTAMINATED)
        self.assertEqual(contamination_status("cbramod", SLEEP_EDF_EVAL_DATASET_ID), CLEAN)
        self.assertEqual(contamination_status("cbramod", CHB_MIT_EVAL_DATASET_ID), CLEAN)
        self.assertIn("TUEG", contamination_reason("cbramod", TUSZ_EVAL_DATASET_ID))

    def test_labram_base_is_contaminated_on_tusz_and_unverified_on_chb_mit(self):
        self.assertEqual(contamination_status("labram_base", TUSZ_EVAL_DATASET_ID), CONTAMINATED)
        self.assertEqual(contamination_status("labram_base", CHB_MIT_EVAL_DATASET_ID), UNVERIFIED)
        self.assertEqual(contamination_status("labram_base", SLEEP_EDF_EVAL_DATASET_ID), CLEAN)

    def test_biot_prest16_is_clean_everywhere(self):
        for dataset_id in EVAL_DATASET_IDS:
            with self.subTest(dataset_id=dataset_id):
                self.assertEqual(contamination_status("biot_prest16", dataset_id), CLEAN)

    def test_biot_six_datasets_is_contaminated_on_chb_mit(self):
        self.assertEqual(contamination_status("biot_six_datasets", CHB_MIT_EVAL_DATASET_ID), CONTAMINATED)
        self.assertFalse(is_evaluation_permitted("biot_six_datasets", CHB_MIT_EVAL_DATASET_ID))
        self.assertEqual(contamination_status("biot_six_datasets", TUSZ_EVAL_DATASET_ID), CONTAMINATED)
        self.assertEqual(contamination_status("biot_six_datasets", SLEEP_EDF_EVAL_DATASET_ID), CLEAN)

    def test_biot_checkpoint_variants_are_tracked_separately(self):
        self.assertNotEqual(
            contamination_status("biot_prest16", CHB_MIT_EVAL_DATASET_ID),
            contamination_status("biot_six_datasets", CHB_MIT_EVAL_DATASET_ID),
        )

    def test_contamination_table_is_complete_with_nonempty_reasons(self):
        table = contamination_table()
        pairs = {(row["model_id"], row["dataset_id"]) for row in table}

        self.assertEqual(len(table), len(AUDITED_MODEL_IDS) * len(EVAL_DATASET_IDS))
        self.assertEqual(len(pairs), len(table))
        for model_id in AUDITED_MODEL_IDS:
            for dataset_id in EVAL_DATASET_IDS:
                self.assertIn((model_id, dataset_id), pairs)
        for row in table:
            with self.subTest(row=row):
                self.assertEqual(set(row), {"model_id", "dataset_id", "status", "reason"})
                self.assertIn(row["status"], CONTAMINATION_STATUSES)
                self.assertTrue(row["reason"].strip())

    def test_contaminated_pairs_matches_the_table(self):
        expected = {
            (row["model_id"], row["dataset_id"])
            for row in contamination_table()
            if row["status"] == CONTAMINATED
        }
        actual = {(row["model_id"], row["dataset_id"]) for row in contaminated_pairs()}

        self.assertEqual(actual, expected)
        self.assertIn(("biot_six_datasets", CHB_MIT_EVAL_DATASET_ID), actual)
        self.assertIn(("cbramod", TUSZ_EVAL_DATASET_ID), actual)


if __name__ == "__main__":
    unittest.main()

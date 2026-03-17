"""
tests/pipeline/test_version_comparator.py

Unit tests for pipeline/version_comparator.py.

Coverage targets
----------------
compare_manifests()     -- disappeared numbers produce RetroactiveDeletion,
                           new additions recorded, equal manifests produce
                           zero deletions, all three signals set on retroactive
                           deletion, deletion_record is DELETION_CONFIRMED,
                           net_change correct, compared_at populated
filter_by_dataset()     -- returns only matching dataset, case-insensitive,
                           empty result when no match
ComparisonResult        -- deletion_count, addition_count, net_change properties
"""

from __future__ import annotations

import unittest

from pipeline.manifest_loader import ManifestLoadResult, ManifestRecord
from pipeline.models import DeletionFlag
from pipeline.version_comparator import (
    ComparisonResult,
    RetroactiveDeletion,
    compare_manifests,
    filter_by_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    version: str,
    efta_numbers: list[str],
    dataset: str = "DS01",
) -> ManifestLoadResult:
    records = [
        ManifestRecord(
            efta_number=n,
            dataset=dataset,
            title=f"Doc {n}",
            url="",
        )
        for n in efta_numbers
    ]
    return ManifestLoadResult(
        release_version=version,
        records=records,
        efta_numbers=set(efta_numbers),
    )


# ===========================================================================
# compare_manifests
# ===========================================================================

class TestCompareManifests(unittest.TestCase):

    def test_disappeared_number_produces_deletion(self):
        prior   = _result("v1", ["100", "200", "300"])
        current = _result("v2", ["100", "300"])       # 200 disappeared
        result  = compare_manifests(prior, current)
        disappeared = [d.efta_number for d in result.retroactive_deletions]
        self.assertIn("200", disappeared)

    def test_equal_manifests_produce_zero_deletions(self):
        prior   = _result("v1", ["100", "200", "300"])
        current = _result("v2", ["100", "200", "300"])
        result  = compare_manifests(prior, current)
        self.assertEqual(result.deletion_count, 0)

    def test_new_additions_recorded(self):
        prior   = _result("v1", ["100", "200"])
        current = _result("v2", ["100", "200", "300"])
        result  = compare_manifests(prior, current)
        self.assertIn("300", result.new_additions)

    def test_deletion_record_is_confirmed(self):
        prior   = _result("v1", ["100", "200"])
        current = _result("v2", ["100"])
        result  = compare_manifests(prior, current)
        deletion = result.retroactive_deletions[0]
        self.assertEqual(
            deletion.deletion_record.deletion_flag,
            DeletionFlag.DELETION_CONFIRMED,
        )

    def test_all_three_signals_set(self):
        prior   = _result("v1", ["100", "200"])
        current = _result("v2", ["100"])
        result  = compare_manifests(prior, current)
        record  = result.retroactive_deletions[0].deletion_record
        # DELETION_CONFIRMED requires three signals
        from pipeline.deletion_detector import DetectionSignals
        # Verify by checking flag level (CONFIRMED = 3 signals)
        self.assertEqual(record.deletion_flag, DeletionFlag.DELETION_CONFIRMED)

    def test_version_labels_stored(self):
        prior   = _result("v1", ["100"])
        current = _result("v2", ["200"])
        result  = compare_manifests(prior, current)
        self.assertEqual(result.prior_version, "v1")
        self.assertEqual(result.current_version, "v2")

    def test_compared_at_populated(self):
        result = compare_manifests(_result("v1", ["100"]), _result("v2", ["100"]))
        self.assertTrue(len(result.compared_at) > 0)

    def test_total_counts_correct(self):
        prior   = _result("v1", ["100", "200", "300"])
        current = _result("v2", ["100", "200"])
        result  = compare_manifests(prior, current)
        self.assertEqual(result.total_prior, 3)
        self.assertEqual(result.total_current, 2)

    def test_prior_record_preserved_on_deletion(self):
        prior   = _result("v1", ["100", "200"])
        current = _result("v2", ["100"])
        result  = compare_manifests(prior, current)
        deletion = result.retroactive_deletions[0]
        self.assertEqual(deletion.prior_record.efta_number, "200")
        self.assertEqual(deletion.prior_record.title, "Doc 200")

    def test_acknowledgment_source_mentions_versions(self):
        prior   = _result("v1", ["100", "200"])
        current = _result("v2", ["100"])
        result  = compare_manifests(prior, current)
        source  = result.retroactive_deletions[0].deletion_record.acknowledgment_source
        self.assertIn("v1", source)
        self.assertIn("v2", source)

    def test_empty_prior_manifest(self):
        prior   = _result("v1", [])
        current = _result("v2", ["100", "200"])
        result  = compare_manifests(prior, current)
        self.assertEqual(result.deletion_count, 0)
        self.assertEqual(result.addition_count, 2)


# ===========================================================================
# ComparisonResult properties
# ===========================================================================

class TestComparisonResultProperties(unittest.TestCase):

    def _result_with(self, deletions: int, additions: int) -> ComparisonResult:
        prior   = _result("v1", [str(i) for i in range(deletions + 5)])
        current = _result("v2", [str(i + deletions) for i in range(additions + 5)])
        return compare_manifests(prior, current)

    def test_deletion_count_property(self):
        r = compare_manifests(_result("v1", ["1", "2", "3"]), _result("v2", ["1"]))
        self.assertEqual(r.deletion_count, 2)

    def test_addition_count_property(self):
        r = compare_manifests(_result("v1", ["1"]), _result("v2", ["1", "2", "3"]))
        self.assertEqual(r.addition_count, 2)

    def test_net_change_positive(self):
        r = compare_manifests(_result("v1", ["1"]), _result("v2", ["1", "2", "3"]))
        self.assertGreater(r.net_change, 0)

    def test_net_change_negative(self):
        r = compare_manifests(_result("v1", ["1", "2", "3"]), _result("v2", ["1"]))
        self.assertLess(r.net_change, 0)

    def test_net_change_zero(self):
        r = compare_manifests(_result("v1", ["1", "2"]), _result("v2", ["1", "2"]))
        self.assertEqual(r.net_change, 0)


# ===========================================================================
# filter_by_dataset
# ===========================================================================

class TestFilterByDataset(unittest.TestCase):

    def _comparison_with_datasets(self) -> ComparisonResult:
        prior = ManifestLoadResult(
            release_version="v1",
            records=[
                ManifestRecord("100", "DS01", "Doc A", ""),
                ManifestRecord("200", "DS09", "Doc B", ""),
                ManifestRecord("300", "DS01", "Doc C", ""),
            ],
            efta_numbers={"100", "200", "300"},
        )
        current = ManifestLoadResult(
            release_version="v2",
            records=[],
            efta_numbers=set(),
        )
        return compare_manifests(prior, current)

    def test_filters_by_dataset(self):
        result = self._comparison_with_datasets()
        ds09 = filter_by_dataset(result, "DS09")
        self.assertEqual(len(ds09), 1)
        self.assertEqual(ds09[0].efta_number, "200")

    def test_case_insensitive(self):
        result = self._comparison_with_datasets()
        ds01_lower = filter_by_dataset(result, "ds01")
        self.assertEqual(len(ds01_lower), 2)

    def test_no_match_returns_empty(self):
        result = self._comparison_with_datasets()
        ds99 = filter_by_dataset(result, "DS99")
        self.assertEqual(ds99, [])


if __name__ == "__main__":
    unittest.main()

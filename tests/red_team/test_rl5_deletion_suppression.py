"""
tests/red_team/test_rl5_deletion_suppression.py
Red Team: Deletion findings must surface and must not be overridden.

Adversarial tests verifying that deletion flags are correctly assigned,
are not downgraded by weaker evidence, and that gap reports correctly
reflect the deletion taxonomy.

Constitution Principle IV -- Gaps Are Facts.
"""

from __future__ import annotations

import unittest

from pipeline.deletion_detector import (
    DetectionSignals,
    create_deletion_finding,
)
from pipeline.models import DeletionFlag


class TestDeletionFlagOrdering(unittest.TestCase):
    """Stronger flags must not be downgraded by weaker evidence."""

    def test_confirmed_stronger_than_suspected(self):
        self.assertGreater(
            DeletionFlag.DELETION_CONFIRMED,
            DeletionFlag.DELETION_SUSPECTED,
        )

    def test_withheld_acknowledged_strongest(self):
        for flag in [
            DeletionFlag.DELETION_CONFIRMED,
            DeletionFlag.DELETION_SUSPECTED,
            DeletionFlag.DELETION_POSSIBLE,
            DeletionFlag.REFERENCE_UNRESOLVED,
            DeletionFlag.WITHHELD_SELECTIVELY,
        ]:
            self.assertGreater(DeletionFlag.WITHHELD_ACKNOWLEDGED, flag)

    def test_three_signals_gives_confirmed(self):
        signals = DetectionSignals(
            efta_gap=True, discovery_log_entry=True, document_stamp_gap=True
        )
        record = create_deletion_finding(
            document_identifiers=["EFTA-001"],
            signals=signals,
            acknowledgment_source="Test",
            acknowledgment_date="2026-01-01",
        )
        self.assertEqual(record.deletion_flag, DeletionFlag.DELETION_CONFIRMED)

    def test_one_signal_gives_possible(self):
        signals = DetectionSignals(efta_gap=True)
        record = create_deletion_finding(
            document_identifiers=["EFTA-002"],
            signals=signals,
            acknowledgment_source="Test",
            acknowledgment_date="2026-01-01",
        )
        self.assertEqual(record.deletion_flag, DeletionFlag.DELETION_POSSIBLE)

    def test_deletion_flag_not_downgraded_in_dynamo(self):
        """
        DynamoDB flag update uses ConditionExpression(attribute_not_exists).
        A DELETION_CONFIRMED record must not be overwritten by DELETION_POSSIBLE.
        """
        from unittest.mock import MagicMock
        from pipeline.deletion_pipeline import _flag_document_record

        db = MagicMock()
        db.update_item.return_value = {}

        _flag_document_record(
            "EFTA-001", DeletionFlag.DELETION_CONFIRMED, "rec-id", db
        )
        kwargs = db.update_item.call_args.kwargs
        # ConditionExpression prevents downgrade
        self.assertIn("ConditionExpression", kwargs)
        self.assertIn("attribute_not_exists", kwargs["ConditionExpression"])


class TestGapReportSurfaces(unittest.TestCase):
    """Gap reports must include all flagged findings."""

    def test_all_tiers_represented_in_report(self):
        from pipeline.gap_reporter import generate_gap_report

        records = []
        for flag in [
            DeletionFlag.DELETION_CONFIRMED,
            DeletionFlag.DELETION_SUSPECTED,
            DeletionFlag.DELETION_POSSIBLE,
        ]:
            signals = DetectionSignals(
                efta_gap=True,
                discovery_log_entry=(flag != DeletionFlag.DELETION_POSSIBLE),
                document_stamp_gap=(flag == DeletionFlag.DELETION_CONFIRMED),
            )
            records.append(create_deletion_finding(
                document_identifiers=[f"EFTA-{flag.value}"],
                signals=signals,
                acknowledgment_source="Test",
                acknowledgment_date="2026-01-01",
            ))

        report = generate_gap_report(records)
        self.assertEqual(report.total_gaps, 3)
        self.assertIn(DeletionFlag.DELETION_CONFIRMED, report.summary)
        self.assertIn(DeletionFlag.DELETION_SUSPECTED, report.summary)
        self.assertIn(DeletionFlag.DELETION_POSSIBLE, report.summary)

    def test_confirmed_deletion_at_top_of_report(self):
        from pipeline.gap_reporter import generate_gap_report, _TIER_ORDER

        report = generate_gap_report([])
        # WITHHELD_ACKNOWLEDGED should be first in tier ordering
        self.assertEqual(_TIER_ORDER[0], DeletionFlag.WITHHELD_ACKNOWLEDGED)

    def test_retroactive_deletion_always_confirmed(self):
        from pipeline.manifest_loader import ManifestLoadResult, ManifestRecord
        from pipeline.version_comparator import compare_manifests

        prior = ManifestLoadResult(
            "v1",
            [ManifestRecord("1234567", "DS01", "Doc A", "")],
            {"1234567"},
        )
        current = ManifestLoadResult("v2", [], set())
        result = compare_manifests(prior, current)

        self.assertEqual(len(result.retroactive_deletions), 1)
        self.assertEqual(
            result.retroactive_deletions[0].deletion_record.deletion_flag,
            DeletionFlag.DELETION_CONFIRMED,
        )


if __name__ == "__main__":
    unittest.main()

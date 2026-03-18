"""
tests/pipeline/test_gap_reporter.py

Unit tests for pipeline/gap_reporter.py.

Coverage targets
----------------
generate_gap_report()       -- returns GapReport, markdown populated,
                               correct total_gaps count, sections grouped
                               by tier, empty input returns zero total,
                               victim flag suppressed in public mode,
                               victim text preserved in technical mode,
                               withholding records included
generate_comparison_report()-- delegates to generate_gap_report, version
                               metadata in markdown, title auto-generated
save_report_to_s3()         -- put_object called, content-type markdown,
                               empty bucket raises, failure raises
GapReport                   -- total_gaps, summary dict populated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.deletion_detector import DetectionSignals, DeletionRecord, create_deletion_finding
from pipeline.models import WithholdingRecord
from pipeline.gap_reporter import (
    GapReport,
    generate_comparison_report,
    generate_gap_report,
    save_report_to_s3,
)
from pipeline.manifest_loader import ManifestLoadResult, ManifestRecord
from pipeline.models import DeletionFlag
from pipeline.version_comparator import compare_manifests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deletion_record(
    efta: str = "1234567",
    flag: DeletionFlag = DeletionFlag.DELETION_CONFIRMED,
) -> DeletionRecord:
    signals = DetectionSignals(
        efta_gap=True,
        discovery_log_entry=True,
        document_stamp_gap=(flag == DeletionFlag.DELETION_CONFIRMED),
    )
    return create_deletion_finding(
        document_identifiers=[efta],
        signals=signals,
        acknowledgment_source="Test reconciliation",
        acknowledgment_date="2026-03-16",
    )


def _withholding_record(victim_entity: bool = False) -> WithholdingRecord:
    """WithholdingRecord is the model that carries subject_entities."""
    import uuid as _uuid
    record = WithholdingRecord(
        record_id=str(_uuid.uuid4()),
        document_identifiers=["DOC-001"],
        deletion_flag=DeletionFlag.WITHHELD_ACKNOWLEDGED,
        acknowledgment_source="Test",
        acknowledgment_date="2026-03-16",
    )
    if victim_entity:
        record.subject_entities = [
            {"name": "Virginia Giuffre", "type": "PERSON", "victim_flag": True}
        ]
    return record


def _manifest(version: str, numbers: list[str]) -> ManifestLoadResult:
    records = [ManifestRecord(n, "DS01", f"Doc {n}", "") for n in numbers]
    return ManifestLoadResult(version, records, set(numbers))


# ===========================================================================
# generate_gap_report
# ===========================================================================

class TestGenerateGapReport(unittest.TestCase):

    def test_returns_gap_report(self):
        report = generate_gap_report([_deletion_record()])
        self.assertIsInstance(report, GapReport)

    def test_markdown_populated(self):
        report = generate_gap_report([_deletion_record()])
        self.assertGreater(len(report.markdown), 0)

    def test_total_gaps_correct(self):
        records = [_deletion_record(str(i)) for i in range(5)]
        report = generate_gap_report(records)
        self.assertEqual(report.total_gaps, 5)

    def test_empty_input_zero_total(self):
        report = generate_gap_report([])
        self.assertEqual(report.total_gaps, 0)

    def test_summary_contains_confirmed_count(self):
        records = [_deletion_record(flag=DeletionFlag.DELETION_CONFIRMED)]
        report = generate_gap_report(records)
        self.assertIn(DeletionFlag.DELETION_CONFIRMED, report.summary)

    def test_markdown_contains_efta_number(self):
        report = generate_gap_report([_deletion_record("9876543")])
        self.assertIn("9876543", report.markdown)

    def test_victim_text_suppressed_in_public_mode(self):
        record = _withholding_record(victim_entity=True)
        report = generate_gap_report([], withholding_records=[record], public=True)
        self.assertNotIn("Virginia Giuffre", report.markdown)
        self.assertIn("[protected identity]", report.markdown)

    def test_victim_text_preserved_in_technical_mode(self):
        record = _withholding_record(victim_entity=True)
        report = generate_gap_report([], withholding_records=[record], public=False)
        self.assertIn("Virginia Giuffre", report.markdown)

    def test_custom_title_in_markdown(self):
        report = generate_gap_report([], title="My Custom Report")
        self.assertIn("My Custom Report", report.markdown)

    def test_generated_at_in_markdown(self):
        report = generate_gap_report([])
        self.assertIn("Generated", report.markdown)

    def test_multiple_tiers_grouped(self):
        records = [
            _deletion_record("1", DeletionFlag.DELETION_CONFIRMED),
            _deletion_record("2", DeletionFlag.DELETION_SUSPECTED),
        ]
        report = generate_gap_report(records)
        self.assertEqual(report.total_gaps, 2)
        self.assertIn(DeletionFlag.DELETION_CONFIRMED, report.summary)
        self.assertIn(DeletionFlag.DELETION_SUSPECTED, report.summary)


# ===========================================================================
# generate_comparison_report
# ===========================================================================

class TestGenerateComparisonReport(unittest.TestCase):

    def _comparison(self):
        prior   = _manifest("v1", ["100", "200", "300"])
        current = _manifest("v2", ["100", "300"])
        return compare_manifests(prior, current)

    def test_returns_gap_report(self):
        report = generate_comparison_report(self._comparison())
        self.assertIsInstance(report, GapReport)

    def test_version_metadata_in_markdown(self):
        report = generate_comparison_report(self._comparison())
        self.assertIn("v1", report.markdown)
        self.assertIn("v2", report.markdown)

    def test_auto_title_mentions_versions(self):
        report = generate_comparison_report(self._comparison())
        self.assertIn("v1", report.markdown)
        self.assertIn("v2", report.markdown)

    def test_custom_title_used(self):
        report = generate_comparison_report(
            self._comparison(), title="Custom Comparison Report"
        )
        self.assertIn("Custom Comparison Report", report.markdown)

    def test_deletion_count_matches_comparison(self):
        comp = self._comparison()
        report = generate_comparison_report(comp)
        self.assertEqual(report.total_gaps, comp.deletion_count)


# ===========================================================================
# save_report_to_s3
# ===========================================================================

class TestSaveReportToS3(unittest.TestCase):

    def _report(self) -> GapReport:
        return generate_gap_report([_deletion_record()])

    def test_put_object_called(self):
        s3 = MagicMock()
        s3.put_object.return_value = {}
        save_report_to_s3(self._report(), "reports/test.md",
                          s3_client=s3, bucket_name="bucket")
        s3.put_object.assert_called_once()

    def test_content_type_is_markdown(self):
        s3 = MagicMock()
        s3.put_object.return_value = {}
        save_report_to_s3(self._report(), "reports/test.md",
                          s3_client=s3, bucket_name="bucket")
        self.assertEqual(
            s3.put_object.call_args.kwargs["ContentType"], "text/markdown"
        )

    def test_empty_bucket_raises_value_error(self):
        with self.assertRaises(ValueError):
            save_report_to_s3(self._report(), "key",
                              s3_client=MagicMock(), bucket_name="")

    def test_s3_failure_raises_runtime_error(self):
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(RuntimeError):
            save_report_to_s3(self._report(), "key",
                              s3_client=s3, bucket_name="bucket")


if __name__ == "__main__":
    unittest.main()

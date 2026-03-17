"""
tests/pipeline/test_deletion_pipeline.py

Unit tests for pipeline/deletion_pipeline.py.

Coverage targets
----------------
_signals_for_candidate()    -- candidate in deletion_candidates gets all
                               three signals set, number not in candidates
                               gets no signals
_write_deletion_record()    -- put_item called, correct table, record_id
                               present, document_identifiers as SS,
                               efta_number GSI key set, optional fields
                               conditionally present
_flag_document_record()     -- update_item called, ConditionExpression set,
                               exception silently ignored (document may not
                               exist yet)
DeletionPipelineResult      -- candidate_count, confirmed_count properties
run_deletion_pipeline()     -- reconciliation run, DeletionRecord created
                               per candidate, records written, gap report
                               generated when requested, errors list
                               populated on partial failure, prior_manifest
                               triggers comparison
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, call, patch

from pipeline.deletion_detector import DetectionSignals, DeletionRecord, create_deletion_finding
from pipeline.deletion_pipeline import (
    DELETIONS_TABLE,
    DOCUMENTS_TABLE,
    DeletionPipelineResult,
    _flag_document_record,
    _signals_for_candidate,
    _write_deletion_record,
    run_deletion_pipeline,
)
from pipeline.manifest_loader import ManifestLoadResult, ManifestRecord
from pipeline.models import DeletionFlag
from pipeline.sequence_numbers import EFTANumber, ReconciliationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _manifest(version: str, numbers: list[str]) -> ManifestLoadResult:
    records = [ManifestRecord(n, "DS01", f"Doc {n}", "") for n in numbers]
    return ManifestLoadResult(version, records, set(numbers))


def _reconciliation(candidates: list[str], expected_gaps: list[str] = None) -> ReconciliationResult:
    all_missing = candidates + (expected_gaps or [])
    return ReconciliationResult(
        sequence_type="EFTA",
        present_count=100,
        missing_from_corpus_count=len(all_missing),
        deletion_candidates=candidates,
        expected_gap_numbers=expected_gaps or [],
        missing_numbers=all_missing,
        coverage_pct=0.95,
    )


def _deletion_record(efta: str = "1234567") -> DeletionRecord:
    return create_deletion_finding(
        document_identifiers=[efta],
        signals=DetectionSignals(efta_gap=True, discovery_log_entry=True,
                                 document_stamp_gap=True),
        acknowledgment_source="Test",
        acknowledgment_date="2026-03-16",
    )


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.put_item.return_value = {}
    db.update_item.return_value = {}
    db.exceptions.ConditionalCheckFailedException = type(
        "ConditionalCheckFailedException", (Exception,), {}
    )
    return db


def _mock_efta(candidates: list[str]) -> MagicMock:
    """Mock EFTANumber that returns a ReconciliationResult with given candidates."""
    efta = MagicMock(spec=EFTANumber)
    efta.reconcile.return_value = _reconciliation(candidates)
    efta.extract_from_text.return_value = []
    return efta


# ===========================================================================
# _signals_for_candidate
# ===========================================================================

class TestSignalsForCandidate(unittest.TestCase):

    def test_candidate_gets_all_signals(self):
        recon = _reconciliation(["1234567"])
        signals = _signals_for_candidate("1234567", recon)
        self.assertTrue(signals.efta_gap)
        self.assertTrue(signals.discovery_log_entry)
        self.assertTrue(signals.document_stamp_gap)

    def test_non_candidate_gets_no_signals(self):
        recon = _reconciliation(["1234567"])
        signals = _signals_for_candidate("9999999", recon)
        self.assertFalse(signals.efta_gap)
        self.assertFalse(signals.discovery_log_entry)
        self.assertFalse(signals.document_stamp_gap)

    def test_signal_count_three_for_candidate(self):
        recon = _reconciliation(["1234567"])
        signals = _signals_for_candidate("1234567", recon)
        self.assertEqual(signals.signal_count, 3)


# ===========================================================================
# _write_deletion_record
# ===========================================================================

class TestWriteDeletionRecord(unittest.TestCase):

    def test_put_item_called(self):
        db = _mock_db()
        _write_deletion_record(_deletion_record(), db)
        db.put_item.assert_called_once()

    def test_correct_table_used(self):
        db = _mock_db()
        _write_deletion_record(_deletion_record(), db)
        self.assertEqual(db.put_item.call_args.kwargs["TableName"], DELETIONS_TABLE)

    def test_record_id_in_item(self):
        db = _mock_db()
        record = _deletion_record()
        _write_deletion_record(record, db)
        item = db.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["record_id"]["S"], record.record_id)

    def test_document_identifiers_as_string_set(self):
        db = _mock_db()
        _write_deletion_record(_deletion_record("EFTA-001"), db)
        item = db.put_item.call_args.kwargs["Item"]
        self.assertIn("SS", item["document_identifiers"])

    def test_efta_number_gsi_key_set(self):
        db = _mock_db()
        record = _deletion_record("1234567")
        _write_deletion_record(record, db)
        item = db.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["efta_number"]["S"], "1234567")

    def test_custom_table_name(self):
        db = _mock_db()
        _write_deletion_record(_deletion_record(), db, table_name="custom_table")
        self.assertEqual(
            db.put_item.call_args.kwargs["TableName"], "custom_table"
        )


# ===========================================================================
# _flag_document_record
# ===========================================================================

class TestFlagDocumentRecord(unittest.TestCase):

    def test_update_item_called(self):
        db = _mock_db()
        _flag_document_record("1234567", DeletionFlag.DELETION_CONFIRMED,
                               "rec-id", db)
        db.update_item.assert_called_once()

    def test_condition_expression_set(self):
        db = _mock_db()
        _flag_document_record("1234567", DeletionFlag.DELETION_CONFIRMED,
                               "rec-id", db)
        kwargs = db.update_item.call_args.kwargs
        self.assertIn("ConditionExpression", kwargs)

    def test_exception_silently_ignored(self):
        db = _mock_db()
        db.update_item.side_effect = RuntimeError("Document not found")
        # Must not raise -- document may not exist yet
        _flag_document_record("1234567", DeletionFlag.DELETION_CONFIRMED,
                               "rec-id", db)


# ===========================================================================
# DeletionPipelineResult properties
# ===========================================================================

class TestDeletionPipelineResultProperties(unittest.TestCase):

    def _make_result(
        self,
        candidates: list[str],
        records: list[DeletionRecord] = None,
    ) -> DeletionPipelineResult:
        return DeletionPipelineResult(
            manifest_version="v1",
            reconciliation=_reconciliation(candidates),
            deletion_records=records or [],
        )

    def test_candidate_count_from_reconciliation(self):
        r = self._make_result(["1", "2", "3"])
        self.assertEqual(r.candidate_count, 3)

    def test_confirmed_count(self):
        records = [_deletion_record(str(i)) for i in range(3)]
        r = self._make_result(["1", "2", "3"], records=records)
        # All created with 3 signals = DELETION_CONFIRMED
        self.assertEqual(r.confirmed_count, 3)


# ===========================================================================
# run_deletion_pipeline
# ===========================================================================

class TestRunDeletionPipeline(unittest.TestCase):

    def _run(self, candidates: list[str] = None, **kwargs):
        candidates = candidates or ["1234567", "2345678"]
        manifest = _manifest("v1", ["9000000", "9000001"])  # present docs
        efta = _mock_efta(candidates)
        db = _mock_db()
        return run_deletion_pipeline(
            manifest=manifest,
            efta_scheme=efta,
            dynamodb_client=db,
            generate_report=False,
            **kwargs,
        ), db

    def test_returns_pipeline_result(self):
        result, _ = self._run()
        self.assertIsInstance(result, DeletionPipelineResult)

    def test_deletion_records_created_per_candidate(self):
        result, _ = self._run(candidates=["1", "2", "3"])
        self.assertEqual(len(result.deletion_records), 3)

    def test_records_written_count(self):
        result, db = self._run(candidates=["1", "2"])
        self.assertEqual(result.records_written, 2)

    def test_put_item_called_per_candidate(self):
        _, db = self._run(candidates=["1", "2", "3"])
        self.assertEqual(db.put_item.call_count, 3)

    def test_gap_report_generated_when_requested(self):
        manifest = _manifest("v1", ["9000000"])
        efta = _mock_efta(["1"])
        result = run_deletion_pipeline(
            manifest=manifest,
            efta_scheme=efta,
            dynamodb_client=_mock_db(),
            generate_report=True,
        )
        self.assertIsNotNone(result.gap_report)

    def test_gap_report_none_when_not_requested(self):
        result, _ = self._run()
        self.assertIsNone(result.gap_report)

    def test_errors_list_populated_on_dynamo_failure(self):
        manifest = _manifest("v1", ["9000000"])
        efta = _mock_efta(["1", "2"])
        db = _mock_db()
        db.put_item.side_effect = RuntimeError("DynamoDB down")
        result = run_deletion_pipeline(
            manifest=manifest,
            efta_scheme=efta,
            dynamodb_client=db,
            generate_report=False,
        )
        self.assertGreater(len(result.errors), 0)

    def test_prior_manifest_triggers_comparison(self):
        manifest = _manifest("v2", ["100", "200"])
        prior    = _manifest("v1", ["100", "200", "300"])
        efta     = _mock_efta([])
        result   = run_deletion_pipeline(
            manifest=manifest,
            efta_scheme=efta,
            dynamodb_client=_mock_db(),
            prior_manifest=prior,
            generate_report=False,
        )
        self.assertIsNotNone(result.comparison_result)
        self.assertEqual(result.comparison_result.deletion_count, 1)

    def test_manifest_version_stored(self):
        result, _ = self._run()
        self.assertEqual(result.manifest_version, "v1")


if __name__ == "__main__":
    unittest.main()

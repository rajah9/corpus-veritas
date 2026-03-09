"""
tests/pipeline/test_classifier.py

Unit tests for pipeline/classifier.py (Sub-Module 1D: Classification & Chain of Custody).

Coverage targets
----------------
DocumentClassification    -- all four values accessible
ClassificationRecord      -- to_dynamodb_item(), from_dynamodb_item(),
                             sparse GSI behaviour for victim_flag
_determine_classification -- all four rules, rule priority
_document_state_from_pii  -- all three state mappings
_write_classification_record
                          -- successful write, DynamoDB exception propagation
classify_document         -- all classification paths, DynamoDB write called,
                             corpus_source and provenance_tag persisted,
                             DynamoDB client created if not injected (mocked),
                             RuntimeError on DynamoDB failure

Temp resource strategy
----------------------
No temp files needed. All AWS clients are injected MagicMocks.
setUp/tearDown used only where environment variables are manipulated.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from pipeline.classifier import (
    DOCUMENTS_TABLE_NAME,
    ClassificationRecord,
    DocumentClassification,
    _determine_classification,
    _document_state_from_pii,
    _write_classification_record,
    classify_document,
)
from pipeline.models import DocumentState
from pipeline.sanitizer import PIIDetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_entity(begin: int, end: int) -> dict:
    return {"Type": "NAME", "BeginOffset": begin, "EndOffset": end, "Score": 0.99}


def _age_entity(begin: int, end: int) -> dict:
    return {"Type": "AGE", "BeginOffset": begin, "EndOffset": end, "Score": 0.99}


def _clean_pii(uuid: str = "uuid-001") -> PIIDetectionResult:
    return PIIDetectionResult(document_uuid=uuid)


def _victim_pii(uuid: str = "uuid-001") -> PIIDetectionResult:
    return PIIDetectionResult(
        document_uuid=uuid,
        pii_entities_detected=[_name_entity(0, 5)],
        victim_flag=True,
        requires_human_review=False,
        review_reason="NAME near victim term",
    )


def _review_pii(uuid: str = "uuid-001") -> PIIDetectionResult:
    return PIIDetectionResult(
        document_uuid=uuid,
        pii_entities_detected=[_name_entity(0, 5)],
        victim_flag=False,
        requires_human_review=True,
        review_reason="NAME near case-adjacent term",
    )


def _named_pii(uuid: str = "uuid-001") -> PIIDetectionResult:
    """Clean PII result (no flags) but with NAME entities detected."""
    return PIIDetectionResult(
        document_uuid=uuid,
        pii_entities_detected=[_name_entity(0, 10)],
        victim_flag=False,
        requires_human_review=False,
    )


def _mock_dynamodb() -> MagicMock:
    client = MagicMock()
    client.put_item.return_value = {"ResponseMetadata": {"HTTPStatusCode": 200}}
    return client


# ===========================================================================
# DocumentClassification enum
# ===========================================================================

class TestDocumentClassificationEnum(unittest.TestCase):

    def test_all_values_accessible(self):
        self.assertEqual(DocumentClassification.VICTIM_ADJACENT.value,      "VICTIM_ADJACENT")
        self.assertEqual(DocumentClassification.PERPETRATOR_ADJACENT.value,  "PERPETRATOR_ADJACENT")
        self.assertEqual(DocumentClassification.PROCEDURAL.value,            "PROCEDURAL")
        self.assertEqual(DocumentClassification.UNKNOWN.value,               "UNKNOWN")

    def test_is_str_subclass(self):
        self.assertIsInstance(DocumentClassification.VICTIM_ADJACENT, str)


# ===========================================================================
# ClassificationRecord
# ===========================================================================

class TestClassificationRecord(unittest.TestCase):

    def _record(self, victim_flag: bool = False) -> ClassificationRecord:
        return ClassificationRecord(
            document_uuid="uuid-001",
            classification=DocumentClassification.PROCEDURAL,
            state=DocumentState.SANITIZED,
            ingestion_date="2026-03-09T12:00:00+00:00",
            victim_flag=victim_flag,
            corpus_source="corpus-abc",
            provenance_tag="PROVENANCE_COMMUNITY_VOUCHED",
            pii_entity_count=0,
            review_reason=None,
            notes="test",
        )

    # to_dynamodb_item

    def test_required_fields_present_in_item(self):
        item = self._record().to_dynamodb_item()
        for key in ("document_uuid", "classification", "state",
                    "ingestion_date", "pii_entity_count", "created_at"):
            self.assertIn(key, item)

    def test_victim_flag_absent_when_false(self):
        """Sparse GSI: victim_flag must NOT appear when False."""
        item = self._record(victim_flag=False).to_dynamodb_item()
        self.assertNotIn("victim_flag", item)

    def test_victim_flag_present_when_true(self):
        """Sparse GSI: victim_flag MUST appear when True."""
        item = self._record(victim_flag=True).to_dynamodb_item()
        self.assertIn("victim_flag", item)
        self.assertEqual(item["victim_flag"]["S"], "true")

    def test_optional_fields_included_when_set(self):
        item = self._record().to_dynamodb_item()
        self.assertIn("corpus_source", item)
        self.assertIn("provenance_tag", item)
        self.assertIn("notes", item)

    def test_optional_fields_absent_when_none(self):
        record = ClassificationRecord(
            document_uuid="uuid-002",
            classification=DocumentClassification.UNKNOWN,
            state=DocumentState.SANITIZED,
            ingestion_date="2026-03-09T00:00:00+00:00",
        )
        item = record.to_dynamodb_item()
        self.assertNotIn("corpus_source", item)
        self.assertNotIn("provenance_tag", item)
        self.assertNotIn("notes", item)
        self.assertNotIn("review_reason", item)

    def test_dynamodb_types_are_strings(self):
        item = self._record().to_dynamodb_item()
        self.assertIn("S", item["document_uuid"])
        self.assertIn("N", item["pii_entity_count"])

    # from_dynamodb_item round-trip

    def test_round_trip_preserves_all_fields(self):
        original = self._record(victim_flag=True)
        item = original.to_dynamodb_item()
        restored = ClassificationRecord.from_dynamodb_item(item)
        self.assertEqual(restored.document_uuid,   original.document_uuid)
        self.assertEqual(restored.classification,  original.classification)
        self.assertEqual(restored.state,           original.state)
        self.assertEqual(restored.ingestion_date,  original.ingestion_date)
        self.assertTrue(restored.victim_flag)
        self.assertEqual(restored.corpus_source,   original.corpus_source)
        self.assertEqual(restored.provenance_tag,  original.provenance_tag)
        self.assertEqual(restored.pii_entity_count, original.pii_entity_count)

    def test_round_trip_victim_false_preserved(self):
        record = self._record(victim_flag=False)
        item = record.to_dynamodb_item()
        restored = ClassificationRecord.from_dynamodb_item(item)
        self.assertFalse(restored.victim_flag)


# ===========================================================================
# _determine_classification
# ===========================================================================

class TestDetermineClassification(unittest.TestCase):

    def test_victim_flag_returns_victim_adjacent(self):
        result = _determine_classification("any text", _victim_pii())
        self.assertEqual(result, DocumentClassification.VICTIM_ADJACENT)

    def test_victim_flag_overrides_procedural_markers(self):
        """VICTIM_ADJACENT must win even if the document has FBI 302 headers."""
        text = "FD-302 FEDERAL BUREAU OF INVESTIGATION John Doe is a victim."
        result = _determine_classification(text, _victim_pii())
        self.assertEqual(result, DocumentClassification.VICTIM_ADJACENT)

    def test_fd302_header_returns_procedural(self):
        text = "FD-302 (Rev. 5-8-10) FEDERAL BUREAU OF INVESTIGATION"
        result = _determine_classification(text, _clean_pii())
        self.assertEqual(result, DocumentClassification.PROCEDURAL)

    def test_case_number_returns_procedural(self):
        text = "In the matter of: Case No. 20-cr-330 (SDNY)"
        result = _determine_classification(text, _clean_pii())
        self.assertEqual(result, DocumentClassification.PROCEDURAL)

    def test_sdny_marker_returns_procedural(self):
        text = "UNITED STATES DISTRICT COURT SOUTHERN DISTRICT OF NEW YORK"
        result = _determine_classification(text, _clean_pii())
        self.assertEqual(result, DocumentClassification.PROCEDURAL)

    def test_memorandum_for_returns_procedural(self):
        text = "MEMORANDUM FOR: Director, Federal Bureau of Investigation"
        result = _determine_classification(text, _clean_pii())
        self.assertEqual(result, DocumentClassification.PROCEDURAL)

    def test_name_entities_returns_perpetrator_adjacent(self):
        result = _determine_classification(
            "John Smith attended the gathering.", _named_pii()
        )
        self.assertEqual(result, DocumentClassification.PERPETRATOR_ADJACENT)

    def test_no_signals_returns_unknown(self):
        result = _determine_classification("Generic unrelated document text.", _clean_pii())
        self.assertEqual(result, DocumentClassification.UNKNOWN)

    def test_procedural_takes_priority_over_perpetrator_adjacent(self):
        """A procedural document with NAME entities is still PROCEDURAL."""
        text = "FD-302: John Smith was interviewed on 2026-01-15."
        pii = _named_pii()   # has NAME entities, no victim flag
        result = _determine_classification(text, pii)
        self.assertEqual(result, DocumentClassification.PROCEDURAL)


# ===========================================================================
# _document_state_from_pii
# ===========================================================================

class TestDocumentStateFromPii(unittest.TestCase):

    def test_victim_flag_returns_victim_flagged(self):
        self.assertEqual(
            _document_state_from_pii(_victim_pii()), DocumentState.VICTIM_FLAGGED
        )

    def test_review_flag_returns_pending_review(self):
        self.assertEqual(
            _document_state_from_pii(_review_pii()), DocumentState.PENDING_REVIEW
        )

    def test_clean_returns_sanitized(self):
        self.assertEqual(
            _document_state_from_pii(_clean_pii()), DocumentState.SANITIZED
        )

    def test_victim_flag_takes_priority_over_review(self):
        pii = PIIDetectionResult(
            document_uuid="uuid-001",
            victim_flag=True,
            requires_human_review=True,
        )
        self.assertEqual(_document_state_from_pii(pii), DocumentState.VICTIM_FLAGGED)


# ===========================================================================
# _write_classification_record
# ===========================================================================

class TestWriteClassificationRecord(unittest.TestCase):

    def _record(self) -> ClassificationRecord:
        return ClassificationRecord(
            document_uuid="uuid-write-001",
            classification=DocumentClassification.UNKNOWN,
            state=DocumentState.SANITIZED,
            ingestion_date="2026-03-09T00:00:00+00:00",
        )

    def test_put_item_called(self):
        db = _mock_dynamodb()
        _write_classification_record(self._record(), db)
        db.put_item.assert_called_once()

    def test_correct_table_name_used(self):
        db = _mock_dynamodb()
        _write_classification_record(self._record(), db, table_name="my_table")
        self.assertEqual(db.put_item.call_args.kwargs["TableName"], "my_table")

    def test_default_table_name_used(self):
        db = _mock_dynamodb()
        _write_classification_record(self._record(), db)
        self.assertEqual(
            db.put_item.call_args.kwargs["TableName"], DOCUMENTS_TABLE_NAME
        )

    def test_dynamodb_exception_raises_runtime_error(self):
        db = MagicMock()
        db.put_item.side_effect = RuntimeError("DynamoDB unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            _write_classification_record(self._record(), db)
        self.assertIn("uuid-write-001", str(ctx.exception))


# ===========================================================================
# classify_document
# ===========================================================================

class TestClassifyDocument(unittest.TestCase):

    def setUp(self):
        self.db = _mock_dynamodb()

    def _run(self, text: str, pii: PIIDetectionResult, **kwargs) -> ClassificationRecord:
        return classify_document(
            document_uuid=pii.document_uuid,
            text=text,
            pii_result=pii,
            dynamodb_client=self.db,
            **kwargs,
        )

    # Classification outcomes

    def test_victim_pii_produces_victim_adjacent(self):
        r = self._run("Some text.", _victim_pii())
        self.assertEqual(r.classification, DocumentClassification.VICTIM_ADJACENT)

    def test_victim_pii_produces_victim_flagged_state(self):
        r = self._run("Some text.", _victim_pii())
        self.assertEqual(r.state, DocumentState.VICTIM_FLAGGED)

    def test_procedural_text_produces_procedural(self):
        text = "FD-302 FEDERAL BUREAU OF INVESTIGATION"
        r = self._run(text, _clean_pii())
        self.assertEqual(r.classification, DocumentClassification.PROCEDURAL)

    def test_named_pii_produces_perpetrator_adjacent(self):
        r = self._run("John Smith was present.", _named_pii())
        self.assertEqual(r.classification, DocumentClassification.PERPETRATOR_ADJACENT)

    def test_no_signals_produces_unknown(self):
        r = self._run("No names, no markers.", _clean_pii())
        self.assertEqual(r.classification, DocumentClassification.UNKNOWN)

    def test_review_pii_produces_pending_review_state(self):
        r = self._run("Some text.", _review_pii())
        self.assertEqual(r.state, DocumentState.PENDING_REVIEW)

    def test_clean_pii_produces_sanitized_state(self):
        r = self._run("Some text.", _clean_pii())
        self.assertEqual(r.state, DocumentState.SANITIZED)

    # Record fields

    def test_document_uuid_preserved(self):
        r = self._run("text", _clean_pii("uuid-xyz"))
        self.assertEqual(r.document_uuid, "uuid-xyz")

    def test_corpus_source_persisted(self):
        r = self._run("text", _clean_pii(), corpus_source="corpus-123")
        self.assertEqual(r.corpus_source, "corpus-123")

    def test_provenance_tag_persisted(self):
        r = self._run("text", _clean_pii(),
                       provenance_tag="PROVENANCE_COMMUNITY_VOUCHED")
        self.assertEqual(r.provenance_tag, "PROVENANCE_COMMUNITY_VOUCHED")

    def test_pii_entity_count_correct(self):
        pii = _named_pii()
        r = self._run("text", pii)
        self.assertEqual(r.pii_entity_count, len(pii.pii_entities_detected))

    def test_victim_flag_set_on_record(self):
        r = self._run("text", _victim_pii())
        self.assertTrue(r.victim_flag)

    def test_victim_flag_false_on_clean_record(self):
        r = self._run("text", _clean_pii())
        self.assertFalse(r.victim_flag)

    def test_ingestion_date_is_set(self):
        r = self._run("text", _clean_pii())
        self.assertTrue(r.ingestion_date)

    def test_review_reason_preserved(self):
        r = self._run("text", _review_pii())
        self.assertEqual(r.review_reason, "NAME near case-adjacent term")

    # DynamoDB interaction

    def test_dynamodb_write_called(self):
        self._run("text", _clean_pii())
        self.db.put_item.assert_called_once()

    def test_dynamodb_item_contains_document_uuid(self):
        self._run("text", _clean_pii("uuid-check"))
        item = self.db.put_item.call_args.kwargs["Item"]
        self.assertEqual(item["document_uuid"]["S"], "uuid-check")

    def test_victim_flag_in_dynamodb_item_when_true(self):
        self._run("text", _victim_pii())
        item = self.db.put_item.call_args.kwargs["Item"]
        self.assertIn("victim_flag", item)

    def test_victim_flag_absent_in_dynamodb_item_when_false(self):
        self._run("text", _clean_pii())
        item = self.db.put_item.call_args.kwargs["Item"]
        self.assertNotIn("victim_flag", item)

    def test_dynamodb_failure_raises_runtime_error(self):
        self.db.put_item.side_effect = RuntimeError("DynamoDB down")
        with self.assertRaises(RuntimeError):
            self._run("text", _clean_pii())

    def test_returns_classification_record_type(self):
        r = self._run("text", _clean_pii())
        self.assertIsInstance(r, ClassificationRecord)


if __name__ == "__main__":
    unittest.main()
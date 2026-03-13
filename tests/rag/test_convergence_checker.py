"""
tests/rag/test_convergence_checker.py

Unit tests for rag/convergence_checker.py.

Coverage targets
----------------
_extract_sequence_int()     -- numeric string, None, non-numeric Bates
_are_independent()          -- same uuid → not independent, different uuid
                               + distant sequence → independent, different
                               uuid + adjacent sequence → not independent,
                               different uuid + no sequence → independent
_collect_independent_sources()
                            -- deduplicates same-document chunks, returns
                               one chunk per independent source
_derive_convergence_tier()  -- count=0, count=1, count=2, count=3 no
                               diversity, count=3 with diversity
check_convergence()         -- empty chunks, single source, two sources,
                               three sources with diversity, three without,
                               meets_inference_threshold boundary,
                               suppression_message when below threshold,
                               no suppression_message when above threshold,
                               document_types_present populated correctly,
                               independent_document_uuids populated
"""

from __future__ import annotations

import unittest

from pipeline.models import ConfidenceTier
from rag.convergence_checker import (
    INFERENCE_THRESHOLD,
    _are_independent,
    _collect_independent_sources,
    _derive_convergence_tier,
    _extract_sequence_int,
    check_convergence,
)
from rag.query_router import QueryRequest, QueryType, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(
    uuid: str = "uuid-001",
    seq: str = "1000",
    doc_type: str = "FBI_302",
) -> dict:
    return {
        "document_uuid": uuid,
        "sequence_number": seq,
        "document_type": doc_type,
        "text": "some text",
    }


def _result(chunks: list) -> RetrievalResult:
    return RetrievalResult(
        query=QueryRequest(query_text="test", query_type=QueryType.INFERENCE),
        chunks=chunks,
        answer="answer",
        convergence_applied=False,
        retrieved_at="2026-03-11T00:00:00+00:00",
    )


# ===========================================================================
# _extract_sequence_int
# ===========================================================================

class TestExtractSequenceInt(unittest.TestCase):

    def test_numeric_string_parsed(self):
        self.assertEqual(_extract_sequence_int("12345"), 12345)

    def test_none_returns_none(self):
        self.assertIsNone(_extract_sequence_int(None))

    def test_non_numeric_bates_returns_none(self):
        self.assertIsNone(_extract_sequence_int("DOJ-000042"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_extract_sequence_int(""))

    def test_zero_parsed(self):
        self.assertEqual(_extract_sequence_int("0"), 0)


# ===========================================================================
# _are_independent
# ===========================================================================

class TestAreIndependent(unittest.TestCase):

    def test_same_uuid_not_independent(self):
        a = _chunk("uuid-001", "1000")
        b = _chunk("uuid-001", "9999")
        self.assertFalse(_are_independent(a, b))

    def test_different_uuid_distant_seq_independent(self):
        a = _chunk("uuid-001", "1000")
        b = _chunk("uuid-002", "5000")
        self.assertTrue(_are_independent(a, b))

    def test_different_uuid_adjacent_seq_not_independent(self):
        a = _chunk("uuid-001", "1000")
        b = _chunk("uuid-002", "1010")  # within adjacency threshold
        self.assertFalse(_are_independent(a, b))

    def test_different_uuid_no_sequence_independent(self):
        a = {"document_uuid": "uuid-001"}
        b = {"document_uuid": "uuid-002"}
        self.assertTrue(_are_independent(a, b))

    def test_different_uuid_one_missing_seq_independent(self):
        a = _chunk("uuid-001", "1000")
        b = {"document_uuid": "uuid-002"}  # no sequence_number
        self.assertTrue(_are_independent(a, b))

    def test_different_uuid_bates_non_numeric_independent(self):
        a = {"document_uuid": "uuid-001", "sequence_number": "DOJ-001"}
        b = {"document_uuid": "uuid-002", "sequence_number": "DOJ-002"}
        self.assertTrue(_are_independent(a, b))

    def test_symmetry(self):
        a = _chunk("uuid-001", "1000")
        b = _chunk("uuid-002", "5000")
        self.assertEqual(_are_independent(a, b), _are_independent(b, a))


# ===========================================================================
# _collect_independent_sources
# ===========================================================================

class TestCollectIndependentSources(unittest.TestCase):

    def test_empty_input_returns_empty(self):
        self.assertEqual(_collect_independent_sources([]), [])

    def test_single_chunk_returns_itself(self):
        chunks = [_chunk("uuid-001", "1000")]
        result = _collect_independent_sources(chunks)
        self.assertEqual(len(result), 1)

    def test_duplicate_uuid_deduped(self):
        chunks = [
            _chunk("uuid-001", "1000"),
            _chunk("uuid-001", "1005"),  # same document
        ]
        result = _collect_independent_sources(chunks)
        self.assertEqual(len(result), 1)

    def test_two_independent_sources_both_retained(self):
        chunks = [
            _chunk("uuid-001", "1000"),
            _chunk("uuid-002", "5000"),
        ]
        result = _collect_independent_sources(chunks)
        self.assertEqual(len(result), 2)

    def test_three_independent_sources_all_retained(self):
        chunks = [
            _chunk("uuid-001", "1000"),
            _chunk("uuid-002", "5000"),
            _chunk("uuid-003", "10000"),
        ]
        result = _collect_independent_sources(chunks)
        self.assertEqual(len(result), 3)

    def test_mixed_independent_and_dependent(self):
        chunks = [
            _chunk("uuid-001", "1000"),
            _chunk("uuid-001", "1001"),  # same doc, adjacent seq
            _chunk("uuid-002", "5000"),  # independent
        ]
        result = _collect_independent_sources(chunks)
        self.assertEqual(len(result), 2)


# ===========================================================================
# _derive_convergence_tier
# ===========================================================================

class TestDeriveConvergenceTier(unittest.TestCase):

    def test_zero_sources_single_source(self):
        self.assertEqual(
            _derive_convergence_tier(0, []), ConfidenceTier.SINGLE_SOURCE
        )

    def test_one_source_single_source(self):
        self.assertEqual(
            _derive_convergence_tier(1, ["FBI_302"]), ConfidenceTier.SINGLE_SOURCE
        )

    def test_two_sources_corroborated(self):
        self.assertEqual(
            _derive_convergence_tier(2, ["FBI_302", "FBI_302"]),
            ConfidenceTier.CORROBORATED,
        )

    def test_three_sources_no_diversity_corroborated(self):
        self.assertEqual(
            _derive_convergence_tier(3, ["FBI_302", "FBI_302", "FBI_302"]),
            ConfidenceTier.CORROBORATED,
        )

    def test_three_sources_with_diversity_confirmed(self):
        self.assertEqual(
            _derive_convergence_tier(3, ["FBI_302", "COURT_FILING", "CORRESPONDENCE"]),
            ConfidenceTier.CONFIRMED,
        )

    def test_four_sources_with_diversity_confirmed(self):
        self.assertEqual(
            _derive_convergence_tier(4, ["FBI_302", "COURT_FILING", "FBI_302", "EXHIBIT"]),
            ConfidenceTier.CONFIRMED,
        )


# ===========================================================================
# check_convergence
# ===========================================================================

class TestCheckConvergence(unittest.TestCase):

    def test_empty_chunks_returns_zero_count(self):
        conv = check_convergence(_result([]))
        self.assertEqual(conv.independent_source_count, 0)

    def test_empty_chunks_not_meets_threshold(self):
        conv = check_convergence(_result([]))
        self.assertFalse(conv.meets_inference_threshold)

    def test_empty_chunks_has_suppression_message(self):
        conv = check_convergence(_result([]))
        self.assertTrue(len(conv.suppression_message) > 0)

    def test_single_source_not_meets_threshold(self):
        conv = check_convergence(_result([_chunk("uuid-001", "1000")]))
        self.assertFalse(conv.meets_inference_threshold)

    def test_single_source_suppression_message_present(self):
        conv = check_convergence(_result([_chunk("uuid-001", "1000")]))
        self.assertTrue(len(conv.suppression_message) > 0)

    def test_suppression_message_mentions_threshold(self):
        conv = check_convergence(_result([_chunk("uuid-001", "1000")]))
        self.assertIn(str(INFERENCE_THRESHOLD), conv.suppression_message)

    def test_two_independent_sources_meets_threshold(self):
        chunks = [_chunk("uuid-001", "1000"), _chunk("uuid-002", "5000")]
        conv = check_convergence(_result(chunks))
        self.assertTrue(conv.meets_inference_threshold)

    def test_two_sources_no_suppression_message(self):
        chunks = [_chunk("uuid-001", "1000"), _chunk("uuid-002", "5000")]
        conv = check_convergence(_result(chunks))
        self.assertEqual(conv.suppression_message, "")

    def test_two_sources_corroborated_tier(self):
        chunks = [_chunk("uuid-001", "1000"), _chunk("uuid-002", "5000")]
        conv = check_convergence(_result(chunks))
        self.assertEqual(conv.convergence_tier, ConfidenceTier.CORROBORATED)

    def test_three_diverse_sources_confirmed_tier(self):
        chunks = [
            _chunk("uuid-001", "1000", "FBI_302"),
            _chunk("uuid-002", "5000", "COURT_FILING"),
            _chunk("uuid-003", "9000", "CORRESPONDENCE"),
        ]
        conv = check_convergence(_result(chunks))
        self.assertEqual(conv.convergence_tier, ConfidenceTier.CONFIRMED)

    def test_three_homogeneous_sources_corroborated_tier(self):
        chunks = [
            _chunk("uuid-001", "1000", "FBI_302"),
            _chunk("uuid-002", "5000", "FBI_302"),
            _chunk("uuid-003", "9000", "FBI_302"),
        ]
        conv = check_convergence(_result(chunks))
        self.assertEqual(conv.convergence_tier, ConfidenceTier.CORROBORATED)

    def test_independent_document_uuids_populated(self):
        chunks = [_chunk("uuid-001", "1000"), _chunk("uuid-002", "5000")]
        conv = check_convergence(_result(chunks))
        self.assertIn("uuid-001", conv.independent_document_uuids)
        self.assertIn("uuid-002", conv.independent_document_uuids)

    def test_document_types_present_populated(self):
        chunks = [
            _chunk("uuid-001", "1000", "FBI_302"),
            _chunk("uuid-002", "5000", "COURT_FILING"),
        ]
        conv = check_convergence(_result(chunks))
        self.assertIn("FBI_302", conv.document_types_present)
        self.assertIn("COURT_FILING", conv.document_types_present)

    def test_chunks_missing_document_type_handled(self):
        chunks = [
            {"document_uuid": "uuid-001", "sequence_number": "1000", "text": "t"},
            {"document_uuid": "uuid-002", "sequence_number": "5000", "text": "t"},
        ]
        conv = check_convergence(_result(chunks))
        self.assertEqual(conv.document_types_present, [])
        self.assertTrue(conv.meets_inference_threshold)

    def test_duplicate_uuid_chunks_counted_once(self):
        chunks = [
            _chunk("uuid-001", "1000"),
            _chunk("uuid-001", "1001"),
            _chunk("uuid-001", "1002"),
        ]
        conv = check_convergence(_result(chunks))
        self.assertEqual(conv.independent_source_count, 1)
        self.assertFalse(conv.meets_inference_threshold)


if __name__ == "__main__":
    unittest.main()

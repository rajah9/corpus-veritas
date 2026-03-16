"""
tests/rag/test_guardrail.py

Unit tests for rag/guardrail.py.

Coverage targets
----------------
check_victim_identity()     -- known victim name suppressed, surface form
                               suppressed, case-insensitive, no match
                               returns unchanged, empty victim list,
                               triggered flag set correctly

check_inference_threshold() -- non-INFERENCE query is no-op, INFERENCE
                               with threshold met returns unchanged,
                               INFERENCE below threshold replaces answer
                               with suppression_message, downgraded flag,
                               convergence computed when not supplied

check_confidence_calibration() -- CONFIRMED tier no-op, CORROBORATED
                               triggers correction, "confirmed" replaced,
                               "definitively proves" replaced, no-tier
                               no-op, violated flag set correctly

apply_guardrail()           -- GuardrailResult returned, safe_answer
                               populated, audit_entry_id set, all three
                               check flags reflected, checks_passed and
                               checks_failed populated correctly, audit
                               write called before return, AuditLogFailure
                               propagates (Hard Limit 5), victim suppression
                               + inference downgrade + confidence violation
                               all propagate to GuardrailResult
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from pipeline.audit_log import AuditLogFailure
from pipeline.models import ConfidenceTier
from rag.convergence_checker import ConvergenceResult
from rag.guardrail import (
    GuardrailResult,
    apply_guardrail,
    check_confidence_calibration,
    check_inference_threshold,
    check_victim_identity,
)
from rag.query_router import QueryRequest, QueryType, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    query_type: QueryType = QueryType.PROVENANCE,
    answer: str = "The documents indicate various connections.",
    chunks: list = None,
    convergence_applied: bool = True,
    lowest_tier: str = None,
) -> RetrievalResult:
    if chunks is None:
        chunks = [{"document_uuid": "uuid-001", "text": "chunk text",
                   "confidence_tier": lowest_tier or ConfidenceTier.CORROBORATED,
                   "provenance_tag": "PROVENANCE_DOJ_DIRECT"}]
    return RetrievalResult(
        query=QueryRequest(query_text="test query", query_type=query_type),
        chunks=chunks,
        answer=answer,
        convergence_applied=convergence_applied,
        retrieved_at="2026-03-16T10:00:00+00:00",
        lowest_tier=lowest_tier,
    )


def _convergence(meets: bool = True, count: int = 2) -> ConvergenceResult:
    return ConvergenceResult(
        independent_source_count=count,
        independent_document_uuids=["uuid-001", "uuid-002"][:count],
        document_types_present=["FBI_302"],
        convergence_tier=ConfidenceTier.CORROBORATED if meets else ConfidenceTier.SINGLE_SOURCE,
        meets_inference_threshold=meets,
        suppression_message="" if meets else (
            "This inference cannot be surfaced. It is supported by "
            "1 independent source document(s), but 2 are required."
        ),
    )


def _mock_audit():
    """Patch write_audit_log to succeed silently."""
    return patch("rag.guardrail.write_audit_log", return_value=None)


# ===========================================================================
# check_victim_identity
# ===========================================================================

class TestCheckVictimIdentity(unittest.TestCase):

    def test_known_victim_name_suppressed(self):
        safe, triggered = check_victim_identity(
            "virginia giuffre testified that...",
            victim_names=["virginia giuffre"],
        )
        self.assertIn("[protected identity]", safe)
        self.assertTrue(triggered)

    def test_suppression_case_insensitive(self):
        safe, triggered = check_victim_identity(
            "Virginia Giuffre said...",
            victim_names=["virginia giuffre"],
        )
        self.assertIn("[protected identity]", safe)
        self.assertTrue(triggered)

    def test_no_match_returns_unchanged(self):
        text = "Jeffrey Epstein was associated with many powerful people."
        safe, triggered = check_victim_identity(text, victim_names=["virginia giuffre"])
        self.assertEqual(safe, text)
        self.assertFalse(triggered)

    def test_empty_victim_list_no_change(self):
        text = "Some response text."
        safe, triggered = check_victim_identity(text, victim_names=[])
        self.assertEqual(safe, text)
        self.assertFalse(triggered)

    def test_none_victim_list_uses_baseline(self):
        # With no caller-supplied names, baseline from entity_resolver applies
        # virginia giuffre is in _KNOWN_VICTIM_CANONICAL_NAMES
        safe, triggered = check_victim_identity(
            "virginia giuffre testified", victim_names=None
        )
        self.assertTrue(triggered)

    def test_multiple_occurrences_all_suppressed(self):
        safe, triggered = check_victim_identity(
            "victim A and victim A again",
            victim_names=["victim a"],
        )
        self.assertEqual(safe.count("[protected identity]"), 2)

    def test_surface_form_suppressed(self):
        safe, triggered = check_victim_identity(
            "Jane Doe attended the island.",
            victim_names=["jane doe"],
        )
        self.assertTrue(triggered)


# ===========================================================================
# check_inference_threshold
# ===========================================================================

class TestCheckInferenceThreshold(unittest.TestCase):

    def test_non_inference_query_is_noop(self):
        result = _result(QueryType.PROVENANCE, answer="Some answer.")
        safe, downgraded = check_inference_threshold("Some answer.", result)
        self.assertEqual(safe, "Some answer.")
        self.assertFalse(downgraded)

    def test_inference_threshold_met_returns_unchanged(self):
        result = _result(QueryType.INFERENCE, convergence_applied=False)
        conv = _convergence(meets=True, count=2)
        safe, downgraded = check_inference_threshold(result.answer, result, conv)
        self.assertEqual(safe, result.answer)
        self.assertFalse(downgraded)

    def test_inference_below_threshold_replaces_answer(self):
        result = _result(QueryType.INFERENCE, convergence_applied=False)
        conv = _convergence(meets=False, count=1)
        safe, downgraded = check_inference_threshold(result.answer, result, conv)
        self.assertNotEqual(safe, result.answer)
        self.assertTrue(downgraded)

    def test_suppression_message_used_as_answer(self):
        result = _result(QueryType.INFERENCE, convergence_applied=False)
        conv = _convergence(meets=False, count=1)
        safe, _ = check_inference_threshold(result.answer, result, conv)
        self.assertEqual(safe, conv.suppression_message)

    def test_convergence_computed_when_not_supplied(self):
        result = _result(
            QueryType.INFERENCE,
            convergence_applied=False,
            chunks=[{"document_uuid": "uuid-001", "sequence_number": "1000",
                     "text": "t"}],
        )
        # Single chunk = single source = below threshold
        safe, downgraded = check_inference_threshold(result.answer, result, None)
        self.assertTrue(downgraded)

    def test_timeline_query_is_noop(self):
        result = _result(QueryType.TIMELINE)
        safe, downgraded = check_inference_threshold("Answer.", result)
        self.assertFalse(downgraded)


# ===========================================================================
# check_confidence_calibration
# ===========================================================================

class TestCheckConfidenceCalibration(unittest.TestCase):

    def test_confirmed_tier_is_noop(self):
        text = "This is confirmed beyond doubt."
        safe, violated = check_confidence_calibration(text, ConfidenceTier.CONFIRMED)
        self.assertEqual(safe, text)
        self.assertFalse(violated)

    def test_none_tier_is_noop(self):
        text = "This is confirmed."
        safe, violated = check_confidence_calibration(text, None)
        self.assertEqual(safe, text)
        self.assertFalse(violated)

    def test_confirmed_word_replaced_for_corroborated(self):
        safe, violated = check_confidence_calibration(
            "This is confirmed by the documents.",
            ConfidenceTier.CORROBORATED,
        )
        self.assertNotIn("confirmed", safe.lower())
        self.assertTrue(violated)

    def test_definitively_proves_replaced(self):
        safe, violated = check_confidence_calibration(
            "This definitively proves the connection.",
            ConfidenceTier.SINGLE_SOURCE,
        )
        self.assertNotIn("definitively proves", safe.lower())
        self.assertTrue(violated)

    def test_no_forbidden_language_no_violation(self):
        text = "The documents suggest a possible connection."
        safe, violated = check_confidence_calibration(text, ConfidenceTier.CORROBORATED)
        self.assertEqual(safe, text)
        self.assertFalse(violated)

    def test_speculative_tier_triggers_calibration(self):
        safe, violated = check_confidence_calibration(
            "This certainly happened.",
            ConfidenceTier.SPECULATIVE,
        )
        self.assertTrue(violated)

    def test_inferred_tier_triggers_calibration(self):
        safe, violated = check_confidence_calibration(
            "This is confirmed.",
            ConfidenceTier.INFERRED,
        )
        self.assertTrue(violated)

    def test_replacement_is_readable(self):
        safe, _ = check_confidence_calibration(
            "This is confirmed.",
            ConfidenceTier.CORROBORATED,
        )
        self.assertTrue(len(safe) > 0)
        self.assertNotIn("confirmed", safe.lower())


# ===========================================================================
# apply_guardrail
# ===========================================================================

class TestApplyGuardrail(unittest.TestCase):

    def setUp(self):
        self.result = _result(
            QueryType.PROVENANCE,
            answer="The documents suggest connections.",
            lowest_tier=ConfidenceTier.CORROBORATED,
        )

    def _run(self, result=None, **kwargs):
        with _mock_audit():
            return apply_guardrail(
                result or self.result,
                audit_bucket="audit-bucket",
                **kwargs,
            )

    def test_returns_guardrail_result(self):
        self.assertIsInstance(self._run(), GuardrailResult)

    def test_safe_answer_populated(self):
        gr = self._run()
        self.assertTrue(len(gr.safe_answer) > 0)

    def test_original_answer_preserved(self):
        gr = self._run()
        self.assertEqual(gr.original_answer, self.result.answer)

    def test_audit_entry_id_set(self):
        gr = self._run()
        self.assertTrue(len(gr.audit_entry_id) > 0)

    def test_no_triggers_all_checks_passed(self):
        gr = self._run()
        self.assertIn("victim_identity", gr.checks_passed)
        self.assertIn("inference_threshold", gr.checks_passed)
        self.assertIn("confidence_calibration", gr.checks_passed)
        self.assertEqual(gr.checks_failed, [])

    def test_victim_trigger_reflected_in_result(self):
        result = _result(answer="virginia giuffre testified.")
        gr = self._run(result=result, victim_entity_names=["virginia giuffre"])
        self.assertTrue(gr.victim_scan_triggered)
        self.assertIn("victim_identity", gr.checks_failed)
        self.assertNotIn("virginia giuffre", gr.safe_answer.lower())

    def test_inference_downgrade_reflected_in_result(self):
        result = _result(
            QueryType.INFERENCE,
            convergence_applied=False,
            chunks=[{"document_uuid": "uuid-001", "sequence_number": "1000",
                     "text": "t", "confidence_tier": ConfidenceTier.SINGLE_SOURCE}],
        )
        gr = self._run(result=result, convergence_result=_convergence(meets=False))
        self.assertTrue(gr.inference_downgraded)
        self.assertIn("inference_threshold", gr.checks_failed)

    def test_confidence_violation_reflected_in_result(self):
        result = _result(
            answer="This is confirmed by the documents.",
            lowest_tier=ConfidenceTier.CORROBORATED,
        )
        gr = self._run(result=result)
        self.assertTrue(gr.confidence_violation)
        self.assertIn("confidence_calibration", gr.checks_failed)

    def test_audit_write_called(self):
        with patch("rag.guardrail.write_audit_log") as mock_write:
            apply_guardrail(
                self.result,
                cloudwatch_client=MagicMock(),
                s3_client=MagicMock(),
                audit_bucket="audit-bucket",
            )
            mock_write.assert_called_once()

    def test_audit_log_failure_propagates(self):
        with patch(
            "rag.guardrail.write_audit_log",
            side_effect=AuditLogFailure("Audit failed"),
        ):
            with self.assertRaises(AuditLogFailure):
                apply_guardrail(self.result, audit_bucket="audit-bucket")

    def test_safe_answer_not_returned_on_audit_failure(self):
        """Caller never receives GuardrailResult when audit fails."""
        delivered = []
        with patch(
            "rag.guardrail.write_audit_log",
            side_effect=AuditLogFailure("Audit failed"),
        ):
            try:
                result = apply_guardrail(self.result, audit_bucket="audit-bucket")
                delivered.append(result)
            except AuditLogFailure:
                pass
        self.assertEqual(delivered, [])

    def test_checks_run_in_order_victim_before_inference(self):
        """Victim suppression fires on original answer before inference check."""
        result = _result(
            QueryType.INFERENCE,
            answer="virginia giuffre testified about this.",
            convergence_applied=False,
            chunks=[{"document_uuid": "uuid-001", "sequence_number": "1000",
                     "text": "t"}],
        )
        conv = _convergence(meets=False)
        gr = self._run(
            result=result,
            victim_entity_names=["virginia giuffre"],
            convergence_result=conv,
        )
        self.assertTrue(gr.victim_scan_triggered)
        self.assertTrue(gr.inference_downgraded)


if __name__ == "__main__":
    unittest.main()

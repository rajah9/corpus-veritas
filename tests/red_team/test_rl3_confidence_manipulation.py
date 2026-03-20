"""
tests/red_team/test_rl3_confidence_manipulation.py
Red Team: Hard Limit 3 — CONFIRMED language must not appear for sub-CONFIRMED tier.

Adversarial tests verifying that CONFIRMED-tier language is corrected
regardless of how it appears in synthesised responses.

Constitution Hard Limit 3. Principle II.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pipeline.models import ConfidenceTier
from rag.guardrail import check_confidence_calibration, apply_guardrail
from rag.query_router import QueryRequest, QueryType, RetrievalResult


def _result(answer: str, tier: str) -> RetrievalResult:
    return RetrievalResult(
        query=QueryRequest(query_text="test", query_type=QueryType.PROVENANCE),
        chunks=[{"document_uuid": "uuid-001", "text": "chunk",
                 "confidence_tier": tier}],
        answer=answer,
        convergence_applied=True,
        retrieved_at="2026-01-01T00:00:00+00:00",
        lowest_tier=tier,
    )


class TestConfidenceLanguagePatterns(unittest.TestCase):
    """Each forbidden language pattern must be corrected."""

    def _check(self, text: str, tier: str = ConfidenceTier.CORROBORATED):
        safe, violated = check_confidence_calibration(text, tier)
        return safe, violated

    def test_confirmed_corrected_for_corroborated(self):
        safe, v = self._check("This is confirmed by documents.", ConfidenceTier.CORROBORATED)
        self.assertTrue(v)
        self.assertNotIn("confirmed", safe.lower())

    def test_definitively_proves_corrected(self):
        safe, v = self._check("This definitively proves the connection.")
        self.assertTrue(v)
        self.assertNotIn("definitively proves", safe.lower())

    def test_certainly_corrected(self):
        safe, v = self._check("He certainly attended the island.", ConfidenceTier.SINGLE_SOURCE)
        self.assertTrue(v)
        self.assertNotIn("certainly", safe.lower())

    def test_without_doubt_corrected(self):
        safe, v = self._check("Without any doubt, this occurred.", ConfidenceTier.INFERRED)
        self.assertTrue(v)

    def test_it_is_a_fact_corrected(self):
        safe, v = self._check("It is a fact that Epstein knew Maxwell.", ConfidenceTier.SINGLE_SOURCE)
        self.assertTrue(v)

    def test_conclusively_corrected(self):
        safe, v = self._check("This conclusively shows the link.", ConfidenceTier.SPECULATIVE)
        self.assertTrue(v)

    def test_confirmed_tier_not_corrected(self):
        """CONFIRMED tier responses must be allowed to use confirmed language."""
        safe, v = self._check("This is confirmed.", ConfidenceTier.CONFIRMED)
        self.assertFalse(v)
        self.assertIn("confirmed", safe.lower())

    def test_hedged_language_not_violated(self):
        safe, v = self._check(
            "The documents suggest a possible connection.", ConfidenceTier.CORROBORATED
        )
        self.assertFalse(v)


class TestGuardrailConfidenceCheck(unittest.TestCase):
    """Full guardrail must correct confidence violations."""

    def test_confirmed_language_corrected_in_corroborated_response(self):
        result = _result(
            "This is confirmed by multiple documents.",
            ConfidenceTier.CORROBORATED,
        )
        with patch("rag.guardrail.write_audit_log"):
            gr = apply_guardrail(result, audit_bucket="bucket")

        self.assertTrue(gr.confidence_violation)
        self.assertNotIn("confirmed", gr.safe_answer.lower())

    def test_confidence_violation_logged_in_audit(self):
        from pipeline.audit_log import AuditLogEntry
        result = _result("This proves the connection.", ConfidenceTier.SPECULATIVE)

        with patch("rag.guardrail.write_audit_log") as mock_write:
            gr = apply_guardrail(result, audit_bucket="bucket")
            entry: AuditLogEntry = mock_write.call_args[0][0]

        self.assertTrue(entry.confidence_violation)

    def test_correction_preserves_rest_of_answer(self):
        answer = "The documents confirmed this meeting. Other details are unclear."
        result = _result(answer, ConfidenceTier.CORROBORATED)
        with patch("rag.guardrail.write_audit_log"):
            gr = apply_guardrail(result, audit_bucket="bucket")

        self.assertIn("Other details are unclear", gr.safe_answer)

    def test_multiple_violations_all_corrected(self):
        answer = "This is confirmed. It is certainly true. Without doubt, it occurred."
        result = _result(answer, ConfidenceTier.SINGLE_SOURCE)
        with patch("rag.guardrail.write_audit_log"):
            gr = apply_guardrail(result, audit_bucket="bucket")

        self.assertTrue(gr.confidence_violation)
        self.assertNotIn("confirmed", gr.safe_answer.lower())
        self.assertNotIn("certainly", gr.safe_answer.lower())
        self.assertNotIn("without doubt", gr.safe_answer.lower())


if __name__ == "__main__":
    unittest.main()

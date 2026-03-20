"""
tests/red_team/test_rl4_audit_circumvention.py
Red Team: Hard Limit 5 — Audit log must never be circumvented.

Adversarial tests verifying that query responses are withheld when the
audit log write fails, that the audit log captures both original and
safe answers, and that the audit trail cannot be bypassed.

Constitution Hard Limit 5. Principle V.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from pipeline.audit_log import AuditLogEntry, AuditLogFailure, write_audit_log
from rag.guardrail import apply_guardrail
from rag.query_router import QueryRequest, QueryType, RetrievalResult


def _result(answer: str = "test answer") -> RetrievalResult:
    return RetrievalResult(
        query=QueryRequest(query_text="test", query_type=QueryType.PROVENANCE),
        chunks=[],
        answer=answer,
        convergence_applied=True,
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


class TestAuditLogFailureBlocksDelivery(unittest.TestCase):
    """AuditLogFailure must prevent GuardrailResult from being returned."""

    def test_cloudwatch_failure_raises_audit_log_failure(self):
        cw = MagicMock()
        cw.put_log_events.side_effect = RuntimeError("CloudWatch down")
        cw.create_log_group.return_value = {}
        cw.create_log_stream.return_value = {}
        s3 = MagicMock()
        s3.put_object.return_value = {}

        with self.assertRaises(AuditLogFailure):
            write_audit_log(
                AuditLogEntry(),
                cloudwatch_client=cw,
                s3_client=s3,
                audit_bucket="bucket",
            )

    def test_s3_failure_raises_audit_log_failure(self):
        cw = MagicMock()
        cw.put_log_events.return_value = {}
        cw.create_log_group.return_value = {}
        cw.create_log_stream.return_value = {}
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 down")

        with self.assertRaises(AuditLogFailure):
            write_audit_log(
                AuditLogEntry(),
                cloudwatch_client=cw,
                s3_client=s3,
                audit_bucket="bucket",
            )

    def test_guardrail_propagates_audit_log_failure(self):
        """apply_guardrail must not return a result when audit fails."""
        delivered = []
        with patch(
            "rag.guardrail.write_audit_log",
            side_effect=AuditLogFailure("Audit failed"),
        ):
            try:
                result = apply_guardrail(_result(), audit_bucket="bucket")
                delivered.append(result)
            except AuditLogFailure:
                pass

        self.assertEqual(delivered, [],
                         "GuardrailResult must not be returned when audit fails")

    def test_api_returns_503_on_audit_failure(self):
        """API /query must return 503 when audit log fails."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
        from handler import handle_query, inject_clients

        import io

        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = lambda **kwargs: {
            "body": io.BytesIO(
                json.dumps({"embedding": [0.1]*1024}).encode()
                if "inputText" in json.loads(kwargs["body"])
                else json.dumps({"content": [{"text": "answer"}]}).encode()
            )
        }
        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": []}}
        inject_clients(bedrock=bedrock, opensearch=os_client)

        with patch(
            "rag.guardrail.write_audit_log",
            side_effect=AuditLogFailure("Audit unavailable"),
        ):
            result = handle_query({"body": json.dumps({
                "query_text": "test query",
                "query_type": "PROVENANCE",
            })})

        self.assertEqual(result["statusCode"], 503)
        body = json.loads(result["body"])
        self.assertIn("audit", body["message"].lower())


class TestAuditLogCompleteness(unittest.TestCase):
    """Audit log must capture complete, accurate information."""

    def test_audit_captures_both_original_and_safe_answer(self):
        """
        The original answer (before guardrail) must be preserved in the
        audit log even when the guardrail modifies it.
        """
        original = "virginia giuffre testified."
        result = _result(original)

        with patch("rag.guardrail.write_audit_log") as mock_write:
            apply_guardrail(result, audit_bucket="bucket")
            entry: AuditLogEntry = mock_write.call_args[0][0]

        self.assertIn("virginia giuffre", entry.original_answer.lower())
        self.assertNotIn("virginia giuffre", entry.safe_answer.lower())

    def test_audit_captures_query_text(self):
        result = RetrievalResult(
            query=QueryRequest(
                query_text="who was on the island?",
                query_type=QueryType.PROVENANCE,
            ),
            chunks=[],
            answer="answer",
            convergence_applied=True,
            retrieved_at="2026-01-01T00:00:00+00:00",
        )
        with patch("rag.guardrail.write_audit_log") as mock_write:
            apply_guardrail(result, audit_bucket="bucket")
            entry: AuditLogEntry = mock_write.call_args[0][0]

        self.assertEqual(entry.query_text, "who was on the island?")

    def test_audit_written_before_response_returned(self):
        """
        The audit write must complete before apply_guardrail returns.
        Verified by confirming write_audit_log is called during apply_guardrail.
        """
        call_order = []

        def mock_write(entry, **kwargs):
            call_order.append("audit_written")

        with patch("rag.guardrail.write_audit_log", side_effect=mock_write):
            gr = apply_guardrail(_result(), audit_bucket="bucket")
            call_order.append("result_returned")

        self.assertEqual(call_order[0], "audit_written")
        self.assertEqual(call_order[1], "result_returned")

    def test_both_write_targets_attempted_on_cloudwatch_failure(self):
        """When CloudWatch fails, S3 write must still be attempted."""
        cw = MagicMock()
        cw.put_log_events.side_effect = RuntimeError("CW down")
        cw.create_log_group.return_value = {}
        cw.create_log_stream.return_value = {}
        s3 = MagicMock()
        s3.put_object.return_value = {}

        try:
            write_audit_log(
                AuditLogEntry(),
                cloudwatch_client=cw,
                s3_client=s3,
                audit_bucket="bucket",
            )
        except AuditLogFailure:
            pass

        s3.put_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()

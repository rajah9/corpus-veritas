"""
tests/red_team/test_rl2_inference_bypass.py
Red Team: Hard Limit 2 — No single-source inference about living individuals.

Adversarial tests attempting to surface inferences about living individuals
from single-source evidence through query rephrasing, claim framing,
and bypass of the convergence check. All must fail.

Constitution Hard Limit 2. Principle III.
"""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from pipeline.models import ConfidenceTier
from rag.convergence_checker import (
    INFERENCE_THRESHOLD,
    ConvergenceResult,
    check_convergence,
)
from rag.query_router import QueryRequest, QueryType, RetrievalResult


def _single_source_result(answer: str = "The documents suggest X.") -> RetrievalResult:
    return RetrievalResult(
        query=QueryRequest(query_text="test", query_type=QueryType.INFERENCE),
        chunks=[{
            "document_uuid":   "uuid-001",
            "sequence_number": "1000",
            "text":            "some text",
        }],
        answer=answer,
        convergence_applied=False,
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


class TestConvergenceEnforcement(unittest.TestCase):
    """Convergence check must suppress single-source inferences."""

    def test_single_source_not_meets_threshold(self):
        result = _single_source_result()
        conv = check_convergence(result)
        self.assertFalse(conv.meets_inference_threshold)

    def test_single_source_suppression_message_returned(self):
        result = _single_source_result()
        conv = check_convergence(result)
        self.assertTrue(len(conv.suppression_message) > 0)
        self.assertIn(str(INFERENCE_THRESHOLD), conv.suppression_message)

    def test_guardrail_replaces_answer_with_suppression(self):
        from rag.guardrail import apply_guardrail
        result = _single_source_result("Prince Andrew attended the island.")
        conv = ConvergenceResult(
            independent_source_count=1,
            independent_document_uuids=["uuid-001"],
            document_types_present=["FBI_302"],
            convergence_tier=ConfidenceTier.SINGLE_SOURCE,
            meets_inference_threshold=False,
            suppression_message="Suppressed: only 1 source.",
        )

        with patch("rag.guardrail.write_audit_log"):
            gr = apply_guardrail(
                result, convergence_result=conv, audit_bucket="bucket"
            )

        self.assertTrue(gr.inference_downgraded)
        self.assertNotEqual(gr.safe_answer, result.answer)
        self.assertIn("Suppressed", gr.safe_answer)

    def test_guardrail_backstop_catches_missed_convergence_check(self):
        """
        Even if the caller forgot to run check_convergence(),
        the guardrail runs it internally as a backstop.
        """
        from rag.guardrail import apply_guardrail
        result = _single_source_result("The accused is definitely guilty.")
        # convergence_result=None forces guardrail to compute it
        with patch("rag.guardrail.write_audit_log"):
            gr = apply_guardrail(
                result, convergence_result=None, audit_bucket="bucket"
            )

        self.assertTrue(gr.inference_downgraded)

    def test_two_sources_meets_threshold(self):
        result = RetrievalResult(
            query=QueryRequest(query_text="test", query_type=QueryType.INFERENCE),
            chunks=[
                {"document_uuid": "uuid-001", "sequence_number": "1000", "text": "a"},
                {"document_uuid": "uuid-002", "sequence_number": "5000", "text": "b"},
            ],
            answer="Two sources support this.",
            convergence_applied=False,
            retrieved_at="2026-01-01T00:00:00+00:00",
        )
        conv = check_convergence(result)
        self.assertTrue(conv.meets_inference_threshold)


class TestInferenceQueryRouting(unittest.TestCase):
    """INFERENCE queries must always set convergence_applied=False."""

    def test_inference_query_sets_convergence_applied_false(self):
        from rag.query_router import route_query
        import io

        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = lambda **kwargs: {
            "body": io.BytesIO(
                __import__("json").dumps({"embedding": [0.1]*1024}).encode()
                if "inputText" in __import__("json").loads(kwargs["body"])
                else __import__("json").dumps({"content": [{"text": "answer"}]}).encode()
            )
        }
        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": []}}

        result = route_query(
            QueryRequest(query_text="test inference", query_type=QueryType.INFERENCE),
            opensearch_client=os_client,
            bedrock_client=bedrock,
        )
        self.assertFalse(result.convergence_applied)

    def test_non_inference_query_sets_convergence_applied_true(self):
        from rag.query_router import route_query
        import io

        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = lambda **kwargs: {
            "body": io.BytesIO(
                __import__("json").dumps({"embedding": [0.1]*1024}).encode()
                if "inputText" in __import__("json").loads(kwargs["body"])
                else __import__("json").dumps({"content": [{"text": "answer"}]}).encode()
            )
        }
        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": []}}

        result = route_query(
            QueryRequest(query_text="test provenance", query_type=QueryType.PROVENANCE),
            opensearch_client=os_client,
            bedrock_client=bedrock,
        )
        self.assertTrue(result.convergence_applied)


class TestAPIInferenceEndpoint(unittest.TestCase):
    """API /query endpoint must enforce convergence for INFERENCE queries."""

    def _query(self, body: dict) -> dict:
        import sys, os, json
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
        from handler import handle_query, inject_clients

        import io

        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = lambda **kwargs: {
            "body": io.BytesIO(
                json.dumps({"embedding": [0.1]*1024}).encode()
                if "inputText" in json.loads(kwargs["body"])
                else json.dumps({"content": [{"text": "inferred answer"}]}).encode()
            )
        }
        os_client = MagicMock()
        os_client.search.return_value = {"hits": {"hits": []}}
        cw = MagicMock()
        cw.create_log_group.return_value = {}
        cw.create_log_stream.return_value = {}
        cw.put_log_events.return_value = {}
        s3 = MagicMock()
        s3.put_object.return_value = {}
        inject_clients(bedrock=bedrock, opensearch=os_client,
                       cloudwatch=cw, s3=s3)

        import handler as _handler
        _handler.AUDIT_S3_BUCKET = "test-audit-bucket"

        return handle_query({"body": json.dumps(body)})

    def test_inference_with_no_sources_returns_suppression_message(self):
        result = self._query({
            "query_text": "Who else was involved?",
            "query_type": "INFERENCE",
        })
        # 200 is correct -- audit succeeds, guardrail suppresses the answer
        # A 503 would only occur if the audit log write itself failed.
        self.assertEqual(result["statusCode"], 200)
        body = json.loads(result["body"])
        # With no chunks, convergence threshold not met -- inference downgraded
        self.assertTrue(body.get("inference_downgraded", False),
                        "INFERENCE with no sources must be downgraded")


if __name__ == "__main__":
    unittest.main()

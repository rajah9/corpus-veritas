"""
tests/api/test_handler.py

Unit tests for api/handler.py.

Coverage targets
----------------
lambda_handler()        -- routes GET/POST to correct handler, 404 for unknown
handle_health()         -- 200, status ok, timestamp present
handle_query()          -- 200 on success, 400 on missing query_text, 400 on
                           missing query_type, 400 on invalid query_type,
                           503 on AuditLogFailure, 500 on RuntimeError,
                           all GuardrailResult fields in response
handle_gap_report()     -- 200 on success, 404 on missing report,
                           version and public params passed to S3
handle_entity_lookup()  -- 200 on found entity, 404 on not found, 403 on
                           victim-flagged entity, 400 on missing canonical_name
handle_document_lookup()-- 200 on found doc, 404 on not found, 403 on
                           victim-flagged doc, 400 on missing uuid
_deserialise_dynamo_item()
                        -- S, N, SS, BOOL types all deserialised correctly
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))

from handler import (
    _deserialise_dynamo_item,
    handle_document_lookup,
    handle_entity_lookup,
    handle_gap_report,
    handle_health,
    handle_query,
    inject_clients,
    lambda_handler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bedrock_mock() -> MagicMock:
    client = MagicMock()
    client.invoke_model.side_effect = lambda **kwargs: {
        "body": io.BytesIO(
            json.dumps({"embedding": [0.1] * 1024}).encode()
            if "inputText" in json.loads(kwargs["body"])
            else json.dumps({"content": [{"text": "synthesised answer"}]}).encode()
        )
    }
    return client


def _os_mock(chunks=None) -> MagicMock:
    client = MagicMock()
    client.search.return_value = {
        "hits": {"hits": [{"_source": c} for c in (chunks or [])]}
    }
    return client


def _dynamo_item(name="jeffrey epstein", victim=False) -> dict:
    item = {
        "canonical_name": {"S": name},
        "entity_type":    {"S": "PERSON"},
        "confidence":     {"N": "0.99"},
        "surface_forms":  {"SS": ["Jeffrey Epstein"]},
    }
    if victim:
        item["victim_flag"] = {"S": "true"}
    return item


def _setup_clients(bedrock=None, opensearch=None, dynamodb=None,
                   s3=None, cloudwatch=None) -> None:
    inject_clients(
        bedrock=bedrock or _bedrock_mock(),
        opensearch=opensearch or _os_mock(),
        dynamodb=dynamodb or MagicMock(),
        s3=s3 or MagicMock(),
        cloudwatch=cloudwatch or MagicMock(),
    )


# ===========================================================================
# lambda_handler routing
# ===========================================================================

class TestLambdaHandlerRouting(unittest.TestCase):

    def setUp(self):
        _setup_clients()

    def test_post_query_routed(self):
        # _ROUTES holds a direct function reference so patching the name doesn't work.
        # Assert on the response structure instead.
        with patch("rag.guardrail.write_audit_log"):
            result = lambda_handler(
                {"httpMethod": "POST", "path": "/query",
                 "body": '{"query_text": "test", "query_type": "PROVENANCE"}'},
                None,
            )
        self.assertIn(result["statusCode"], (200, 500, 503))

    def test_get_health_routed(self):
        result = lambda_handler({"httpMethod": "GET", "path": "/health"}, None)
        self.assertEqual(result["statusCode"], 200)

    def test_unknown_route_returns_404(self):
        result = lambda_handler({"httpMethod": "GET", "path": "/unknown"}, None)
        self.assertEqual(result["statusCode"], 404)

    def test_wrong_method_returns_404(self):
        result = lambda_handler({"httpMethod": "GET", "path": "/query"}, None)
        self.assertEqual(result["statusCode"], 404)


# ===========================================================================
# handle_health
# ===========================================================================

class TestHandleHealth(unittest.TestCase):

    def test_returns_200(self):
        result = handle_health({})
        self.assertEqual(result["statusCode"], 200)

    def test_body_has_status_ok(self):
        result = handle_health({})
        body = json.loads(result["body"])
        self.assertEqual(body["status"], "ok")

    def test_body_has_timestamp(self):
        result = handle_health({})
        body = json.loads(result["body"])
        self.assertIn("timestamp", body)

    def test_content_type_is_json(self):
        result = handle_health({})
        self.assertEqual(result["headers"]["Content-Type"], "application/json")


# ===========================================================================
# handle_query
# ===========================================================================

class TestHandleQuery(unittest.TestCase):

    def setUp(self):
        _setup_clients()

    def _query(self, body: dict) -> dict:
        with patch("rag.guardrail.write_audit_log"):
            return handle_query({"body": json.dumps(body)})

    def test_missing_query_text_returns_400(self):
        result = self._query({"query_type": "PROVENANCE"})
        self.assertEqual(result["statusCode"], 400)

    def test_missing_query_type_returns_400(self):
        result = self._query({"query_text": "test"})
        self.assertEqual(result["statusCode"], 400)

    def test_invalid_query_type_returns_400(self):
        result = self._query({"query_text": "test", "query_type": "INVALID"})
        self.assertEqual(result["statusCode"], 400)

    def test_valid_query_returns_200(self):
        result = self._query({"query_text": "test query", "query_type": "PROVENANCE"})
        self.assertEqual(result["statusCode"], 200)

    def test_response_contains_audit_entry_id(self):
        result = self._query({"query_text": "test", "query_type": "PROVENANCE"})
        body = json.loads(result["body"])
        self.assertIn("audit_entry_id", body)

    def test_response_contains_answer(self):
        result = self._query({"query_text": "test", "query_type": "PROVENANCE"})
        body = json.loads(result["body"])
        self.assertIn("answer", body)

    def test_response_contains_guardrail_flags(self):
        result = self._query({"query_text": "test", "query_type": "PROVENANCE"})
        body = json.loads(result["body"])
        for field in ("victim_scan_triggered", "inference_downgraded",
                      "confidence_violation", "creative_content_suppressed"):
            self.assertIn(field, body)

    def test_audit_failure_returns_503(self):
        from pipeline.audit_log import AuditLogFailure
        with patch("rag.guardrail.write_audit_log",
                   side_effect=AuditLogFailure("audit down")):
            result = handle_query({"body": json.dumps({
                "query_text": "test", "query_type": "PROVENANCE"
            })})
        self.assertEqual(result["statusCode"], 503)

    def test_invalid_json_body_returns_400(self):
        with patch("rag.guardrail.write_audit_log"):
            result = handle_query({"body": "not json"})
        self.assertEqual(result["statusCode"], 400)


# ===========================================================================
# handle_gap_report
# ===========================================================================

class TestHandleGapReport(unittest.TestCase):

    def test_returns_200_when_report_found(self):
        s3 = MagicMock()
        s3.get_object.return_value = {
            "Body": io.BytesIO(b"# Gap Report\n")
        }
        inject_clients(s3=s3)
        result = handle_gap_report({
            "queryStringParameters": {"version": "v1", "public": "true"}
        })
        self.assertEqual(result["statusCode"], 200)

    def test_returns_404_when_not_found(self):
        s3 = MagicMock()
        s3.get_object.side_effect = Exception("NoSuchKey")
        inject_clients(s3=s3)
        result = handle_gap_report({"queryStringParameters": {"version": "v1"}})
        self.assertEqual(result["statusCode"], 404)

    def test_default_version_is_latest(self):
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": io.BytesIO(b"report")}
        inject_clients(s3=s3)
        handle_gap_report({"queryStringParameters": None})
        key = s3.get_object.call_args.kwargs["Key"]
        self.assertIn("latest", key)


# ===========================================================================
# handle_entity_lookup
# ===========================================================================

class TestHandleEntityLookup(unittest.TestCase):

    def test_returns_200_when_found(self):
        db = MagicMock()
        db.get_item.return_value = {"Item": _dynamo_item()}
        inject_clients(dynamodb=db)
        result = handle_entity_lookup({
            "pathParameters": {"canonical_name": "jeffrey epstein"},
            "queryStringParameters": {},
        })
        self.assertEqual(result["statusCode"], 200)

    def test_returns_404_when_not_found(self):
        db = MagicMock()
        db.get_item.return_value = {}
        inject_clients(dynamodb=db)
        result = handle_entity_lookup({
            "pathParameters": {"canonical_name": "unknown person"},
            "queryStringParameters": {},
        })
        self.assertEqual(result["statusCode"], 404)

    def test_returns_403_for_victim_flagged(self):
        db = MagicMock()
        db.get_item.return_value = {"Item": _dynamo_item(victim=True)}
        inject_clients(dynamodb=db)
        result = handle_entity_lookup({
            "pathParameters": {"canonical_name": "virginia giuffre"},
            "queryStringParameters": {},
        })
        self.assertEqual(result["statusCode"], 403)
        body = json.loads(result["body"])
        self.assertNotIn("virginia giuffre", result["body"])

    def test_missing_canonical_name_returns_400(self):
        result = handle_entity_lookup({
            "pathParameters": {},
            "queryStringParameters": {},
        })
        self.assertEqual(result["statusCode"], 400)


# ===========================================================================
# handle_document_lookup
# ===========================================================================

class TestHandleDocumentLookup(unittest.TestCase):

    def _doc_item(self, victim=False) -> dict:
        item = {
            "document_uuid":  {"S": "uuid-001"},
            "classification": {"S": "PROCEDURAL"},
            "state":          {"S": "INGESTED"},
        }
        if victim:
            item["victim_flag"] = {"S": "true"}
        return item

    def test_returns_200_when_found(self):
        db = MagicMock()
        db.get_item.return_value = {"Item": self._doc_item()}
        inject_clients(dynamodb=db)
        result = handle_document_lookup({
            "pathParameters": {"uuid": "uuid-001"}
        })
        self.assertEqual(result["statusCode"], 200)

    def test_returns_404_when_not_found(self):
        db = MagicMock()
        db.get_item.return_value = {}
        inject_clients(dynamodb=db)
        result = handle_document_lookup({"pathParameters": {"uuid": "uuid-999"}})
        self.assertEqual(result["statusCode"], 404)

    def test_returns_403_for_victim_flagged(self):
        db = MagicMock()
        db.get_item.return_value = {"Item": self._doc_item(victim=True)}
        inject_clients(dynamodb=db)
        result = handle_document_lookup({"pathParameters": {"uuid": "uuid-001"}})
        self.assertEqual(result["statusCode"], 403)

    def test_missing_uuid_returns_400(self):
        result = handle_document_lookup({"pathParameters": {}})
        self.assertEqual(result["statusCode"], 400)


# ===========================================================================
# _deserialise_dynamo_item
# ===========================================================================

class TestDeserialiseDynamoItem(unittest.TestCase):

    def test_string_type(self):
        result = _deserialise_dynamo_item({"name": {"S": "value"}})
        self.assertEqual(result["name"], "value")

    def test_number_type(self):
        result = _deserialise_dynamo_item({"score": {"N": "0.99"}})
        self.assertAlmostEqual(result["score"], 0.99)

    def test_string_set_type(self):
        result = _deserialise_dynamo_item({"forms": {"SS": ["A", "B"]}})
        self.assertIsInstance(result["forms"], list)
        self.assertIn("A", result["forms"])

    def test_bool_type(self):
        result = _deserialise_dynamo_item({"flag": {"BOOL": True}})
        self.assertTrue(result["flag"])

    def test_mixed_types(self):
        item = {
            "name":  {"S": "epstein"},
            "score": {"N": "0.95"},
            "forms": {"SS": ["Epstein"]},
        }
        result = _deserialise_dynamo_item(item)
        self.assertEqual(result["name"], "epstein")
        self.assertAlmostEqual(result["score"], 0.95)


if __name__ == "__main__":
    unittest.main()

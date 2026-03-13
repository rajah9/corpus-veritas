"""
tests/rag/test_query_router.py

Unit tests for rag/query_router.py.

Coverage targets
----------------
QueryRequest        -- construction, validation (empty text, top_k <= 0)
build_query()       -- dispatches to correct builder per QueryType,
                       victim_flag suppressed in every DSL, date range
                       in TIMELINE, entity filter in TIMELINE and
                       RELATIONSHIP, no entity filter falls back gracefully
retrieve_chunks()   -- search called with correct index/body, _source
                       fields returned, empty hits returns [], exception
                       raises RuntimeError
_lowest_confidence_tier()
                    -- returns weakest tier, None when absent
synthesise_answer() -- invoke_model called, correct model ID, prompt
                       contains query text and chunk text, empty chunks
                       returns empty string, exception raises RuntimeError
route_query()       -- full pipeline (embed→build→retrieve→synthesise),
                       RetrievalResult fields set correctly, INFERENCE
                       sets convergence_applied=False, non-INFERENCE sets
                       True, no chunks → empty answer, embedding failure
                       propagates, retrieval failure propagates
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock

from config import EmbeddingConfig
from pipeline.models import ConfidenceTier
from rag.query_router import (
    QueryRequest,
    QueryType,
    RetrievalResult,
    _lowest_confidence_tier,
    build_query,
    retrieve_chunks,
    route_query,
    synthesise_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_bedrock_embed(vector: list = None) -> MagicMock:
    """Bedrock mock that returns a fixed embedding vector."""
    if vector is None:
        vector = [0.1] * 1024
    body_bytes = json.dumps({"embedding": vector}).encode()
    client = MagicMock()
    client.invoke_model.side_effect = lambda **kwargs: {
        "body": io.BytesIO(body_bytes)
    }
    return client


def _mock_bedrock_synthesise(answer: str = "Synthesised answer.") -> MagicMock:
    """Bedrock mock that returns a fixed synthesis answer."""
    body_bytes = json.dumps({
        "content": [{"text": answer}]
    }).encode()
    client = MagicMock()
    client.invoke_model.side_effect = lambda **kwargs: {
        "body": io.BytesIO(body_bytes)
    }
    return client


def _mock_bedrock_full(
    vector: list = None,
    answer: str = "Synthesised answer.",
) -> MagicMock:
    """
    Bedrock mock for route_query() -- handles both embed and synthesis calls.
    Distinguishes by presence of 'inputText' vs 'messages' in the body.
    """
    if vector is None:
        vector = [0.1] * 1024
    embed_bytes = json.dumps({"embedding": vector}).encode()
    synth_bytes = json.dumps({"content": [{"text": answer}]}).encode()

    def side_effect(**kwargs):
        body = json.loads(kwargs["body"])
        if "inputText" in body:
            return {"body": io.BytesIO(embed_bytes)}
        return {"body": io.BytesIO(synth_bytes)}

    client = MagicMock()
    client.invoke_model.side_effect = side_effect
    return client


def _mock_opensearch(chunks: list = None) -> MagicMock:
    if chunks is None:
        chunks = [{"document_uuid": "uuid-001", "text": "some text"}]
    client = MagicMock()
    client.search.return_value = {
        "hits": {"hits": [{"_source": c} for c in chunks]}
    }
    return client


def _request(
    query_type: QueryType = QueryType.PROVENANCE,
    text: str = "What happened?",
    **kwargs,
) -> QueryRequest:
    return QueryRequest(query_text=text, query_type=query_type, **kwargs)


# ===========================================================================
# QueryRequest
# ===========================================================================

class TestQueryRequest(unittest.TestCase):

    def test_minimal_construction(self):
        req = QueryRequest(query_text="test", query_type=QueryType.PROVENANCE)
        self.assertIsInstance(req, QueryRequest)

    def test_default_top_k(self):
        req = _request()
        self.assertGreater(req.top_k, 0)

    def test_empty_query_text_raises(self):
        with self.assertRaises(ValueError):
            QueryRequest(query_text="", query_type=QueryType.PROVENANCE)

    def test_whitespace_query_text_raises(self):
        with self.assertRaises(ValueError):
            QueryRequest(query_text="   ", query_type=QueryType.PROVENANCE)

    def test_zero_top_k_raises(self):
        with self.assertRaises(ValueError):
            QueryRequest(query_text="test", query_type=QueryType.PROVENANCE, top_k=0)

    def test_negative_top_k_raises(self):
        with self.assertRaises(ValueError):
            QueryRequest(query_text="test", query_type=QueryType.PROVENANCE, top_k=-1)

    def test_optional_fields_default_none(self):
        req = _request()
        self.assertIsNone(req.entity_names)
        self.assertIsNone(req.date_from)
        self.assertIsNone(req.date_to)


# ===========================================================================
# build_query -- victim flag suppression
# ===========================================================================

class TestBuildQueryVictimFlagSuppression(unittest.TestCase):
    """victim_flag must_not filter present in every query type."""

    def _must_not(self, query_type: QueryType) -> list:
        dsl = build_query(_request(query_type), [0.1] * 1024)
        return dsl["query"]["bool"].get("must_not", [])

    def test_timeline_suppresses_victim_flag(self):
        must_not = self._must_not(QueryType.TIMELINE)
        terms = [f.get("term", {}) for f in must_not]
        self.assertTrue(any("victim_flag" in t for t in terms))

    def test_provenance_suppresses_victim_flag(self):
        must_not = self._must_not(QueryType.PROVENANCE)
        terms = [f.get("term", {}) for f in must_not]
        self.assertTrue(any("victim_flag" in t for t in terms))

    def test_inference_suppresses_victim_flag(self):
        must_not = self._must_not(QueryType.INFERENCE)
        terms = [f.get("term", {}) for f in must_not]
        self.assertTrue(any("victim_flag" in t for t in terms))

    def test_relationship_suppresses_victim_flag(self):
        must_not = self._must_not(QueryType.RELATIONSHIP)
        terms = [f.get("term", {}) for f in must_not]
        self.assertTrue(any("victim_flag" in t for t in terms))


# ===========================================================================
# build_query -- TIMELINE
# ===========================================================================

class TestBuildQueryTimeline(unittest.TestCase):

    def test_has_sort_by_document_date(self):
        dsl = build_query(
            _request(QueryType.TIMELINE, date_from="2000-01-01"),
            [0.1] * 1024,
        )
        self.assertIn("sort", dsl)
        self.assertEqual(dsl["sort"][0]["document_date"]["order"], "asc")

    def test_date_from_included_in_filter(self):
        dsl = build_query(
            _request(QueryType.TIMELINE, date_from="2000-01-01"),
            [0.1] * 1024,
        )
        filters = dsl["query"]["bool"].get("filter", [])
        date_filters = [f for f in filters if "range" in f]
        self.assertTrue(any(
            "gte" in f["range"].get("document_date", {})
            for f in date_filters
        ))

    def test_date_to_included_in_filter(self):
        dsl = build_query(
            _request(QueryType.TIMELINE, date_to="2010-01-01"),
            [0.1] * 1024,
        )
        filters = dsl["query"]["bool"].get("filter", [])
        date_filters = [f for f in filters if "range" in f]
        self.assertTrue(any(
            "lte" in f["range"].get("document_date", {})
            for f in date_filters
        ))

    def test_no_date_range_produces_no_date_filter(self):
        dsl = build_query(_request(QueryType.TIMELINE), [0.1] * 1024)
        filters = dsl["query"]["bool"].get("filter", [])
        date_filters = [f for f in filters if "range" in f]
        self.assertEqual(len(date_filters), 0)

    def test_entity_names_filter_applied(self):
        req = _request(QueryType.TIMELINE, entity_names=["John Doe"])
        dsl = build_query(req, [0.1] * 1024)
        filters = dsl["query"]["bool"].get("filter", [])
        entity_filters = [f for f in filters if "terms" in f]
        self.assertTrue(len(entity_filters) > 0)


# ===========================================================================
# build_query -- RELATIONSHIP
# ===========================================================================

class TestBuildQueryRelationship(unittest.TestCase):

    def test_entity_names_filter_applied(self):
        req = _request(QueryType.RELATIONSHIP, entity_names=["Jane Doe"])
        dsl = build_query(req, [0.1] * 1024)
        filters = dsl["query"]["bool"].get("filter", [])
        self.assertTrue(len(filters) > 0)

    def test_no_entity_names_produces_no_filter(self):
        req = _request(QueryType.RELATIONSHIP, entity_names=None)
        dsl = build_query(req, [0.1] * 1024)
        filters = dsl["query"]["bool"].get("filter", [])
        self.assertEqual(len(filters), 0)


# ===========================================================================
# retrieve_chunks
# ===========================================================================

class TestRetrieveChunks(unittest.TestCase):

    def test_search_called_with_correct_index(self):
        client = _mock_opensearch()
        retrieve_chunks({"query": {}}, client, index_name="documents")
        self.assertEqual(client.search.call_args.kwargs["index"], "documents")

    def test_search_called_with_body(self):
        client = _mock_opensearch()
        dsl = {"query": {"match_all": {}}}
        retrieve_chunks(dsl, client)
        self.assertEqual(client.search.call_args.kwargs["body"], dsl)

    def test_returns_source_fields(self):
        chunks = [{"document_uuid": "a"}, {"document_uuid": "b"}]
        client = _mock_opensearch(chunks)
        result = retrieve_chunks({"query": {}}, client)
        self.assertEqual(result, chunks)

    def test_empty_hits_returns_empty_list(self):
        client = MagicMock()
        client.search.return_value = {"hits": {"hits": []}}
        result = retrieve_chunks({"query": {}}, client)
        self.assertEqual(result, [])

    def test_search_exception_raises_runtime_error(self):
        client = MagicMock()
        client.search.side_effect = RuntimeError("OS unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            retrieve_chunks({"query": {}}, client)
        self.assertIn("retrieval failed", str(ctx.exception))


# ===========================================================================
# _lowest_confidence_tier
# ===========================================================================

class TestLowestConfidenceTier(unittest.TestCase):

    def test_returns_none_when_no_tiers(self):
        self.assertIsNone(_lowest_confidence_tier([{"text": "x"}]))

    def test_returns_none_for_empty_list(self):
        self.assertIsNone(_lowest_confidence_tier([]))

    def test_single_tier_returned(self):
        chunks = [{"confidence_tier": ConfidenceTier.CONFIRMED}]
        self.assertEqual(_lowest_confidence_tier(chunks), ConfidenceTier.CONFIRMED)

    def test_returns_weakest_tier(self):
        chunks = [
            {"confidence_tier": ConfidenceTier.CONFIRMED},
            {"confidence_tier": ConfidenceTier.SPECULATIVE},
            {"confidence_tier": ConfidenceTier.CORROBORATED},
        ]
        self.assertEqual(_lowest_confidence_tier(chunks), ConfidenceTier.SPECULATIVE)

    def test_mixed_present_absent_tiers(self):
        chunks = [
            {"confidence_tier": ConfidenceTier.CONFIRMED},
            {"text": "no tier here"},
        ]
        self.assertEqual(_lowest_confidence_tier(chunks), ConfidenceTier.CONFIRMED)


# ===========================================================================
# synthesise_answer
# ===========================================================================

class TestSynthesiseAnswer(unittest.TestCase):

    def test_returns_empty_string_for_no_chunks(self):
        result = synthesise_answer(_request(), [], MagicMock())
        self.assertEqual(result, "")

    def test_invoke_model_called(self):
        bedrock = _mock_bedrock_synthesise()
        synthesise_answer(_request(), [{"text": "chunk"}], bedrock)
        bedrock.invoke_model.assert_called_once()

    def test_correct_model_id_used(self):
        bedrock = _mock_bedrock_synthesise()
        synthesise_answer(_request(), [{"text": "chunk"}], bedrock)
        self.assertIn(
            "claude",
            bedrock.invoke_model.call_args.kwargs["modelId"].lower(),
        )

    def test_returns_answer_string(self):
        bedrock = _mock_bedrock_synthesise("Answer text here.")
        result = synthesise_answer(_request(), [{"text": "chunk"}], bedrock)
        self.assertEqual(result, "Answer text here.")

    def test_prompt_contains_query_text(self):
        bedrock = _mock_bedrock_synthesise()
        req = _request(text="Tell me about the flight logs.")
        synthesise_answer(req, [{"text": "chunk"}], bedrock)
        body = json.loads(bedrock.invoke_model.call_args.kwargs["body"])
        prompt = body["messages"][0]["content"]
        self.assertIn("flight logs", prompt)

    def test_prompt_contains_chunk_text(self):
        bedrock = _mock_bedrock_synthesise()
        synthesise_answer(_request(), [{"text": "specific chunk content"}], bedrock)
        body = json.loads(bedrock.invoke_model.call_args.kwargs["body"])
        prompt = body["messages"][0]["content"]
        self.assertIn("specific chunk content", prompt)

    def test_bedrock_exception_raises_runtime_error(self):
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = RuntimeError("Bedrock down")
        with self.assertRaises(RuntimeError) as ctx:
            synthesise_answer(_request(), [{"text": "chunk"}], bedrock)
        self.assertIn("synthesis failed", str(ctx.exception))


# ===========================================================================
# route_query
# ===========================================================================

class TestRouteQuery(unittest.TestCase):

    def setUp(self):
        self.chunks = [
            {"document_uuid": "uuid-001", "text": "chunk text",
             "confidence_tier": ConfidenceTier.CORROBORATED}
        ]
        self.bedrock = _mock_bedrock_full()
        self.os_client = _mock_opensearch(self.chunks)

    def _run(self, query_type=QueryType.PROVENANCE, **kwargs):
        return route_query(
            request=_request(query_type, **kwargs),
            opensearch_client=self.os_client,
            bedrock_client=self.bedrock,
        )

    def test_returns_retrieval_result(self):
        self.assertIsInstance(self._run(), RetrievalResult)

    def test_chunks_populated(self):
        result = self._run()
        self.assertEqual(result.chunks, self.chunks)

    def test_answer_populated(self):
        result = self._run()
        self.assertIsInstance(result.answer, str)
        self.assertTrue(len(result.answer) > 0)

    def test_retrieved_at_set(self):
        result = self._run()
        self.assertTrue(result.retrieved_at)

    def test_query_stored_on_result(self):
        result = self._run()
        self.assertEqual(result.query.query_type, QueryType.PROVENANCE)

    def test_inference_convergence_applied_false(self):
        result = self._run(QueryType.INFERENCE)
        self.assertFalse(result.convergence_applied)

    def test_provenance_convergence_applied_true(self):
        result = self._run(QueryType.PROVENANCE)
        self.assertTrue(result.convergence_applied)

    def test_timeline_convergence_applied_true(self):
        result = self._run(QueryType.TIMELINE)
        self.assertTrue(result.convergence_applied)

    def test_relationship_convergence_applied_true(self):
        result = self._run(QueryType.RELATIONSHIP)
        self.assertTrue(result.convergence_applied)

    def test_lowest_tier_set_from_chunks(self):
        result = self._run()
        self.assertEqual(result.lowest_tier, ConfidenceTier.CORROBORATED)

    def test_no_chunks_produces_empty_answer(self):
        self.os_client.search.return_value = {"hits": {"hits": []}}
        result = self._run()
        self.assertEqual(result.answer, "")
        self.assertIsNone(result.lowest_tier)

    def test_embedding_failure_raises_runtime_error(self):
        self.bedrock.invoke_model.side_effect = RuntimeError("Bedrock down")
        with self.assertRaises(RuntimeError):
            self._run()

    def test_retrieval_failure_raises_runtime_error(self):
        self.os_client.search.side_effect = RuntimeError("OS down")
        with self.assertRaises(RuntimeError):
            self._run()

    def test_opensearch_search_called(self):
        self._run()
        self.os_client.search.assert_called_once()


if __name__ == "__main__":
    unittest.main()

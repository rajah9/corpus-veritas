"""
tests/pipeline/test_ingestor.py

Unit tests for pipeline/ingestor.py (Layer 2: chunking, embedding, storage).

Coverage targets
----------------
chunk_text()          -- short text (one chunk), multi-chunk, overlap,
                         empty text, exact chunk boundary
embed_text()          -- successful call, correct model_id used, request
                         body format, response parsed, Bedrock exception
index_chunk()         -- client.index called, correct index/id/body,
                         OpenSearch exception propagation
_assert_document_cleared
                      -- VICTIM_FLAGGED raises, PENDING_REVIEW raises,
                         SANITIZED passes
ingest_document()     -- full pipeline (chunk -> embed -> index),
                         chunk_ids returned, correct count, idempotent _id,
                         VICTIM_FLAGGED guard, missing opensearch_client raises,
                         Bedrock failure propagates, OpenSearch failure propagates
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, call, patch

from config import ChunkingConfig, EmbeddingConfig
from pipeline.chunk_schema import ChunkMetadata
from pipeline.classifier import ClassificationRecord, DocumentClassification
from pipeline.ingestor import (
    _assert_document_cleared,
    chunk_text,
    embed_text,
    index_chunk,
    ingest_document,
)
from pipeline.models import DocumentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    state: DocumentState = DocumentState.SANITIZED,
    victim_flag: bool = False,
    uuid: str = "uuid-001",
) -> ClassificationRecord:
    return ClassificationRecord(
        document_uuid=uuid,
        classification=DocumentClassification.PROCEDURAL,
        state=state,
        ingestion_date="2026-03-11T00:00:00+00:00",
        victim_flag=victim_flag,
        corpus_source="DOJ_DIRECT",
        provenance_tag=None,
    )


def _mock_bedrock(vector: list = None) -> MagicMock:
    """Return a Bedrock client mock that yields a fixed embedding vector.

    Uses side_effect so a fresh BytesIO is created on every invoke_model()
    call. A single BytesIO instance is exhausted after one .read() -- using
    return_value would silently return b"" on every chunk after the first,
    causing JSON decode errors in multi-chunk documents.
    """
    if vector is None:
        vector = [0.1] * 1024
    body_bytes = json.dumps({"embedding": vector}).encode()
    client = MagicMock()
    client.invoke_model.side_effect = lambda **kwargs: {"body": io.BytesIO(body_bytes)}
    return client


def _mock_opensearch() -> MagicMock:
    client = MagicMock()
    client.index.return_value = {"result": "created"}
    return client


def _minimal_chunk(uuid: str = "uuid-001", idx: int = 0) -> ChunkMetadata:
    return ChunkMetadata(
        document_uuid=uuid,
        chunk_index=idx,
        text="some chunk text",
        vector=[0.1] * 1024,
        classification="PROCEDURAL",
        ingestion_date="2026-03-11T00:00:00+00:00",
    )


# ===========================================================================
# chunk_text
# ===========================================================================

class TestChunkText(unittest.TestCase):

    def test_short_text_returns_one_chunk(self):
        cfg = ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=50)
        chunks = chunk_text("hello world", cfg)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "hello world")

    def test_empty_text_returns_list_with_original(self):
        cfg = ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=50)
        chunks = chunk_text("", cfg)
        self.assertEqual(chunks, [""])

    def test_long_text_produces_multiple_chunks(self):
        cfg = ChunkingConfig(chunk_size_tokens=5, chunk_overlap_tokens=1)
        text = " ".join([f"word{i}" for i in range(20)])
        chunks = chunk_text(text, cfg)
        self.assertGreater(len(chunks), 1)

    def test_all_words_covered(self):
        """Every word in the source must appear in at least one chunk."""
        cfg = ChunkingConfig(chunk_size_tokens=5, chunk_overlap_tokens=2)
        words = [f"word{i}" for i in range(30)]
        text = " ".join(words)
        chunks = chunk_text(text, cfg)
        all_chunk_words = set()
        for chunk in chunks:
            all_chunk_words.update(chunk.split())
        for word in words:
            self.assertIn(word, all_chunk_words)

    def test_overlap_words_appear_in_consecutive_chunks(self):
        """Words in the overlap window must appear in both adjacent chunks."""
        cfg = ChunkingConfig(chunk_size_tokens=5, chunk_overlap_tokens=2)
        text = " ".join([f"w{i}" for i in range(20)])
        chunks = chunk_text(text, cfg)
        self.assertGreater(len(chunks), 1)
        # Last 2 words of chunk[0] should appear at start of chunk[1]
        tail = chunks[0].split()[-cfg.chunk_overlap_tokens:]
        head = chunks[1].split()[:cfg.chunk_overlap_tokens]
        self.assertEqual(tail, head)

    def test_no_overlap_produces_non_overlapping_chunks(self):
        cfg = ChunkingConfig(chunk_size_tokens=4, chunk_overlap_tokens=0)
        text = "a b c d e f g h"
        chunks = chunk_text(text, cfg)
        self.assertEqual(chunks[0], "a b c d")
        self.assertEqual(chunks[1], "e f g h")

    def test_chunk_size_exactly_equals_text_length(self):
        cfg = ChunkingConfig(chunk_size_tokens=5, chunk_overlap_tokens=1)
        text = "a b c d e"
        chunks = chunk_text(text, cfg)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)


# ===========================================================================
# embed_text
# ===========================================================================

class TestEmbedText(unittest.TestCase):

    def test_returns_list_of_floats(self):
        bedrock = _mock_bedrock()
        cfg = EmbeddingConfig()
        result = embed_text("hello", bedrock, cfg)
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(v, float) for v in result))

    def test_vector_length_matches_config(self):
        vector = [0.5] * 1024
        bedrock = _mock_bedrock(vector)
        result = embed_text("text", bedrock, EmbeddingConfig())
        self.assertEqual(len(result), 1024)

    def test_correct_model_id_used(self):
        bedrock = _mock_bedrock()
        cfg = EmbeddingConfig(model_id="amazon.titan-embed-text-v2:0")
        embed_text("text", bedrock, cfg)
        call_kwargs = bedrock.invoke_model.call_args.kwargs
        self.assertEqual(call_kwargs["modelId"], "amazon.titan-embed-text-v2:0")

    def test_request_body_contains_input_text(self):
        bedrock = _mock_bedrock()
        embed_text("my chunk text", bedrock, EmbeddingConfig())
        body_str = bedrock.invoke_model.call_args.kwargs["body"]
        body = json.loads(body_str)
        self.assertEqual(body["inputText"], "my chunk text")

    def test_bedrock_exception_raises_runtime_error(self):
        bedrock = MagicMock()
        bedrock.invoke_model.side_effect = RuntimeError("Bedrock unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            embed_text("text", bedrock, EmbeddingConfig())
        self.assertIn("Bedrock embedding failed", str(ctx.exception))

    def test_malformed_response_raises_runtime_error(self):
        bedrock = MagicMock()
        bad_bytes = json.dumps({"wrong_key": []}).encode()
        bedrock.invoke_model.side_effect = lambda **kwargs: {"body": io.BytesIO(bad_bytes)}
        with self.assertRaises(RuntimeError):
            embed_text("text", bedrock, EmbeddingConfig())


# ===========================================================================
# index_chunk
# ===========================================================================

class TestIndexChunk(unittest.TestCase):

    def test_client_index_called(self):
        os_client = _mock_opensearch()
        chunk = _minimal_chunk()
        index_chunk(chunk, os_client)
        os_client.index.assert_called_once()

    def test_correct_index_name_used(self):
        os_client = _mock_opensearch()
        index_chunk(_minimal_chunk(), os_client, index_name="my-index")
        self.assertEqual(
            os_client.index.call_args.kwargs["index"], "my-index"
        )

    def test_chunk_id_used_as_document_id(self):
        os_client = _mock_opensearch()
        chunk = _minimal_chunk(uuid="uuid-001", idx=3)
        index_chunk(chunk, os_client)
        self.assertEqual(
            os_client.index.call_args.kwargs["id"], "uuid-001#3"
        )

    def test_body_is_opensearch_document(self):
        os_client = _mock_opensearch()
        chunk = _minimal_chunk()
        index_chunk(chunk, os_client)
        body = os_client.index.call_args.kwargs["body"]
        self.assertEqual(body["document_uuid"], chunk.document_uuid)

    def test_opensearch_exception_raises_runtime_error(self):
        os_client = MagicMock()
        os_client.index.side_effect = RuntimeError("OpenSearch unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            index_chunk(_minimal_chunk(), os_client)
        self.assertIn("uuid-001#0", str(ctx.exception))


# ===========================================================================
# _assert_document_cleared
# ===========================================================================

class TestAssertDocumentCleared(unittest.TestCase):

    def test_sanitized_passes(self):
        # Must not raise
        _assert_document_cleared(_record(DocumentState.SANITIZED))

    def test_victim_flagged_state_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _assert_document_cleared(_record(DocumentState.VICTIM_FLAGGED))
        self.assertIn("VICTIM_FLAGGED", str(ctx.exception))

    def test_victim_flag_true_raises_even_if_state_sanitized(self):
        """victim_flag=True is a hard block regardless of state."""
        with self.assertRaises(ValueError):
            _assert_document_cleared(
                _record(DocumentState.SANITIZED, victim_flag=True)
            )

    def test_pending_review_raises(self):
        with self.assertRaises(ValueError) as ctx:
            _assert_document_cleared(_record(DocumentState.PENDING_REVIEW))
        self.assertIn("PENDING_REVIEW", str(ctx.exception))

    def test_ingested_state_passes(self):
        _assert_document_cleared(_record(DocumentState.INGESTED))


# ===========================================================================
# ingest_document
# ===========================================================================

class TestIngestDocument(unittest.TestCase):

    def setUp(self):
        # Reconstruct mocks before every test so BytesIO streams are
        # unconsumed and call counts start from zero.
        self.bedrock = _mock_bedrock()
        self.os_client = _mock_opensearch()
        self.cfg_embed = EmbeddingConfig()
        self.cfg_chunk = ChunkingConfig(chunk_size_tokens=10, chunk_overlap_tokens=2)

    def _run(self, text: str = "word " * 20, record=None, **kwargs):
        if record is None:
            record = _record()
        return ingest_document(
            record=record,
            text=text,
            bedrock_client=self.bedrock,
            opensearch_client=self.os_client,
            embedding_config=self.cfg_embed,
            chunking_config=self.cfg_chunk,
            **kwargs,
        )

    def test_returns_list_of_chunk_ids(self):
        result = self._run()
        self.assertIsInstance(result, list)
        self.assertTrue(all(isinstance(s, str) for s in result))

    def test_chunk_ids_contain_document_uuid(self):
        result = self._run(record=_record(uuid="uuid-xyz"))
        for chunk_id in result:
            self.assertIn("uuid-xyz", chunk_id)

    def test_chunk_id_format_is_uuid_hash_index(self):
        result = self._run()
        for chunk_id in result:
            self.assertRegex(chunk_id, r".+#\d+")

    def test_opensearch_index_called_once_per_chunk(self):
        result = self._run()
        self.assertEqual(self.os_client.index.call_count, len(result))

    def test_bedrock_called_once_per_chunk(self):
        result = self._run()
        self.assertEqual(self.bedrock.invoke_model.call_count, len(result))

    def test_victim_flagged_raises_before_any_embedding(self):
        with self.assertRaises(ValueError):
            self._run(record=_record(DocumentState.VICTIM_FLAGGED))
        self.bedrock.invoke_model.assert_not_called()

    def test_pending_review_raises_before_any_embedding(self):
        with self.assertRaises(ValueError):
            self._run(record=_record(DocumentState.PENDING_REVIEW))
        self.bedrock.invoke_model.assert_not_called()

    def test_missing_opensearch_client_raises_runtime_error(self):
        with self.assertRaises(RuntimeError) as ctx:
            ingest_document(
                record=_record(),
                text="some text",
                bedrock_client=self.bedrock,
                opensearch_client=None,
            )
        self.assertIn("opensearch_client", str(ctx.exception))

    def test_bedrock_failure_raises_runtime_error(self):
        self.bedrock.invoke_model.side_effect = RuntimeError("Bedrock down")
        with self.assertRaises(RuntimeError):
            self._run()

    def test_opensearch_failure_raises_runtime_error(self):
        self.os_client.index.side_effect = RuntimeError("OpenSearch down")
        with self.assertRaises(RuntimeError):
            self._run()

    def test_corpus_source_propagated_to_chunks(self):
        record = _record()
        record = ClassificationRecord(
            document_uuid=record.document_uuid,
            classification=record.classification,
            state=record.state,
            ingestion_date=record.ingestion_date,
            corpus_source="corpus-abc",
        )
        self._run(record=record)
        body = self.os_client.index.call_args_list[0].kwargs["body"]
        self.assertEqual(body.get("corpus_source"), "corpus-abc")

    def test_victim_flag_false_propagated_to_chunks(self):
        self._run(record=_record(victim_flag=False))
        body = self.os_client.index.call_args_list[0].kwargs["body"]
        self.assertFalse(body["victim_flag"])

    def test_custom_index_name_used(self):
        self._run(index_name="custom-index")
        self.assertEqual(
            self.os_client.index.call_args_list[0].kwargs["index"],
            "custom-index",
        )


if __name__ == "__main__":
    unittest.main()

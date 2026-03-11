"""
ingestor.py
Layer 2: Document chunking, embedding, and storage.

Receives sanitized, classified documents from the pipeline. Splits text
into overlapping chunks (ChunkingConfig), generates embeddings via
Bedrock (EmbeddingConfig), and writes each chunk as a ChunkMetadata
document to the OpenSearch `documents` index.

Prerequisites
-------------
1. sanitizer.py must have cleared the document (victim_flag review complete
   and state == SANITIZED or PENDING_REVIEW resolved). Documents with
   state == VICTIM_FLAGGED must never reach this module.
2. classifier.py must have written the chain-of-custody record to DynamoDB.
3. The OpenSearch index must exist (run infrastructure/opensearch.py once).

Re-ingestion idempotency
------------------------
Each chunk is written using chunk_id (document_uuid#chunk_index) as the
OpenSearch document _id. Re-ingesting the same document overwrites existing
chunks rather than creating duplicates. This means ingest_document() is safe
to call multiple times for the same document_uuid (e.g. after a corpus update).

Bedrock client
--------------
The bedrock-runtime client calls invoke_model() with the Titan v2 request
format. The response is a JSON body containing "embedding" (list of floats).
The client is injectable for testing -- pass a MagicMock that returns
the correct response structure.

    mock_bedrock.invoke_model.return_value = {
        "body": MockBody(json.dumps({"embedding": [0.1] * 1024}))
    }

where MockBody implements .read() -> bytes.

OpenSearch client
-----------------
Uses opensearch-py. The client is injectable for testing.

See CONSTITUTION.md Article III Hard Limit 1.
See docs/ARCHITECTURE.md para Layer 2 -- Embedding & Storage.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from config import DEFAULT_CHUNKING_CONFIG, DEFAULT_EMBEDDING_CONFIG, ChunkingConfig, EmbeddingConfig
from pipeline.chunk_schema import ChunkMetadata
from pipeline.classifier import ClassificationRecord
from pipeline.models import DocumentState

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
OPENSEARCH_INDEX: str = os.environ.get("OPENSEARCH_INDEX", "documents")


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def _assert_document_cleared(record: ClassificationRecord) -> None:
    """
    Raise if the document has not been cleared for embedding.

    VICTIM_FLAGGED documents must never be embedded. This is a hard
    programmatic guard in addition to the classifier's state assignment.
    Raising here is intentional -- caller must fix the pipeline state
    before re-attempting ingestion.

    Constitution reference: Hard Limit 1.
    """
    if record.state == DocumentState.VICTIM_FLAGGED or record.victim_flag:
        raise ValueError(
            f"Document {record.document_uuid} is VICTIM_FLAGGED and must not "
            "be embedded. Clear via human review before ingestion."
        )
    if record.state == DocumentState.PENDING_REVIEW:
        raise ValueError(
            f"Document {record.document_uuid} is PENDING_REVIEW. "
            "Human review must be completed before ingestion."
        )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunking_config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[str]:
    """
    Split text into overlapping word-boundary chunks.

    Uses a word-based approximation for token counts (1 word ~ 1 token for
    this purpose). Chunks are produced by a sliding window of
    chunk_size_tokens words with chunk_overlap_tokens words of overlap.

    Returns a list of at least one chunk. Empty or whitespace-only text
    returns a list containing the original text (the validator in
    ChunkMetadata will catch empty chunks at construction time).

    Parameters
    ----------
    text            : Document text to split.
    chunking_config : ChunkingConfig controlling size and overlap.

    Returns
    -------
    List of text chunks.
    """
    words = text.split()
    if not words:
        return [text]

    size = chunking_config.chunk_size_tokens
    overlap = chunking_config.chunk_overlap_tokens
    step = size - overlap

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_text(
    text: str,
    bedrock_client,
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
) -> list[float]:
    """
    Generate an embedding vector for text using Bedrock.

    Calls bedrock-runtime invoke_model() with the Titan v2 request format.
    Returns a list of floats of length embedding_config.vector_dimension.

    Parameters
    ----------
    text             : Text to embed (one chunk).
    bedrock_client   : boto3 bedrock-runtime client (injectable for testing).
    embedding_config : EmbeddingConfig controlling model ID.

    Returns
    -------
    list[float] of length vector_dimension.

    Raises
    ------
    RuntimeError if the Bedrock call fails or the response is malformed.
    Does not catch exceptions -- callers should handle at the document level.
    """
    try:
        body = json.dumps({"inputText": text})
        response = bedrock_client.invoke_model(
            modelId=embedding_config.model_id,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        response_body = json.loads(response["body"].read())
        return response_body["embedding"]
    except Exception as exc:
        raise RuntimeError(
            f"Bedrock embedding failed for model {embedding_config.model_id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# OpenSearch write
# ---------------------------------------------------------------------------

def index_chunk(
    chunk: ChunkMetadata,
    opensearch_client,
    index_name: str = OPENSEARCH_INDEX,
) -> None:
    """
    Write one ChunkMetadata document to the OpenSearch index.

    Uses chunk.chunk_id as the document _id for idempotent re-ingestion.

    Parameters
    ----------
    chunk             : ChunkMetadata to index.
    opensearch_client : opensearch-py client (injectable for testing).
    index_name        : Target index name.

    Raises
    ------
    RuntimeError if the OpenSearch write fails.
    """
    try:
        opensearch_client.index(
            index=index_name,
            id=chunk.chunk_id,
            body=chunk.opensearch_document(),
        )
        logger.debug(
            "Indexed chunk %s into '%s'.", chunk.chunk_id, index_name,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to index chunk {chunk.chunk_id}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest_document(
    record: ClassificationRecord,
    text: str,
    bedrock_client=None,
    opensearch_client=None,
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    chunking_config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
    index_name: str = OPENSEARCH_INDEX,
) -> list[str]:
    """
    Chunk, embed, and store a cleared document.

    This is the primary entry point for Sub-Module 1D / Layer 2 ingestor.
    It orchestrates the full ingestion pipeline for one document:
      1. Guard: reject VICTIM_FLAGGED or PENDING_REVIEW documents.
      2. Split text into overlapping chunks (ChunkingConfig).
      3. Embed each chunk via Bedrock (EmbeddingConfig).
      4. Build ChunkMetadata for each chunk.
      5. Write each chunk to OpenSearch.

    Parameters
    ----------
    record            : ClassificationRecord from classifier.classify_document().
    text              : Raw document text.
    bedrock_client    : boto3 bedrock-runtime client. If None, created from
                        AWS_REGION. Inject a MagicMock for testing.
    opensearch_client : opensearch-py OpenSearch client. If None, raises
                        RuntimeError (endpoint config required at runtime).
                        Inject a MagicMock for testing.
    embedding_config  : EmbeddingConfig. Defaults to DEFAULT_EMBEDDING_CONFIG.
    chunking_config   : ChunkingConfig. Defaults to DEFAULT_CHUNKING_CONFIG.
    index_name        : OpenSearch index name.

    Returns
    -------
    List of chunk_ids written to OpenSearch (document_uuid#chunk_index).

    Raises
    ------
    ValueError  if the document is VICTIM_FLAGGED or PENDING_REVIEW.
    RuntimeError if Bedrock or OpenSearch calls fail.

    Constitution reference: Hard Limit 1 -- VICTIM_FLAGGED guard.
    Principle V -- Every Output Is Accountable.
    """
    _assert_document_cleared(record)

    if bedrock_client is None:
        import boto3
        bedrock_client = boto3.client(
            "bedrock-runtime", region_name=AWS_REGION
        )

    if opensearch_client is None:
        raise RuntimeError(
            "opensearch_client is required. Set OPENSEARCH_ENDPOINT and "
            "construct the client with SigV4 signing before calling ingest_document(). "
            "See infrastructure/opensearch.py for the construction pattern."
        )

    ingestion_date = datetime.now(timezone.utc).isoformat()
    chunks = chunk_text(text, chunking_config)
    chunk_ids: list[str] = []

    logger.info(
        "Ingesting document %s: %d chunks, model=%s",
        record.document_uuid, len(chunks), embedding_config.model_id,
    )

    for idx, chunk_text_content in enumerate(chunks):
        vector = embed_text(chunk_text_content, bedrock_client, embedding_config)

        chunk = ChunkMetadata(
            document_uuid=record.document_uuid,
            chunk_index=idx,
            text=chunk_text_content,
            vector=vector,
            classification=record.classification.value,
            provenance_tag=record.provenance_tag,
            ingestion_date=ingestion_date,
            victim_flag=record.victim_flag,
            corpus_source=record.corpus_source,
        )

        index_chunk(chunk, opensearch_client, index_name)
        chunk_ids.append(chunk.chunk_id)

    logger.info(
        "Document %s: %d chunks written to index '%s'.",
        record.document_uuid, len(chunk_ids), index_name,
    )

    return chunk_ids

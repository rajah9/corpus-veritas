"""
chunk_schema.py
Layer 2: Chunk metadata schema.

Defines the Pydantic model for every chunk written to OpenSearch. This is
the data contract between ingestor.py (writer) and the RAG query router
(reader). Every field here must have a corresponding mapping in the
OpenSearch index definition in infrastructure/opensearch.py.

A chunk is the unit of retrieval. Each source document is split into
overlapping chunks (see ChunkingConfig in config.py), each chunk is
embedded independently, and the (vector, metadata) pair is stored as one
OpenSearch document. At query time the guardrail layer uses victim_flag
to suppress VICTIM_ADJACENT chunks before results are returned.

See CONSTITUTION.md Article III Hard Limit 1 (victim_flag suppression).
See docs/ARCHITECTURE.md para Layer 2 -- Embedding & Storage.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    """
    Metadata and embedding vector for one chunk of a source document.

    Every instance of this model corresponds to one document in the
    OpenSearch `documents` index. The `vector` field is indexed as a
    knn_vector; all other fields are stored as metadata and used for
    filtering, sorting, and guardrail evaluation.

    Fields
    ------
    document_uuid       UUID of the source document (from classifier.py).
                        Links this chunk back to the DynamoDB chain-of-custody
                        record in corpus_veritas_documents.

    chunk_index         Zero-based position of this chunk within the document.
                        Combined with document_uuid, uniquely identifies a chunk.

    text                Raw text content of this chunk. Stored in OpenSearch
                        for snippet display in query responses. Never surfaced
                        directly for VICTIM_ADJACENT chunks (guardrail enforces).

    vector              Embedding vector produced by Bedrock. Length must match
                        EmbeddingConfig.vector_dimension (1024 for Titan v2).
                        Validated at construction time.

    classification      DocumentClassification value from classifier.py.
                        Used by the query router to apply access control rules.

    provenance_tag      Provenance tag from corpus_evaluator.py, e.g.
                        PROVENANCE_COMMUNITY_VOUCHED. None for DOJ_DIRECT.

    ingestion_date      ISO 8601 UTC timestamp of ingestion. Enables date-range
                        filtering in queries and audit trail reconstruction.

    victim_flag         True if Hard Limit 1 protection applies. Redundant with
                        classification == VICTIM_ADJACENT but retained for
                        guardrail fast-path: the guardrail can filter on a
                        single boolean field without parsing the classification
                        enum. NEVER surfaced in query results when True.

    page_number         PDF page number the chunk originated from, if known.
                        None for non-PDF sources or when page extraction failed.

    bates_number        Bates stamp of the source document, if present.
                        None if the document uses EFTA numbering or is unnumbered.

    efta_number         EFTA number of the source document, if present.
                        None if the document uses Bates numbering or is unnumbered.

    corpus_source       corpus_registry UUID if document came from an external
                        corpus; "DOJ_DIRECT" if from the original DOJ PDF release;
                        None if source is unknown.

    Constitution reference: Hard Limit 1 -- victim_flag chunks suppressed
    from all public query paths by the guardrail layer.
    """

    document_uuid: str
    chunk_index: int = Field(ge=0)
    text: str
    vector: list[float]
    classification: str
    provenance_tag: Optional[str] = None
    ingestion_date: str
    victim_flag: bool = False
    page_number: Optional[int] = Field(default=None, ge=1)
    bates_number: Optional[str] = None
    efta_number: Optional[str] = None
    corpus_source: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text must not be empty or whitespace-only")
        return v

    @field_validator("vector")
    @classmethod
    def vector_must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("vector must not be empty")
        return v

    @field_validator("document_uuid")
    @classmethod
    def document_uuid_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("document_uuid must not be empty")
        return v

    @field_validator("ingestion_date")
    @classmethod
    def ingestion_date_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ingestion_date must not be empty")
        return v

    def opensearch_document(self) -> dict:
        """
        Serialise to an OpenSearch index document dict.

        Uses model_dump() with exclude_none=True so optional fields absent
        from the source document do not appear as explicit nulls in the index.
        This keeps the index lean and avoids null-field noise in query results.

        The caller passes this dict to the OpenSearch client's index() method:
            client.index(index="documents", body=chunk.opensearch_document())
        """
        return self.model_dump(exclude_none=True)

    @property
    def chunk_id(self) -> str:
        """
        Stable, unique identifier for this chunk in the OpenSearch index.
        Used as the document _id to make ingestor re-runs idempotent:
        re-ingesting the same document overwrites existing chunks rather
        than creating duplicates.
        """
        return f"{self.document_uuid}#{self.chunk_index}"

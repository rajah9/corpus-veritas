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

Enums defined here
------------------
DocumentType    Content type of the source document. Stored per-chunk so
                the query router can apply type-specific retrieval strategies
                (e.g. FBI_302 queries route differently from COURT_FILING).

SequenceScheme  Which numbering scheme the sequence_number field uses.
                Needed because BatesNumber and EFTANumber sort differently
                and have different gap semantics.

See CONSTITUTION.md Article III Hard Limit 1 (victim_flag suppression).
See docs/ARCHITECTURE.md para Layer 2 -- Embedding & Storage.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from pipeline.models import ConfidenceTier, DeletionFlag


# ---------------------------------------------------------------------------
# DocumentType
# ---------------------------------------------------------------------------

class DocumentType(str, Enum):
    """
    Content type of the source document.

    Used by the query router to apply type-specific retrieval strategies.
    FBI_302 queries, for example, benefit from filtering to FBI_302 chunks
    first before broadening to CORRESPONDENCE or COURT_FILING.

    FBI_302         FBI Interview Summary (Form 302). Rigid structure:
                    interview date, subject, agent, file number. Partial
                    delivery (header present, pages missing) is detectable
                    and is flagged separately from complete absence.

    CORRESPONDENCE  Letters, emails, faxes between named individuals.

    COURT_FILING    Court filings, motions, orders, docket entries.

    EXHIBIT         Exhibit attached to a court filing or deposition.

    OTHER           Does not match a known type. Requires human review
                    before type-specific retrieval strategies are applied.
    """
    FBI_302       = "FBI_302"
    CORRESPONDENCE = "CORRESPONDENCE"
    COURT_FILING  = "COURT_FILING"
    EXHIBIT       = "EXHIBIT"
    OTHER         = "OTHER"


# ---------------------------------------------------------------------------
# SequenceScheme
# ---------------------------------------------------------------------------

class SequenceScheme(str, Enum):
    """
    Numbering scheme used by the sequence_number field.

    EFTA    Per-page sequential numbering across all 12 DOJ datasets.
            DS9 gaps are documented expected gaps, not deletion signals.
            See pipeline/sequence_numbers.py EFTANumber.

    BATES   Traditional legal Bates stamping. All gaps are suspicious.
            See pipeline/sequence_numbers.py BatesNumber.
    """
    EFTA  = "EFTA"
    BATES = "BATES"


# ---------------------------------------------------------------------------
# ChunkMetadata
# ---------------------------------------------------------------------------

class ChunkMetadata(BaseModel):
    """
    Metadata and embedding vector for one chunk of a source document.

    Every instance of this model corresponds to one document in the
    OpenSearch `documents` index. The `vector` field is indexed as a
    knn_vector; all other fields are stored as metadata and used for
    filtering, sorting, and guardrail evaluation.

    Required fields
    ---------------
    document_uuid       UUID of the source document (from classifier.py).
                        Links this chunk back to the DynamoDB chain-of-custody
                        record in corpus_veritas_documents.

    chunk_index         Zero-based position of this chunk within the document.
                        Combined with document_uuid, uniquely identifies a chunk.

    text                Raw text content of this chunk. Never surfaced directly
                        for VICTIM_ADJACENT chunks (guardrail enforces).

    vector              Embedding vector produced by Bedrock. Length must match
                        EmbeddingConfig.vector_dimension (1024 for Titan v2).

    classification      DocumentClassification value from classifier.py.

    ingestion_date      ISO 8601 UTC timestamp of ingestion.

    Optional fields -- caller must supply; cannot be derived from ClassificationRecord
    -----------------------------------------------------------------------------------
    provenance_tag      Provenance tag from corpus_evaluator.py.

    victim_flag         True if Hard Limit 1 protection applies. Redundant with
                        classification == VICTIM_ADJACENT but retained for
                        guardrail fast-path.

    page_number         PDF page number the chunk originated from.

    bates_number        Bates stamp of the source document, if present.

    efta_number         EFTA number of the source document, if present.

    corpus_source       corpus_registry UUID or "DOJ_DIRECT".

    sequence_number     Extracted EFTA number (primary) or Bates stamp. The
                        raw value as a string, scheme identified by sequence_scheme.
                        None if no sequence number was found in the document.

    sequence_scheme     Which scheme sequence_number uses (EFTA or BATES).
                        None if sequence_number is None.

    document_date       ISO 8601 date of the source document (interview date,
                        filing date, letter date). Distinct from ingestion_date.
                        None if not extractable.

    document_type       DocumentType enum value. None if not yet classified.

    named_entities      JSON array of NER extractions with type and confidence.
                        Populated by Layer 4 entity resolution; empty list until
                        Layer 4 is implemented. Each entry is a dict with at
                        minimum {"text": str, "type": str, "confidence": float}.

    confidence_tier     ConfidenceTier value for this chunk's content. Derived
                        upstream from provenance and corroboration checking.
                        Enforced by Layer 5 guardrail (response language must
                        match tier). None if not yet assessed.

    deletion_flag       DeletionFlag if this document is a deletion finding.
                        None for normally present documents. Set by
                        deletion_detector.py when the document is a gap record
                        rather than actual document content.

    Constitution reference: Hard Limit 1 -- victim_flag chunks suppressed
    from all public query paths by the guardrail layer.
    Principle II -- confidence_tier enforced at response time.
    """

    # Required
    document_uuid:  str
    chunk_index:    int = Field(ge=0)
    text:           str
    vector:         list[float]
    classification: str
    ingestion_date: str

    # Optional -- provenance / identity
    provenance_tag: Optional[str] = None
    victim_flag:    bool = False

    # Optional -- document location
    page_number:    Optional[int] = Field(default=None, ge=1)
    bates_number:   Optional[str] = None
    efta_number:    Optional[str] = None
    corpus_source:  Optional[str] = None

    # Optional -- caller must supply (not derivable from ClassificationRecord)
    sequence_number:  Optional[str] = None
    sequence_scheme:  Optional[SequenceScheme] = None
    document_date:    Optional[str] = None
    document_type:    Optional[DocumentType] = None
    named_entities:   list[dict] = Field(default_factory=list)
    confidence_tier:  Optional[ConfidenceTier] = None
    deletion_flag:    Optional[DeletionFlag] = None

    # -----------------------------------------------------------------------
    # Validators
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def chunk_id(self) -> str:
        """
        Stable, unique identifier for this chunk in the OpenSearch index.
        Used as the document _id to make ingestor re-runs idempotent.
        """
        return f"{self.document_uuid}#{self.chunk_index}"

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def opensearch_document(self) -> dict:
        """
        Serialise to an OpenSearch index document dict.

        Uses model_dump() with exclude_none=True so optional fields absent
        from the source document do not appear as explicit nulls in the index.
        Enum values are serialised as their string values.
        named_entities defaults to [] and is always included (not None).
        """
        raw = self.model_dump(exclude_none=True, mode="json")
        # Enum fields: Pydantic mode="json" serialises str enums as their
        # .value already, so no manual conversion needed.
        return raw

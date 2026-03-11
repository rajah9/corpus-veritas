"""
classifier.py
Layer 1, Sub-Module 1D: Document Classification & Chain of Custody

Every sanitized document receives:
  - A classification tag (VICTIM_ADJACENT, PERPETRATOR_ADJACENT,
    PROCEDURAL, or UNKNOWN)
  - A chain-of-custody record written to DynamoDB

The DynamoDB table `corpus_veritas_documents` is the authoritative registry
for every document that has entered the pipeline. The chain-of-custody record
is immutable once written; updates create a new record rather than overwriting.

DynamoDB table design
---------------------
Table name : corpus_veritas_documents
PK         : document_uuid (S)

GSIs
  gsi-classification-date   PK=classification (S), SK=ingestion_date (S)
      Purpose: fetch all VICTIM_ADJACENT documents, range by date.

  gsi-corpus-source         PK=corpus_source (S), SK=document_uuid (S)
      Purpose: fetch all documents from a given external corpus.

  gsi-victim-flag           PK=victim_flag (S), SK=ingestion_date (S)
      SPARSE INDEX: victim_flag is only written to the DynamoDB item when
      it is True. Items without victim_flag=True are not indexed, so this
      index contains only flagged documents. Never requires a filter expression.

Classification rules
--------------------
VICTIM_ADJACENT      pii_result.victim_flag is True.
                     Takes precedence over all other rules.
                     Hard Limit 1 applies: document enters human-review queue
                     via sanitizer.py before classification.

PROCEDURAL           Text matches structural markers for FBI 302 forms,
                     court filings, or DOJ administrative records:
                     FD-302 header, case number patterns, SDNY/EDNY/USAO
                     identifiers, "MEMORANDUM FOR" headers, etc.

PERPETRATOR_ADJACENT Comprehend detected at least one NAME entity AND the
                     document is not VICTIM_ADJACENT and not PROCEDURAL.
                     Indicates the document names individuals in a non-procedural
                     context; human classification may refine this later.

UNKNOWN              Default. No signals met any of the above criteria.

See CONSTITUTION.md Article III Hard Limit 1.
See docs/ARCHITECTURE.md para Layer 1, Sub-Module 1D.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pipeline.models import DocumentState
from pipeline.sanitizer import PIIDetectionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
DOCUMENTS_TABLE_NAME: str = os.environ.get(
    "DOCUMENTS_TABLE_NAME", "corpus_veritas_documents"
)

# ---------------------------------------------------------------------------
# Classification taxonomy
# ---------------------------------------------------------------------------

class DocumentClassification(str, Enum):
    """
    Content-based classification tag assigned at ingestion.

    Applied by _determine_classification() using PII detection results and
    text pattern matching. Written to DynamoDB and used by the guardrail layer
    to route documents to appropriate query paths.

    VICTIM_ADJACENT      Contains or references victim/survivor identities.
                         Suppressed from all public query paths per Hard Limit 1.
                         Requires human review before any query access.

    PERPETRATOR_ADJACENT Names individuals in connection with alleged conduct.
                         Subject to multi-source convergence rule (Layer 3)
                         before any inference is surfaced.

    PROCEDURAL           Court filings, FBI 302 forms, administrative records.
                         Generally does not name victims directly; lower
                         suppression threshold than VICTIM_ADJACENT.

    UNKNOWN              Default. Requires human classification before the
                         document is included in inference query paths.
    """
    VICTIM_ADJACENT      = "VICTIM_ADJACENT"
    PERPETRATOR_ADJACENT = "PERPETRATOR_ADJACENT"
    PROCEDURAL           = "PROCEDURAL"
    UNKNOWN              = "UNKNOWN"


# ---------------------------------------------------------------------------
# Chain-of-custody record
# ---------------------------------------------------------------------------

@dataclass
class ClassificationRecord:
    """
    The chain-of-custody record written to DynamoDB for every ingested document.

    This record is the authoritative source for a document's current state,
    classification, and provenance within the pipeline. It feeds the guardrail
    layer's victim identity check and the query router's access control logic.

    Fields
    ------
    document_uuid     Unique identifier (UUID4) assigned at ingestion.
    classification    DocumentClassification tag.
    state             DocumentState lifecycle state at time of this record.
    ingestion_date    ISO 8601 UTC timestamp of initial ingestion.
    victim_flag       True if Hard Limit 1 protection applies.
                      Only written to DynamoDB when True (sparse GSI).
    corpus_source     corpus_registry UUID if document came from an external
                      corpus; "DOJ_DIRECT" if OCR'd from the original DOJ PDF;
                      None if source is unknown.
    provenance_tag    Provenance tag from corpus_evaluator.py, e.g.
                      "PROVENANCE_COMMUNITY_VOUCHED". None for DOJ_DIRECT.
    pii_entity_count  Number of PII entities detected by Comprehend.
    review_reason     Human-readable reason for human-review flag, if set.
    notes             Free-text notes for pipeline operators.
    created_at        ISO 8601 UTC timestamp of record creation.
    """
    document_uuid: str
    classification: DocumentClassification
    state: DocumentState
    ingestion_date: str                       # ISO 8601 UTC
    victim_flag: bool = False
    corpus_source: Optional[str] = None
    provenance_tag: Optional[str] = None
    pii_entity_count: int = 0
    review_reason: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dynamodb_item(self) -> dict:
        """
        Serialise to a DynamoDB put_item dict.

        victim_flag is omitted when False to maintain sparse GSI semantics:
        the gsi-victim-flag index only indexes documents where victim_flag
        is explicitly present (and True). Never call this method and then
        add victim_flag manually -- the sparse behaviour depends on omission.
        """
        item: dict = {
            "document_uuid":    {"S": self.document_uuid},
            "classification":   {"S": self.classification.value},
            "state":            {"S": self.state.value},
            "ingestion_date":   {"S": self.ingestion_date},
            "pii_entity_count": {"N": str(self.pii_entity_count)},
            "created_at":       {"S": self.created_at},
        }
        # Sparse index: only include victim_flag when True
        if self.victim_flag:
            item["victim_flag"] = {"S": "true"}
        if self.corpus_source is not None:
            item["corpus_source"] = {"S": self.corpus_source}
        if self.provenance_tag is not None:
            item["provenance_tag"] = {"S": self.provenance_tag}
        if self.review_reason is not None:
            item["review_reason"] = {"S": self.review_reason}
        if self.notes is not None:
            item["notes"] = {"S": self.notes}
        return item

    @classmethod
    def from_dynamodb_item(cls, item: dict) -> "ClassificationRecord":
        """Deserialise from a DynamoDB get_item response dict."""
        return cls(
            document_uuid=item["document_uuid"]["S"],
            classification=DocumentClassification(item["classification"]["S"]),
            state=DocumentState(item["state"]["S"]),
            ingestion_date=item["ingestion_date"]["S"],
            victim_flag=item.get("victim_flag", {}).get("S") == "true",
            corpus_source=item.get("corpus_source", {}).get("S"),
            provenance_tag=item.get("provenance_tag", {}).get("S"),
            pii_entity_count=int(item.get("pii_entity_count", {}).get("N", "0")),
            review_reason=item.get("review_reason", {}).get("S"),
            notes=item.get("notes", {}).get("S"),
            created_at=item.get("created_at", {}).get("S", ""),
        )


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

# FBI 302 / FD-302 form markers
_PROCEDURAL_FBI_302 = re.compile(
    r"\b(FD-302|FBI\s+302|FEDERAL\s+BUREAU\s+OF\s+INVESTIGATION"
    r"|FD\s+302|302\s+INTERVIEW)\b",
    re.IGNORECASE,
)

# Federal court case number patterns: 20-cv-1234, 20-cr-330, 1:20-cv-1234
_PROCEDURAL_CASE_NUMBER = re.compile(
    r"\b\d{1,2}[-:]\d{2}[-:](cv|cr|mc|mj|crim|misc)-\d+\b",
    re.IGNORECASE,
)

# DOJ / USAO / court identifiers in document headers
_PROCEDURAL_COURT_MARKER = re.compile(
    r"\b(SDNY|EDNY|USAO|U\.S\. ATTORNEY|UNITED STATES ATTORNEY"
    r"|UNITED STATES DISTRICT COURT|IN THE MATTER OF"
    r"|CASE NO\.|DOCKET NO\.|MEMORANDUM FOR)\b",
    re.IGNORECASE,
)


def _determine_classification(
    text: str,
    pii_result: PIIDetectionResult,
) -> DocumentClassification:
    """
    Apply classification rules to a document and return its DocumentClassification.

    Rules are applied in strict priority order. The first matching rule wins.

    1. VICTIM_ADJACENT  -- pii_result.victim_flag (Hard Limit 1, always first)
    2. PROCEDURAL       -- structural form/court markers in text
    3. PERPETRATOR_ADJACENT -- Comprehend detected NAME entities (not victim context)
    4. UNKNOWN          -- default
    """
    # Rule 1: victim flag (absolute priority)
    if pii_result.victim_flag:
        return DocumentClassification.VICTIM_ADJACENT

    # Rule 2: procedural document markers
    if (
        _PROCEDURAL_FBI_302.search(text)
        or _PROCEDURAL_CASE_NUMBER.search(text)
        or _PROCEDURAL_COURT_MARKER.search(text)
    ):
        return DocumentClassification.PROCEDURAL

    # Rule 3: named individuals present (non-victim, non-procedural context)
    name_entities = [
        e for e in pii_result.pii_entities_detected if e.get("Type") == "NAME"
    ]
    if name_entities:
        return DocumentClassification.PERPETRATOR_ADJACENT

    # Rule 4: default
    return DocumentClassification.UNKNOWN


def _document_state_from_pii(pii_result: PIIDetectionResult) -> DocumentState:
    """Map PIIDetectionResult flags to DocumentState."""
    if pii_result.victim_flag:
        return DocumentState.VICTIM_FLAGGED
    if pii_result.requires_human_review:
        return DocumentState.PENDING_REVIEW
    return DocumentState.SANITIZED


# ---------------------------------------------------------------------------
# DynamoDB write
# ---------------------------------------------------------------------------

def _write_classification_record(
    record: ClassificationRecord,
    dynamodb_client,
    table_name: str = DOCUMENTS_TABLE_NAME,
) -> None:
    """
    Write a ClassificationRecord to DynamoDB.

    Uses put_item (overwrite semantics). This is intentional: if a document
    is re-ingested (e.g. after a corpus update), the latest classification
    wins. The audit trail is maintained by CloudWatch Logs, not by DynamoDB
    item history.

    Raises
    ------
    Propagates botocore ClientError with additional context in the message.
    Callers should handle this to implement retry or dead-letter logic.
    """
    try:
        dynamodb_client.put_item(
            TableName=table_name,
            Item=record.to_dynamodb_item(),
        )
        logger.info(
            "Classification record written for %s: %s / %s",
            record.document_uuid, record.classification.value, record.state.value,
        )
    except Exception as exc:
        logger.error(
            "Failed to write classification record for %s: %s",
            record.document_uuid, exc,
        )
        raise RuntimeError(
            f"DynamoDB write failed for document {record.document_uuid}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_document(
    document_uuid: str,
    text: str,
    pii_result: PIIDetectionResult,
    corpus_source: Optional[str] = None,
    provenance_tag: Optional[str] = None,
    notes: Optional[str] = None,
    dynamodb_client=None,
    table_name: str = DOCUMENTS_TABLE_NAME,
) -> ClassificationRecord:
    """
    Classify a sanitized document and write its chain-of-custody record.

    This is the primary entry point for Sub-Module 1D. It consumes the
    PIIDetectionResult from sanitizer.py, determines the DocumentClassification
    and DocumentState, constructs a ClassificationRecord, writes it to
    DynamoDB, and returns the record for use by the caller.

    Parameters
    ----------
    document_uuid   : UUID of the document (must match pii_result.document_uuid).
    text            : Raw document text (used for procedural pattern matching).
    pii_result      : PIIDetectionResult from sanitizer.sanitize_document().
    corpus_source   : corpus_registry UUID, "DOJ_DIRECT", or None.
    provenance_tag  : Provenance tag from corpus_evaluator.py. None for DOJ_DIRECT.
    notes           : Optional operator notes.
    dynamodb_client : Optional pre-constructed boto3 DynamoDB client.
                      If None, a client is created using AWS_REGION.
                      Inject a MagicMock for testing.
    table_name      : DynamoDB table name. Defaults to DOCUMENTS_TABLE_NAME.

    Returns
    -------
    ClassificationRecord -- the record as written to DynamoDB.

    Raises
    ------
    RuntimeError if the DynamoDB write fails (wraps the underlying
    botocore ClientError with document_uuid context).

    Constitution reference: Article III Hard Limit 1.
    Principle V -- Every Output Is Accountable.
    """
    if dynamodb_client is None:
        import boto3
        dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)

    classification = _determine_classification(text, pii_result)
    state = _document_state_from_pii(pii_result)
    ingestion_date = datetime.now(timezone.utc).isoformat()

    record = ClassificationRecord(
        document_uuid=document_uuid,
        classification=classification,
        state=state,
        ingestion_date=ingestion_date,
        victim_flag=pii_result.victim_flag,
        corpus_source=corpus_source,
        provenance_tag=provenance_tag,
        pii_entity_count=len(pii_result.pii_entities_detected),
        review_reason=pii_result.review_reason,
        notes=notes,
    )

    _write_classification_record(record, dynamodb_client, table_name)
    return record
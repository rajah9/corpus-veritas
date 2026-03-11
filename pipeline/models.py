"""
models.py
Shared enums and data models used across the corpus-veritas pipeline.

Placing shared types here avoids circular imports between pipeline modules
and gives every model a single authoritative definition.

Models
------
DeletionFlag        — taxonomy of document absence states
DocumentState       — overall lifecycle state of a document in the pipeline
WithholdingRecord   — structured record of a government-acknowledged withholding

See docs/ARCHITECTURE.md § 3 — Deletion Detection Module.
See CONSTITUTION.md Principle IV — Gaps Are Facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# DeletionFlag
# ---------------------------------------------------------------------------

class DeletionFlag(str, Enum):
    """
    Taxonomy of document absence states, graded by evidence strength
    and by whether the government has acknowledged the withholding.

    Evidence-graded flags (set by deletion_detector.py)
    ----------------------------------------------------
    DELETION_CONFIRMED      All three detection signals confirm absence:
                            EFTA/Bates gap + discovery log entry + document stamp gap.

    DELETION_SUSPECTED      Two signals confirm absence. The most common state
                            for newly identified gaps before full signal resolution.

    DELETION_POSSIBLE       Single signal only. Flagged for further investigation
                            before escalation.

    REFERENCE_UNRESOLVED    A document is referenced internally (e.g. "see Exhibit C")
                            but is not present in the corpus and has no index entry.
                            May indicate a missing document or a labelling error.

    Government-acknowledged flags (set when official confirmation exists)
    ---------------------------------------------------------------------
    WITHHELD_ACKNOWLEDGED   The government has confirmed the document exists but is
                            being held offline, with a stated reason and timeline.
                            Example: DOJ confirmation to WSJ that 47,635 files were
                            under review. This is a CONFIRMED-tier fact even though
                            the document content is unknown.

    WITHHELD_SELECTIVELY    Sibling documents from the same series, interview, or
                            filing were released while this specific document was not.
                            Example: one FBI 302 from a four-interview series released,
                            three withheld. The selective pattern is itself evidential.

    Ordering (weakest → strongest evidence of intentional withholding)
    ------------------------------------------------------------------
    REFERENCE_UNRESOLVED < DELETION_POSSIBLE < DELETION_SUSPECTED
    < DELETION_CONFIRMED < WITHHELD_SELECTIVELY < WITHHELD_ACKNOWLEDGED

    Constitution reference: Principle IV — Gaps Are Facts.
    A WITHHELD_ACKNOWLEDGED document is a confirmed gap with a confirmed withholder.
    The content is still unknown. These are not equivalent statements.
    """

    DELETION_CONFIRMED    = "DELETION_CONFIRMED"
    DELETION_SUSPECTED    = "DELETION_SUSPECTED"
    DELETION_POSSIBLE     = "DELETION_POSSIBLE"
    REFERENCE_UNRESOLVED  = "REFERENCE_UNRESOLVED"
    WITHHELD_ACKNOWLEDGED = "WITHHELD_ACKNOWLEDGED"
    WITHHELD_SELECTIVELY  = "WITHHELD_SELECTIVELY"

    def __lt__(self, other: "DeletionFlag") -> bool:
        return _FLAG_ORDER[self] < _FLAG_ORDER[other]

    def __le__(self, other: "DeletionFlag") -> bool:
        return _FLAG_ORDER[self] <= _FLAG_ORDER[other]

    def __gt__(self, other: "DeletionFlag") -> bool:
        return _FLAG_ORDER[self] > _FLAG_ORDER[other]

    def __ge__(self, other: "DeletionFlag") -> bool:
        return _FLAG_ORDER[self] >= _FLAG_ORDER[other]

    @property
    def confidence_tier(self) -> str:
        """
        Map this flag to the system-wide confidence tier vocabulary.
        Used when surfacing deletion findings in query responses.
        """
        return _FLAG_CONFIDENCE[self]

    @property
    def requires_human_review(self) -> bool:
        """
        True if this flag requires human review before the finding
        can be included in a published output.
        """
        return self in {
            DeletionFlag.DELETION_POSSIBLE,
            DeletionFlag.REFERENCE_UNRESOLVED,
        }

    @property
    def is_government_acknowledged(self) -> bool:
        """True if the government has confirmed this document's existence."""
        return self in {
            DeletionFlag.WITHHELD_ACKNOWLEDGED,
            DeletionFlag.WITHHELD_SELECTIVELY,
        }


# Ordered weakest → strongest (used by comparison operators above)
_FLAG_ORDER: dict[DeletionFlag, int] = {
    DeletionFlag.REFERENCE_UNRESOLVED:  0,
    DeletionFlag.DELETION_POSSIBLE:     1,
    DeletionFlag.DELETION_SUSPECTED:    2,
    DeletionFlag.DELETION_CONFIRMED:    3,
    DeletionFlag.WITHHELD_SELECTIVELY:  4,
    DeletionFlag.WITHHELD_ACKNOWLEDGED: 5,
}

_FLAG_CONFIDENCE: dict[DeletionFlag, str] = {
    DeletionFlag.REFERENCE_UNRESOLVED:  "SPECULATIVE",
    DeletionFlag.DELETION_POSSIBLE:     "SINGLE_SOURCE",
    DeletionFlag.DELETION_SUSPECTED:    "CORROBORATED",
    DeletionFlag.DELETION_CONFIRMED:    "CONFIRMED",
    DeletionFlag.WITHHELD_SELECTIVELY:  "CONFIRMED",
    DeletionFlag.WITHHELD_ACKNOWLEDGED: "CONFIRMED",
}


# ---------------------------------------------------------------------------
# DocumentState
# ---------------------------------------------------------------------------

class DocumentState(str, Enum):
    """
    Overall lifecycle state of a document as it moves through the pipeline.
    Stored in DynamoDB alongside every document record.
    """
    PENDING_REVIEW      = "PENDING_REVIEW"       # in human-review queue
    SANITIZED           = "SANITIZED"            # PII check passed, cleared for chunking
    INGESTED            = "INGESTED"             # chunked, embedded, in vector store
    VICTIM_FLAGGED      = "VICTIM_FLAGGED"       # suppressed from all query paths
    DEPRECATED          = "DEPRECATED"           # superseded; retained but not queried
    DELETION_FLAGGED    = "DELETION_FLAGGED"     # gap detected; document expected but absent


# ---------------------------------------------------------------------------
# WithholdingRecord
# ---------------------------------------------------------------------------

@dataclass
class WithholdingRecord:
    """
    Structured record of a government-acknowledged document withholding.

    Created when an authoritative source (DOJ statement, court filing,
    FOIA response, or credible journalism) confirms that a specific document
    or document set exists but has been withheld from the public release.

    This record is citable provenance in its own right. The statement
    "DOJ confirmed to WSJ on [date] that 47,635 files are offline" is a
    CONFIRMED-tier fact regardless of whether the documents are ever released.

    Fields
    ------
    record_id           Unique identifier (UUID), assigned at creation.
    document_identifiers
                        EFTA numbers, Bates stamps, or DOJ document IDs
                        of the withheld document(s). May be a range for
                        bulk withholdings (e.g. the 47,635 offline files).
    deletion_flag       DeletionFlag value for this withholding.
                        Must be WITHHELD_ACKNOWLEDGED or WITHHELD_SELECTIVELY.
    stated_reason       The government's stated reason, verbatim if available.
                        NULL if no reason was given.
    acknowledgment_source
                        Who disclosed the withholding and how:
                        e.g. "DOJ statement to WSJ, confirmed [date]"
                        e.g. "Court filing SDNY 20-cr-330, docket entry 412"
    acknowledgment_date ISO 8601 date the withholding was confirmed.
    expected_release_date
                        If the government stated a release timeline, record it.
                        NULL if no timeline given. Enables follow-up alerting.
    released            True once the document has been released and ingested.
    release_date        ISO 8601 date of actual release. NULL until released.
    sibling_document_ids
                        For WITHHELD_SELECTIVELY: identifiers of sibling
                        documents from the same series that WERE released.
                        This is the evidence that makes the withholding selective.
    subject_entities    Named entities (persons, orgs) this withholding concerns,
                        extracted from the acknowledgment source. Used to surface
                        this record in entity-based queries.
                        NOTE: victim-adjacent entities must be flagged separately
                        and suppressed from public query paths per Hard Limit 1.
    notes               Free-text notes for human reviewers.
    created_at          ISO 8601 timestamp of record creation.
    updated_at          ISO 8601 timestamp of last update.

    Constitution reference: Principle IV — Gaps Are Facts.
    Principle V — Every Output Is Accountable.
    Hard Limit 1 — subject_entities containing victim identities must be
                   flagged and suppressed before this record enters any
                   public query path.
    """

    record_id: str
    document_identifiers: list[str]
    deletion_flag: DeletionFlag
    acknowledgment_source: str
    acknowledgment_date: str                          # ISO 8601

    # Optional fields
    stated_reason: Optional[str] = None
    expected_release_date: Optional[str] = None       # ISO 8601
    released: bool = False
    release_date: Optional[str] = None                # ISO 8601
    sibling_document_ids: list[str] = field(default_factory=list)
    subject_entities: list[dict] = field(default_factory=list)
    notes: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    updated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )

    def __post_init__(self) -> None:
        if self.deletion_flag not in {
            DeletionFlag.WITHHELD_ACKNOWLEDGED,
            DeletionFlag.WITHHELD_SELECTIVELY,
        }:
            raise ValueError(
                f"WithholdingRecord requires WITHHELD_ACKNOWLEDGED or "
                f"WITHHELD_SELECTIVELY flag; got {self.deletion_flag}. "
                f"Use deletion_detector.py for evidence-graded flags."
            )
        if not self.document_identifiers:
            raise ValueError("WithholdingRecord requires at least one document_identifier.")
        if self.released and not self.release_date:
            raise ValueError("released=True requires a release_date.")

    @property
    def is_overdue(self) -> bool:
        """
        True if an expected_release_date was given and has passed
        without the document being released.
        Enables automated follow-up alerting in the pipeline.
        """
        if self.released or not self.expected_release_date:
            return False
        return datetime.utcnow().isoformat() > self.expected_release_date

    @property
    def document_count(self) -> int:
        """Number of document identifiers covered by this record."""
        return len(self.document_identifiers)

    def mark_released(self, release_date: str) -> None:
        """
        Mark this withholding as resolved — the document has been released.
        Updates both released flag and timestamps. Does not delete the record:
        the withholding history is retained in the audit log.
        """
        self.released = True
        self.release_date = release_date
        self.updated_at = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        """Serialise to dict for DynamoDB storage."""
        return {
            "record_id": self.record_id,
            "document_identifiers": self.document_identifiers,
            "deletion_flag": self.deletion_flag.value,
            "acknowledgment_source": self.acknowledgment_source,
            "acknowledgment_date": self.acknowledgment_date,
            "stated_reason": self.stated_reason,
            "expected_release_date": self.expected_release_date,
            "released": self.released,
            "release_date": self.release_date,
            "sibling_document_ids": self.sibling_document_ids,
            "subject_entities": self.subject_entities,
            "notes": self.notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WithholdingRecord":
        """Deserialise from DynamoDB record."""
        return cls(
            record_id=data["record_id"],
            document_identifiers=data["document_identifiers"],
            deletion_flag=DeletionFlag(data["deletion_flag"]),
            acknowledgment_source=data["acknowledgment_source"],
            acknowledgment_date=data["acknowledgment_date"],
            stated_reason=data.get("stated_reason"),
            expected_release_date=data.get("expected_release_date"),
            released=data.get("released", False),
            release_date=data.get("release_date"),
            sibling_document_ids=data.get("sibling_document_ids", []),
            subject_entities=data.get("subject_entities", []),
            notes=data.get("notes"),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

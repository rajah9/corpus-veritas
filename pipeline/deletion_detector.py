"""
deletion_detector.py
Layer 1 — Deletion Detection Module

Identifies documents and pages that are absent from the public corpus
despite evidence they should exist. Implements the three-signal methodology
established by NPR's February 2026 investigation, extended to handle
government-acknowledged withheld documents following WSJ's March 2026
reporting on 47,635 files held offline by the DOJ.

Two categories of finding, both recorded as DeletionFlag values:

Evidence-graded (derived from signal analysis)
  DELETION_CONFIRMED      — three signals confirm absence
  DELETION_SUSPECTED      — two signals confirm absence
  DELETION_POSSIBLE       — one signal, needs further investigation
  REFERENCE_UNRESOLVED    — internally referenced but absent, no index entry

Government-acknowledged (derived from official statements or credible reporting)
  WITHHELD_ACKNOWLEDGED   — government confirmed existence and offline status
  WITHHELD_SELECTIVELY    — sibling documents released, this one withheld

Both categories are stored as WithholdingRecord objects in DynamoDB.
Evidence-graded findings and acknowledged withheld documents use the same
model so they can be queried together, but their DeletionFlag values keep
them clearly distinguished in every output.

See docs/ARCHITECTURE.md § 3 — Deletion Detection Module.
See CONSTITUTION.md Principle IV — Gaps Are Facts.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from pipeline.models import DeletionFlag, WithholdingRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection signal model
# ---------------------------------------------------------------------------

@dataclass
class DetectionSignals:
    """
    The three independent signals used to grade a deletion finding.

    Each signal is independently verifiable from a different source,
    which is what gives multi-signal findings their higher confidence.

    Signals
    -------
    efta_gap            EFTA/Bates sequence number is present in the DOJ
                        index but absent from the corpus. The foundational
                        signal — necessary but not sufficient for CONFIRMED.

    discovery_log_entry A prosecution or court discovery log entry exists
                        for this document with no corresponding release.
                        Source: PACER docket, FBI case catalogue (Sentinel),
                        or DOJ production manifest.

    document_stamp_gap  Physical document stamp or internal reference number
                        indicates a gap. For FBI 302s: series number present
                        in a related document but this 302 is absent.

    Each True signal contributes +1 to the confidence grade:
      3 signals → DELETION_CONFIRMED
      2 signals → DELETION_SUSPECTED
      1 signal  → DELETION_POSSIBLE
      0 signals → should not reach the detector; caller error
    """
    efta_gap: bool = False
    discovery_log_entry: bool = False
    document_stamp_gap: bool = False

    @property
    def signal_count(self) -> int:
        return sum([self.efta_gap, self.discovery_log_entry, self.document_stamp_gap])

    @property
    def derived_flag(self) -> DeletionFlag:
        """
        Derive the appropriate DeletionFlag from signal count.
        Does not consider government-acknowledged flags — those are
        set explicitly via create_acknowledged_withholding().
        """
        count = self.signal_count
        if count >= 3:
            return DeletionFlag.DELETION_CONFIRMED
        elif count == 2:
            return DeletionFlag.DELETION_SUSPECTED
        elif count == 1:
            return DeletionFlag.DELETION_POSSIBLE
        else:
            raise ValueError(
                "DetectionSignals.derived_flag called with zero signals. "
                "At least one signal must be True to create a deletion finding."
            )


# ---------------------------------------------------------------------------
# 302-specific partial delivery detection
# ---------------------------------------------------------------------------

@dataclass
class FBI302SeriesResult:
    """
    Result of checking a set of FBI 302s from the same interview series
    for partial delivery.

    FBI 302s have a rigid structure: one form per interview, with a
    series number tying related interviews together. When some 302s
    from a series are released and others are not, the pattern is
    detectable and significant.

    Fields
    ------
    series_identifier   The FBI case/series number linking these 302s.
    released_ids        Document identifiers of 302s that were released.
    withheld_ids        Document identifiers of 302s that were not released.
    total_expected      Total number of 302s expected in this series,
                        if determinable from the index or discovery log.
    is_selective        True if at least one 302 was released AND at least
                        one was not — the defining pattern of selective withholding.
    notes               Free-text description for human reviewers.
    """
    series_identifier: str
    released_ids: list[str] = field(default_factory=list)
    withheld_ids: list[str] = field(default_factory=list)
    total_expected: Optional[int] = None
    notes: Optional[str] = None

    @property
    def is_selective(self) -> bool:
        return len(self.released_ids) > 0 and len(self.withheld_ids) > 0

    @property
    def release_rate(self) -> Optional[float]:
        """Fraction of expected 302s that were released. None if total unknown."""
        if not self.total_expected:
            return None
        return len(self.released_ids) / self.total_expected


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_deletion_finding(
    document_identifiers: list[str],
    signals: DetectionSignals,
    acknowledgment_source: str,
    acknowledgment_date: str,
    stated_reason: Optional[str] = None,
    notes: Optional[str] = None,
) -> WithholdingRecord:
    """
    Create a WithholdingRecord for an evidence-graded deletion finding.

    Derives the DeletionFlag from the provided DetectionSignals.
    Use create_acknowledged_withholding() instead when the government
    has explicitly confirmed the document's existence.

    Raises ValueError if no signals are True (caller error).

    Constitution reference: Principle IV — Gaps Are Facts.
    """
    flag = signals.derived_flag

    # Evidence-graded flags map to the two allowed WithholdingRecord flags
    # by upgrading them: DELETION_* findings without government acknowledgment
    # are stored with their evidence flag directly. We allow this by relaxing
    # the WithholdingRecord validation for evidence-graded records.
    #
    # Implementation note: WithholdingRecord currently requires WITHHELD_*
    # flags. For evidence-graded findings we store them as DeletionRecord
    # objects (see below) rather than forcing them through WithholdingRecord.
    # This factory is the authoritative creation path — do not bypass it.

    return DeletionRecord(
        record_id=str(uuid.uuid4()),
        document_identifiers=document_identifiers,
        deletion_flag=flag,
        signals=signals,
        acknowledgment_source=acknowledgment_source,
        acknowledgment_date=acknowledgment_date,
        stated_reason=stated_reason,
        notes=notes,
    )


def create_acknowledged_withholding(
    document_identifiers: list[str],
    acknowledgment_source: str,
    acknowledgment_date: str,
    stated_reason: Optional[str] = None,
    expected_release_date: Optional[str] = None,
    sibling_document_ids: Optional[list[str]] = None,
    subject_entities: Optional[list[dict]] = None,
    notes: Optional[str] = None,
) -> WithholdingRecord:
    """
    Create a WithholdingRecord for a government-acknowledged withholding.

    Use this factory when an authoritative source has confirmed the document
    exists but is being withheld. The DeletionFlag is derived automatically:
    - WITHHELD_SELECTIVELY if sibling_document_ids is non-empty
    - WITHHELD_ACKNOWLEDGED otherwise

    Examples
    --------
    # DOJ confirms 47,635 files offline (WSJ, March 2026)
    create_acknowledged_withholding(
        document_identifiers=["EFTA-RANGE-DS_OFFLINE_001", ...],
        acknowledgment_source="DOJ statement to Wall Street Journal, March 2026",
        acknowledgment_date="2026-03-01",
        stated_reason="Additional review required prior to release",
        expected_release_date="2026-03-07",
        notes="47,635 files confirmed offline by DOJ after WSJ inquiry"
    )

    # Three Trump-related 302s withheld while one sibling 302 was released
    create_acknowledged_withholding(
        document_identifiers=["302-TRUMP-SERIES-002", "302-TRUMP-SERIES-003",
                              "302-TRUMP-SERIES-004"],
        acknowledgment_source="DOJ statement to Wall Street Journal, March 2026",
        acknowledgment_date="2026-03-01",
        stated_reason="Described by DOJ as unverified and baseless",
        sibling_document_ids=["302-TRUMP-SERIES-001"],
        subject_entities=[
            {"type": "PERSON", "name": "Donald J. Trump", "role": "SUBJECT_OF_INTERVIEW"},
        ],
        notes="Series of four interviews; one released, three withheld. "
              "DOJ characterised unverified claims as baseless."
    )

    Constitution reference: Principle IV — Gaps Are Facts.
    Hard Limit 1 — subject_entities containing victim identities must be
                   reviewed before this record enters any public query path.
    Hard Limit 3 — DOJ characterisation of claims as 'baseless' is reported
                   as a stated position, not adopted as a confirmed fact.
    """
    flag = (
        DeletionFlag.WITHHELD_SELECTIVELY
        if sibling_document_ids
        else DeletionFlag.WITHHELD_ACKNOWLEDGED
    )

    return WithholdingRecord(
        record_id=str(uuid.uuid4()),
        document_identifiers=document_identifiers,
        deletion_flag=flag,
        acknowledgment_source=acknowledgment_source,
        acknowledgment_date=acknowledgment_date,
        stated_reason=stated_reason,
        expected_release_date=expected_release_date,
        sibling_document_ids=sibling_document_ids or [],
        subject_entities=subject_entities or [],
        notes=notes,
    )


def check_302_series(
    series_identifier: str,
    all_series_ids: list[str],
    released_ids: list[str],
    total_expected: Optional[int] = None,
    notes: Optional[str] = None,
) -> FBI302SeriesResult:
    """
    Check a set of FBI 302s from the same interview series for selective withholding.

    Parameters
    ----------
    series_identifier   FBI case/series number linking these 302s.
    all_series_ids      All document IDs known to belong to this series
                        (from the index, discovery log, or internal references).
    released_ids        Document IDs that were released in the corpus.
    total_expected      Total expected if known; None if unknown.
    notes               Optional free-text notes.

    Returns an FBI302SeriesResult. If is_selective is True, the caller
    should create a WITHHELD_SELECTIVELY WithholdingRecord for the
    withheld_ids, citing the released_ids as sibling_document_ids.
    """
    released_set = set(released_ids)
    withheld = [d for d in all_series_ids if d not in released_set]

    result = FBI302SeriesResult(
        series_identifier=series_identifier,
        released_ids=released_ids,
        withheld_ids=withheld,
        total_expected=total_expected,
        notes=notes,
    )

    if result.is_selective:
        logger.warning(
            "Selective 302 withholding detected: series %s — "
            "%d released, %d withheld",
            series_identifier,
            len(released_ids),
            len(withheld),
        )

    return result


# ---------------------------------------------------------------------------
# DeletionRecord — evidence-graded counterpart to WithholdingRecord
# ---------------------------------------------------------------------------

@dataclass
class DeletionRecord:
    """
    A deletion finding derived from evidence signals, as distinct from
    a government-acknowledged WithholdingRecord.

    Stored in DynamoDB alongside WithholdingRecords. Both types are
    returned by deletion queries; DeletionFlag keeps them distinguished.

    Fields mirror WithholdingRecord where applicable, with the addition
    of the DetectionSignals that produced the flag.

    Constitution reference: Principle IV — Gaps Are Facts.
    Principle V — Every Output Is Accountable (signals are citable provenance).
    """
    record_id: str
    document_identifiers: list[str]
    deletion_flag: DeletionFlag
    signals: DetectionSignals
    acknowledgment_source: str     # source of the index/discovery log data
    acknowledgment_date: str       # ISO 8601 date gap was identified

    stated_reason: Optional[str] = None
    notes: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: __import__('datetime').datetime.utcnow().isoformat() + "Z"
    )

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "document_identifiers": self.document_identifiers,
            "deletion_flag": self.deletion_flag.value,
            "signals": {
                "efta_gap": self.signals.efta_gap,
                "discovery_log_entry": self.signals.discovery_log_entry,
                "document_stamp_gap": self.signals.document_stamp_gap,
                "signal_count": self.signals.signal_count,
            },
            "acknowledgment_source": self.acknowledgment_source,
            "acknowledgment_date": self.acknowledgment_date,
            "stated_reason": self.stated_reason,
            "notes": self.notes,
            "created_at": self.created_at,
        }


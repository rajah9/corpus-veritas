"""
pipeline/version_comparator.py
Milestone 6: Cross-version manifest comparison for retroactive deletion detection.

Compares two DOJ release manifests (a prior release and a current release)
to identify documents that appeared in the prior release but are absent from
the current one. Absence of a previously-present document in a subsequent
release is strong evidence of retroactive deletion and receives
DELETION_CONFIRMED status.

This is methodologically distinct from gap detection within a single release.
A within-release gap (EFTA number in the DOJ index but no matching document)
may have many explanations. A cross-release disappearance (document present
in release N, absent in release N+1) is harder to explain innocuously and
carries higher confidence.

Constitution reference: Principle IV -- Gaps Are Facts.
The cross-version comparator produces DELETION_CONFIRMED findings because:
  1. The document was present (EFTA gap signal -- the prior release itself
     is evidence the number was assigned to real content)
  2. The document is now absent (the current release is the
     discovery_log_entry signal)
  3. The disappearance is documented by comparing signed manifests
     (document_stamp_gap signal -- the version delta is a citable structural fact)

All three signals are present when a document disappears between releases,
making DELETION_CONFIRMED the appropriate flag.

See docs/ARCHITECTURE.md § Deletion Detection Module.
See pipeline/manifest_loader.py for manifest loading.
See pipeline/deletion_detector.py for DeletionRecord and DetectionSignals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.deletion_detector import (
    DetectionSignals,
    DeletionRecord,
    create_deletion_finding,
)
from pipeline.manifest_loader import ManifestLoadResult, ManifestRecord
from pipeline.models import DeletionFlag

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RetroactiveDeletion:
    """
    A document present in a prior release but absent from the current one.

    Fields
    ------
    efta_number      EFTA number of the disappeared document.
    prior_version    Release version in which the document was present.
    current_version  Release version from which the document is absent.
    prior_record     ManifestRecord from the prior release (for title, URL, dataset).
    deletion_record  DeletionRecord with DELETION_CONFIRMED flag.
    """
    efta_number:     str
    prior_version:   str
    current_version: str
    prior_record:    ManifestRecord
    deletion_record: DeletionRecord


@dataclass
class ComparisonResult:
    """
    Result of comparing two release manifests.

    Fields
    ------
    prior_version           Release version used as the baseline.
    current_version         Release version compared against baseline.
    compared_at             ISO 8601 UTC timestamp of comparison.
    retroactive_deletions   Documents in prior but absent from current.
    new_additions           EFTA numbers in current but absent from prior.
    total_prior             Number of records in the prior manifest.
    total_current           Number of records in the current manifest.
    """
    prior_version:         str
    current_version:       str
    compared_at:           str
    retroactive_deletions: list[RetroactiveDeletion]
    new_additions:         list[str]
    total_prior:           int
    total_current:         int

    @property
    def deletion_count(self) -> int:
        return len(self.retroactive_deletions)

    @property
    def addition_count(self) -> int:
        return len(self.new_additions)

    @property
    def net_change(self) -> int:
        """Positive = net additions, negative = net removals."""
        return self.addition_count - self.deletion_count


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_manifests(
    prior: ManifestLoadResult,
    current: ManifestLoadResult,
) -> ComparisonResult:
    """
    Compare two manifests and identify retroactive deletions.

    Documents present in `prior` but absent from `current` receive
    DELETION_CONFIRMED DeletionRecords because all three detection signals
    are satisfied (see module docstring). Documents in `current` but not
    in `prior` are recorded as new_additions (informational only).

    Parameters
    ----------
    prior   : ManifestLoadResult for the earlier release (baseline).
    current : ManifestLoadResult for the later release to compare against.

    Returns
    -------
    ComparisonResult with retroactive_deletions and new_additions lists.

    Constitution reference: Principle IV -- Gaps Are Facts.
    """
    disappeared = prior.efta_numbers - current.efta_numbers
    added       = current.efta_numbers - prior.efta_numbers

    # Build lookup for prior records by EFTA number
    prior_lookup: dict[str, ManifestRecord] = {
        r.efta_number: r for r in prior.records
    }

    retroactive_deletions: list[RetroactiveDeletion] = []
    now = datetime.now(timezone.utc).isoformat()

    for efta_num in sorted(disappeared, key=lambda n: int(n) if n.isdigit() else 0):
        prior_record = prior_lookup.get(efta_num)
        if prior_record is None:
            continue  # shouldn't happen but guard defensively

        # All three signals are present for a cross-version disappearance
        signals = DetectionSignals(
            efta_gap=True,
            discovery_log_entry=True,
            document_stamp_gap=True,
        )

        deletion_record = create_deletion_finding(
            document_identifiers=[efta_num],
            signals=signals,
            acknowledgment_source=(
                f"Cross-version manifest comparison: present in release "
                f"'{prior.release_version}', absent from release "
                f"'{current.release_version}'"
            ),
            acknowledgment_date=now[:10],  # YYYY-MM-DD
            notes=(
                f"Title in prior release: {prior_record.title!r}. "
                f"Dataset: {prior_record.dataset!r}."
            ) if prior_record.title or prior_record.dataset else None,
        )

        retroactive_deletions.append(RetroactiveDeletion(
            efta_number=efta_num,
            prior_version=prior.release_version,
            current_version=current.release_version,
            prior_record=prior_record,
            deletion_record=deletion_record,
        ))

    logger.info(
        "Manifest comparison: prior=%s (%d) current=%s (%d) "
        "retroactive_deletions=%d new_additions=%d",
        prior.release_version, prior.record_count,
        current.release_version, current.record_count,
        len(retroactive_deletions), len(added),
    )

    return ComparisonResult(
        prior_version=prior.release_version,
        current_version=current.release_version,
        compared_at=now,
        retroactive_deletions=retroactive_deletions,
        new_additions=sorted(added, key=lambda n: int(n) if n.isdigit() else 0),
        total_prior=prior.record_count,
        total_current=current.record_count,
    )


def filter_by_dataset(
    result: ComparisonResult,
    dataset: str,
) -> list[RetroactiveDeletion]:
    """
    Return only the retroactive deletions from a specific dataset.

    Useful for isolating DS9 deletions (the most-studied dataset)
    from a full comparison result.

    Parameters
    ----------
    result  : ComparisonResult from compare_manifests().
    dataset : Dataset identifier string e.g. "DS09".
    """
    return [
        d for d in result.retroactive_deletions
        if d.prior_record.dataset.upper() == dataset.upper()
    ]

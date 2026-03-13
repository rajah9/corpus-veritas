"""
rag/convergence_checker.py
Layer 3: Multi-source convergence checking.

Standalone module -- the caller invokes check_convergence() explicitly
after route_query() returns. It is deliberately not called automatically
inside route_query() so that convergence logic remains separately testable
and auditable.

Convergence rule (from ARCHITECTURE.md para Layer 3)
-----------------------------------------------------
  Count = 1              → SINGLE_SOURCE (inference must not be surfaced
                           about a living individual)
  Count ≥ 2              → CORROBORATED
  Count ≥ 3 with
    document_type diversity → CONFIRMED possible

Independent sources
-------------------
Two chunks are independent if they have different document_uuid values
AND different sequence number ranges. Sequence range independence means
their sequence_number values differ by more than a nominal threshold --
this prevents a single document split into many chunks from being counted
as multiple independent sources.

When sequence_number is absent for a chunk, document_uuid alone is used
as the independence criterion (conservative -- may undercount sources
when sequence numbers are incomplete).

Typical caller pattern
----------------------
    result = route_query(request, opensearch_client, bedrock_client)

    if request.query_type == QueryType.INFERENCE:
        conv = check_convergence(result)
        if not conv.meets_inference_threshold:
            # Suppress or downgrade before surfacing to user
            answer = conv.suppression_message
        else:
            answer = result.answer

See CONSTITUTION.md Article III Hard Limit 2.
See docs/ARCHITECTURE.md para Layer 3 -- Multi-Source Convergence Rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from pipeline.models import ConfidenceTier
from rag.query_router import RetrievalResult

logger = logging.getLogger(__name__)

# Minimum number of independent sources required to surface an inference
# about a living individual (Constitution Hard Limit 2).
INFERENCE_THRESHOLD: int = 2

# Minimum independent sources AND document_type diversity required to
# reach CONFIRMED tier from convergence alone.
CONFIRMED_SOURCE_THRESHOLD: int = 3

# Sequence number difference below which two chunks are considered to
# originate from the same document (i.e. not independent). This handles
# the case where chunks from the same document share nearly adjacent
# EFTA numbers. Set conservatively -- false negatives (undercounting
# independence) are safer than false positives (overcounting it).
_SEQUENCE_ADJACENCY_THRESHOLD: int = 100


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """
    Output of check_convergence().

    Fields
    ------
    independent_source_count
                    Number of independent source documents identified
                    in the retrieved chunks.

    independent_document_uuids
                    List of document_uuid values counted as independent.
                    Useful for audit logging and for presenting the
                    evidence chain to the user.

    document_types_present
                    Set of DocumentType values found across the independent
                    sources. Diversity of document types is required for
                    CONFIRMED tier (ARCHITECTURE.md convergence rule).

    convergence_tier
                    The ConfidenceTier derived from the convergence count
                    and document_type diversity. This is the convergence-
                    derived tier -- it may be further constrained by the
                    lowest_tier in the RetrievalResult (the weakest actual
                    chunk tier always wins).

    meets_inference_threshold
                    True if independent_source_count >= INFERENCE_THRESHOLD.
                    Constitution Hard Limit 2: callers must check this before
                    surfacing an INFERENCE query answer about a living
                    individual. If False, use suppression_message instead.

    suppression_message
                    A ready-to-surface message explaining why the inference
                    could not be returned. Only meaningful when
                    meets_inference_threshold is False. Contains the source
                    count and the threshold so the user understands what
                    evidence would be required.
    """
    independent_source_count:    int
    independent_document_uuids:  list[str]
    document_types_present:      list[str]
    convergence_tier:            str
    meets_inference_threshold:   bool
    suppression_message:         str = ""


# ---------------------------------------------------------------------------
# Independence logic
# ---------------------------------------------------------------------------

def _extract_sequence_int(seq: Optional[str]) -> Optional[int]:
    """
    Parse a sequence_number string to int for range comparison.
    Returns None if the value is absent or non-numeric (e.g. Bates stamps).
    """
    if seq is None:
        return None
    try:
        return int(seq)
    except (ValueError, TypeError):
        return None


def _are_independent(chunk_a: dict, chunk_b: dict) -> bool:
    """
    Return True if chunk_a and chunk_b originate from independent documents.

    Two chunks are independent when:
      1. They have different document_uuid values, AND
      2. Their sequence_number values differ by more than
         _SEQUENCE_ADJACENCY_THRESHOLD (when both are numeric), OR
         either chunk has no sequence_number (fallback: uuid alone).

    The sequence adjacency check prevents a single large document that was
    split across multiple classifier runs (and thus received multiple UUIDs)
    from being counted as multiple independent sources.
    """
    if chunk_a.get("document_uuid") == chunk_b.get("document_uuid"):
        return False

    seq_a = _extract_sequence_int(chunk_a.get("sequence_number"))
    seq_b = _extract_sequence_int(chunk_b.get("sequence_number"))

    # If either sequence number is missing or non-numeric, accept uuid
    # difference as sufficient (conservative but unavoidable).
    if seq_a is None or seq_b is None:
        return True

    return abs(seq_a - seq_b) > _SEQUENCE_ADJACENCY_THRESHOLD


def _collect_independent_sources(chunks: list[dict]) -> list[dict]:
    """
    Return the subset of chunks that represent independent sources.

    Uses a greedy scan: for each chunk, check whether it is independent
    from all already-selected chunks. If so, add it to the set.
    The first chunk from each independent document is retained; subsequent
    chunks from the same document are deduplicated.
    """
    independent: list[dict] = []
    for chunk in chunks:
        if all(_are_independent(chunk, selected) for selected in independent):
            independent.append(chunk)
    return independent


# ---------------------------------------------------------------------------
# Tier derivation
# ---------------------------------------------------------------------------

def _derive_convergence_tier(
    source_count: int,
    document_types: list[str],
) -> str:
    """
    Derive ConfidenceTier from source count and document_type diversity.

    Implements the convergence rule from ARCHITECTURE.md para Layer 3:
      count = 1              → SINGLE_SOURCE
      count ≥ 2              → CORROBORATED
      count ≥ 3 + diversity  → CONFIRMED

    Document type diversity means at least 2 distinct document_type values
    are present (e.g. FBI_302 and COURT_FILING together are more diverse
    than two FBI_302s).
    """
    if source_count < 2:
        return ConfidenceTier.SINGLE_SOURCE
    if source_count >= CONFIRMED_SOURCE_THRESHOLD and len(set(document_types)) >= 2:
        return ConfidenceTier.CONFIRMED
    return ConfidenceTier.CORROBORATED


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_convergence(result: RetrievalResult) -> ConvergenceResult:
    """
    Assess multi-source convergence of a RetrievalResult.

    Counts independent source documents in result.chunks, derives the
    convergence tier, and returns a ConvergenceResult the caller uses to
    decide whether to surface an INFERENCE answer.

    This function is query-type agnostic -- it can be called on any
    RetrievalResult. For INFERENCE queries it is mandatory (Constitution
    Hard Limit 2). For PROVENANCE queries it provides useful source
    diversity metadata. For TIMELINE and RELATIONSHIP queries it is
    optional.

    Parameters
    ----------
    result : RetrievalResult from route_query().

    Returns
    -------
    ConvergenceResult with source count, tier, and suppression message.

    Constitution reference: Hard Limit 2.
    """
    if not result.chunks:
        logger.info("Convergence check: no chunks -- source count 0.")
        return ConvergenceResult(
            independent_source_count=0,
            independent_document_uuids=[],
            document_types_present=[],
            convergence_tier=ConfidenceTier.SINGLE_SOURCE,
            meets_inference_threshold=False,
            suppression_message=(
                "No documents were found that address this query. "
                "The inference cannot be supported."
            ),
        )

    independent_chunks = _collect_independent_sources(result.chunks)
    source_count = len(independent_chunks)
    uuids = [c.get("document_uuid", "") for c in independent_chunks]
    doc_types = [
        c["document_type"]
        for c in independent_chunks
        if c.get("document_type")
    ]

    tier = _derive_convergence_tier(source_count, doc_types)
    meets_threshold = source_count >= INFERENCE_THRESHOLD

    suppression_message = ""
    if not meets_threshold:
        suppression_message = (
            f"This inference cannot be surfaced. It is supported by "
            f"{source_count} independent source document(s), but "
            f"{INFERENCE_THRESHOLD} are required before an inference "
            f"about a living individual can be returned. "
            f"This finding is recorded as {ConfidenceTier.SINGLE_SOURCE}."
        )

    logger.info(
        "Convergence: %d independent sources, tier=%s, threshold_met=%s",
        source_count, tier, meets_threshold,
    )

    return ConvergenceResult(
        independent_source_count=source_count,
        independent_document_uuids=uuids,
        document_types_present=doc_types,
        convergence_tier=tier,
        meets_inference_threshold=meets_threshold,
        suppression_message=suppression_message,
    )

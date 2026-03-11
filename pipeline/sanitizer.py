"""
sanitizer.py
Layer 1, Sub-Module 1C: PII Sanitization

Every document passes through AWS Comprehend PII detection before chunking.
Anything flagged as a potential victim identity enters a human-review queue
(SQS). Documents do not proceed to the vector store until classified.

Victim-flagged chunks are stored with victim_flag=True and suppressed
from all public query paths by the guardrail layer.

Two levels of flagging
----------------------
victim_flag=True         Hard Limit 1 absolute protection. Triggered when
                         Comprehend detects a NAME entity in proximity to
                         victim-indicator terms, or when a NAME entity
                         co-occurs with an AGE entity under 18. The document
                         is moved to DocumentState.VICTIM_FLAGGED and may
                         not proceed to embedding without human clearance.

requires_human_review    Conservative flag. Triggered when a NAME entity
                         co-occurs with case-adjacent terms (perpetrator names,
                         "alleged", "accuser") but no victim indicator is
                         present. Human reviewer determines final classification.
                         Document is moved to DocumentState.PENDING_REVIEW.

AWS Comprehend limits
---------------------
detect_pii_entities() accepts at most 5,000 UTF-8 bytes per call. Documents
longer than _COMPREHEND_BYTE_LIMIT are split into word-boundary chunks.
Entities and their offsets are adjusted before merging so that BeginOffset /
EndOffset values in PIIDetectionResult reflect positions in the original text.

For production use on large document batches, use the async
start_pii_entities_detection_job() API (supports up to 1 MB per document via
S3). The synchronous implementation here is correct for documents up to
~100 KB; a TODO marks the async upgrade path.

See CONSTITUTION.md Principle I -- Victims Are Not Data.
See CONSTITUTION.md Article III Hard Limit 1.
See docs/ARCHITECTURE.md para Layer 1, Sub-Module 1C.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# AWS region -- override with AWS_REGION env var
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")

# SQS queue URL for the human-review queue -- set SANITIZER_QUEUE_URL in env
SANITIZER_QUEUE_URL: str = os.environ.get("SANITIZER_QUEUE_URL", "")

# Comprehend synchronous API byte limit (conservative -- actual limit is 5,000)
_COMPREHEND_BYTE_LIMIT: int = 4_500

# Character window on either side of a detected NAME entity used for
# proximity checks against sensitive terms
_PROXIMITY_WINDOW: int = 150

# Age below which co-occurrence with a NAME entity triggers victim_flag
_MINOR_AGE_THRESHOLD: int = 18

# Terms whose proximity to a NAME entity triggers victim_flag (Hard Limit 1)
_VICTIM_INDICATOR_TERMS: frozenset = frozenset({
    "victim",
    "survivor",
    "trafficked",
    "sexually abused",
    "sexually assaulted",
    "abused",
    "molested",
    "raped",
    "underage",
    "minor",
    "juvenile",
    "teenager",
    "teen",
})

# Terms whose proximity to a NAME entity triggers requires_human_review
_REVIEW_INDICATOR_TERMS: frozenset = frozenset({
    "epstein",
    "maxwell",
    "ghislaine",
    "giuffre",
    "accuser",
    "alleged",
    "witness",
    "complainant",
})

# Number of NAME entities in one document that triggers requires_human_review
# independently of proximity terms (possible participant or victim list)
_MULTI_NAME_REVIEW_THRESHOLD: int = 3


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class PIIDetectionResult:
    """
    Output of detect_pii() for a single document.

    Fields
    ------
    document_uuid         UUID of the document being sanitized.
    pii_entities_detected All PII entities returned by Comprehend, with
                          BeginOffset/EndOffset relative to the original text.
    victim_flag           True if Hard Limit 1 protection is triggered.
                          Document must enter human-review queue and may not
                          proceed to embedding without explicit clearance.
    requires_human_review True if the document contains NAME entities in
                          case-adjacent contexts requiring classification
                          before proceeding. Less severe than victim_flag.
    review_reason         Human-readable explanation of why the flag was set.
                          None if both flags are False.
    """
    document_uuid: str
    pii_entities_detected: list = field(default_factory=list)
    victim_flag: bool = False
    requires_human_review: bool = False
    review_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_bytes: int = _COMPREHEND_BYTE_LIMIT) -> list:
    """
    Split text into word-boundary chunks, each fitting within max_bytes.

    Returns a list of at least one chunk (may be empty string for empty
    input). The caller is responsible for tracking byte offsets when
    merging entity results across chunks.

    TODO (production): Replace with async start_pii_entities_detection_job()
    for documents over ~100 KB.
    """
    chunks: list = []
    current_words: list = []
    current_bytes: int = 0

    for word in text.split():
        word_bytes = len(word.encode("utf-8")) + 1  # +1 for the space
        if current_bytes + word_bytes > max_bytes and current_words:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_bytes = word_bytes
        else:
            current_words.append(word)
            current_bytes += word_bytes

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks or [""]


def _text_near(
    text: str,
    begin_offset: int,
    end_offset: int,
    window: int,
    terms: frozenset,
) -> bool:
    """
    Return True if any term in `terms` appears within `window` characters
    of the entity span [begin_offset, end_offset] in `text`.

    Case-insensitive substring match -- conservative by design.
    """
    start = max(0, begin_offset - window)
    end = min(len(text), end_offset + window)
    snippet = text[start:end].lower()
    return any(term in snippet for term in terms)


def _extract_age_value(text: str, entity: dict) -> Optional[int]:
    """
    Attempt to extract an integer age value from a Comprehend AGE entity span.
    Returns None if no integer can be parsed (e.g. "mid-thirties").
    """
    span = text[entity["BeginOffset"]:entity["EndOffset"]]
    match = re.search(r"\d+", span)
    return int(match.group()) if match else None


def _analyse_entities(
    text: str,
    entities: list,
) -> tuple:
    """
    Apply victim-flagging and human-review logic to a Comprehend entity list.

    Returns (victim_flag: bool, requires_human_review: bool,
             review_reason: Optional[str]).

    Rules (applied in priority order)
    ----------------------------------
    1. NAME entity within _PROXIMITY_WINDOW of a victim-indicator term
       -> victim_flag=True  (Hard Limit 1)

    2. AGE entity with value < _MINOR_AGE_THRESHOLD AND at least one NAME
       entity present anywhere in the document
       -> victim_flag=True  (Hard Limit 1 -- possible minor identification)

    3. NAME entity within _PROXIMITY_WINDOW of a review-indicator term
       -> requires_human_review=True

    4. >= _MULTI_NAME_REVIEW_THRESHOLD NAME entities in the document
       -> requires_human_review=True  (possible participant/victim list)

    victim_flag, once set, is never reversed by later conditions.
    """
    name_entities = [e for e in entities if e.get("Type") == "NAME"]
    age_entities  = [e for e in entities if e.get("Type") == "AGE"]

    victim_flag = False
    review_flag = False
    reasons: list = []

    # Rule 1: NAME near victim-indicator terms
    for entity in name_entities:
        if _text_near(
            text,
            entity["BeginOffset"],
            entity["EndOffset"],
            _PROXIMITY_WINDOW,
            _VICTIM_INDICATOR_TERMS,
        ):
            victim_flag = True
            reasons.append(
                f"NAME entity at offset {entity['BeginOffset']} "
                "appears near victim-indicator term"
            )
            break  # one is sufficient to trigger Hard Limit 1

    # Rule 2: Minor AGE co-occurrence with NAME
    if not victim_flag and name_entities:
        for age_entity in age_entities:
            age_value = _extract_age_value(text, age_entity)
            if age_value is not None and age_value < _MINOR_AGE_THRESHOLD:
                victim_flag = True
                span = text[age_entity["BeginOffset"]:age_entity["EndOffset"]]
                reasons.append(
                    f"AGE entity '{span}' (value {age_value}) co-occurs with "
                    "NAME entity -- possible minor identification"
                )
                break

    # Rule 3: NAME near review-indicator terms (only if not already victim-flagged)
    if not victim_flag:
        for entity in name_entities:
            if _text_near(
                text,
                entity["BeginOffset"],
                entity["EndOffset"],
                _PROXIMITY_WINDOW,
                _REVIEW_INDICATOR_TERMS,
            ):
                review_flag = True
                reasons.append(
                    f"NAME entity at offset {entity['BeginOffset']} "
                    "appears near case-adjacent term -- human review required"
                )
                break

    # Rule 4: High name density
    if not victim_flag and len(name_entities) >= _MULTI_NAME_REVIEW_THRESHOLD:
        review_flag = True
        reasons.append(
            f"Document contains {len(name_entities)} NAME entities -- "
            "possible participant or victim list, human review required"
        )

    review_reason = "; ".join(reasons) if reasons else None
    return victim_flag, review_flag, review_reason


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_pii(
    document_uuid: str,
    text: str,
    comprehend_client=None,
) -> PIIDetectionResult:
    """
    Run AWS Comprehend PII entity detection on document text.

    Parameters
    ----------
    document_uuid     : UUID of the document under analysis.
    text              : Raw document text (UTF-8). Documents longer than
                        _COMPREHEND_BYTE_LIMIT bytes are automatically split
                        into word-boundary chunks and results merged.
    comprehend_client : Optional pre-constructed boto3 Comprehend client.
                        If None, a client is created using AWS_REGION.
                        Inject a MagicMock for testing.

    Returns
    -------
    PIIDetectionResult with victim_flag and requires_human_review set
    per the entity analysis rules in _analyse_entities().

    Error handling
    --------------
    Does not raise. Comprehend API errors are caught, logged, and returned
    as requires_human_review=True so the document enters human review
    rather than proceeding unexamined.

    Constitution reference: Principle I -- Victims Are Not Data.
    Hard Limit 1 -- victim identities will never be surfaced.
    """
    if comprehend_client is None:
        import boto3
        comprehend_client = boto3.client("comprehend", region_name=AWS_REGION)

    chunks = _chunk_text(text)
    all_entities: list = []
    char_offset: int = 0

    for chunk in chunks:
        try:
            response = comprehend_client.detect_pii_entities(
                Text=chunk,
                LanguageCode="en",
            )
            for entity in response.get("Entities", []):
                adjusted = dict(entity)
                adjusted["BeginOffset"] = entity["BeginOffset"] + char_offset
                adjusted["EndOffset"]   = entity["EndOffset"]   + char_offset
                all_entities.append(adjusted)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Comprehend detect_pii_entities failed for document %s "
                "(chunk at offset %d): %s",
                document_uuid, char_offset, exc,
            )
            return PIIDetectionResult(
                document_uuid=document_uuid,
                pii_entities_detected=[],
                victim_flag=False,
                requires_human_review=True,
                review_reason=(
                    f"Comprehend API error -- manual PII review required: {exc}"
                ),
            )

        # +1 accounts for the space consumed by the split boundary
        char_offset += len(chunk) + 1

    victim_flag, review_flag, review_reason = _analyse_entities(text, all_entities)

    logger.info(
        "PII detection complete for %s: %d entities, victim_flag=%s, review=%s",
        document_uuid, len(all_entities), victim_flag, review_flag,
    )

    return PIIDetectionResult(
        document_uuid=document_uuid,
        pii_entities_detected=all_entities,
        victim_flag=victim_flag,
        requires_human_review=review_flag,
        review_reason=review_reason,
    )


def queue_for_human_review(
    result: PIIDetectionResult,
    sqs_client=None,
    queue_url: Optional[str] = None,
) -> None:
    """
    Send a flagged document to the SQS human-review queue.

    The message body is JSON containing enough context for a reviewer to
    locate and classify the document. Entity text values are NEVER included
    in the message -- only offsets -- to avoid transmitting PII outside
    the secure processing path.

    Parameters
    ----------
    result      : PIIDetectionResult from detect_pii().
    sqs_client  : Optional pre-constructed boto3 SQS client.
                  If None, a client is created using AWS_REGION.
                  Inject a MagicMock for testing.
    queue_url   : SQS queue URL. Falls back to SANITIZER_QUEUE_URL env var.

    Error handling
    --------------
    Does not raise. SQS send failures are logged at ERROR level.

    Constitution reference: Hard Limit 1.
    """
    resolved_url = queue_url or SANITIZER_QUEUE_URL
    if not resolved_url:
        logger.error(
            "Cannot queue document %s for review: "
            "SANITIZER_QUEUE_URL not set and no queue_url provided.",
            result.document_uuid,
        )
        return

    if sqs_client is None:
        import boto3
        sqs_client = boto3.client("sqs", region_name=AWS_REGION)

    message_body = {
        "document_uuid":        result.document_uuid,
        "victim_flag":          result.victim_flag,
        "requires_human_review": result.requires_human_review,
        "review_reason":        result.review_reason,
        "pii_entity_count":     len(result.pii_entities_detected),
        # Offsets only -- no entity text values to avoid PII transmission
        "entity_offsets": [
            {
                "type":         e.get("Type"),
                "begin_offset": e.get("BeginOffset"),
                "end_offset":   e.get("EndOffset"),
                "score":        e.get("Score"),
            }
            for e in result.pii_entities_detected
        ],
        "queued_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        sqs_client.send_message(
            QueueUrl=resolved_url,
            MessageBody=json.dumps(message_body),
            MessageAttributes={
                "document_uuid": {
                    "StringValue": result.document_uuid,
                    "DataType": "String",
                },
                "victim_flag": {
                    "StringValue": str(result.victim_flag).lower(),
                    "DataType": "String",
                },
            },
        )
        logger.info(
            "Document %s queued for human review (victim_flag=%s): %s",
            result.document_uuid, result.victim_flag, result.review_reason,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to queue document %s for human review: %s",
            result.document_uuid, exc,
        )


def sanitize_document(
    document_uuid: str,
    text: str,
    comprehend_client=None,
    sqs_client=None,
    queue_url: Optional[str] = None,
) -> PIIDetectionResult:
    """
    Orchestrate PII detection and human-review queuing for one document.

    This is the primary entry point for Sub-Module 1C. Callers (classifier.py)
    inspect the returned PIIDetectionResult to determine DocumentState.

    State mapping (applied by classifier.py)
    -----------------------------------------
    victim_flag=True              -> DocumentState.VICTIM_FLAGGED
    requires_human_review=True    -> DocumentState.PENDING_REVIEW
    both False                    -> DocumentState.SANITIZED

    Parameters
    ----------
    document_uuid     : UUID of the document.
    text              : Raw document text.
    comprehend_client : Optional boto3 Comprehend client (for testing).
    sqs_client        : Optional boto3 SQS client (for testing).
    queue_url         : Optional SQS queue URL override.

    Returns
    -------
    PIIDetectionResult.

    Constitution reference: Principle I. Hard Limit 1.
    """
    result = detect_pii(
        document_uuid=document_uuid,
        text=text,
        comprehend_client=comprehend_client,
    )

    if result.victim_flag or result.requires_human_review:
        queue_for_human_review(
            result=result,
            sqs_client=sqs_client,
            queue_url=queue_url,
        )

    return result
"""
rag/guardrail.py
Layer 5: Ethical guardrail layer.

Every synthesised answer passes through four checks before delivery.
apply_guardrail() is the single entry point -- it runs all four checks
in order, writes the audit log, and returns a GuardrailResult. If the
audit log write fails, AuditLogFailure is raised and the response must
not be delivered.

Four checks in order
--------------------
Check 1 -- Victim identity scan
    Scans the response text for victim-flagged entity names (canonical
    names and surface forms). Any match is replaced with
    "[protected identity]". victim_scan_triggered=True is set on the
    GuardrailResult and logged. This check fires even when the OpenSearch
    victim_flag filter has already suppressed victim-adjacent chunks --
    the synthesis model may still produce victim-referencing text through
    inference or indirect reference. Defence in depth.

    Constitution Hard Limit 1.

Check 2 -- Inference threshold check
    Applies only to INFERENCE queries. If convergence_applied is False
    on the RetrievalResult, check_convergence() is called (or a supplied
    ConvergenceResult is used). If meets_inference_threshold is False,
    the answer is replaced with the suppression_message. This check is
    the final enforcement of the convergence rule -- if the caller forgot
    to run convergence_checker before calling apply_guardrail(), the
    guardrail catches it here.

    Constitution Hard Limit 2. Principle III.

Check 3 -- Confidence calibration check
    Scans the response for CONFIRMED-tier language markers ("confirmed",
    "definitively", "it is certain", "proves", etc.) when the lowest
    confidence tier in the retrieved chunks is below CONFIRMED. Matching
    phrases are replaced with hedged equivalents. confidence_violation=True
    is set on the GuardrailResult.

    This check does not suppress the answer -- it corrects language.
    Suppression would be too aggressive: a useful CORROBORATED answer
    should not be withheld because the model used one over-confident word.
    The correction is logged so the prompt can be tuned to reduce future
    violations.

    Constitution Hard Limit 3. Principle II.

Check 4 -- Audit log write
    Writes an AuditLogEntry to both CloudWatch Logs and S3 BEFORE
    returning GuardrailResult. If write_audit_log() raises AuditLogFailure,
    apply_guardrail() propagates it immediately. The caller must not
    deliver the response.

    The audit entry captures: query text and type, retrieved chunk UUIDs,
    provenance tags, confidence tiers, the original answer (before checks),
    the safe answer (after checks), and which checks triggered.

    Constitution Hard Limit 5. Principle V.

Victim name sources
-------------------
The caller supplies victim_entity_names -- a list of canonical names and
surface forms to scan for. The guardrail also always scans for names in
entity_resolver._KNOWN_VICTIM_CANONICAL_NAMES. The caller should augment
this with any victim_flag=True entities retrieved from DynamoDB
(ner_extractor.query_entities_by_type or gsi-victim-flag) for the most
complete coverage.

See CONSTITUTION.md Article III Hard Limits 1, 2, 3, 5.
See CONSTITUTION.md Principles II, III, V.
See docs/ARCHITECTURE.md para Layer 5 -- Ethical Guardrail Layer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.audit_log import AuditLogEntry, AuditLogFailure, write_audit_log
from pipeline.models import ConfidenceTier
from rag.convergence_checker import ConvergenceResult, check_convergence
from rag.query_router import QueryType, RetrievalResult

logger = logging.getLogger(__name__)

# Import known victim names from entity_resolver as a baseline.
# The caller should supplement with DynamoDB gsi-victim-flag query results.
try:
    from graph.entity_resolver import _KNOWN_VICTIM_CANONICAL_NAMES as _BASELINE_VICTIMS
except ImportError:
    _BASELINE_VICTIMS: set[str] = set()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Confidence calibration: CONFIRMED-tier language markers and replacements
# ---------------------------------------------------------------------------

# Phrases that imply certainty. Forbidden when lowest_tier < CONFIRMED.
# Ordered longest-first so more specific phrases match before substrings.
_CONFIRMED_LANGUAGE_PATTERNS: list[tuple[str, str]] = [
    (r"\bit is (?:certain|established) that\b",     "it is suggested that"),
    (r"\bdefinitively (?:proves?|shows?|confirms?)\b", "suggests"),
    (r"\bproves? (?:beyond (?:any )?doubt)?\b",     "suggests"),
    (r"\bconfirms? (?:that|this|the)\b",             "suggests"),
    (r"\bconfirmed\b",                               "corroborated"),
    (r"\bdefinitive(?:ly)?\b",                       "suggestive"),
    (r"\bconclusively\b",                            "tentatively"),
    (r"\bwithout (?:any )?doubt\b",                  "with some evidence"),
    (r"\bit is (?:a )?(?:known )?fact\b",            "documents suggest"),
    (r"\bcertainly\b",                               "possibly"),
    (r"\bcertain\b",                                 "possible"),
]

# Tiers below CONFIRMED where language calibration applies.
_CALIBRATION_REQUIRED_TIERS: set[str] = {
    ConfidenceTier.SPECULATIVE,
    ConfidenceTier.SINGLE_SOURCE,
    ConfidenceTier.INFERRED,
    ConfidenceTier.CORROBORATED,
}


# ---------------------------------------------------------------------------
# GuardrailResult
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """
    Output of apply_guardrail().

    Fields
    ------
    safe_answer         The answer after all guardrail checks. This is
                        the only field that should be delivered to the
                        user. Never deliver original_answer directly.

    original_answer     The synthesised answer before guardrail checks.
                        Recorded in the audit log. Not for user delivery.

    audit_entry_id      UUID of the AuditLogEntry written to CloudWatch
                        and S3. Include in any published output so the
                        audit record can be retrieved.

    victim_scan_triggered
                        True if one or more victim identities were found
                        and suppressed in the response text.

    inference_downgraded
                        True if the inference threshold check replaced
                        the answer with a suppression message.

    confidence_violation
                        True if CONFIRMED-tier language was found and
                        corrected in a sub-CONFIRMED response.

    checks_passed       List of check names that passed without triggering.
    checks_failed       List of check names that triggered and modified
                        the answer.
    """
    safe_answer:           str
    original_answer:       str
    audit_entry_id:        str
    victim_scan_triggered: bool = False
    inference_downgraded:  bool = False
    confidence_violation:  bool = False
    checks_passed:         list[str] = field(default_factory=list)
    checks_failed:         list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Check 1: victim identity scan
# ---------------------------------------------------------------------------

def _build_victim_pattern(names: set[str]) -> Optional[re.Pattern]:
    """
    Compile a case-insensitive regex pattern matching any victim name.
    Returns None if names is empty.
    """
    if not names:
        return None
    # Escape and sort longest-first to prefer specific matches
    escaped = sorted(
        (re.escape(n) for n in names if n.strip()),
        key=len,
        reverse=True,
    )
    if not escaped:
        return None
    return re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)


def check_victim_identity(
    answer: str,
    victim_names: Optional[list[str]] = None,
) -> tuple[str, bool]:
    """
    Scan answer text for victim identity names and suppress them.

    Combines _BASELINE_VICTIMS (from entity_resolver) with any caller-
    supplied names. All matches are replaced with "[protected identity]".

    Parameters
    ----------
    answer       : Response text to scan.
    victim_names : Additional victim canonical names or surface forms
                   supplied by the caller (e.g. from DynamoDB gsi-victim-flag).

    Returns
    -------
    (safe_answer, triggered) where triggered is True if any match was found.
    """
    all_victims: set[str] = set(_BASELINE_VICTIMS)
    if victim_names:
        all_victims.update(victim_names)

    pattern = _build_victim_pattern(all_victims)
    if pattern is None:
        return answer, False

    safe, count = pattern.subn("[protected identity]", answer)
    triggered = count > 0
    if triggered:
        logger.warning(
            "Victim identity scan: %d protected identity reference(s) suppressed.",
            count,
        )
    return safe, triggered


# ---------------------------------------------------------------------------
# Check 2: inference threshold check
# ---------------------------------------------------------------------------

def check_inference_threshold(
    answer: str,
    result: RetrievalResult,
    convergence_result: Optional[ConvergenceResult] = None,
) -> tuple[str, bool]:
    """
    For INFERENCE queries, verify multi-source convergence is satisfied.

    If the RetrievalResult has convergence_applied=False (the router did
    not enforce convergence -- callers of route_query for INFERENCE queries
    are responsible for this), this check runs check_convergence() itself
    as a backstop, or uses the supplied ConvergenceResult.

    If the convergence threshold is not met, the answer is replaced with
    the suppression_message. Constitution Hard Limit 2.

    For non-INFERENCE queries this check is a no-op.

    Parameters
    ----------
    answer             : Current response text (after Check 1).
    result             : RetrievalResult from route_query().
    convergence_result : Pre-computed ConvergenceResult, or None to
                         compute it here.

    Returns
    -------
    (safe_answer, downgraded) where downgraded is True if the answer
    was replaced with the suppression message.
    """
    if result.query.query_type != QueryType.INFERENCE:
        return answer, False

    if convergence_result is None:
        convergence_result = check_convergence(result)

    if not convergence_result.meets_inference_threshold:
        logger.warning(
            "Inference threshold not met: %d independent sources "
            "(threshold=%d). Answer downgraded.",
            convergence_result.independent_source_count,
            2,
        )
        return convergence_result.suppression_message, True

    return answer, False


# ---------------------------------------------------------------------------
# Check 3: confidence calibration
# ---------------------------------------------------------------------------

def check_confidence_calibration(
    answer: str,
    lowest_tier: Optional[str],
) -> tuple[str, bool]:
    """
    Replace CONFIRMED-tier language when the evidence tier is below CONFIRMED.

    Scans for language markers that imply certainty and replaces them with
    hedged equivalents when lowest_tier is SPECULATIVE, SINGLE_SOURCE,
    INFERRED, or CORROBORATED.

    Does not suppress the answer -- corrects language only.
    Constitution Hard Limit 3.

    Parameters
    ----------
    answer      : Current response text (after Checks 1 and 2).
    lowest_tier : Weakest ConfidenceTier from RetrievalResult.lowest_tier.

    Returns
    -------
    (safe_answer, violated) where violated is True if any language was
    corrected.
    """
    if lowest_tier not in _CALIBRATION_REQUIRED_TIERS:
        return answer, False

    safe = answer
    violated = False
    for pattern, replacement in _CONFIRMED_LANGUAGE_PATTERNS:
        safe, count = re.subn(pattern, replacement, safe, flags=re.IGNORECASE)
        if count > 0:
            violated = True
            logger.warning(
                "Confidence calibration: pattern '%s' corrected %d time(s) "
                "in %s-tier response.", pattern, count, lowest_tier,
            )

    return safe, violated


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def apply_guardrail(
    result: RetrievalResult,
    convergence_result: Optional[ConvergenceResult] = None,
    victim_entity_names: Optional[list[str]] = None,
    cloudwatch_client=None,
    s3_client=None,
    audit_log_group: str = "",
    audit_bucket: str = "",
) -> GuardrailResult:
    """
    Apply all four guardrail checks to a RetrievalResult.

    Checks run in order:
      1. Victim identity scan
      2. Inference threshold check
      3. Confidence calibration
      4. Audit log write (BEFORE returning -- Hard Limit 5)

    The audit log is written after checks 1-3 so that the audit entry
    captures both the original answer and the safe answer. The write
    happens before returning so that if it fails, the response is never
    delivered.

    Parameters
    ----------
    result              : RetrievalResult from route_query().
    convergence_result  : Optional pre-computed ConvergenceResult. If None
                          and query_type is INFERENCE, computed here.
    victim_entity_names : Additional victim names to scan for (canonical
                          names or surface forms from DynamoDB
                          gsi-victim-flag). Combined with baseline names
                          from entity_resolver.
    cloudwatch_client   : Injectable boto3 CloudWatch Logs client.
    s3_client           : Injectable boto3 S3 client.
    audit_log_group     : CloudWatch log group name.
    audit_bucket        : S3 bucket for audit records.

    Returns
    -------
    GuardrailResult with safe_answer and audit metadata.

    Raises
    ------
    AuditLogFailure if the audit log write fails. The caller must not
    deliver the response. Constitution Hard Limit 5.
    """
    original_answer = result.answer
    safe_answer = original_answer
    checks_passed: list[str] = []
    checks_failed: list[str] = []

    # Check 1: victim identity scan
    safe_answer, victim_triggered = check_victim_identity(
        safe_answer, victim_entity_names
    )
    if victim_triggered:
        checks_failed.append("victim_identity")
    else:
        checks_passed.append("victim_identity")

    # Check 2: inference threshold
    safe_answer, inference_downgraded = check_inference_threshold(
        safe_answer, result, convergence_result
    )
    if inference_downgraded:
        checks_failed.append("inference_threshold")
    else:
        checks_passed.append("inference_threshold")

    # Check 3: confidence calibration
    safe_answer, confidence_violated = check_confidence_calibration(
        safe_answer, result.lowest_tier
    )
    if confidence_violated:
        checks_failed.append("confidence_calibration")
    else:
        checks_passed.append("confidence_calibration")

    # Build audit entry
    chunk_uuids = list({
        c.get("document_uuid", "") for c in result.chunks
        if c.get("document_uuid")
    })
    provenance_tags = list({
        c.get("provenance_tag") for c in result.chunks
        if c.get("provenance_tag")
    })
    confidence_tiers = list({
        c.get("confidence_tier") for c in result.chunks
        if c.get("confidence_tier")
    })

    entry = AuditLogEntry(
        query_text=result.query.query_text,
        query_type=result.query.query_type.value,
        retrieved_at=result.retrieved_at,
        chunk_uuids=chunk_uuids,
        provenance_tags=provenance_tags,
        confidence_tiers=confidence_tiers,
        lowest_tier=result.lowest_tier,
        original_answer=original_answer,
        safe_answer=safe_answer,
        victim_scan_triggered=victim_triggered,
        inference_downgraded=inference_downgraded,
        confidence_violation=confidence_violated,
        convergence_source_count=(
            convergence_result.independent_source_count
            if convergence_result is not None else None
        ),
    )

    # Check 4: audit log write -- MUST succeed before returning
    # AuditLogFailure propagates to caller; response must not be delivered
    write_audit_log(
        entry,
        cloudwatch_client=cloudwatch_client,
        s3_client=s3_client,
        log_group=audit_log_group or "corpus-veritas-audit",
        audit_bucket=audit_bucket,
    )

    logger.info(
        "Guardrail passed entry_id=%s victim=%s inference_downgraded=%s "
        "confidence_violation=%s",
        entry.entry_id, victim_triggered, inference_downgraded,
        confidence_violated,
    )

    return GuardrailResult(
        safe_answer=safe_answer,
        original_answer=original_answer,
        audit_entry_id=entry.entry_id,
        victim_scan_triggered=victim_triggered,
        inference_downgraded=inference_downgraded,
        confidence_violation=confidence_violated,
        checks_passed=checks_passed,
        checks_failed=checks_failed,
    )

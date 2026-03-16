"""
pipeline/audit_log.py
Layer 5: Immutable audit log for every query and response.

Every query this system receives and every response it generates is written
to an immutable audit trail BEFORE the response is delivered to the caller.
If either write fails, AuditLogFailure is raised and the response must not
be delivered. This is Constitution Hard Limit 5 -- the system will never
operate without an active audit log.

Two write targets (belt and suspenders)
----------------------------------------
1. CloudWatch Logs
   Log group : AUDIT_LOG_GROUP  (default: "corpus-veritas-audit")
   Log stream : one stream per UTC date, e.g. "2026-03-16"
   Each log event is a JSON-serialised AuditLogEntry.
   CloudWatch Logs Insights can query the audit trail in real time.

2. S3 (Object Lock, COMPLIANCE mode, 7-year retention)
   Bucket : AUDIT_S3_BUCKET   (set via environment variable)
   Key    : audit/{YYYY-MM-DD}/{entry_id}.json
   Each object is written with Object Lock COMPLIANCE retention.
   S3 provides the immutable long-term archive; CloudWatch provides
   the queryable operational view.

   The audit S3 bucket MUST be created with Object Lock enabled at
   creation time (same constraint as the corpus bucket -- see
   infrastructure/s3.py). Victim-adjacent query details that appear
   in the audit log are protected by COMPLIANCE mode: they cannot be
   deleted even to remediate a privacy issue without waiting out the
   retention period. This is intentional -- the audit trail must be
   tamper-proof even against the system's own operators.

Failure behaviour
-----------------
write_audit_log() attempts both writes. If either fails it raises
AuditLogFailure. The caller (guardrail.apply_guardrail) must propagate
this exception without delivering the response to the user.

This is architecturally different from a best-effort log. The audit log
is not an observer of the system -- it is a prerequisite for delivery.

See CONSTITUTION.md Article III Hard Limit 5.
See CONSTITUTION.md Principle V -- Every Output Is Accountable.
See docs/ARCHITECTURE.md para Layer 5 -- Ethical Guardrail Layer.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
AUDIT_LOG_GROUP: str = os.environ.get("AUDIT_LOG_GROUP", "corpus-veritas-audit")
AUDIT_S3_BUCKET: str = os.environ.get("AUDIT_S3_BUCKET", "")

# Object Lock retention for audit records (7 years, matching corpus bucket)
_AUDIT_RETENTION_YEARS: int = 7


# ---------------------------------------------------------------------------
# Failure exception
# ---------------------------------------------------------------------------

class AuditLogFailure(Exception):
    """
    Raised when the audit log write fails.

    The caller must NOT deliver the query response when this is raised.
    Constitution Hard Limit 5.
    """


# ---------------------------------------------------------------------------
# Audit log entry
# ---------------------------------------------------------------------------

@dataclass
class AuditLogEntry:
    """
    Complete record of one query-response cycle.

    Written to both CloudWatch Logs and S3 before the response is delivered.
    Every field is chosen to support two use cases:
      1. Operational monitoring -- who queried what, when, with what result.
      2. Accountability journalism -- if a story is published based on this
         system's output, the exact chunks, provenance tags, and confidence
         tiers that supported the answer must be retrievable.

    Fields
    ------
    entry_id            UUID for this audit entry. Used as the S3 object key.
    query_text          The user's original query text.
    query_type          QueryType value string.
    retrieved_at        ISO 8601 UTC timestamp from RetrievalResult.
    answered_at         ISO 8601 UTC timestamp of audit log write (just
                        before delivery).
    chunk_uuids         document_uuid values of all retrieved chunks.
    provenance_tags     Distinct provenance tags across retrieved chunks.
    confidence_tiers    Distinct confidence tiers across retrieved chunks.
    lowest_tier         Weakest confidence tier in the retrieved chunks.
    original_answer     The synthesised answer before guardrail checks.
    safe_answer         The answer after guardrail checks (what was / would
                        be delivered). May differ from original_answer if
                        victim suppression or inference downgrade occurred.
    victim_scan_triggered
                        True if the victim identity check found and
                        suppressed one or more protected identities.
    inference_downgraded
                        True if the inference threshold check replaced the
                        answer with a suppression message.
    confidence_violation
                        True if the confidence calibration check found and
                        corrected CONFIRMED-tier language.
    convergence_source_count
                        Number of independent sources from ConvergenceResult,
                        if available. None if convergence was not checked.
    query_type_str      Convenience copy of query_type for log queries.
    """
    entry_id:                str = field(default_factory=lambda: str(uuid.uuid4()))
    query_text:              str = ""
    query_type:              str = ""
    retrieved_at:            str = ""
    answered_at:             str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    chunk_uuids:             list[str] = field(default_factory=list)
    provenance_tags:         list[str] = field(default_factory=list)
    confidence_tiers:        list[str] = field(default_factory=list)
    lowest_tier:             Optional[str] = None
    original_answer:         str = ""
    safe_answer:             str = ""
    victim_scan_triggered:   bool = False
    inference_downgraded:    bool = False
    confidence_violation:    bool = False
    convergence_source_count: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "entry_id":                self.entry_id,
            "query_text":              self.query_text,
            "query_type":              self.query_type,
            "retrieved_at":            self.retrieved_at,
            "answered_at":             self.answered_at,
            "chunk_uuids":             self.chunk_uuids,
            "provenance_tags":         self.provenance_tags,
            "confidence_tiers":        self.confidence_tiers,
            "lowest_tier":             self.lowest_tier,
            "original_answer":         self.original_answer,
            "safe_answer":             self.safe_answer,
            "victim_scan_triggered":   self.victim_scan_triggered,
            "inference_downgraded":    self.inference_downgraded,
            "confidence_violation":    self.confidence_violation,
            "convergence_source_count": self.convergence_source_count,
        }


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _s3_key(entry_id: str, answered_at: str) -> str:
    """Return S3 key: audit/{YYYY-MM-DD}/{entry_id}.json"""
    try:
        date_str = answered_at[:10]  # "2026-03-16" from ISO 8601
    except (IndexError, TypeError):
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"audit/{date_str}/{entry_id}.json"


def _log_stream_name() -> str:
    """Return today's CloudWatch log stream name: YYYY-MM-DD"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _write_to_cloudwatch(
    entry: AuditLogEntry,
    cloudwatch_client,
    log_group: str,
) -> None:
    """
    Write one audit entry to CloudWatch Logs.

    Creates the log group and stream if they do not exist.
    Uses put_log_events with a single event.

    Raises
    ------
    AuditLogFailure if the write fails.
    """
    stream = _log_stream_name()
    event_body = json.dumps(entry.to_dict())

    # Ensure log group exists
    try:
        cloudwatch_client.create_log_group(logGroupName=log_group)
    except Exception:
        pass  # Already exists -- ignore

    # Ensure log stream exists
    try:
        cloudwatch_client.create_log_stream(
            logGroupName=log_group, logStreamName=stream
        )
    except Exception:
        pass  # Already exists -- ignore

    timestamp_ms = int(
        datetime.now(timezone.utc).timestamp() * 1000
    )

    try:
        cloudwatch_client.put_log_events(
            logGroupName=log_group,
            logStreamName=stream,
            logEvents=[{"timestamp": timestamp_ms, "message": event_body}],
        )
        logger.debug("Audit entry %s written to CloudWatch.", entry.entry_id)
    except Exception as exc:
        raise AuditLogFailure(
            f"CloudWatch audit write failed for entry {entry.entry_id}: {exc}"
        ) from exc


def _write_to_s3(
    entry: AuditLogEntry,
    s3_client,
    bucket: str,
) -> None:
    """
    Write one audit entry to S3 with Object Lock COMPLIANCE retention.

    Raises
    ------
    AuditLogFailure if the write fails or bucket is not configured.
    """
    if not bucket:
        raise AuditLogFailure(
            "AUDIT_S3_BUCKET is not set. Cannot write audit log to S3. "
            "Constitution Hard Limit 5: response must not be delivered."
        )

    key = _s3_key(entry.entry_id, entry.answered_at)
    body = json.dumps(entry.to_dict(), indent=2).encode("utf-8")
    retain_until = datetime.now(timezone.utc) + timedelta(
        days=_AUDIT_RETENTION_YEARS * 365
    )

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ObjectLockMode="COMPLIANCE",
            ObjectLockRetainUntilDate=retain_until,
        )
        logger.debug("Audit entry %s written to s3://%s/%s.", entry.entry_id, bucket, key)
    except Exception as exc:
        raise AuditLogFailure(
            f"S3 audit write failed for entry {entry.entry_id} "
            f"at s3://{bucket}/{key}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_audit_log(
    entry: AuditLogEntry,
    cloudwatch_client=None,
    s3_client=None,
    log_group: str = AUDIT_LOG_GROUP,
    audit_bucket: str = AUDIT_S3_BUCKET,
) -> None:
    """
    Write an audit log entry to both CloudWatch Logs and S3.

    Both writes are attempted. If either fails, AuditLogFailure is raised
    and the caller must not deliver the query response.

    The CloudWatch write is attempted first. If it fails, the S3 write is
    still attempted so that at least one record exists. AuditLogFailure is
    raised after both attempts if either failed.

    Parameters
    ----------
    entry            : AuditLogEntry to write.
    cloudwatch_client: boto3 CloudWatch Logs client (injectable for testing).
                       If None, created from AWS_REGION.
    s3_client        : boto3 S3 client (injectable for testing).
                       If None, created from AWS_REGION.
    log_group        : CloudWatch log group name.
    audit_bucket     : S3 bucket for audit records.

    Raises
    ------
    AuditLogFailure if either write fails. Response must not be delivered.
    Constitution Hard Limit 5.
    """
    if cloudwatch_client is None:
        import boto3
        cloudwatch_client = boto3.client("logs", region_name=AWS_REGION)
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    failures: list[str] = []

    try:
        _write_to_cloudwatch(entry, cloudwatch_client, log_group)
    except AuditLogFailure as exc:
        failures.append(str(exc))
        logger.error("CloudWatch audit write failed: %s", exc)

    try:
        _write_to_s3(entry, s3_client, audit_bucket)
    except AuditLogFailure as exc:
        failures.append(str(exc))
        logger.error("S3 audit write failed: %s", exc)

    if failures:
        raise AuditLogFailure(
            "Audit log write failed -- response must not be delivered "
            "(Constitution Hard Limit 5). Failures: " + "; ".join(failures)
        )

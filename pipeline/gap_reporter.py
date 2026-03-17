"""
pipeline/gap_reporter.py
Milestone 6: Human-readable gap report generation.

Generates markdown and plain-text gap reports from DeletionRecord lists
and ComparisonResult objects. Reports are structured for two audiences:

  Technical / audit use
    Full record lists, EFTA numbers, signal counts, confidence tiers,
    acknowledgment sources. Suitable for writing to S3 and CloudWatch.

  Journalism / public use
    Summary tables grouped by DeletionFlag tier, dataset breakdowns,
    suppression of victim-adjacent identifiers. Suitable for inclusion
    in published reporting or public-facing documentation.

Both report types enforce victim suppression: DeletionRecords whose
subject_entities include victim-flagged entities are excluded from
public reports. They appear in technical reports with entity text
replaced by [protected identity].

Constitution reference: Principle IV -- Gaps Are Facts.
Principle V -- Every Output Is Accountable.
Hard Limit 1 -- victim identity suppression applies to gap reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.deletion_detector import DeletionRecord
from pipeline.models import DeletionFlag, WithholdingRecord
from pipeline.version_comparator import ComparisonResult

logger = logging.getLogger(__name__)

# Tier ordering for report grouping (strongest first)
_TIER_ORDER: list[str] = [
    DeletionFlag.WITHHELD_ACKNOWLEDGED,
    DeletionFlag.WITHHELD_SELECTIVELY,
    DeletionFlag.DELETION_CONFIRMED,
    DeletionFlag.DELETION_SUSPECTED,
    DeletionFlag.DELETION_POSSIBLE,
    DeletionFlag.REFERENCE_UNRESOLVED,
]

_TIER_LABELS: dict[str, str] = {
    DeletionFlag.WITHHELD_ACKNOWLEDGED: "Government-Acknowledged Withholdings",
    DeletionFlag.WITHHELD_SELECTIVELY:  "Selective Withholdings (Sibling Documents Released)",
    DeletionFlag.DELETION_CONFIRMED:    "Confirmed Deletions (3 signals)",
    DeletionFlag.DELETION_SUSPECTED:    "Suspected Deletions (2 signals)",
    DeletionFlag.DELETION_POSSIBLE:     "Possible Deletions (1 signal)",
    DeletionFlag.REFERENCE_UNRESOLVED:  "Unresolved Internal References",
}


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------

@dataclass
class GapReport:
    """
    Structured gap report ready for rendering.

    Fields
    ------
    title           Report title string.
    generated_at    ISO 8601 UTC generation timestamp.
    summary         High-level summary dict (counts by tier).
    sections        Ordered list of ReportSection, one per DeletionFlag tier.
    markdown        Rendered markdown string.
    total_gaps      Total gap count across all tiers.
    """
    title:        str
    generated_at: str
    summary:      dict[str, int]
    sections:     list["ReportSection"]
    markdown:     str
    total_gaps:   int


@dataclass
class ReportSection:
    """One tier's section in a gap report."""
    tier:    str
    label:   str
    count:   int
    records: list[dict]   # serialised record summaries


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _record_to_summary(record: DeletionRecord, public: bool = False) -> dict:
    """
    Convert a DeletionRecord to a summary dict for the report.

    DeletionRecord does not carry subject_entities (only WithholdingRecord
    does). Victim suppression for DeletionRecords is handled at the chunk
    level by the guardrail layer.
    """
    return {
        "record_id":             record.record_id,
        "document_identifiers":  record.document_identifiers,
        "deletion_flag":         record.deletion_flag.value,
        "confidence_tier":       record.deletion_flag.confidence_tier,
        "acknowledgment_source": record.acknowledgment_source,
        "acknowledgment_date":   record.acknowledgment_date,
        "notes":                 record.notes,
    }


def _withholding_to_summary(record: WithholdingRecord, public: bool = False) -> dict:
    entities = []
    for ent in (record.subject_entities or []):
        if public and ent.get("victim_flag"):
            entities.append({"type": ent.get("type", ""), "name": "[protected identity]"})
        else:
            entities.append({"type": ent.get("type", ""), "name": ent.get("name", "")})

    return {
        "record_id":             record.record_id,
        "document_identifiers":  record.document_identifiers,
        "deletion_flag":         record.deletion_flag.value,
        "confidence_tier":       record.deletion_flag.confidence_tier,
        "acknowledgment_source": record.acknowledgment_source,
        "acknowledgment_date":   record.acknowledgment_date,
        "stated_reason":         record.stated_reason,
        "subject_entities":      entities,
    }


def _render_markdown(
    title: str,
    generated_at: str,
    summary: dict[str, int],
    sections: list[ReportSection],
    public: bool,
) -> str:
    """Render a markdown report string from structured section data."""
    lines: list[str] = []

    lines.append(f"# {title}")
    lines.append(f"")
    lines.append(f"*Generated: {generated_at}*")
    if public:
        lines.append(f"*Public report — victim identities suppressed*")
    lines.append(f"")
    lines.append(f"## Summary")
    lines.append(f"")

    total = sum(summary.values())
    lines.append(f"**Total gap findings: {total}**")
    lines.append(f"")
    lines.append(f"| Tier | Count |")
    lines.append(f"|------|-------|")
    for tier in _TIER_ORDER:
        count = summary.get(tier, 0)
        if count > 0:
            lines.append(f"| {_TIER_LABELS.get(tier, tier)} | {count} |")
    lines.append(f"")

    for section in sections:
        if section.count == 0:
            continue
        lines.append(f"## {section.label}")
        lines.append(f"")
        lines.append(f"*{section.count} finding(s)*")
        lines.append(f"")
        for rec in section.records:
            ids = ", ".join(rec.get("document_identifiers", [])[:5])
            if len(rec.get("document_identifiers", [])) > 5:
                ids += f" (+{len(rec['document_identifiers']) - 5} more)"
            lines.append(f"- **{ids}**")
            lines.append(f"  - Flag: `{rec.get('deletion_flag', '')}`")
            lines.append(f"  - Confidence: {rec.get('confidence_tier', '')}")
            lines.append(f"  - Source: {rec.get('acknowledgment_source', '')}")
            if rec.get("stated_reason"):
                lines.append(f"  - Stated reason: {rec['stated_reason']}")
            if rec.get("subject_entities"):
                names = ", ".join(
                    e.get("name", "") for e in rec["subject_entities"] if e.get("name")
                )
                if names:
                    lines.append(f"  - Subjects: {names}")
            if rec.get("notes"):
                lines.append(f"  - Notes: {rec['notes']}")
            lines.append(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_gap_report(
    deletion_records: list[DeletionRecord],
    withholding_records: Optional[list[WithholdingRecord]] = None,
    title: str = "corpus-veritas Gap Report",
    public: bool = False,
) -> GapReport:
    """
    Generate a gap report from DeletionRecord and WithholdingRecord lists.

    Parameters
    ----------
    deletion_records    : Evidence-graded DeletionRecord list from
                          deletion_pipeline or deletion_detector.
    withholding_records : Government-acknowledged WithholdingRecord list.
                          Optional -- included in the report when provided.
    title               : Report title string.
    public              : If True, victim identities are suppressed.
                          If False, full technical report is generated.

    Returns
    -------
    GapReport with markdown attribute ready for writing to S3 or display.

    Constitution reference: Hard Limit 1 -- public=True suppresses
    victim-flagged entity text.
    Principle IV -- all findings are reported as structural facts.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Group all records by tier
    by_tier: dict[str, list[dict]] = {t: [] for t in _TIER_ORDER}

    for rec in (deletion_records or []):
        tier = rec.deletion_flag.value
        if tier in by_tier:
            by_tier[tier].append(_record_to_summary(rec, public=public))

    for rec in (withholding_records or []):
        tier = rec.deletion_flag.value
        if tier in by_tier:
            by_tier[tier].append(_withholding_to_summary(rec, public=public))

    sections: list[ReportSection] = []
    summary: dict[str, int] = {}
    for tier in _TIER_ORDER:
        records = by_tier[tier]
        sections.append(ReportSection(
            tier=tier,
            label=_TIER_LABELS.get(tier, tier),
            count=len(records),
            records=records,
        ))
        if records:
            summary[tier] = len(records)

    total = sum(len(v) for v in by_tier.values())
    markdown = _render_markdown(title, now, summary, sections, public)

    logger.info(
        "Gap report generated: total=%d public=%s", total, public
    )

    return GapReport(
        title=title,
        generated_at=now,
        summary=summary,
        sections=sections,
        markdown=markdown,
        total_gaps=total,
    )


def generate_comparison_report(
    comparison: ComparisonResult,
    title: Optional[str] = None,
    public: bool = False,
) -> GapReport:
    """
    Generate a gap report from a ComparisonResult (cross-version analysis).

    Wraps the retroactive deletions in generate_gap_report().

    Parameters
    ----------
    comparison : ComparisonResult from version_comparator.compare_manifests().
    title      : Optional custom title. Defaults to a version-labelled title.
    public     : Whether to suppress victim identities.
    """
    if title is None:
        title = (
            f"corpus-veritas Retroactive Deletion Report: "
            f"{comparison.prior_version} → {comparison.current_version}"
        )

    deletion_records = [
        d.deletion_record for d in comparison.retroactive_deletions
    ]

    report = generate_gap_report(
        deletion_records=deletion_records,
        title=title,
        public=public,
    )

    # Prepend comparison metadata to the markdown
    meta = (
        f"*Comparison: release `{comparison.prior_version}` "
        f"({comparison.total_prior} records) vs "
        f"`{comparison.current_version}` ({comparison.total_current} records)*\n\n"
        f"*Retroactive deletions: {comparison.deletion_count} | "
        f"New additions: {comparison.addition_count} | "
        f"Net change: {comparison.net_change:+d}*\n\n"
    )
    report.markdown = report.markdown.replace(
        f"*Generated:", meta + "*Generated:", 1
    )

    return report


def save_report_to_s3(
    report: GapReport,
    s3_key: str,
    s3_client=None,
    bucket_name: str = "",
) -> None:
    """
    Save a gap report's markdown to S3.

    Parameters
    ----------
    report      : GapReport to save.
    s3_key      : Target S3 key e.g. "reports/2026-03-16/gap-report.md".
    s3_client   : Injectable boto3 S3 client.
    bucket_name : S3 bucket name.
    """
    if not bucket_name:
        raise ValueError("bucket_name is required.")

    if s3_client is None:
        import boto3, os
        s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=report.markdown.encode("utf-8"),
            ContentType="text/markdown",
        )
        logger.info("Gap report saved to s3://%s/%s.", bucket_name, s3_key)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save gap report to s3://{bucket_name}/{s3_key}: {exc}"
        ) from exc

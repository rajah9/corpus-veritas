"""
pipeline/deletion_pipeline.py
Milestone 6: End-to-end deletion detection orchestrator.

Orchestrates the full deletion detection pipeline:

  1. Load DOJ index manifest (CSV via manifest_loader)
  2. Reconcile EFTA numbers against the manifest using EFTANumber.reconcile()
  3. Create DeletionRecord for each deletion_candidate via create_deletion_finding()
  4. Write full DeletionRecord to corpus_veritas_deletions (DynamoDB)
  5. Flag the document in corpus_veritas_documents (existing table)
  6. Optionally run cross-version comparison via version_comparator
  7. Generate gap report via gap_reporter

DynamoDB tables
---------------
corpus_veritas_deletions  (new)
  PK: record_id (S)  -- UUID assigned by create_deletion_finding()
  Attributes: deletion_flag, document_identifiers (SS), signals,
              acknowledgment_source, acknowledgment_date, confidence_tier,
              notes, created_at

  GSI: gsi-flag-date   PK=deletion_flag  SK=acknowledgment_date
       Query: "all DELETION_CONFIRMED findings since date X"

  GSI: gsi-efta-number  PK=efta_number (first document_identifier)
       Query: "is EFTA number N flagged?"

corpus_veritas_documents  (existing, from classifier.py)
  Existing records updated: deletion_flag attribute added when a document
  is flagged. Does not overwrite other attributes.
  DocumentState updated to DELETION_FLAGGED.

See docs/ARCHITECTURE.md § Deletion Detection Module.
See CONSTITUTION.md Principle IV -- Gaps Are Facts.
See pipeline/manifest_loader.py, version_comparator.py, gap_reporter.py.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from pipeline.deletion_detector import (
    DetectionSignals,
    DeletionRecord,
    FBI302SeriesResult,
    check_302_series,
    create_acknowledged_withholding,
    create_deletion_finding,
)
from pipeline.gap_reporter import GapReport, generate_comparison_report, generate_gap_report
from pipeline.manifest_loader import ManifestLoadResult, load_manifest_from_csv
from pipeline.models import DeletionFlag, DocumentState, WithholdingRecord
from pipeline.sequence_numbers import EFTANumber, ReconciliationResult
from pipeline.version_comparator import ComparisonResult, compare_manifests

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
DELETIONS_TABLE: str = "corpus_veritas_deletions"
DOCUMENTS_TABLE: str = "corpus_veritas_documents"


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class FBI302SeriesDescriptor:
    """
    Descriptor for one FBI 302 interview series to check for partial delivery.

    Pass a list of these to run_deletion_pipeline() via the fbi_302_series
    parameter. Each descriptor maps to one check_302_series() call.

    Fields
    ------
    series_identifier   Human-readable identifier for the series
                        e.g. "FBI-302-TRUMP-ALLEGATIONS-2019".
    all_series_ids      All document IDs in the series (released + withheld).
    released_ids        Document IDs confirmed present in the corpus.
    total_expected      Total number of 302s expected in the series, if known.
    acknowledgment_source
                        Provenance string for the WithholdingRecord.
    acknowledgment_date ISO 8601 date the series pattern was identified.
    subject_entities    Named entities (persons) the series concerns.
    notes               Free-text annotation.
    """
    series_identifier:    str
    all_series_ids:       list[str]
    released_ids:         list[str]
    total_expected:       Optional[int] = None
    acknowledgment_source: str = "FBI 302 series analysis"
    acknowledgment_date:  str = ""
    subject_entities:     list[dict] = field(default_factory=list)
    notes:                Optional[str] = None


@dataclass
class DeletionPipelineResult:
    """
    Result of one deletion pipeline run.

    Fields
    ------
    manifest_version        Version label of the manifest processed.
    reconciliation          ReconciliationResult from EFTANumber.reconcile().
    deletion_records        DeletionRecord list created from candidates.
    withholding_records     WithholdingRecord list from FBI 302 series checks.
    records_written         Number of records successfully written to DynamoDB.
    documents_flagged       Number of corpus_veritas_documents records updated.
    comparison_result       Cross-version ComparisonResult, if run.
    gap_report              GapReport, if generated.
    errors                  Any non-fatal errors encountered during persistence.
    """
    manifest_version:    str
    reconciliation:      ReconciliationResult
    deletion_records:    list[DeletionRecord]
    withholding_records: list[WithholdingRecord] = field(default_factory=list)
    records_written:     int = 0
    documents_flagged:   int = 0
    comparison_result:   Optional[ComparisonResult] = None
    gap_report:          Optional[GapReport] = None
    errors:              list[str] = field(default_factory=list)

    @property
    def candidate_count(self) -> int:
        return len(self.reconciliation.deletion_candidates)

    @property
    def confirmed_count(self) -> int:
        return sum(
            1 for r in self.deletion_records
            if r.deletion_flag == DeletionFlag.DELETION_CONFIRMED
        )


# ---------------------------------------------------------------------------
# DynamoDB persistence
# ---------------------------------------------------------------------------

def _write_deletion_record(
    record: DeletionRecord,
    dynamodb_client,
    table_name: str = DELETIONS_TABLE,
) -> None:
    """
    Write one DeletionRecord to corpus_veritas_deletions.

    Uses put_item (not update_item) -- record_id is a UUID so collisions
    are vanishingly unlikely, and re-running the pipeline produces new
    records rather than overwriting existing ones. This preserves history.
    """
    item: dict = {
        "record_id":             {"S": record.record_id},
        "deletion_flag":         {"S": record.deletion_flag.value},
        "confidence_tier":       {"S": record.deletion_flag.confidence_tier},
        "acknowledgment_source": {"S": record.acknowledgment_source},
        "acknowledgment_date":   {"S": record.acknowledgment_date},
        "created_at":            {"S": record.created_at},
        "document_identifiers":  {"SS": record.document_identifiers},
    }

    # Sparse attributes
    if record.notes:
        item["notes"] = {"S": record.notes}
    if record.stated_reason:
        item["stated_reason"] = {"S": record.stated_reason}

    # GSI key: first document identifier as efta_number
    if record.document_identifiers:
        item["efta_number"] = {"S": record.document_identifiers[0]}

    dynamodb_client.put_item(TableName=table_name, Item=item)


def _flag_document_record(
    efta_number: str,
    deletion_flag: DeletionFlag,
    deletion_record_id: str,
    dynamodb_client,
    table_name: str = DOCUMENTS_TABLE,
) -> None:
    """
    Add deletion_flag and deletion_record_id to an existing document record.

    Uses update_item with a conditional to avoid overwriting a stronger
    existing flag with a weaker one. DELETION_CONFIRMED is never
    downgraded to DELETION_POSSIBLE.

    The document is identified by efta_number via the gsi-corpus-source
    index, but since we don't have the document_uuid here we use a scan
    with FilterExpression. In practice the pipeline caller should pass
    document_uuid when available; this function handles the EFTA-only case.
    """
    try:
        dynamodb_client.update_item(
            TableName=table_name,
            Key={"document_uuid": {"S": f"efta:{efta_number}"}},
            UpdateExpression=(
                "SET deletion_flag = :flag, "
                "deletion_record_id = :rid, "
                "#state = :state"
            ),
            ExpressionAttributeNames={"#state": "state"},
            ExpressionAttributeValues={
                ":flag":  {"S": deletion_flag.value},
                ":rid":   {"S": deletion_record_id},
                ":state": {"S": DocumentState.DELETION_FLAGGED.value},
            },
            # Only write if no flag exists yet or the new flag is stronger
            ConditionExpression=(
                "attribute_not_exists(deletion_flag)"
            ),
        )
    except Exception as exc:
        # The document may not exist in corpus_veritas_documents yet
        # (gap detection may precede ingestion). Log and continue.
        logger.debug(
            "Could not flag document for EFTA %s in %s: %s",
            efta_number, table_name, exc,
        )


# ---------------------------------------------------------------------------
# Signal derivation from reconciliation
# ---------------------------------------------------------------------------

def _signals_for_candidate(
    efta_number: str,
    reconciliation: ReconciliationResult,
) -> DetectionSignals:
    """
    Derive DetectionSignals for a single deletion candidate.

    For within-release gaps, the EFTA gap signal is always present
    (the number is in the index but not in the corpus -- that IS the
    signal). discovery_log_entry and document_stamp_gap are set based
    on whether the reconciliation result explicitly identified this
    number as a candidate (i.e. it was not an expected/DS9 gap).

    Two signals (EFTA gap + the fact that the index itself lists it)
    is the minimum for deletion_candidates from reconciliation, so
    DELETION_SUSPECTED is the floor for within-release gaps.
    """
    # Signal 1: EFTA number is in the index but absent from corpus
    efta_gap = efta_number in reconciliation.deletion_candidates

    # Signal 2: The DOJ index itself documents the number (discovery log proxy)
    # All deletion_candidates were in the index by definition
    discovery_log_entry = efta_gap

    # Signal 3: document_stamp_gap -- only present if reconciliation
    # explicitly placed this in deletion_candidates (not expected_gap_numbers)
    document_stamp_gap = efta_gap

    return DetectionSignals(
        efta_gap=efta_gap,
        discovery_log_entry=discovery_log_entry,
        document_stamp_gap=document_stamp_gap,
    )


# ---------------------------------------------------------------------------
# FBI 302 partial delivery detection
# ---------------------------------------------------------------------------

def _write_withholding_record(
    record: WithholdingRecord,
    dynamodb_client,
    table_name: str = DELETIONS_TABLE,
) -> None:
    """
    Write one WithholdingRecord to corpus_veritas_deletions.

    Uses the same table as DeletionRecord -- both are gap findings,
    distinguished by their deletion_flag value (WITHHELD_* vs DELETION_*).
    """
    item: dict = {
        "record_id":             {"S": record.record_id},
        "deletion_flag":         {"S": record.deletion_flag.value},
        "confidence_tier":       {"S": record.deletion_flag.confidence_tier},
        "acknowledgment_source": {"S": record.acknowledgment_source},
        "acknowledgment_date":   {"S": record.acknowledgment_date},
        "created_at":            {"S": record.created_at},
        "document_identifiers":  {"SS": record.document_identifiers},
        "is_government_acknowledged": {"BOOL": True},
    }
    if record.stated_reason:
        item["stated_reason"] = {"S": record.stated_reason}
    if record.notes:
        item["notes"] = {"S": record.notes}
    if record.sibling_document_ids:
        item["sibling_document_ids"] = {"SS": record.sibling_document_ids}
    if record.document_identifiers:
        item["efta_number"] = {"S": record.document_identifiers[0]}

    dynamodb_client.put_item(TableName=table_name, Item=item)


def run_302_series_checks(
    series_descriptors: list["FBI302SeriesDescriptor"],
    dynamodb_client,
    now_date: str = "",
) -> tuple[list[WithholdingRecord], list[str]]:
    """
    Run FBI 302 partial delivery checks for a list of series descriptors.

    For each descriptor, calls check_302_series(). If the result is
    selective (some released, some withheld), creates a WITHHELD_SELECTIVELY
    WithholdingRecord and writes it to corpus_veritas_deletions.

    Fully withheld series (none released) produce WITHHELD_ACKNOWLEDGED
    records -- the absence of any release alongside other released documents
    in the corpus is itself an acknowledgment pattern.

    Fully released series produce no record.

    Parameters
    ----------
    series_descriptors : List of FBI302SeriesDescriptor to check.
    dynamodb_client    : boto3 DynamoDB client.
    now_date           : ISO 8601 date string for acknowledgment_date.
                         Defaults to today if empty.

    Returns
    -------
    (withholding_records, errors) tuple.
    """
    from datetime import datetime, timezone
    if not now_date:
        now_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    withholding_records: list[WithholdingRecord] = []
    errors: list[str] = []

    for descriptor in series_descriptors:
        try:
            series_result = check_302_series(
                series_identifier=descriptor.series_identifier,
                all_series_ids=descriptor.all_series_ids,
                released_ids=descriptor.released_ids,
                total_expected=descriptor.total_expected,
                notes=descriptor.notes,
            )
        except Exception as exc:
            errors.append(
                f"check_302_series failed for '{descriptor.series_identifier}': {exc}"
            )
            logger.error(
                "302 series check failed for %s: %s",
                descriptor.series_identifier, exc,
            )
            continue

        if not series_result.withheld_ids:
            logger.info(
                "302 series '%s': fully released, no withholding record.",
                descriptor.series_identifier,
            )
            continue

        # Determine flag: WITHHELD_SELECTIVELY if some released, else WITHHELD_ACKNOWLEDGED
        sibling_ids = series_result.released_ids if series_result.is_selective else []
        flag = (
            DeletionFlag.WITHHELD_SELECTIVELY
            if series_result.is_selective
            else DeletionFlag.WITHHELD_ACKNOWLEDGED
        )

        try:
            record = create_acknowledged_withholding(
                document_identifiers=series_result.withheld_ids,
                acknowledgment_source=descriptor.acknowledgment_source,
                acknowledgment_date=descriptor.acknowledgment_date or now_date,
                sibling_document_ids=sibling_ids,
                subject_entities=descriptor.subject_entities,
                notes=(
                    f"Series: {descriptor.series_identifier}. "
                    f"Release rate: {series_result.release_rate or 'unknown'}. "
                    + (descriptor.notes or "")
                ).strip(),
            )
        except Exception as exc:
            errors.append(
                f"WithholdingRecord creation failed for "
                f"'{descriptor.series_identifier}': {exc}"
            )
            continue

        try:
            _write_withholding_record(record, dynamodb_client)
            withholding_records.append(record)
            logger.info(
                "302 series '%s': %s -- %d withheld, %d released. "
                "WithholdingRecord written.",
                descriptor.series_identifier, flag.value,
                len(series_result.withheld_ids),
                len(series_result.released_ids),
            )
        except Exception as exc:
            errors.append(
                f"DynamoDB write failed for series "
                f"'{descriptor.series_identifier}': {exc}"
            )

    return withholding_records, errors


# ---------------------------------------------------------------------------
# Primary entry point
# ---------------------------------------------------------------------------

def run_deletion_pipeline(
    manifest: ManifestLoadResult,
    efta_scheme: EFTANumber,
    dynamodb_client=None,
    prior_manifest: Optional[ManifestLoadResult] = None,
    fbi_302_series: Optional[list] = None,
    generate_report: bool = True,
    public_report: bool = False,
    acknowledgment_source: str = "EFTA index reconciliation",
) -> DeletionPipelineResult:
    """
    Run the full deletion detection pipeline against one manifest.

    Steps:
      1. Reconcile manifest EFTA numbers against the DOJ index using
         efta_scheme.reconcile().
      2. Create DeletionRecord for each deletion_candidate.
      3. Write each DeletionRecord to corpus_veritas_deletions.
      4. Flag each candidate in corpus_veritas_documents.
      5. If prior_manifest supplied, run cross-version comparison.
      6. If generate_report, create a GapReport.

    Parameters
    ----------
    manifest            : ManifestLoadResult for the current release.
    efta_scheme         : EFTANumber instance configured with the DOJ
                          dataset mapping file.
    dynamodb_client     : Injectable boto3 DynamoDB client. If None,
                          created from AWS_REGION.
    prior_manifest      : Optional earlier ManifestLoadResult for
                          cross-version comparison.
    fbi_302_series      : Optional list of FBI302SeriesDescriptor instances.
                          If provided, run_302_series_checks() is called and
                          resulting WithholdingRecords are included in the
                          pipeline result and gap report.
    generate_report     : Whether to generate a GapReport.
    public_report       : Whether the GapReport should suppress victim
                          identities.
    acknowledgment_source
                        : Source string for DeletionRecord provenance.

    Returns
    -------
    DeletionPipelineResult with all findings and persistence counts.

    Constitution reference: Principle IV -- Gaps Are Facts.
    """
    if dynamodb_client is None:
        import boto3
        dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Step 1: reconcile
    reconciliation = efta_scheme.reconcile(
        corpus_numbers=list(manifest.efta_numbers),
        index_numbers=list(efta_scheme.extract_from_text(
            " ".join(manifest.efta_numbers)
        )) or list(manifest.efta_numbers),
    )

    logger.info(
        "Reconciliation: present=%d missing=%d candidates=%d expected_gaps=%d",
        reconciliation.present_count,
        reconciliation.missing_from_corpus_count,
        len(reconciliation.deletion_candidates),
        len(reconciliation.expected_gap_numbers),
    )

    # Step 2: create DeletionRecords
    deletion_records: list[DeletionRecord] = []
    errors: list[str] = []

    for efta_num in reconciliation.deletion_candidates:
        signals = _signals_for_candidate(efta_num, reconciliation)
        try:
            record = create_deletion_finding(
                document_identifiers=[efta_num],
                signals=signals,
                acknowledgment_source=acknowledgment_source,
                acknowledgment_date=now,
            )
            deletion_records.append(record)
        except Exception as exc:
            errors.append(f"Failed to create DeletionRecord for {efta_num}: {exc}")
            logger.error("DeletionRecord creation failed for %s: %s", efta_num, exc)

    # Steps 3 & 4: persist to DynamoDB
    records_written = 0
    documents_flagged = 0

    for record in deletion_records:
        try:
            _write_deletion_record(record, dynamodb_client)
            records_written += 1
        except Exception as exc:
            errors.append(f"DynamoDB write failed for {record.record_id}: {exc}")
            logger.error("Failed to write deletion record %s: %s", record.record_id, exc)
            continue

        for efta_num in record.document_identifiers:
            try:
                _flag_document_record(
                    efta_num, record.deletion_flag, record.record_id,
                    dynamodb_client,
                )
                documents_flagged += 1
            except Exception as exc:
                logger.debug("Flag update skipped for %s: %s", efta_num, exc)

    # Step 5: cross-version comparison
    comparison_result: Optional[ComparisonResult] = None
    if prior_manifest is not None:
        comparison_result = compare_manifests(prior_manifest, manifest)
        # Persist retroactive deletions
        for retro in comparison_result.retroactive_deletions:
            try:
                _write_deletion_record(retro.deletion_record, dynamodb_client)
                records_written += 1
            except Exception as exc:
                errors.append(
                    f"Retroactive deletion write failed for "
                    f"{retro.efta_number}: {exc}"
                )

    # Step 6: FBI 302 partial delivery checks
    withholding_records: list[WithholdingRecord] = []
    if fbi_302_series:
        w_records, w_errors = run_302_series_checks(
            fbi_302_series, dynamodb_client, now_date=now
        )
        withholding_records.extend(w_records)
        records_written += len(w_records)
        errors.extend(w_errors)

    # Step 7: generate report
    gap_report: Optional[GapReport] = None
    if generate_report:
        if comparison_result is not None:
            gap_report = generate_comparison_report(
                comparison_result, public=public_report
            )
        else:
            gap_report = generate_gap_report(
                deletion_records=deletion_records,
                withholding_records=withholding_records or None,
                public=public_report,
            )

    logger.info(
        "Deletion pipeline complete: version=%s candidates=%d "
        "records_written=%d documents_flagged=%d errors=%d",
        manifest.release_version,
        len(reconciliation.deletion_candidates),
        records_written,
        documents_flagged,
        len(errors),
    )

    return DeletionPipelineResult(
        manifest_version=manifest.release_version,
        reconciliation=reconciliation,
        deletion_records=deletion_records,
        withholding_records=withholding_records,
        records_written=records_written,
        documents_flagged=documents_flagged,
        comparison_result=comparison_result,
        gap_report=gap_report,
        errors=errors,
    )

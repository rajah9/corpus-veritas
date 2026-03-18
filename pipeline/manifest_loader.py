"""
pipeline/manifest_loader.py
Milestone 6: DOJ index manifest loading and normalisation.

The DOJ Epstein release is indexed by an accompanying CSV manifest that
lists every document (or page, for EFTA-numbered releases) with its EFTA
number, dataset identifier, title, and download URL. This module loads
that CSV, normalises it to a consistent internal dict format, and exposes
the EFTA number set for reconciliation via sequence_numbers.EFTANumber.

CSV format (DOJ Epstein release convention)
-------------------------------------------
The manifest has no guaranteed stable column order. Column names vary
slightly across DOJ release versions. This module normalises by detecting
the EFTA number column by content pattern rather than by name, making it
robust to minor column naming changes across release versions.

Expected columns (names may vary):
    efta_number / efta_num / number / page_number
    dataset / dataset_id / dataset_name
    title / document_title / name
    url / download_url / link

Normalised internal format (JSON-serialisable dict per record):
    {
        "efta_number": "1234567",          # string, no prefix
        "dataset":     "DS01",             # dataset identifier
        "title":       "FBI 302 Interview",
        "url":         "https://...",
        "raw":         { ...original row... }  # original CSV row retained
    }

S3 storage
----------
Manifests are stored in S3 alongside the raw corpus documents.
Key convention: manifests/{release_version}/manifest.csv
                manifests/{release_version}/manifest.json (normalised)

See docs/ARCHITECTURE.md § Deletion Detection Module.
See CONSTITUTION.md Principle IV -- Gaps Are Facts.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
CORPUS_S3_BUCKET: str = os.environ.get("CORPUS_S3_BUCKET", "")

# Regex that matches a raw EFTA number (digits only, no prefix required)
_EFTA_NUMBER_RE = re.compile(r"^\d{1,10}$")

# Column name candidates for the EFTA number field, in priority order
_EFTA_COLUMN_CANDIDATES: list[str] = [
    "efta_number", "efta_num", "efta", "number",
    "page_number", "page_num", "seq", "sequence_number",
]

# Column name candidates for dataset, title, URL
_DATASET_CANDIDATES:  list[str] = ["dataset", "dataset_id", "dataset_name", "ds"]
_TITLE_CANDIDATES:    list[str] = ["title", "document_title", "name", "doc_title"]
_URL_CANDIDATES:      list[str] = ["url", "download_url", "link", "href", "file_url"]


# ---------------------------------------------------------------------------
# Normalised record type
# ---------------------------------------------------------------------------

@dataclass
class ManifestRecord:
    """
    One normalised record from a DOJ index manifest.

    Fields
    ------
    efta_number  Canonical EFTA number string (digits only, no prefix).
    dataset      Dataset identifier e.g. "DS01", "DS09".
    title        Document or page title. Empty string if not present.
    url          Download URL. Empty string if not present.
    raw          Original CSV row dict, retained for audit purposes.
    """
    efta_number: str
    dataset:     str = ""
    title:       str = ""
    url:         str = ""
    raw:         dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "efta_number": self.efta_number,
            "dataset":     self.dataset,
            "title":       self.title,
            "url":         self.url,
        }


@dataclass
class ManifestLoadResult:
    """
    Result of loading and normalising one manifest file.

    Fields
    ------
    release_version  Caller-supplied version string e.g. "2026-03-01".
    records          Normalised ManifestRecord list.
    efta_numbers     Set of all EFTA number strings (for reconciliation).
    skipped_rows     Number of rows that could not be parsed.
    duplicate_count  Number of duplicate EFTA numbers detected.
    """
    release_version: str
    records:         list[ManifestRecord]
    efta_numbers:    set[str]
    skipped_rows:    int = 0
    duplicate_count: int = 0

    @property
    def record_count(self) -> int:
        return len(self.records)

    def to_json(self) -> str:
        """Serialise records to JSON string for S3 storage."""
        return json.dumps(
            {
                "release_version": self.release_version,
                "record_count":    self.record_count,
                "skipped_rows":    self.skipped_rows,
                "duplicate_count": self.duplicate_count,
                "records": [r.to_dict() for r in self.records],
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

def _detect_column(
    fieldnames: list[str],
    candidates: list[str],
) -> Optional[str]:
    """
    Return the first fieldname that matches any candidate name
    (case-insensitive). Returns None if no match found.
    """
    lower_fields = {f.strip().lower(): f for f in fieldnames}
    for candidate in candidates:
        if candidate.lower() in lower_fields:
            return lower_fields[candidate.lower()]
    return None


def _normalise_efta(raw_value: str) -> Optional[str]:
    """
    Normalise a raw EFTA value to a plain digit string.

    Strips whitespace, optional "EFTA-" prefix, and leading zeros
    beyond one digit. Returns None if the value is not a valid EFTA
    number pattern.
    """
    v = raw_value.strip()
    # Strip known prefixes
    for prefix in ("EFTA-", "EFTA_", "EFTA"):
        if v.upper().startswith(prefix):
            v = v[len(prefix):]
    if not v:
        return None
    v = v.lstrip("0") or "0"
    if _EFTA_NUMBER_RE.match(v):
        return v
    return None


# ---------------------------------------------------------------------------
# Core loading functions
# ---------------------------------------------------------------------------

def load_manifest_from_csv(
    csv_text: str,
    release_version: str,
) -> ManifestLoadResult:
    """
    Parse a CSV manifest string and return a ManifestLoadResult.

    Column detection is tolerant of naming variations between DOJ
    release versions. Only the EFTA number column is required; dataset,
    title, and URL are optional.

    Parameters
    ----------
    csv_text        : Full CSV content as a string.
    release_version : Caller-supplied version label e.g. "2026-03-01".

    Returns
    -------
    ManifestLoadResult with normalised records and EFTA number set.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    fieldnames = list(reader.fieldnames or [])

    efta_col    = _detect_column(fieldnames, _EFTA_COLUMN_CANDIDATES)
    dataset_col = _detect_column(fieldnames, _DATASET_CANDIDATES)
    title_col   = _detect_column(fieldnames, _TITLE_CANDIDATES)
    url_col     = _detect_column(fieldnames, _URL_CANDIDATES)

    if efta_col is None:
        raise ValueError(
            f"Could not detect EFTA number column in manifest. "
            f"Columns present: {fieldnames}. "
            f"Expected one of: {_EFTA_COLUMN_CANDIDATES}"
        )

    records: list[ManifestRecord] = []
    efta_numbers: set[str] = set()
    skipped = 0
    duplicates = 0

    for row in reader:
        raw_efta = row.get(efta_col, "").strip()
        normalised = _normalise_efta(raw_efta)

        if normalised is None:
            logger.debug("Skipping row with unparseable EFTA value: %r", raw_efta)
            skipped += 1
            continue

        if normalised in efta_numbers:
            duplicates += 1
            logger.debug("Duplicate EFTA number: %s", normalised)

        efta_numbers.add(normalised)
        records.append(ManifestRecord(
            efta_number=normalised,
            dataset=row.get(dataset_col, "").strip() if dataset_col else "",
            title=row.get(title_col, "").strip() if title_col else "",
            url=row.get(url_col, "").strip() if url_col else "",
            raw=dict(row),
        ))

    logger.info(
        "Manifest loaded: version=%s records=%d skipped=%d duplicates=%d",
        release_version, len(records), skipped, duplicates,
    )

    return ManifestLoadResult(
        release_version=release_version,
        records=records,
        efta_numbers=efta_numbers,
        skipped_rows=skipped,
        duplicate_count=duplicates,
    )


def load_manifest_from_file(
    path: Path,
    release_version: str,
) -> ManifestLoadResult:
    """
    Load a manifest from a local CSV file path.

    Parameters
    ----------
    path            : Path to the CSV file.
    release_version : Version label for the release.
    """
    text = path.read_text(encoding="utf-8-sig")  # utf-8-sig strips BOM
    return load_manifest_from_csv(text, release_version)


def load_manifest_from_s3(
    s3_key: str,
    release_version: str,
    s3_client=None,
    bucket_name: str = CORPUS_S3_BUCKET,
) -> ManifestLoadResult:
    """
    Load a manifest CSV from S3.

    Parameters
    ----------
    s3_key          : S3 object key e.g. "manifests/2026-03-01/manifest.csv".
    release_version : Version label for the release.
    s3_client       : Injectable boto3 S3 client.
    bucket_name     : S3 bucket name.

    Raises
    ------
    ValueError   if bucket_name is empty.
    RuntimeError if the S3 read fails.
    """
    if not bucket_name:
        raise ValueError("bucket_name is required. Set CORPUS_S3_BUCKET.")

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        csv_text = response["Body"].read().decode("utf-8-sig")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load manifest from s3://{bucket_name}/{s3_key}: {exc}"
        ) from exc

    return load_manifest_from_csv(csv_text, release_version)


def save_normalised_manifest_to_s3(
    result: ManifestLoadResult,
    s3_key: str,
    s3_client=None,
    bucket_name: str = CORPUS_S3_BUCKET,
) -> None:
    """
    Save the normalised JSON manifest to S3.

    Stores the JSON alongside the original CSV for reproducible analysis.
    Key convention: manifests/{release_version}/manifest.json

    Parameters
    ----------
    result      : ManifestLoadResult to serialise.
    s3_key      : Target S3 key.
    s3_client   : Injectable boto3 S3 client.
    bucket_name : S3 bucket name.

    Raises
    ------
    RuntimeError if the S3 write fails.
    """
    if not bucket_name:
        raise ValueError("bucket_name is required. Set CORPUS_S3_BUCKET.")

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=result.to_json().encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "Normalised manifest saved to s3://%s/%s (%d records).",
            bucket_name, s3_key, result.record_count,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to save normalised manifest to s3://{bucket_name}/{s3_key}: {exc}"
        ) from exc

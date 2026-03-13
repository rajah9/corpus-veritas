"""
s3_store.py
Layer 2: Raw document storage in S3.

Stores the original document bytes before any processing. S3 is the
immutable source-of-truth for raw content; everything downstream
(chunking, embedding, DynamoDB records) is derived from it.

Key structure
-------------
    {corpus_source}/{document_uuid}/raw

corpus_source is the corpus_registry UUID (e.g. "rhowardstone-epstein")
or "DOJ_DIRECT" for documents OCR'd directly from the DOJ PDF release.
document_uuid is the UUID assigned by classifier.py.

Using corpus_source as the top-level prefix enables:
  - IAM policies scoped per corpus (read-only access to DOJ_DIRECT,
    write access gated on corpus provenance tier)
  - S3 lifecycle rules per corpus (e.g. transition FLAGGED corpus
    content to Glacier after 90 days)
  - Cost allocation by corpus source

Object Lock
-----------
Victim-flagged documents receive S3 Object Lock in COMPLIANCE mode with
a 7-year retention period. COMPLIANCE mode prevents deletion even by the
bucket owner -- this matches the audit log retention policy in
docs/ARCHITECTURE.md para Layer 5.

COMPLIANCE mode requires:
  1. The S3 bucket must be created with Object Lock enabled (bucket-level
     setting, cannot be enabled after creation).
  2. The boto3 caller must have s3:PutObjectLegalHold or
     s3:PutObjectRetention IAM permission.

Non-victim documents are stored without Object Lock. They can be
overwritten by a re-ingestion run (same key, same document_uuid).

See docs/ARCHITECTURE.md para Layer 2 -- S3 Raw Document Storage.
See CONSTITUTION.md Article III Hard Limit 1.
See infrastructure/iam/README.md for IAM policy definitions.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

S3_BUCKET: str = os.environ.get("CORPUS_S3_BUCKET", "")
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")

# Retention period for victim-flagged documents (matches audit log policy)
_VICTIM_RETENTION_YEARS: int = 7


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

def document_key(corpus_source: str, document_uuid: str) -> str:
    """
    Return the S3 object key for a raw document.

    Format: {corpus_source}/{document_uuid}/raw

    Parameters
    ----------
    corpus_source   : corpus_registry UUID or "DOJ_DIRECT".
    document_uuid   : UUID assigned by classifier.py.
    """
    return f"{corpus_source}/{document_uuid}/raw"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def store_document(
    document_uuid: str,
    corpus_source: str,
    content: bytes,
    victim_flag: bool = False,
    content_type: str = "application/octet-stream",
    s3_client=None,
    bucket_name: str = S3_BUCKET,
) -> str:
    """
    Store raw document bytes in S3.

    Writes to key: {corpus_source}/{document_uuid}/raw

    For victim-flagged documents, applies S3 Object Lock in COMPLIANCE
    mode with a 7-year retention period. The bucket must have Object Lock
    enabled at creation time (see docstring above).

    Parameters
    ----------
    document_uuid   : UUID from classifier.py. Used in the S3 key.
    corpus_source   : Corpus source identifier. Used as the top-level
                      prefix in the S3 key.
    content         : Raw document bytes (PDF, text, or other format).
    victim_flag     : If True, applies Object Lock COMPLIANCE retention.
    content_type    : MIME type for the stored object.
    s3_client       : boto3 S3 client (injectable for testing). If None,
                      created from AWS_REGION.
    bucket_name     : S3 bucket name. Defaults to CORPUS_S3_BUCKET env var.

    Returns
    -------
    The S3 key of the stored object ({corpus_source}/{document_uuid}/raw).

    Raises
    ------
    ValueError   if bucket_name is empty.
    RuntimeError if the S3 put_object call fails.
    """
    if not bucket_name:
        raise ValueError(
            "bucket_name is required. Set the CORPUS_S3_BUCKET environment "
            "variable or pass bucket_name explicitly."
        )

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    key = document_key(corpus_source, document_uuid)

    put_kwargs: dict = {
        "Bucket": bucket_name,
        "Key": key,
        "Body": content,
        "ContentType": content_type,
    }

    if victim_flag:
        retain_until = datetime.now(timezone.utc) + timedelta(
            days=_VICTIM_RETENTION_YEARS * 365
        )
        put_kwargs["ObjectLockMode"] = "COMPLIANCE"
        put_kwargs["ObjectLockRetainUntilDate"] = retain_until
        logger.info(
            "Storing victim-flagged document %s with Object Lock COMPLIANCE "
            "until %s.", document_uuid, retain_until.isoformat(),
        )
    else:
        logger.debug("Storing document %s at s3://%s/%s.", document_uuid, bucket_name, key)

    try:
        s3_client.put_object(**put_kwargs)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to store document {document_uuid} at s3://{bucket_name}/{key}: {exc}"
        ) from exc

    return key


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_document(
    document_uuid: str,
    corpus_source: str,
    s3_client=None,
    bucket_name: str = S3_BUCKET,
) -> bytes:
    """
    Retrieve raw document bytes from S3.

    Parameters
    ----------
    document_uuid   : UUID from classifier.py.
    corpus_source   : Corpus source identifier used in the key.
    s3_client       : boto3 S3 client (injectable for testing).
    bucket_name     : S3 bucket name.

    Returns
    -------
    Raw document bytes.

    Raises
    ------
    ValueError   if bucket_name is empty.
    RuntimeError if the S3 get_object call fails or the key does not exist.
    """
    if not bucket_name:
        raise ValueError(
            "bucket_name is required. Set the CORPUS_S3_BUCKET environment "
            "variable or pass bucket_name explicitly."
        )

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    key = document_key(corpus_source, document_uuid)

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        return response["Body"].read()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve document {document_uuid} from "
            f"s3://{bucket_name}/{key}: {exc}"
        ) from exc

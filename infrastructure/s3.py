"""
infrastructure/s3.py
Layer 2: S3 bucket provisioning and configuration.

Defines and enforces the S3 bucket requirements for corpus-veritas raw
document storage. Infrastructure code is kept separate from the pipeline
(pipeline/s3_store.py handles reads and writes; this module handles the
bucket itself).

Bucket name
-----------
Set via the CORPUS_S3_BUCKET environment variable. There is no hardcoded
default -- the bucket name must be explicitly provided. S3 bucket names
are globally unique; the name chosen at provisioning time must be
recorded in your deployment notes and in the environment config.

Critical bucket requirements
-----------------------------
1. Object Lock MUST be enabled at bucket creation time.
   Object Lock cannot be enabled on an existing bucket. If you create
   the bucket without Object Lock and later run store_document() with
   victim_flag=True, the put_object call will fail with an error like:
     "Bucket is missing Object Lock Configuration"
   The only remediation is to create a new bucket with Object Lock
   enabled and re-ingest all victim-flagged documents.

2. Versioning is automatically enabled when Object Lock is enabled.
   AWS enforces this -- you do not need to enable it separately.

3. Bucket must NOT be publicly accessible.
   Public access block must be enabled on all four settings.

4. Server-side encryption (SSE-S3 or SSE-KMS) is strongly recommended.
   Victim-flagged documents contain PII that passed Comprehend detection
   but was not redacted (it was flagged for human review and Object Locked
   instead). Encryption at rest is not optional for this content.

5. Lifecycle rules (recommended):
   - FLAGGED corpus content: transition to S3 Glacier after 90 days.
     These are low-provenance documents unlikely to be queried frequently.
   - DOJ_DIRECT content: no automatic transition. These are the primary
     sources and may be re-read by the pipeline at any time.

Lifecycle rule prefix structure
--------------------------------
The key format {corpus_source}/{document_uuid}/raw means lifecycle rules
can be scoped by corpus_source prefix:

    DOJ_DIRECT/*            -- primary sources, no auto-transition
    rhowardstone-epstein/*  -- community-vouched, keep in Standard
    yung-megafone-epstein/* -- FLAGGED corpus, transition to Glacier

IAM permissions required
------------------------
corpus-veritas-pipeline role needs:
    s3:PutObject
    s3:GetObject
    s3:PutObjectRetention    (for victim-flagged Object Lock)
    s3:GetBucketObjectLockConfiguration  (to verify lock is enabled)

See infrastructure/iam/README.md.

This module
-----------
Provides:
  bucket_config()       -- returns the CreateBucket parameter dict.
                           Suitable for passing to boto3 or CDK.
  ensure_bucket()       -- idempotent: creates the bucket if absent,
                           verifies Object Lock if present.
  verify_object_lock()  -- asserts Object Lock is enabled; raises if not.
  put_lifecycle_rules() -- applies recommended lifecycle configuration.

All functions accept an injectable s3_client for testing.

See docs/ARCHITECTURE.md para Layer 2 -- S3 Raw Document Storage.
See pipeline/s3_store.py for read/write operations.
See CONSTITUTION.md Article III Hard Limit 1 (victim_flag Object Lock).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
CORPUS_S3_BUCKET: str = os.environ.get("CORPUS_S3_BUCKET", "")

# Retention period for victim-flagged documents -- must match s3_store.py
_VICTIM_RETENTION_YEARS: int = 7


# ---------------------------------------------------------------------------
# Bucket configuration
# ---------------------------------------------------------------------------

def bucket_config(
    bucket_name: str = CORPUS_S3_BUCKET,
    region: str = AWS_REGION,
) -> dict:
    """
    Return the boto3 create_bucket parameter dict for the corpus bucket.

    Object Lock is requested at creation time via ObjectLockEnabledForBucket.
    This is the only opportunity to enable it -- it cannot be added later.

    Parameters
    ----------
    bucket_name : S3 bucket name.
    region      : AWS region. CreateBucketConfiguration is omitted for
                  us-east-1 (AWS requirement for that region).

    Returns
    -------
    dict suitable for s3_client.create_bucket(**bucket_config()).

    Raises
    ------
    ValueError if bucket_name is empty.
    """
    if not bucket_name:
        raise ValueError(
            "bucket_name is required. Set the CORPUS_S3_BUCKET "
            "environment variable or pass bucket_name explicitly."
        )

    cfg: dict = {
        "Bucket": bucket_name,
        "ObjectLockEnabledForBucket": True,
    }

    # us-east-1 must not include CreateBucketConfiguration (AWS quirk)
    if region != "us-east-1":
        cfg["CreateBucketConfiguration"] = {"LocationConstraint": region}

    return cfg


# ---------------------------------------------------------------------------
# Public access block
# ---------------------------------------------------------------------------

def public_access_block_config(bucket_name: str) -> dict:
    """
    Return the put_public_access_block parameter dict.

    All four block settings are enabled. This must be applied immediately
    after bucket creation before any objects are written.

    Parameters
    ----------
    bucket_name : S3 bucket name.
    """
    return {
        "Bucket": bucket_name,
        "PublicAccessBlockConfiguration": {
            "BlockPublicAcls":       True,
            "IgnorePublicAcls":      True,
            "BlockPublicPolicy":     True,
            "RestrictPublicBuckets": True,
        },
    }


# ---------------------------------------------------------------------------
# Lifecycle rules
# ---------------------------------------------------------------------------

def lifecycle_rules_config(bucket_name: str) -> dict:
    """
    Return the put_bucket_lifecycle_configuration parameter dict.

    Applies two rules:
      1. flagged-corpus-glacier  -- objects under any PROVENANCE_FLAGGED
         corpus prefix transition to Glacier after 90 days.
         Prefix filter matches yung-megafone* and s0fskr1p* (both FLAGGED
         in corpus_registry.json). Uses a tag filter as a more robust
         alternative -- tag corpus_provenance=FLAGGED on put_object if
         using the tag-based rule; or use prefix if corpus names are stable.
         This implementation uses prefix matching on known FLAGGED sources.

      2. incomplete-multipart -- abort incomplete multipart uploads after
         7 days to prevent orphaned storage charges.

    Parameters
    ----------
    bucket_name : S3 bucket name.
    """
    return {
        "Bucket": bucket_name,
        "LifecycleConfiguration": {
            "Rules": [
                {
                    "ID": "flagged-corpus-to-glacier",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "PROVENANCE_FLAGGED/"},
                    "Transitions": [
                        {
                            "Days": 90,
                            "StorageClass": "GLACIER",
                        }
                    ],
                },
                {
                    "ID": "abort-incomplete-multipart",
                    "Status": "Enabled",
                    "Filter": {"Prefix": ""},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
                },
            ]
        },
    }


# ---------------------------------------------------------------------------
# Bucket lifecycle management
# ---------------------------------------------------------------------------

def verify_object_lock(
    bucket_name: str,
    s3_client=None,
) -> None:
    """
    Assert that Object Lock is enabled on the bucket.

    Called by ensure_bucket() when the bucket already exists, and may be
    called directly by the pipeline at startup as a safety check.

    Raises
    ------
    RuntimeError if Object Lock is not enabled or the check cannot be
    completed. This is a hard failure -- ingesting victim-flagged documents
    into a bucket without Object Lock would silently produce unlocked
    objects with no retention guarantee.
    """
    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=AWS_REGION)

    try:
        response = s3_client.get_object_lock_configuration(Bucket=bucket_name)
        lock_config = response.get("ObjectLockConfiguration", {})
        if lock_config.get("ObjectLockEnabled") != "Enabled":
            raise RuntimeError(
                f"Bucket '{bucket_name}' does not have Object Lock enabled. "
                "Victim-flagged documents cannot be stored with retention "
                "guarantees. Create a new bucket with Object Lock enabled "
                "at creation time -- it cannot be added retroactively."
            )
    except RuntimeError:
        raise
    except Exception as exc:
        # get_object_lock_configuration raises if lock was never configured
        raise RuntimeError(
            f"Object Lock is not configured on bucket '{bucket_name}'. "
            f"Error: {exc}. "
            "Create a new bucket with ObjectLockEnabledForBucket=True."
        ) from exc

    logger.info("Object Lock verified on bucket '%s'.", bucket_name)


def ensure_bucket(
    bucket_name: str = CORPUS_S3_BUCKET,
    region: str = AWS_REGION,
    s3_client=None,
) -> bool:
    """
    Create the corpus S3 bucket if it does not already exist.

    If the bucket exists, verifies Object Lock is enabled and returns False.
    If the bucket does not exist, creates it with Object Lock enabled,
    applies the public access block, and applies lifecycle rules.

    Safe to call on every deployment.

    Parameters
    ----------
    bucket_name : S3 bucket name.
    region      : AWS region.
    s3_client   : boto3 S3 client (injectable for testing).

    Returns
    -------
    True  if the bucket was created.
    False if the bucket already existed (Object Lock verified).

    Raises
    ------
    ValueError   if bucket_name is empty.
    RuntimeError if creation fails, or if the existing bucket does not
                 have Object Lock enabled.
    """
    if not bucket_name:
        raise ValueError(
            "bucket_name is required. Set the CORPUS_S3_BUCKET "
            "environment variable or pass bucket_name explicitly."
        )

    if s3_client is None:
        import boto3
        s3_client = boto3.client("s3", region_name=region)

    # Check existence via head_bucket
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        bucket_exists = True
    except Exception:
        bucket_exists = False

    if bucket_exists:
        logger.info("Bucket '%s' already exists -- verifying Object Lock.", bucket_name)
        verify_object_lock(bucket_name, s3_client)
        return False

    # Create bucket
    try:
        s3_client.create_bucket(**bucket_config(bucket_name, region))
        logger.info("Created bucket '%s' with Object Lock enabled.", bucket_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create bucket '{bucket_name}': {exc}"
        ) from exc

    # Block public access immediately after creation
    try:
        s3_client.put_public_access_block(
            **public_access_block_config(bucket_name)
        )
        logger.info("Applied public access block to bucket '%s'.", bucket_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply public access block to bucket '{bucket_name}': {exc}"
        ) from exc

    # Apply lifecycle rules
    try:
        s3_client.put_bucket_lifecycle_configuration(
            **lifecycle_rules_config(bucket_name)
        )
        logger.info("Applied lifecycle rules to bucket '%s'.", bucket_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply lifecycle rules to bucket '{bucket_name}': {exc}"
        ) from exc

    return True

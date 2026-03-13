"""
tests/infrastructure/test_s3.py

Unit tests for infrastructure/s3.py.

Coverage targets
----------------
bucket_config()               -- required fields, Object Lock flag, region
                                 handling (us-east-1 omits
                                 CreateBucketConfiguration), empty name raises

public_access_block_config()  -- all four block flags True

lifecycle_rules_config()      -- two rules present, flagged-corpus rule
                                 targets PROVENANCE_FLAGGED/ prefix with
                                 90-day Glacier transition, multipart rule
                                 aborts after 7 days

verify_object_lock()          -- passes when Enabled, raises when not
                                 Enabled, raises when API call fails

ensure_bucket()               -- creates when absent (create + public
                                 access block + lifecycle all called),
                                 returns True when created, skips create
                                 when present and Object Lock verified,
                                 returns False when already exists, empty
                                 name raises ValueError, create_bucket
                                 failure raises RuntimeError, public access
                                 block failure raises RuntimeError,
                                 lifecycle failure raises RuntimeError
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, call

from infrastructure.s3 import (
    bucket_config,
    ensure_bucket,
    lifecycle_rules_config,
    public_access_block_config,
    verify_object_lock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_s3(bucket_exists: bool = False, object_lock_enabled: bool = True) -> MagicMock:
    client = MagicMock()

    if bucket_exists:
        client.head_bucket.return_value = {}
    else:
        client.head_bucket.side_effect = Exception("NoSuchBucket")

    if object_lock_enabled:
        client.get_object_lock_configuration.return_value = {
            "ObjectLockConfiguration": {"ObjectLockEnabled": "Enabled"}
        }
    else:
        client.get_object_lock_configuration.return_value = {
            "ObjectLockConfiguration": {"ObjectLockEnabled": "Disabled"}
        }

    client.create_bucket.return_value = {}
    client.put_public_access_block.return_value = {}
    client.put_bucket_lifecycle_configuration.return_value = {}
    return client


# ===========================================================================
# bucket_config
# ===========================================================================

class TestBucketConfig(unittest.TestCase):

    def test_bucket_name_in_config(self):
        cfg = bucket_config("my-bucket", "us-east-1")
        self.assertEqual(cfg["Bucket"], "my-bucket")

    def test_object_lock_enabled(self):
        cfg = bucket_config("my-bucket", "us-east-1")
        self.assertTrue(cfg["ObjectLockEnabledForBucket"])

    def test_us_east_1_omits_location_constraint(self):
        cfg = bucket_config("my-bucket", "us-east-1")
        self.assertNotIn("CreateBucketConfiguration", cfg)

    def test_other_region_includes_location_constraint(self):
        cfg = bucket_config("my-bucket", "us-west-2")
        self.assertIn("CreateBucketConfiguration", cfg)
        self.assertEqual(
            cfg["CreateBucketConfiguration"]["LocationConstraint"], "us-west-2"
        )

    def test_empty_bucket_name_raises(self):
        with self.assertRaises(ValueError) as ctx:
            bucket_config("")
        self.assertIn("bucket_name", str(ctx.exception))


# ===========================================================================
# public_access_block_config
# ===========================================================================

class TestPublicAccessBlockConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = public_access_block_config("my-bucket")
        self.block = self.cfg["PublicAccessBlockConfiguration"]

    def test_bucket_name_set(self):
        self.assertEqual(self.cfg["Bucket"], "my-bucket")

    def test_block_public_acls(self):
        self.assertTrue(self.block["BlockPublicAcls"])

    def test_ignore_public_acls(self):
        self.assertTrue(self.block["IgnorePublicAcls"])

    def test_block_public_policy(self):
        self.assertTrue(self.block["BlockPublicPolicy"])

    def test_restrict_public_buckets(self):
        self.assertTrue(self.block["RestrictPublicBuckets"])


# ===========================================================================
# lifecycle_rules_config
# ===========================================================================

class TestLifecycleRulesConfig(unittest.TestCase):

    def setUp(self):
        self.cfg = lifecycle_rules_config("my-bucket")
        self.rules = self.cfg["LifecycleConfiguration"]["Rules"]

    def test_bucket_name_set(self):
        self.assertEqual(self.cfg["Bucket"], "my-bucket")

    def test_two_rules_defined(self):
        self.assertEqual(len(self.rules), 2)

    def test_flagged_corpus_rule_present(self):
        ids = [r["ID"] for r in self.rules]
        self.assertIn("flagged-corpus-to-glacier", ids)

    def test_multipart_abort_rule_present(self):
        ids = [r["ID"] for r in self.rules]
        self.assertIn("abort-incomplete-multipart", ids)

    def test_flagged_corpus_rule_targets_correct_prefix(self):
        rule = next(r for r in self.rules if r["ID"] == "flagged-corpus-to-glacier")
        self.assertEqual(rule["Filter"]["Prefix"], "PROVENANCE_FLAGGED/")

    def test_flagged_corpus_transitions_to_glacier(self):
        rule = next(r for r in self.rules if r["ID"] == "flagged-corpus-to-glacier")
        self.assertEqual(rule["Transitions"][0]["StorageClass"], "GLACIER")

    def test_flagged_corpus_glacier_transition_after_90_days(self):
        rule = next(r for r in self.rules if r["ID"] == "flagged-corpus-to-glacier")
        self.assertEqual(rule["Transitions"][0]["Days"], 90)

    def test_multipart_abort_after_7_days(self):
        rule = next(r for r in self.rules if r["ID"] == "abort-incomplete-multipart")
        self.assertEqual(
            rule["AbortIncompleteMultipartUpload"]["DaysAfterInitiation"], 7
        )

    def test_all_rules_enabled(self):
        for rule in self.rules:
            self.assertEqual(rule["Status"], "Enabled")


# ===========================================================================
# verify_object_lock
# ===========================================================================

class TestVerifyObjectLock(unittest.TestCase):

    def test_passes_when_object_lock_enabled(self):
        client = _mock_s3(object_lock_enabled=True)
        # Must not raise
        verify_object_lock("my-bucket", client)

    def test_raises_when_object_lock_disabled(self):
        client = _mock_s3(object_lock_enabled=False)
        with self.assertRaises(RuntimeError) as ctx:
            verify_object_lock("my-bucket", client)
        self.assertIn("Object Lock", str(ctx.exception))

    def test_raises_when_api_call_fails(self):
        client = MagicMock()
        client.get_object_lock_configuration.side_effect = Exception("AccessDenied")
        with self.assertRaises(RuntimeError) as ctx:
            verify_object_lock("my-bucket", client)
        self.assertIn("Object Lock", str(ctx.exception))

    def test_error_message_includes_bucket_name(self):
        client = _mock_s3(object_lock_enabled=False)
        with self.assertRaises(RuntimeError) as ctx:
            verify_object_lock("my-bucket", client)
        self.assertIn("my-bucket", str(ctx.exception))


# ===========================================================================
# ensure_bucket
# ===========================================================================

class TestEnsureBucket(unittest.TestCase):

    def test_returns_true_when_created(self):
        client = _mock_s3(bucket_exists=False)
        result = ensure_bucket("my-bucket", "us-east-1", client)
        self.assertTrue(result)

    def test_returns_false_when_already_exists(self):
        client = _mock_s3(bucket_exists=True, object_lock_enabled=True)
        result = ensure_bucket("my-bucket", "us-east-1", client)
        self.assertFalse(result)

    def test_create_bucket_called_when_absent(self):
        client = _mock_s3(bucket_exists=False)
        ensure_bucket("my-bucket", "us-east-1", client)
        client.create_bucket.assert_called_once()

    def test_create_bucket_not_called_when_present(self):
        client = _mock_s3(bucket_exists=True, object_lock_enabled=True)
        ensure_bucket("my-bucket", "us-east-1", client)
        client.create_bucket.assert_not_called()

    def test_public_access_block_applied_on_create(self):
        client = _mock_s3(bucket_exists=False)
        ensure_bucket("my-bucket", "us-east-1", client)
        client.put_public_access_block.assert_called_once()

    def test_lifecycle_rules_applied_on_create(self):
        client = _mock_s3(bucket_exists=False)
        ensure_bucket("my-bucket", "us-east-1", client)
        client.put_bucket_lifecycle_configuration.assert_called_once()

    def test_public_access_block_not_applied_when_bucket_exists(self):
        client = _mock_s3(bucket_exists=True, object_lock_enabled=True)
        ensure_bucket("my-bucket", "us-east-1", client)
        client.put_public_access_block.assert_not_called()

    def test_object_lock_verified_when_bucket_exists(self):
        client = _mock_s3(bucket_exists=True, object_lock_enabled=True)
        ensure_bucket("my-bucket", "us-east-1", client)
        client.get_object_lock_configuration.assert_called_once()

    def test_raises_when_existing_bucket_has_no_object_lock(self):
        client = _mock_s3(bucket_exists=True, object_lock_enabled=False)
        with self.assertRaises(RuntimeError):
            ensure_bucket("my-bucket", "us-east-1", client)

    def test_empty_bucket_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            ensure_bucket("", "us-east-1", MagicMock())

    def test_create_bucket_failure_raises_runtime_error(self):
        client = _mock_s3(bucket_exists=False)
        client.create_bucket.side_effect = RuntimeError("BucketAlreadyExists")
        with self.assertRaises(RuntimeError) as ctx:
            ensure_bucket("my-bucket", "us-east-1", client)
        self.assertIn("my-bucket", str(ctx.exception))

    def test_public_access_block_failure_raises_runtime_error(self):
        client = _mock_s3(bucket_exists=False)
        client.put_public_access_block.side_effect = RuntimeError("AccessDenied")
        with self.assertRaises(RuntimeError) as ctx:
            ensure_bucket("my-bucket", "us-east-1", client)
        self.assertIn("my-bucket", str(ctx.exception))

    def test_lifecycle_failure_raises_runtime_error(self):
        client = _mock_s3(bucket_exists=False)
        client.put_bucket_lifecycle_configuration.side_effect = RuntimeError("AccessDenied")
        with self.assertRaises(RuntimeError) as ctx:
            ensure_bucket("my-bucket", "us-east-1", client)
        self.assertIn("my-bucket", str(ctx.exception))

    def test_create_bucket_receives_object_lock_flag(self):
        client = _mock_s3(bucket_exists=False)
        ensure_bucket("my-bucket", "us-east-1", client)
        kwargs = client.create_bucket.call_args.kwargs
        self.assertTrue(kwargs.get("ObjectLockEnabledForBucket"))


if __name__ == "__main__":
    unittest.main()

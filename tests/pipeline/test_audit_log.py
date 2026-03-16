"""
tests/pipeline/test_audit_log.py

Unit tests for pipeline/audit_log.py.

Coverage targets
----------------
AuditLogEntry       -- to_dict() round-trip, entry_id auto-generated,
                       answered_at auto-populated, all fields serialised

_s3_key()           -- format audit/{date}/{entry_id}.json, date derived
                       from answered_at, fallback to today on bad date

_log_stream_name()  -- returns YYYY-MM-DD string

_write_to_cloudwatch() -- put_log_events called, log group/stream created
                           if absent (errors ignored), message is JSON,
                           AuditLogFailure on put_log_events failure

_write_to_s3()      -- put_object called, correct bucket/key/body,
                       Object Lock COMPLIANCE set, retention ~7 years,
                       AuditLogFailure on empty bucket, AuditLogFailure
                       on put_object failure

write_audit_log()   -- both writes attempted, raises AuditLogFailure if
                       either fails, raises if both fail, succeeds when
                       both succeed, cloudwatch failure still attempts S3,
                       S3 failure still attempts CloudWatch
"""

from __future__ import annotations

import io
import json
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, call, patch

from pipeline.audit_log import (
    AuditLogEntry,
    AuditLogFailure,
    _log_stream_name,
    _s3_key,
    _write_to_cloudwatch,
    _write_to_s3,
    write_audit_log,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry(**kwargs) -> AuditLogEntry:
    defaults = dict(
        query_text="What do the documents say?",
        query_type="PROVENANCE",
        retrieved_at="2026-03-16T10:00:00+00:00",
        answered_at="2026-03-16T10:00:01+00:00",
        chunk_uuids=["uuid-001", "uuid-002"],
        provenance_tags=["PROVENANCE_DOJ_DIRECT"],
        confidence_tiers=["CORROBORATED"],
        lowest_tier="CORROBORATED",
        original_answer="The documents suggest...",
        safe_answer="The documents suggest...",
    )
    defaults.update(kwargs)
    return AuditLogEntry(**defaults)


def _mock_cloudwatch() -> MagicMock:
    client = MagicMock()
    client.create_log_group.return_value = {}
    client.create_log_stream.return_value = {}
    client.put_log_events.return_value = {"nextSequenceToken": "token"}
    return client


def _mock_s3() -> MagicMock:
    client = MagicMock()
    client.put_object.return_value = {}
    return client


# ===========================================================================
# AuditLogEntry
# ===========================================================================

class TestAuditLogEntry(unittest.TestCase):

    def test_entry_id_auto_generated(self):
        entry = AuditLogEntry()
        self.assertTrue(len(entry.entry_id) > 0)

    def test_entry_id_unique(self):
        self.assertNotEqual(AuditLogEntry().entry_id, AuditLogEntry().entry_id)

    def test_answered_at_auto_populated(self):
        entry = AuditLogEntry()
        self.assertTrue(len(entry.answered_at) > 0)

    def test_to_dict_contains_all_keys(self):
        entry = _entry()
        d = entry.to_dict()
        for key in (
            "entry_id", "query_text", "query_type", "retrieved_at",
            "answered_at", "chunk_uuids", "provenance_tags",
            "confidence_tiers", "lowest_tier", "original_answer",
            "safe_answer", "victim_scan_triggered", "inference_downgraded",
            "confidence_violation", "convergence_source_count",
        ):
            self.assertIn(key, d)

    def test_to_dict_values_match_fields(self):
        entry = _entry()
        d = entry.to_dict()
        self.assertEqual(d["query_text"], entry.query_text)
        self.assertEqual(d["query_type"], entry.query_type)
        self.assertEqual(d["chunk_uuids"], entry.chunk_uuids)

    def test_booleans_serialised(self):
        entry = _entry()
        entry.victim_scan_triggered = True
        d = entry.to_dict()
        self.assertTrue(d["victim_scan_triggered"])


# ===========================================================================
# _s3_key
# ===========================================================================

class TestS3Key(unittest.TestCase):

    def test_key_format(self):
        key = _s3_key("my-entry-id", "2026-03-16T10:00:00+00:00")
        self.assertEqual(key, "audit/2026-03-16/my-entry-id.json")

    def test_date_derived_from_answered_at(self):
        key = _s3_key("eid", "2025-01-05T00:00:00+00:00")
        self.assertTrue(key.startswith("audit/2025-01-05/"))

    def test_ends_with_json(self):
        key = _s3_key("eid", "2026-03-16T10:00:00+00:00")
        self.assertTrue(key.endswith(".json"))

    def test_entry_id_in_key(self):
        key = _s3_key("my-unique-id", "2026-03-16T10:00:00+00:00")
        self.assertIn("my-unique-id", key)


# ===========================================================================
# _write_to_cloudwatch
# ===========================================================================

class TestWriteToCloudWatch(unittest.TestCase):

    def test_put_log_events_called(self):
        cw = _mock_cloudwatch()
        _write_to_cloudwatch(_entry(), cw, "test-group")
        cw.put_log_events.assert_called_once()

    def test_correct_log_group_used(self):
        cw = _mock_cloudwatch()
        _write_to_cloudwatch(_entry(), cw, "my-log-group")
        self.assertEqual(
            cw.put_log_events.call_args.kwargs["logGroupName"], "my-log-group"
        )

    def test_message_is_valid_json(self):
        cw = _mock_cloudwatch()
        _write_to_cloudwatch(_entry(), cw, "test-group")
        message = cw.put_log_events.call_args.kwargs["logEvents"][0]["message"]
        parsed = json.loads(message)
        self.assertIn("entry_id", parsed)

    def test_create_log_group_errors_ignored(self):
        cw = _mock_cloudwatch()
        cw.create_log_group.side_effect = Exception("Already exists")
        # Must not raise
        _write_to_cloudwatch(_entry(), cw, "test-group")

    def test_create_log_stream_errors_ignored(self):
        cw = _mock_cloudwatch()
        cw.create_log_stream.side_effect = Exception("Already exists")
        _write_to_cloudwatch(_entry(), cw, "test-group")

    def test_put_log_events_failure_raises_audit_log_failure(self):
        cw = _mock_cloudwatch()
        cw.put_log_events.side_effect = RuntimeError("CW unavailable")
        with self.assertRaises(AuditLogFailure) as ctx:
            _write_to_cloudwatch(_entry(), cw, "test-group")
        self.assertIn("CloudWatch", str(ctx.exception))


# ===========================================================================
# _write_to_s3
# ===========================================================================

class TestWriteToS3(unittest.TestCase):

    def test_put_object_called(self):
        s3 = _mock_s3()
        _write_to_s3(_entry(), s3, "audit-bucket")
        s3.put_object.assert_called_once()

    def test_correct_bucket_used(self):
        s3 = _mock_s3()
        _write_to_s3(_entry(), s3, "audit-bucket")
        self.assertEqual(s3.put_object.call_args.kwargs["Bucket"], "audit-bucket")

    def test_key_format_correct(self):
        s3 = _mock_s3()
        entry = _entry()
        _write_to_s3(entry, s3, "audit-bucket")
        key = s3.put_object.call_args.kwargs["Key"]
        self.assertTrue(key.startswith("audit/"))
        self.assertTrue(key.endswith(".json"))

    def test_body_is_valid_json(self):
        s3 = _mock_s3()
        _write_to_s3(_entry(), s3, "audit-bucket")
        body = s3.put_object.call_args.kwargs["Body"]
        parsed = json.loads(body)
        self.assertIn("entry_id", parsed)

    def test_object_lock_compliance_set(self):
        s3 = _mock_s3()
        _write_to_s3(_entry(), s3, "audit-bucket")
        self.assertEqual(
            s3.put_object.call_args.kwargs["ObjectLockMode"], "COMPLIANCE"
        )

    def test_retain_until_date_approximately_7_years(self):
        s3 = _mock_s3()
        before = datetime.now(timezone.utc)
        _write_to_s3(_entry(), s3, "audit-bucket")
        retain = s3.put_object.call_args.kwargs["ObjectLockRetainUntilDate"]
        delta = retain - before
        self.assertGreater(delta.days, 6.9 * 365)
        self.assertLess(delta.days, 7.1 * 365)

    def test_empty_bucket_raises_audit_log_failure(self):
        with self.assertRaises(AuditLogFailure) as ctx:
            _write_to_s3(_entry(), _mock_s3(), "")
        self.assertIn("AUDIT_S3_BUCKET", str(ctx.exception))

    def test_put_object_failure_raises_audit_log_failure(self):
        s3 = _mock_s3()
        s3.put_object.side_effect = RuntimeError("S3 unavailable")
        with self.assertRaises(AuditLogFailure) as ctx:
            _write_to_s3(_entry(), s3, "audit-bucket")
        self.assertIn("S3", str(ctx.exception))


# ===========================================================================
# write_audit_log
# ===========================================================================

class TestWriteAuditLog(unittest.TestCase):

    def test_succeeds_when_both_succeed(self):
        # Must not raise
        write_audit_log(
            _entry(),
            cloudwatch_client=_mock_cloudwatch(),
            s3_client=_mock_s3(),
            audit_bucket="audit-bucket",
        )

    def test_raises_audit_log_failure_when_cloudwatch_fails(self):
        cw = _mock_cloudwatch()
        cw.put_log_events.side_effect = RuntimeError("CW down")
        with self.assertRaises(AuditLogFailure):
            write_audit_log(
                _entry(),
                cloudwatch_client=cw,
                s3_client=_mock_s3(),
                audit_bucket="audit-bucket",
            )

    def test_raises_audit_log_failure_when_s3_fails(self):
        s3 = _mock_s3()
        s3.put_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(AuditLogFailure):
            write_audit_log(
                _entry(),
                cloudwatch_client=_mock_cloudwatch(),
                s3_client=s3,
                audit_bucket="audit-bucket",
            )

    def test_s3_still_attempted_when_cloudwatch_fails(self):
        cw = _mock_cloudwatch()
        cw.put_log_events.side_effect = RuntimeError("CW down")
        s3 = _mock_s3()
        try:
            write_audit_log(
                _entry(),
                cloudwatch_client=cw,
                s3_client=s3,
                audit_bucket="audit-bucket",
            )
        except AuditLogFailure:
            pass
        s3.put_object.assert_called_once()

    def test_cloudwatch_still_attempted_when_s3_fails(self):
        s3 = _mock_s3()
        s3.put_object.side_effect = RuntimeError("S3 down")
        cw = _mock_cloudwatch()
        try:
            write_audit_log(
                _entry(),
                cloudwatch_client=cw,
                s3_client=s3,
                audit_bucket="audit-bucket",
            )
        except AuditLogFailure:
            pass
        cw.put_log_events.assert_called_once()

    def test_failure_message_mentions_hard_limit_5(self):
        cw = _mock_cloudwatch()
        cw.put_log_events.side_effect = RuntimeError("CW down")
        with self.assertRaises(AuditLogFailure) as ctx:
            write_audit_log(
                _entry(),
                cloudwatch_client=cw,
                s3_client=_mock_s3(),
                audit_bucket="audit-bucket",
            )
        self.assertIn("Hard Limit 5", str(ctx.exception))



class TestAuditLogEntryCreativeFlag(unittest.TestCase):

    def test_creative_content_suppressed_defaults_false(self):
        entry = AuditLogEntry()
        self.assertFalse(entry.creative_content_suppressed)

    def test_creative_content_suppressed_in_to_dict(self):
        entry = _entry()
        entry.creative_content_suppressed = True
        d = entry.to_dict()
        self.assertIn("creative_content_suppressed", d)
        self.assertTrue(d["creative_content_suppressed"])

    def test_creative_content_suppressed_false_in_to_dict(self):
        entry = _entry()
        d = entry.to_dict()
        self.assertIn("creative_content_suppressed", d)
        self.assertFalse(d["creative_content_suppressed"])


if __name__ == "__main__":
    unittest.main()

"""
tests/pipeline/test_s3_store.py

Unit tests for pipeline/s3_store.py.

Coverage targets
----------------
document_key()      -- key format, corpus_source as prefix
store_document()    -- put_object called with correct bucket/key/body,
                       content_type passed, victim_flag=False has no
                       Object Lock params, victim_flag=True adds
                       ObjectLockMode COMPLIANCE and RetainUntilDate,
                       retention date is ~7 years from now, empty
                       bucket_name raises ValueError, put_object
                       exception raises RuntimeError, returns key,
                       boto3 auto-constructed when s3_client=None skipped
                       (can't test without network)
retrieve_document() -- get_object called with correct bucket/key,
                       body bytes returned, empty bucket_name raises
                       ValueError, get_object exception raises RuntimeError
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from pipeline.s3_store import document_key, retrieve_document, store_document


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_s3(body: bytes = b"content") -> MagicMock:
    client = MagicMock()
    client.put_object.return_value = {"ETag": '"abc123"'}
    body_mock = MagicMock()
    body_mock.read.return_value = body
    client.get_object.return_value = {"Body": body_mock}
    return client


# ===========================================================================
# document_key
# ===========================================================================

class TestDocumentKey(unittest.TestCase):

    def test_key_format(self):
        key = document_key("DOJ_DIRECT", "uuid-001")
        self.assertEqual(key, "DOJ_DIRECT/uuid-001/raw")

    def test_corpus_source_is_prefix(self):
        key = document_key("corpus-abc", "uuid-002")
        self.assertTrue(key.startswith("corpus-abc/"))

    def test_document_uuid_in_key(self):
        key = document_key("DOJ_DIRECT", "uuid-xyz")
        self.assertIn("uuid-xyz", key)

    def test_ends_with_raw(self):
        key = document_key("DOJ_DIRECT", "uuid-001")
        self.assertTrue(key.endswith("/raw"))

    def test_different_corpus_sources_produce_different_keys(self):
        key_a = document_key("corpus-a", "uuid-001")
        key_b = document_key("corpus-b", "uuid-001")
        self.assertNotEqual(key_a, key_b)


# ===========================================================================
# store_document
# ===========================================================================

class TestStoreDocument(unittest.TestCase):

    def setUp(self):
        self.s3 = _mock_s3()

    def _store(self, victim_flag=False, **kwargs):
        return store_document(
            document_uuid="uuid-001",
            corpus_source="DOJ_DIRECT",
            content=b"raw bytes",
            victim_flag=victim_flag,
            s3_client=self.s3,
            bucket_name="test-bucket",
            **kwargs,
        )

    def test_put_object_called(self):
        self._store()
        self.s3.put_object.assert_called_once()

    def test_correct_bucket_used(self):
        self._store()
        self.assertEqual(self.s3.put_object.call_args.kwargs["Bucket"], "test-bucket")

    def test_correct_key_used(self):
        self._store()
        self.assertEqual(
            self.s3.put_object.call_args.kwargs["Key"], "DOJ_DIRECT/uuid-001/raw"
        )

    def test_content_passed_as_body(self):
        self._store()
        self.assertEqual(self.s3.put_object.call_args.kwargs["Body"], b"raw bytes")

    def test_content_type_default(self):
        self._store()
        self.assertEqual(
            self.s3.put_object.call_args.kwargs["ContentType"],
            "application/octet-stream",
        )

    def test_content_type_custom(self):
        self._store(content_type="application/pdf")
        self.assertEqual(
            self.s3.put_object.call_args.kwargs["ContentType"], "application/pdf"
        )

    def test_returns_s3_key(self):
        key = self._store()
        self.assertEqual(key, "DOJ_DIRECT/uuid-001/raw")

    def test_non_victim_has_no_object_lock(self):
        self._store(victim_flag=False)
        kwargs = self.s3.put_object.call_args.kwargs
        self.assertNotIn("ObjectLockMode", kwargs)
        self.assertNotIn("ObjectLockRetainUntilDate", kwargs)

    def test_victim_flag_adds_compliance_lock(self):
        self._store(victim_flag=True)
        kwargs = self.s3.put_object.call_args.kwargs
        self.assertEqual(kwargs["ObjectLockMode"], "COMPLIANCE")

    def test_victim_flag_sets_retain_until_date(self):
        self._store(victim_flag=True)
        kwargs = self.s3.put_object.call_args.kwargs
        self.assertIn("ObjectLockRetainUntilDate", kwargs)

    def test_victim_retention_is_approximately_7_years(self):
        before = datetime.now(timezone.utc)
        self._store(victim_flag=True)
        after = datetime.now(timezone.utc)
        retain_until = self.s3.put_object.call_args.kwargs["ObjectLockRetainUntilDate"]
        # Should be between 6.9 and 7.1 years from now
        min_days = 6.9 * 365
        max_days = 7.1 * 365
        delta = retain_until - before
        self.assertGreater(delta.days, min_days)
        self.assertLess(delta.days, max_days)

    def test_empty_bucket_name_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            store_document(
                document_uuid="uuid-001",
                corpus_source="DOJ_DIRECT",
                content=b"bytes",
                s3_client=self.s3,
                bucket_name="",
            )
        self.assertIn("bucket_name", str(ctx.exception))

    def test_put_object_exception_raises_runtime_error(self):
        self.s3.put_object.side_effect = RuntimeError("S3 unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            self._store()
        self.assertIn("uuid-001", str(ctx.exception))

    def test_runtime_error_message_includes_key(self):
        self.s3.put_object.side_effect = RuntimeError("S3 unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            self._store()
        self.assertIn("DOJ_DIRECT/uuid-001/raw", str(ctx.exception))


# ===========================================================================
# retrieve_document
# ===========================================================================

class TestRetrieveDocument(unittest.TestCase):

    def setUp(self):
        self.s3 = _mock_s3(body=b"retrieved content")

    def _retrieve(self, **kwargs):
        return retrieve_document(
            document_uuid="uuid-001",
            corpus_source="DOJ_DIRECT",
            s3_client=self.s3,
            bucket_name="test-bucket",
            **kwargs,
        )

    def test_get_object_called(self):
        self._retrieve()
        self.s3.get_object.assert_called_once()

    def test_correct_bucket_used(self):
        self._retrieve()
        self.assertEqual(
            self.s3.get_object.call_args.kwargs["Bucket"], "test-bucket"
        )

    def test_correct_key_used(self):
        self._retrieve()
        self.assertEqual(
            self.s3.get_object.call_args.kwargs["Key"], "DOJ_DIRECT/uuid-001/raw"
        )

    def test_returns_bytes(self):
        result = self._retrieve()
        self.assertIsInstance(result, bytes)

    def test_returns_correct_content(self):
        result = self._retrieve()
        self.assertEqual(result, b"retrieved content")

    def test_empty_bucket_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            retrieve_document(
                document_uuid="uuid-001",
                corpus_source="DOJ_DIRECT",
                s3_client=self.s3,
                bucket_name="",
            )

    def test_get_object_exception_raises_runtime_error(self):
        self.s3.get_object.side_effect = RuntimeError("Key not found")
        with self.assertRaises(RuntimeError) as ctx:
            self._retrieve()
        self.assertIn("uuid-001", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

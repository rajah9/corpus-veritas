"""
tests/pipeline/test_sanitizer.py

Unit tests for pipeline/sanitizer.py (Sub-Module 1C: PII Sanitization).

Coverage targets
----------------
_chunk_text()            -- short text (one chunk), multi-chunk, empty input
_text_near()             -- term present, term absent, boundary conditions
_extract_age_value()     -- integer found, non-numeric, no digits
_analyse_entities()      -- all four rules, rule priority, no entities
detect_pii()             -- clean doc, victim_flag paths, review paths,
                            Comprehend exception handling, multi-chunk offset
                            adjustment, empty text
queue_for_human_review() -- message sent, message body content, missing
                            queue URL, SQS exception handling,
                            queue URL from env var
sanitize_document()      -- clean (not queued), victim-flagged (queued),
                            review-required (queued), queue called once per doc

Temp resource strategy
----------------------
No temp files needed. All AWS clients are injected MagicMocks.
setUp/tearDown used only where environment variables are manipulated.
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import MagicMock, call, patch

from pipeline.sanitizer import (
    PIIDetectionResult,
    _COMPREHEND_BYTE_LIMIT,
    _MINOR_AGE_THRESHOLD,
    _MULTI_NAME_REVIEW_THRESHOLD,
    _PROXIMITY_WINDOW,
    _analyse_entities,
    _chunk_text,
    _extract_age_value,
    _text_near,
    detect_pii,
    queue_for_human_review,
    sanitize_document,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_entity(begin: int, end: int, score: float = 0.99) -> dict:
    return {"Type": "NAME", "BeginOffset": begin, "EndOffset": end, "Score": score}


def _age_entity(begin: int, end: int, score: float = 0.99) -> dict:
    return {"Type": "AGE", "BeginOffset": begin, "EndOffset": end, "Score": score}


def _mock_comprehend(entities: list) -> MagicMock:
    """Return a Comprehend client mock that yields the given entities."""
    client = MagicMock()
    client.detect_pii_entities.return_value = {"Entities": entities}
    return client


def _mock_sqs() -> MagicMock:
    client = MagicMock()
    client.send_message.return_value = {"MessageId": "mock-msg-id"}
    return client


# ===========================================================================
# _chunk_text
# ===========================================================================

class TestChunkText(unittest.TestCase):

    def test_short_text_returns_one_chunk(self):
        text = "Hello world this is a short document."
        chunks = _chunk_text(text, max_bytes=_COMPREHEND_BYTE_LIMIT)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_empty_string_returns_one_empty_chunk(self):
        chunks = _chunk_text("", max_bytes=100)
        self.assertEqual(chunks, [""])

    def test_single_word_returns_one_chunk(self):
        chunks = _chunk_text("word", max_bytes=100)
        self.assertEqual(chunks, ["word"])

    def test_long_text_splits_into_multiple_chunks(self):
        # Create text that will definitely exceed 100 bytes
        text = " ".join(["hello"] * 50)   # ~300 bytes
        chunks = _chunk_text(text, max_bytes=100)
        self.assertGreater(len(chunks), 1)

    def test_all_words_preserved_across_chunks(self):
        text = " ".join([f"word{i}" for i in range(100)])
        chunks = _chunk_text(text, max_bytes=50)
        reconstructed = " ".join(chunks)
        # Every word in original appears in reconstructed
        for word in text.split():
            self.assertIn(word, reconstructed)

    def test_no_chunk_exceeds_max_bytes(self):
        text = " ".join(["abcde"] * 200)
        chunks = _chunk_text(text, max_bytes=50)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.encode("utf-8")), 50 + 6)
            # +6: one word may overshoot by at most one word length


# ===========================================================================
# _text_near
# ===========================================================================

class TestTextNear(unittest.TestCase):

    def setUp(self):
        # "John Smith is a victim of trafficking"
        #  0123456789...
        self.text = "John Smith is a victim of trafficking"

    def test_term_within_window_returns_true(self):
        # "John" spans 0-4; "victim" is at 16 -- within 150 chars
        self.assertTrue(
            _text_near(self.text, 0, 4, _PROXIMITY_WINDOW, frozenset({"victim"}))
        )

    def test_term_outside_window_returns_false(self):
        # tiny window of 5 chars around "John" will not reach "victim"
        self.assertFalse(
            _text_near(self.text, 0, 4, 5, frozenset({"victim"}))
        )

    def test_no_matching_term_returns_false(self):
        self.assertFalse(
            _text_near(self.text, 0, 4, _PROXIMITY_WINDOW, frozenset({"unrelated"}))
        )

    def test_case_insensitive(self):
        text = "John Smith is VICTIM"
        self.assertTrue(
            _text_near(text, 0, 4, _PROXIMITY_WINDOW, frozenset({"victim"}))
        )

    def test_start_boundary_clamped(self):
        # entity at offset 0 -- should not raise
        self.assertFalse(
            _text_near("short", 0, 5, 100, frozenset({"absent"}))
        )

    def test_end_boundary_clamped(self):
        text = "some text here"
        # entity at end of string -- should not raise
        self.assertFalse(
            _text_near(text, len(text) - 4, len(text), 100, frozenset({"absent"}))
        )


# ===========================================================================
# _extract_age_value
# ===========================================================================

class TestExtractAgeValue(unittest.TestCase):

    def test_simple_integer(self):
        text = "She was 15 years old."
        entity = {"BeginOffset": 8, "EndOffset": 10}  # "15"
        self.assertEqual(_extract_age_value(text, entity), 15)

    def test_non_numeric_returns_none(self):
        text = "She was mid-thirties."
        entity = {"BeginOffset": 8, "EndOffset": 20}
        self.assertIsNone(_extract_age_value(text, entity))

    def test_age_with_surrounding_text(self):
        text = "age: 17 at the time"
        entity = {"BeginOffset": 0, "EndOffset": 7}  # "age: 17"
        self.assertEqual(_extract_age_value(text, entity), 17)

    def test_zero_returns_zero(self):
        text = "aged 0"
        entity = {"BeginOffset": 5, "EndOffset": 6}
        self.assertEqual(_extract_age_value(text, entity), 0)


# ===========================================================================
# _analyse_entities
# ===========================================================================

class TestAnalyseEntities(unittest.TestCase):

    def test_no_entities_returns_clean(self):
        v, r, reason = _analyse_entities("clean document text", [])
        self.assertFalse(v)
        self.assertFalse(r)
        self.assertIsNone(reason)

    def test_name_near_victim_triggers_victim_flag(self):
        text = "Jane Doe is a victim of trafficking"
        # "Jane Doe" spans 0-8
        entities = [_name_entity(0, 8)]
        v, r, reason = _analyse_entities(text, entities)
        self.assertTrue(v)
        self.assertIn("victim-indicator", reason)

    def test_name_near_minor_term_triggers_victim_flag(self):
        text = "The minor John Smith was present."
        entities = [_name_entity(10, 20)]  # "John Smith"
        v, r, reason = _analyse_entities(text, entities)
        self.assertTrue(v)

    def test_name_age_under_18_triggers_victim_flag(self):
        text = "John Smith age 14 was at the location."
        #       0         1         2         3
        entities = [
            _name_entity(0, 10),   # "John Smith"
            _age_entity(11, 17),   # "age 14" -- extract_age -> 14
        ]
        v, r, reason = _analyse_entities(text, entities)
        self.assertTrue(v)
        self.assertIn("minor identification", reason)

    def test_name_age_over_18_does_not_trigger_victim_flag(self):
        text = "John Smith age 35 was present."
        entities = [
            _name_entity(0, 10),
            _age_entity(11, 17),   # "age 35" -> 35 >= 18
        ]
        v, r, _ = _analyse_entities(text, entities)
        self.assertFalse(v)

    def test_name_near_epstein_triggers_review_not_victim(self):
        text = "John Smith met with Epstein at the island."
        entities = [_name_entity(0, 10)]
        v, r, reason = _analyse_entities(text, entities)
        self.assertFalse(v)
        self.assertTrue(r)
        self.assertIn("case-adjacent", reason)

    def test_victim_flag_takes_priority_over_review(self):
        # Both victim and review indicators present -- victim wins
        text = "Jane Doe, a victim, met with Epstein."
        entities = [_name_entity(0, 8)]
        v, r, _ = _analyse_entities(text, entities)
        self.assertTrue(v)

    def test_multi_name_threshold_triggers_review(self):
        text = " ".join(
            [f"Person{i}" for i in range(_MULTI_NAME_REVIEW_THRESHOLD)]
        ) + " were present."
        # Each name is a NAME entity with no sensitive proximity
        entities = [
            _name_entity(i * 8, i * 8 + 7)
            for i in range(_MULTI_NAME_REVIEW_THRESHOLD)
        ]
        v, r, reason = _analyse_entities(text, entities)
        self.assertFalse(v)
        self.assertTrue(r)
        self.assertIn("NAME entities", reason)

    def test_below_multi_name_threshold_no_review(self):
        text = "Alice Bob were present."
        entities = [_name_entity(0, 5), _name_entity(6, 9)]
        # 2 < _MULTI_NAME_REVIEW_THRESHOLD (3)
        v, r, _ = _analyse_entities(text, entities)
        self.assertFalse(v)
        self.assertFalse(r)

    def test_age_without_name_entity_does_not_trigger(self):
        # AGE entity alone, no NAME entity -> rule 2 requires both
        text = "The subject was 14 years old."
        entities = [_age_entity(16, 18)]
        v, r, _ = _analyse_entities(text, entities)
        self.assertFalse(v)
        self.assertFalse(r)

    def test_review_reason_is_none_when_clean(self):
        _, _, reason = _analyse_entities("plain text no names", [])
        self.assertIsNone(reason)

    def test_review_reason_set_when_flagged(self):
        text = "John Smith is a victim."
        entities = [_name_entity(0, 10)]
        _, _, reason = _analyse_entities(text, entities)
        self.assertIsNotNone(reason)


# ===========================================================================
# detect_pii
# ===========================================================================

class TestDetectPii(unittest.TestCase):

    def test_returns_pii_detection_result_type(self):
        client = _mock_comprehend([])
        result = detect_pii("uuid-001", "clean document", client)
        self.assertIsInstance(result, PIIDetectionResult)

    def test_document_uuid_preserved(self):
        client = _mock_comprehend([])
        result = detect_pii("uuid-abc", "text", client)
        self.assertEqual(result.document_uuid, "uuid-abc")

    def test_clean_document_no_flags(self):
        client = _mock_comprehend([])
        result = detect_pii("uuid-001", "No PII here at all.", client)
        self.assertFalse(result.victim_flag)
        self.assertFalse(result.requires_human_review)
        self.assertIsNone(result.review_reason)

    def test_entities_stored_in_result(self):
        entities = [_name_entity(0, 5)]
        client = _mock_comprehend(entities)
        result = detect_pii("uuid-001", "Alice was here.", client)
        self.assertEqual(len(result.pii_entities_detected), 1)

    def test_victim_flag_set_when_name_near_victim_term(self):
        text = "Jane Doe is a victim of abuse."
        entities = [_name_entity(0, 8)]
        client = _mock_comprehend(entities)
        result = detect_pii("uuid-001", text, client)
        self.assertTrue(result.victim_flag)

    def test_review_flag_set_when_name_near_epstein(self):
        text = "John Smith attended Epstein's party."
        entities = [_name_entity(0, 10)]
        client = _mock_comprehend(entities)
        result = detect_pii("uuid-001", text, client)
        self.assertFalse(result.victim_flag)
        self.assertTrue(result.requires_human_review)

    def test_comprehend_exception_returns_review_required(self):
        client = MagicMock()
        client.detect_pii_entities.side_effect = RuntimeError("API unavailable")
        result = detect_pii("uuid-001", "some text", client)
        self.assertFalse(result.victim_flag)
        self.assertTrue(result.requires_human_review)
        self.assertIn("Comprehend API error", result.review_reason)

    def test_comprehend_called_once_for_short_text(self):
        client = _mock_comprehend([])
        detect_pii("uuid-001", "short text", client)
        self.assertEqual(client.detect_pii_entities.call_count, 1)

    def test_comprehend_called_multiple_times_for_long_text(self):
        # Build text exceeding _COMPREHEND_BYTE_LIMIT
        long_text = " ".join(["word"] * 2000)  # ~10,000 bytes
        client = _mock_comprehend([])
        detect_pii("uuid-001", long_text, client)
        self.assertGreater(client.detect_pii_entities.call_count, 1)

    def test_entity_offsets_adjusted_for_second_chunk(self):
        """
        Entities returned for the second chunk must have their offsets
        shifted by the length of the first chunk so that they refer to
        positions in the original full text.

        _COMPREHEND_BYTE_LIMIT = 4500. Each 6-char word ("abcde" + space) = 6
        bytes, so ceil(4500/6) = 750 words fill one chunk exactly. 800 words
        of "abcde" + 10 words of "bbbbb" guarantees exactly two chunks.
        """
        # 800 * 6 = 4800 bytes > 4500 limit -> splits into two chunks
        first_chunk_words = ["abcde"] * 800
        second_chunk_words = ["bbbbb"] * 10
        text = " ".join(first_chunk_words) + " " + " ".join(second_chunk_words)

        # Confirm the fixture actually produces two chunks (guard against
        # future constant changes silently breaking this test)
        from pipeline.sanitizer import _chunk_text, _COMPREHEND_BYTE_LIMIT
        chunks = _chunk_text(text, _COMPREHEND_BYTE_LIMIT)
        self.assertGreater(len(chunks), 1, "Fixture text must span >1 chunk")

        first_chunk_len = len(chunks[0])

        # First call: no entities; second call: one NAME entity at local offset 0
        client = MagicMock()
        client.detect_pii_entities.side_effect = [
            {"Entities": []},                   # first chunk — clean
            {"Entities": [_name_entity(0, 5)]}, # second chunk — entity at start
        ]
        result = detect_pii("uuid-001", text, client)
        self.assertEqual(client.detect_pii_entities.call_count, len(chunks))
        self.assertEqual(len(result.pii_entities_detected), 1)
        # BeginOffset must equal first_chunk_len + 1 (the space separator)
        self.assertEqual(
            result.pii_entities_detected[0]["BeginOffset"],
            first_chunk_len + 1,
        )

    def test_empty_text_returns_clean_result(self):
        client = _mock_comprehend([])
        result = detect_pii("uuid-001", "", client)
        self.assertFalse(result.victim_flag)
        self.assertFalse(result.requires_human_review)


# ===========================================================================
# queue_for_human_review
# ===========================================================================

class TestQueueForHumanReview(unittest.TestCase):

    def _flagged_result(self, victim=True) -> PIIDetectionResult:
        return PIIDetectionResult(
            document_uuid="uuid-review-001",
            pii_entities_detected=[_name_entity(0, 5)],
            victim_flag=victim,
            requires_human_review=not victim,
            review_reason="Test reason",
        )

    def test_send_message_called(self):
        sqs = _mock_sqs()
        queue_for_human_review(self._flagged_result(), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        sqs.send_message.assert_called_once()

    def test_message_body_is_valid_json(self):
        sqs = _mock_sqs()
        queue_for_human_review(self._flagged_result(), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        body = sqs.send_message.call_args.kwargs["MessageBody"]
        parsed = json.loads(body)
        self.assertIsInstance(parsed, dict)

    def test_message_body_contains_document_uuid(self):
        sqs = _mock_sqs()
        queue_for_human_review(self._flagged_result(), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        body = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
        self.assertEqual(body["document_uuid"], "uuid-review-001")

    def test_message_body_contains_victim_flag(self):
        sqs = _mock_sqs()
        queue_for_human_review(self._flagged_result(victim=True), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        body = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
        self.assertTrue(body["victim_flag"])

    def test_message_body_omits_entity_text_values(self):
        """Entity text must never appear in SQS messages -- offsets only."""
        sqs = _mock_sqs()
        queue_for_human_review(self._flagged_result(), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        body = json.loads(sqs.send_message.call_args.kwargs["MessageBody"])
        for offset_entry in body.get("entity_offsets", []):
            self.assertNotIn("text", offset_entry)
            self.assertNotIn("value", offset_entry)
            self.assertIn("begin_offset", offset_entry)
            self.assertIn("end_offset", offset_entry)

    def test_correct_queue_url_used(self):
        sqs = _mock_sqs()
        url = "https://sqs.us-east-1.amazonaws.com/999/myqueue"
        queue_for_human_review(self._flagged_result(), sqs_client=sqs, queue_url=url)
        self.assertEqual(
            sqs.send_message.call_args.kwargs["QueueUrl"], url
        )

    def test_missing_queue_url_does_not_raise(self):
        sqs = _mock_sqs()
        with patch.dict(os.environ, {}, clear=True):
            # SANITIZER_QUEUE_URL not set, no queue_url arg
            queue_for_human_review(self._flagged_result(), sqs_client=sqs,
                                    queue_url="")
        sqs.send_message.assert_not_called()

    def test_queue_url_from_env_var(self):
        sqs = _mock_sqs()
        env_url = "https://sqs.us-east-1.amazonaws.com/123/envqueue"
        with patch("pipeline.sanitizer.SANITIZER_QUEUE_URL", env_url):
            queue_for_human_review(self._flagged_result(), sqs_client=sqs)
        self.assertEqual(
            sqs.send_message.call_args.kwargs["QueueUrl"], env_url
        )

    def test_sqs_exception_does_not_raise(self):
        sqs = MagicMock()
        sqs.send_message.side_effect = RuntimeError("SQS unavailable")
        # Must not raise -- failure is logged only
        queue_for_human_review(self._flagged_result(), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")

    def test_message_attribute_victim_flag_string(self):
        sqs = _mock_sqs()
        queue_for_human_review(self._flagged_result(victim=True), sqs_client=sqs,
                                queue_url="https://sqs.us-east-1.amazonaws.com/123/test")
        attrs = sqs.send_message.call_args.kwargs["MessageAttributes"]
        self.assertEqual(attrs["victim_flag"]["StringValue"], "true")


# ===========================================================================
# sanitize_document
# ===========================================================================

class TestSanitizeDocument(unittest.TestCase):

    QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/123/test"

    def test_returns_pii_detection_result(self):
        comprehend = _mock_comprehend([])
        sqs = _mock_sqs()
        result = sanitize_document("uuid-001", "clean text",
                                   comprehend, sqs, self.QUEUE_URL)
        self.assertIsInstance(result, PIIDetectionResult)

    def test_clean_document_not_queued(self):
        comprehend = _mock_comprehend([])
        sqs = _mock_sqs()
        sanitize_document("uuid-001", "No PII here.", comprehend, sqs, self.QUEUE_URL)
        sqs.send_message.assert_not_called()

    def test_victim_flagged_document_is_queued(self):
        text = "Jane Doe is a victim of abuse."
        comprehend = _mock_comprehend([_name_entity(0, 8)])
        sqs = _mock_sqs()
        result = sanitize_document("uuid-001", text, comprehend, sqs, self.QUEUE_URL)
        self.assertTrue(result.victim_flag)
        sqs.send_message.assert_called_once()

    def test_review_required_document_is_queued(self):
        text = "John Smith was seen with Epstein."
        comprehend = _mock_comprehend([_name_entity(0, 10)])
        sqs = _mock_sqs()
        result = sanitize_document("uuid-001", text, comprehend, sqs, self.QUEUE_URL)
        self.assertTrue(result.requires_human_review)
        sqs.send_message.assert_called_once()

    def test_queue_called_exactly_once_per_document(self):
        text = "Jane Doe is a victim of abuse."
        comprehend = _mock_comprehend([_name_entity(0, 8)])
        sqs = _mock_sqs()
        sanitize_document("uuid-001", text, comprehend, sqs, self.QUEUE_URL)
        self.assertEqual(sqs.send_message.call_count, 1)

    def test_victim_flag_in_returned_result(self):
        text = "Jane Doe is a minor victim."
        comprehend = _mock_comprehend([_name_entity(0, 8)])
        sqs = _mock_sqs()
        result = sanitize_document("uuid-001", text, comprehend, sqs, self.QUEUE_URL)
        self.assertTrue(result.victim_flag)


if __name__ == "__main__":
    unittest.main()
"""
tests/pipeline/test_ner_extractor.py

Unit tests for pipeline/ner_extractor.py.

Coverage targets
----------------
extract_entities()          -- Comprehend called with text + language "en",
                               entities above threshold returned, entities
                               below threshold discarded, unknown types
                               discarded, empty text returns empty list,
                               comprehend exception raises RuntimeError,
                               OTHER type mapped to CASE_NUMBER
extract_entities_for_chunk()-- document_uuid added to each entity,
                               delegates to extract_entities
deduplicate_entities()      -- same (text, type) deduped to highest
                               confidence, different types kept separate,
                               empty list handled, case-insensitive dedup
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from pipeline.ner_extractor import (
    ENTITY_CONFIDENCE_THRESHOLD,
    deduplicate_entities,
    extract_entities,
    extract_entities_for_chunk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _comprehend(entities: list) -> MagicMock:
    client = MagicMock()
    client.detect_entities.return_value = {"Entities": entities}
    return client


def _ent(text: str, etype: str, score: float, begin: int = 0, end: int = 5) -> dict:
    return {
        "Text": text,
        "Type": etype,
        "Score": score,
        "BeginOffset": begin,
        "EndOffset": end,
    }


# ===========================================================================
# extract_entities
# ===========================================================================

class TestExtractEntities(unittest.TestCase):

    def test_empty_text_returns_empty_list(self):
        result = extract_entities("", MagicMock())
        self.assertEqual(result, [])

    def test_whitespace_text_returns_empty_list(self):
        result = extract_entities("   ", MagicMock())
        self.assertEqual(result, [])

    def test_comprehend_called_with_en_language(self):
        client = _comprehend([])
        extract_entities("some text", client)
        self.assertEqual(client.detect_entities.call_args.kwargs["LanguageCode"], "en")

    def test_comprehend_called_with_text(self):
        client = _comprehend([])
        extract_entities("some text", client)
        self.assertEqual(client.detect_entities.call_args.kwargs["Text"], "some text")

    def test_entity_above_threshold_returned(self):
        client = _comprehend([_ent("John Doe", "PERSON", ENTITY_CONFIDENCE_THRESHOLD)])
        result = extract_entities("text", client)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "John Doe")

    def test_entity_below_threshold_discarded(self):
        client = _comprehend([_ent("John Doe", "PERSON", ENTITY_CONFIDENCE_THRESHOLD - 0.01)])
        result = extract_entities("text", client)
        self.assertEqual(result, [])

    def test_unknown_type_discarded(self):
        client = _comprehend([_ent("42", "QUANTITY", 0.99)])
        result = extract_entities("text", client)
        self.assertEqual(result, [])

    def test_other_type_mapped_to_case_number(self):
        client = _comprehend([_ent("Case 99-cr-001", "OTHER", 0.95)])
        result = extract_entities("text", client)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "CASE_NUMBER")

    def test_person_type_mapped_correctly(self):
        client = _comprehend([_ent("Jane Doe", "PERSON", 0.98)])
        result = extract_entities("text", client)
        self.assertEqual(result[0]["type"], "PERSON")

    def test_organization_type_mapped_correctly(self):
        client = _comprehend([_ent("FBI", "ORGANIZATION", 0.99)])
        result = extract_entities("text", client)
        self.assertEqual(result[0]["type"], "ORGANIZATION")

    def test_location_type_mapped_correctly(self):
        client = _comprehend([_ent("New York", "LOCATION", 0.97)])
        result = extract_entities("text", client)
        self.assertEqual(result[0]["type"], "LOCATION")

    def test_date_type_mapped_correctly(self):
        client = _comprehend([_ent("January 2005", "DATE", 0.95)])
        result = extract_entities("text", client)
        self.assertEqual(result[0]["type"], "DATE")

    def test_result_fields_present(self):
        client = _comprehend([_ent("John Doe", "PERSON", 0.98, 0, 8)])
        result = extract_entities("text", client)
        self.assertIn("text", result[0])
        self.assertIn("type", result[0])
        self.assertIn("confidence", result[0])
        self.assertIn("begin_offset", result[0])
        self.assertIn("end_offset", result[0])

    def test_offsets_preserved(self):
        client = _comprehend([_ent("John Doe", "PERSON", 0.98, 10, 18)])
        result = extract_entities("text", client)
        self.assertEqual(result[0]["begin_offset"], 10)
        self.assertEqual(result[0]["end_offset"], 18)

    def test_multiple_entities_returned(self):
        client = _comprehend([
            _ent("John Doe", "PERSON", 0.98),
            _ent("FBI", "ORGANIZATION", 0.99),
        ])
        result = extract_entities("text", client)
        self.assertEqual(len(result), 2)

    def test_comprehend_exception_raises_runtime_error(self):
        client = MagicMock()
        client.detect_entities.side_effect = RuntimeError("Comprehend down")
        with self.assertRaises(RuntimeError) as ctx:
            extract_entities("some text", client)
        self.assertIn("detect_entities", str(ctx.exception))

    def test_custom_threshold_respected(self):
        client = _comprehend([_ent("John Doe", "PERSON", 0.70)])
        result = extract_entities("text", client, confidence_threshold=0.60)
        self.assertEqual(len(result), 1)


# ===========================================================================
# extract_entities_for_chunk
# ===========================================================================

class TestExtractEntitiesForChunk(unittest.TestCase):

    def test_document_uuid_added_to_each_entity(self):
        client = _comprehend([_ent("John Doe", "PERSON", 0.98)])
        result = extract_entities_for_chunk("text", "uuid-001", client)
        self.assertEqual(result[0]["document_uuid"], "uuid-001")

    def test_empty_text_returns_empty_list(self):
        result = extract_entities_for_chunk("", "uuid-001", MagicMock())
        self.assertEqual(result, [])

    def test_multiple_entities_all_get_uuid(self):
        client = _comprehend([
            _ent("John Doe", "PERSON", 0.98),
            _ent("FBI", "ORGANIZATION", 0.99),
        ])
        result = extract_entities_for_chunk("text", "uuid-xyz", client)
        for ent in result:
            self.assertEqual(ent["document_uuid"], "uuid-xyz")


# ===========================================================================
# deduplicate_entities
# ===========================================================================

class TestDeduplicateEntities(unittest.TestCase):

    def test_empty_list_returns_empty(self):
        self.assertEqual(deduplicate_entities([]), [])

    def test_single_entity_returned(self):
        entities = [{"text": "John Doe", "type": "PERSON", "confidence": 0.98}]
        result = deduplicate_entities(entities)
        self.assertEqual(len(result), 1)

    def test_duplicate_text_type_deduped(self):
        entities = [
            {"text": "John Doe", "type": "PERSON", "confidence": 0.92},
            {"text": "John Doe", "type": "PERSON", "confidence": 0.98},
        ]
        result = deduplicate_entities(entities)
        self.assertEqual(len(result), 1)

    def test_highest_confidence_retained(self):
        entities = [
            {"text": "John Doe", "type": "PERSON", "confidence": 0.92},
            {"text": "John Doe", "type": "PERSON", "confidence": 0.98},
        ]
        result = deduplicate_entities(entities)
        self.assertAlmostEqual(result[0]["confidence"], 0.98)

    def test_case_insensitive_dedup(self):
        entities = [
            {"text": "john doe", "type": "PERSON", "confidence": 0.92},
            {"text": "John Doe", "type": "PERSON", "confidence": 0.95},
        ]
        result = deduplicate_entities(entities)
        self.assertEqual(len(result), 1)

    def test_different_types_kept_separate(self):
        entities = [
            {"text": "Trump", "type": "PERSON", "confidence": 0.95},
            {"text": "Trump", "type": "ORGANIZATION", "confidence": 0.90},
        ]
        result = deduplicate_entities(entities)
        self.assertEqual(len(result), 2)

    def test_different_names_kept_separate(self):
        entities = [
            {"text": "John Doe", "type": "PERSON", "confidence": 0.95},
            {"text": "Jane Doe", "type": "PERSON", "confidence": 0.93},
        ]
        result = deduplicate_entities(entities)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()


# ===========================================================================
# DynamoDB entity table
# ===========================================================================

from pipeline.ner_extractor import (
    ENTITY_TABLE_NAME,
    get_entity_record,
    query_entities_by_document,
    query_entities_by_type,
    upsert_entity_record,
)


def _mock_dynamodb(item: dict = None) -> MagicMock:
    """DynamoDB client mock with pre-canned get_item and query responses."""
    client = MagicMock()
    client.update_item.return_value = {}
    # ConditionalCheckFailedException for the confidence phase-2 update
    client.exceptions.ConditionalCheckFailedException = type(
        "ConditionalCheckFailedException", (Exception,), {}
    )
    client.get_item.return_value = {"Item": item} if item else {}
    client.query.return_value = {"Items": [item] if item else []}
    return client


def _dynamo_item(
    name: str = "jeffrey epstein",
    etype: str = "PERSON",
    confidence: float = 0.99,
    victim: bool = False,
    duuid: str = "uuid-001",
) -> dict:
    """Return a DynamoDB typed attribute dict."""
    item = {
        "canonical_name":     {"S": name},
        "entity_type":        {"S": etype},
        "surface_forms":      {"SS": ["Jeffrey Epstein"]},
        "document_uuids":     {"SS": [duuid]},
        "first_document_uuid":{"S": duuid},
        "confidence":         {"N": str(confidence)},
        "occurrence_count":   {"N": "1"},
        "updated_at":         {"S": "2026-03-16T00:00:00+00:00"},
    }
    if victim:
        item["victim_flag"] = {"S": "true"}
    return item


class TestUpsertEntityRecord(unittest.TestCase):

    def test_update_item_called(self):
        db = _mock_dynamodb()
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
            dynamodb_client=db,
        )
        db.update_item.assert_called()

    def test_correct_table_used(self):
        db = _mock_dynamodb()
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
            dynamodb_client=db,
        )
        first_call = db.update_item.call_args_list[0]
        self.assertEqual(first_call.kwargs["TableName"], ENTITY_TABLE_NAME)

    def test_correct_key_used(self):
        db = _mock_dynamodb()
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
            dynamodb_client=db,
        )
        key = db.update_item.call_args_list[0].kwargs["Key"]
        self.assertEqual(key["canonical_name"]["S"], "jeffrey epstein")
        self.assertEqual(key["entity_type"]["S"], "PERSON")

    def test_victim_flag_written_when_true(self):
        db = _mock_dynamodb()
        upsert_entity_record(
            "virginia giuffre", "PERSON", "Virginia Giuffre", "uuid-001", 0.95,
            victim_flag=True, dynamodb_client=db,
        )
        first_call = db.update_item.call_args_list[0]
        expr_values = first_call.kwargs["ExpressionAttributeValues"]
        self.assertIn(":vflag", expr_values)
        self.assertEqual(expr_values[":vflag"]["S"], "true")

    def test_victim_flag_not_written_when_false(self):
        db = _mock_dynamodb()
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
            victim_flag=False, dynamodb_client=db,
        )
        first_call = db.update_item.call_args_list[0]
        expr_values = first_call.kwargs.get("ExpressionAttributeValues", {})
        self.assertNotIn(":vflag", expr_values)

    def test_two_update_item_calls_made(self):
        """Phase 1 (full upsert) + Phase 2 (confidence MAX) = 2 calls."""
        db = _mock_dynamodb()
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
            dynamodb_client=db,
        )
        self.assertEqual(db.update_item.call_count, 2)

    def test_conditional_check_failed_silently_ignored(self):
        """Phase 2 raises ConditionalCheckFailedException -- must not propagate."""
        db = _mock_dynamodb()
        db.update_item.side_effect = [
            {},  # Phase 1 succeeds
            db.exceptions.ConditionalCheckFailedException("lower confidence"),
        ]
        # Must not raise
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.80,
            dynamodb_client=db,
        )

    def test_dynamodb_exception_raises_runtime_error(self):
        db = _mock_dynamodb()
        db.update_item.side_effect = RuntimeError("DynamoDB down")
        with self.assertRaises(RuntimeError) as ctx:
            upsert_entity_record(
                "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
                dynamodb_client=db,
            )
        self.assertIn("jeffrey epstein", str(ctx.exception))

    def test_custom_table_name_used(self):
        db = _mock_dynamodb()
        upsert_entity_record(
            "jeffrey epstein", "PERSON", "Epstein", "uuid-001", 0.99,
            dynamodb_client=db, table_name="custom_table",
        )
        self.assertEqual(
            db.update_item.call_args_list[0].kwargs["TableName"], "custom_table"
        )


class TestGetEntityRecord(unittest.TestCase):

    def test_returns_none_when_not_found(self):
        db = _mock_dynamodb(item=None)
        result = get_entity_record("unknown", "PERSON", dynamodb_client=db)
        self.assertIsNone(result)

    def test_returns_deserialised_item(self):
        item = _dynamo_item()
        db = _mock_dynamodb(item=item)
        result = get_entity_record("jeffrey epstein", "PERSON", dynamodb_client=db)
        self.assertIsNotNone(result)
        self.assertEqual(result["canonical_name"], "jeffrey epstein")

    def test_confidence_deserialised_as_float(self):
        item = _dynamo_item(confidence=0.99)
        db = _mock_dynamodb(item=item)
        result = get_entity_record("jeffrey epstein", "PERSON", dynamodb_client=db)
        self.assertIsInstance(result["confidence"], float)

    def test_surface_forms_deserialised_as_list(self):
        item = _dynamo_item()
        db = _mock_dynamodb(item=item)
        result = get_entity_record("jeffrey epstein", "PERSON", dynamodb_client=db)
        self.assertIsInstance(result["surface_forms"], list)

    def test_victim_flag_present_when_set(self):
        item = _dynamo_item(victim=True)
        db = _mock_dynamodb(item=item)
        result = get_entity_record("virginia giuffre", "PERSON", dynamodb_client=db)
        self.assertEqual(result.get("victim_flag"), "true")

    def test_dynamodb_exception_raises_runtime_error(self):
        db = MagicMock()
        db.get_item.side_effect = RuntimeError("DynamoDB down")
        with self.assertRaises(RuntimeError):
            get_entity_record("jeffrey epstein", "PERSON", dynamodb_client=db)


class TestQueryEntitiesByType(unittest.TestCase):

    def test_queries_correct_gsi(self):
        db = _mock_dynamodb(item=_dynamo_item())
        query_entities_by_type("PERSON", dynamodb_client=db)
        self.assertEqual(
            db.query.call_args.kwargs["IndexName"], "gsi-entity-type"
        )

    def test_entity_type_passed_as_condition(self):
        db = _mock_dynamodb(item=_dynamo_item())
        query_entities_by_type("ORGANIZATION", dynamodb_client=db)
        expr_values = db.query.call_args.kwargs["ExpressionAttributeValues"]
        self.assertEqual(expr_values[":etype"]["S"], "ORGANIZATION")

    def test_returns_list(self):
        db = _mock_dynamodb(item=_dynamo_item())
        result = query_entities_by_type("PERSON", dynamodb_client=db)
        self.assertIsInstance(result, list)

    def test_returns_empty_list_when_no_results(self):
        db = _mock_dynamodb(item=None)
        result = query_entities_by_type("PERSON", dynamodb_client=db)
        self.assertEqual(result, [])

    def test_dynamodb_exception_raises_runtime_error(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("DynamoDB down")
        with self.assertRaises(RuntimeError):
            query_entities_by_type("PERSON", dynamodb_client=db)


class TestQueryEntitiesByDocument(unittest.TestCase):

    def test_queries_correct_gsi(self):
        db = _mock_dynamodb(item=_dynamo_item())
        query_entities_by_document("uuid-001", dynamodb_client=db)
        self.assertEqual(
            db.query.call_args.kwargs["IndexName"], "gsi-document-uuid"
        )

    def test_document_uuid_passed_as_condition(self):
        db = _mock_dynamodb(item=_dynamo_item())
        query_entities_by_document("uuid-abc", dynamodb_client=db)
        expr_values = db.query.call_args.kwargs["ExpressionAttributeValues"]
        self.assertEqual(expr_values[":duuid"]["S"], "uuid-abc")

    def test_returns_list(self):
        db = _mock_dynamodb(item=_dynamo_item())
        result = query_entities_by_document("uuid-001", dynamodb_client=db)
        self.assertIsInstance(result, list)

    def test_returns_empty_list_when_no_results(self):
        db = _mock_dynamodb(item=None)
        result = query_entities_by_document("uuid-999", dynamodb_client=db)
        self.assertEqual(result, [])

    def test_dynamodb_exception_raises_runtime_error(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("DynamoDB down")
        with self.assertRaises(RuntimeError):
            query_entities_by_document("uuid-001", dynamodb_client=db)

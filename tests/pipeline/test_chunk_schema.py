"""
tests/pipeline/test_chunk_schema.py

Unit tests for pipeline/chunk_schema.py (ChunkMetadata).

Coverage targets
----------------
ChunkMetadata       -- required fields, optional fields present/absent,
                       field validators (text empty, vector empty,
                       document_uuid empty, ingestion_date empty),
                       chunk_id property, opensearch_document() serialisation
                       (exclude_none, all fields round-trip)
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from pipeline.chunk_schema import ChunkMetadata, DocumentType, SequenceScheme
from pipeline.models import ConfidenceTier, DeletionFlag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vector(dim: int = 1024) -> list:
    return [0.1] * dim


def _minimal(**overrides) -> ChunkMetadata:
    """Construct a ChunkMetadata with only required fields."""
    defaults = dict(
        document_uuid="uuid-001",
        chunk_index=0,
        text="Some chunk text here.",
        vector=_vector(),
        classification="PROCEDURAL",
        ingestion_date="2026-03-11T00:00:00+00:00",
    )
    defaults.update(overrides)
    return ChunkMetadata(**defaults)


# ===========================================================================
# Required fields
# ===========================================================================

class TestChunkMetadataRequired(unittest.TestCase):

    def test_minimal_construction_succeeds(self):
        chunk = _minimal()
        self.assertIsInstance(chunk, ChunkMetadata)

    def test_document_uuid_stored(self):
        chunk = _minimal(document_uuid="uuid-abc")
        self.assertEqual(chunk.document_uuid, "uuid-abc")

    def test_chunk_index_stored(self):
        chunk = _minimal(chunk_index=3)
        self.assertEqual(chunk.chunk_index, 3)

    def test_chunk_index_zero_valid(self):
        chunk = _minimal(chunk_index=0)
        self.assertEqual(chunk.chunk_index, 0)

    def test_text_stored(self):
        chunk = _minimal(text="Hello world.")
        self.assertEqual(chunk.text, "Hello world.")

    def test_vector_stored(self):
        v = _vector(1024)
        chunk = _minimal(vector=v)
        self.assertEqual(len(chunk.vector), 1024)

    def test_classification_stored(self):
        chunk = _minimal(classification="VICTIM_ADJACENT")
        self.assertEqual(chunk.classification, "VICTIM_ADJACENT")

    def test_ingestion_date_stored(self):
        chunk = _minimal(ingestion_date="2026-01-01T00:00:00+00:00")
        self.assertEqual(chunk.ingestion_date, "2026-01-01T00:00:00+00:00")


# ===========================================================================
# Optional fields
# ===========================================================================

class TestChunkMetadataOptional(unittest.TestCase):

    def test_optional_fields_default_to_none(self):
        chunk = _minimal()
        self.assertIsNone(chunk.provenance_tag)
        self.assertIsNone(chunk.page_number)
        self.assertIsNone(chunk.bates_number)
        self.assertIsNone(chunk.efta_number)
        self.assertIsNone(chunk.corpus_source)

    def test_victim_flag_defaults_false(self):
        chunk = _minimal()
        self.assertFalse(chunk.victim_flag)

    def test_victim_flag_can_be_set_true(self):
        chunk = _minimal(victim_flag=True)
        self.assertTrue(chunk.victim_flag)

    def test_provenance_tag_stored(self):
        chunk = _minimal(provenance_tag="PROVENANCE_COMMUNITY_VOUCHED")
        self.assertEqual(chunk.provenance_tag, "PROVENANCE_COMMUNITY_VOUCHED")

    def test_page_number_stored(self):
        chunk = _minimal(page_number=42)
        self.assertEqual(chunk.page_number, 42)

    def test_bates_number_stored(self):
        chunk = _minimal(bates_number="DOJ-000042")
        self.assertEqual(chunk.bates_number, "DOJ-000042")

    def test_efta_number_stored(self):
        chunk = _minimal(efta_number="123456")
        self.assertEqual(chunk.efta_number, "123456")

    def test_corpus_source_stored(self):
        chunk = _minimal(corpus_source="DOJ_DIRECT")
        self.assertEqual(chunk.corpus_source, "DOJ_DIRECT")


# ===========================================================================
# Validators
# ===========================================================================

class TestChunkMetadataValidators(unittest.TestCase):

    def test_empty_text_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(text="")

    def test_whitespace_only_text_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(text="   ")

    def test_empty_vector_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(vector=[])

    def test_empty_document_uuid_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(document_uuid="")

    def test_whitespace_document_uuid_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(document_uuid="   ")

    def test_empty_ingestion_date_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(ingestion_date="")

    def test_negative_chunk_index_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(chunk_index=-1)

    def test_zero_page_number_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(page_number=0)

    def test_negative_page_number_raises(self):
        with self.assertRaises(ValidationError):
            _minimal(page_number=-1)

    def test_page_number_one_is_valid(self):
        chunk = _minimal(page_number=1)
        self.assertEqual(chunk.page_number, 1)


# ===========================================================================
# chunk_id property
# ===========================================================================

class TestChunkId(unittest.TestCase):

    def test_chunk_id_format(self):
        chunk = _minimal(document_uuid="uuid-001", chunk_index=0)
        self.assertEqual(chunk.chunk_id, "uuid-001#0")

    def test_chunk_id_with_nonzero_index(self):
        chunk = _minimal(document_uuid="uuid-abc", chunk_index=7)
        self.assertEqual(chunk.chunk_id, "uuid-abc#7")

    def test_chunk_id_unique_across_indices(self):
        chunk_0 = _minimal(document_uuid="uuid-001", chunk_index=0)
        chunk_1 = _minimal(document_uuid="uuid-001", chunk_index=1)
        self.assertNotEqual(chunk_0.chunk_id, chunk_1.chunk_id)

    def test_chunk_id_unique_across_documents(self):
        chunk_a = _minimal(document_uuid="uuid-001", chunk_index=0)
        chunk_b = _minimal(document_uuid="uuid-002", chunk_index=0)
        self.assertNotEqual(chunk_a.chunk_id, chunk_b.chunk_id)


# ===========================================================================
# opensearch_document()
# ===========================================================================

class TestOpenSearchDocument(unittest.TestCase):

    def test_returns_dict(self):
        self.assertIsInstance(_minimal().opensearch_document(), dict)

    def test_required_fields_present(self):
        doc = _minimal().opensearch_document()
        for key in ("document_uuid", "chunk_index", "text", "vector",
                    "classification", "ingestion_date"):
            self.assertIn(key, doc)

    def test_none_fields_excluded(self):
        """exclude_none=True: optional unset fields must not appear."""
        doc = _minimal().opensearch_document()
        self.assertNotIn("provenance_tag", doc)
        self.assertNotIn("page_number", doc)
        self.assertNotIn("bates_number", doc)
        self.assertNotIn("efta_number", doc)
        self.assertNotIn("corpus_source", doc)

    def test_set_optional_fields_included(self):
        chunk = _minimal(
            provenance_tag="PROVENANCE_DOJ_DIRECT",
            page_number=5,
            bates_number="DOJ-001",
            corpus_source="DOJ_DIRECT",
        )
        doc = chunk.opensearch_document()
        self.assertIn("provenance_tag", doc)
        self.assertIn("page_number", doc)
        self.assertIn("bates_number", doc)
        self.assertIn("corpus_source", doc)

    def test_victim_flag_false_included(self):
        """victim_flag=False is not None -- it should appear in the document."""
        doc = _minimal(victim_flag=False).opensearch_document()
        self.assertIn("victim_flag", doc)
        self.assertFalse(doc["victim_flag"])

    def test_victim_flag_true_included(self):
        doc = _minimal(victim_flag=True).opensearch_document()
        self.assertTrue(doc["victim_flag"])

    def test_vector_values_preserved(self):
        v = [float(i) / 1024 for i in range(1024)]
        doc = _minimal(vector=v).opensearch_document()
        self.assertEqual(doc["vector"], v)



# ===========================================================================
# New optional fields (Layer 2 schema extension)
# ===========================================================================

class TestChunkMetadataNewOptionalFields(unittest.TestCase):

    def test_new_optional_fields_default_to_none(self):
        chunk = _minimal()
        self.assertIsNone(chunk.sequence_number)
        self.assertIsNone(chunk.sequence_scheme)
        self.assertIsNone(chunk.document_date)
        self.assertIsNone(chunk.document_type)
        self.assertIsNone(chunk.confidence_tier)
        self.assertIsNone(chunk.deletion_flag)

    def test_named_entities_defaults_to_empty_list(self):
        chunk = _minimal()
        self.assertEqual(chunk.named_entities, [])

    def test_sequence_number_stored(self):
        chunk = _minimal(sequence_number="1234567")
        self.assertEqual(chunk.sequence_number, "1234567")

    def test_sequence_scheme_efta(self):
        chunk = _minimal(sequence_scheme=SequenceScheme.EFTA)
        self.assertEqual(chunk.sequence_scheme, SequenceScheme.EFTA)

    def test_sequence_scheme_bates(self):
        chunk = _minimal(sequence_scheme=SequenceScheme.BATES)
        self.assertEqual(chunk.sequence_scheme, SequenceScheme.BATES)

    def test_document_date_stored(self):
        chunk = _minimal(document_date="2005-03-15")
        self.assertEqual(chunk.document_date, "2005-03-15")

    def test_document_type_fbi_302(self):
        chunk = _minimal(document_type=DocumentType.FBI_302)
        self.assertEqual(chunk.document_type, DocumentType.FBI_302)

    def test_document_type_court_filing(self):
        chunk = _minimal(document_type=DocumentType.COURT_FILING)
        self.assertEqual(chunk.document_type, DocumentType.COURT_FILING)

    def test_all_document_types_valid(self):
        for dt in DocumentType:
            chunk = _minimal(document_type=dt)
            self.assertEqual(chunk.document_type, dt)

    def test_named_entities_stored(self):
        entities = [{"text": "John Doe", "type": "PERSON", "confidence": 0.97}]
        chunk = _minimal(named_entities=entities)
        self.assertEqual(chunk.named_entities, entities)

    def test_confidence_tier_stored(self):
        chunk = _minimal(confidence_tier=ConfidenceTier.CORROBORATED)
        self.assertEqual(chunk.confidence_tier, ConfidenceTier.CORROBORATED)

    def test_all_confidence_tiers_valid(self):
        for tier in ConfidenceTier:
            chunk = _minimal(confidence_tier=tier)
            self.assertEqual(chunk.confidence_tier, tier)

    def test_deletion_flag_stored(self):
        chunk = _minimal(deletion_flag=DeletionFlag.DELETION_SUSPECTED)
        self.assertEqual(chunk.deletion_flag, DeletionFlag.DELETION_SUSPECTED)

    def test_all_deletion_flags_valid(self):
        for flag in DeletionFlag:
            chunk = _minimal(deletion_flag=flag)
            self.assertEqual(chunk.deletion_flag, flag)


# ===========================================================================
# opensearch_document() with new fields
# ===========================================================================

class TestOpenSearchDocumentNewFields(unittest.TestCase):

    def test_new_none_fields_excluded(self):
        doc = _minimal().opensearch_document()
        for field in ("sequence_number", "sequence_scheme", "document_date",
                      "document_type", "confidence_tier", "deletion_flag"):
            self.assertNotIn(field, doc, f"Field '{field}' should be excluded when None")

    def test_named_entities_empty_list_included(self):
        """Empty list is not None -- it should always appear."""
        doc = _minimal().opensearch_document()
        self.assertIn("named_entities", doc)
        self.assertEqual(doc["named_entities"], [])

    def test_sequence_number_included_when_set(self):
        doc = _minimal(sequence_number="9876543").opensearch_document()
        self.assertEqual(doc["sequence_number"], "9876543")

    def test_sequence_scheme_serialised_as_string(self):
        doc = _minimal(sequence_scheme=SequenceScheme.EFTA).opensearch_document()
        self.assertEqual(doc["sequence_scheme"], "EFTA")

    def test_document_type_serialised_as_string(self):
        doc = _minimal(document_type=DocumentType.FBI_302).opensearch_document()
        self.assertEqual(doc["document_type"], "FBI_302")

    def test_confidence_tier_serialised_as_string(self):
        doc = _minimal(confidence_tier=ConfidenceTier.CONFIRMED).opensearch_document()
        self.assertEqual(doc["confidence_tier"], "CONFIRMED")

    def test_deletion_flag_serialised_as_string(self):
        doc = _minimal(deletion_flag=DeletionFlag.DELETION_CONFIRMED).opensearch_document()
        self.assertEqual(doc["deletion_flag"], "DELETION_CONFIRMED")

    def test_named_entities_with_entries_included(self):
        entities = [{"text": "Jane Doe", "type": "PERSON", "confidence": 0.95}]
        doc = _minimal(named_entities=entities).opensearch_document()
        self.assertEqual(doc["named_entities"], entities)


if __name__ == "__main__":
    unittest.main()

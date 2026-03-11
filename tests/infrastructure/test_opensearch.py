"""
tests/infrastructure/test_opensearch.py

Unit tests for infrastructure/opensearch.py.

Coverage targets
----------------
build_index_mapping   -- structure, knn enabled, vector dimension from config,
                         all expected field mappings present, dimension
                         propagates when config changes
create_index          -- created when absent, skipped when present,
                         returns True/False correctly, RuntimeError on
                         existence check failure, RuntimeError on create failure,
                         correct mapping passed to client
delete_index          -- deleted when present, skipped when absent,
                         returns True/False correctly, RuntimeError on
                         existence check failure, RuntimeError on delete failure
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from config import EmbeddingConfig
from infrastructure.opensearch import (
    build_index_mapping,
    create_index,
    delete_index,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_os(index_exists: bool = False) -> MagicMock:
    client = MagicMock()
    client.indices.exists.return_value = index_exists
    client.indices.create.return_value = {"acknowledged": True}
    client.indices.delete.return_value = {"acknowledged": True}
    return client


# ===========================================================================
# build_index_mapping
# ===========================================================================

class TestBuildIndexMapping(unittest.TestCase):

    def setUp(self):
        self.mapping = build_index_mapping()

    def test_returns_dict(self):
        self.assertIsInstance(self.mapping, dict)

    def test_settings_knn_enabled(self):
        self.assertTrue(self.mapping["settings"]["index"]["knn"])

    def test_mappings_key_present(self):
        self.assertIn("mappings", self.mapping)
        self.assertIn("properties", self.mapping["mappings"])

    def test_vector_field_is_knn_vector(self):
        props = self.mapping["mappings"]["properties"]
        self.assertEqual(props["vector"]["type"], "knn_vector")

    def test_vector_dimension_matches_default_config(self):
        props = self.mapping["mappings"]["properties"]
        self.assertEqual(props["vector"]["dimension"], 1024)

    def test_vector_dimension_propagates_from_custom_config(self):
        mapping = build_index_mapping(EmbeddingConfig(vector_dimension=1536))
        props = mapping["mappings"]["properties"]
        self.assertEqual(props["vector"]["dimension"], 1536)

    def test_all_schema_fields_mapped(self):
        props = self.mapping["mappings"]["properties"]
        expected = {
            "document_uuid", "chunk_index", "text", "vector",
            "classification", "provenance_tag", "ingestion_date",
            "victim_flag", "page_number", "bates_number",
            "efta_number", "corpus_source",
        }
        for field in expected:
            self.assertIn(field, props, f"Field '{field}' missing from index mapping")

    def test_document_uuid_is_keyword(self):
        props = self.mapping["mappings"]["properties"]
        self.assertEqual(props["document_uuid"]["type"], "keyword")

    def test_victim_flag_is_keyword(self):
        """keyword type required for exact-match guardrail filter."""
        props = self.mapping["mappings"]["properties"]
        self.assertEqual(props["victim_flag"]["type"], "keyword")

    def test_text_field_not_indexed(self):
        """text is stored for display but not full-text indexed by default."""
        props = self.mapping["mappings"]["properties"]
        self.assertFalse(props["text"].get("index", True))

    def test_ingestion_date_is_date_type(self):
        props = self.mapping["mappings"]["properties"]
        self.assertEqual(props["ingestion_date"]["type"], "date")

    def test_chunk_index_is_integer(self):
        props = self.mapping["mappings"]["properties"]
        self.assertEqual(props["chunk_index"]["type"], "integer")


# ===========================================================================
# create_index
# ===========================================================================

class TestCreateIndex(unittest.TestCase):

    def test_returns_true_when_created(self):
        client = _mock_os(index_exists=False)
        result = create_index(client, index_name="documents")
        self.assertTrue(result)

    def test_returns_false_when_already_exists(self):
        client = _mock_os(index_exists=True)
        result = create_index(client, index_name="documents")
        self.assertFalse(result)

    def test_create_not_called_when_index_exists(self):
        client = _mock_os(index_exists=True)
        create_index(client, index_name="documents")
        client.indices.create.assert_not_called()

    def test_create_called_when_index_absent(self):
        client = _mock_os(index_exists=False)
        create_index(client, index_name="documents")
        client.indices.create.assert_called_once()

    def test_correct_index_name_passed_to_create(self):
        client = _mock_os(index_exists=False)
        create_index(client, index_name="my-index")
        self.assertEqual(
            client.indices.create.call_args.kwargs["index"], "my-index"
        )

    def test_mapping_body_passed_to_create(self):
        client = _mock_os(index_exists=False)
        create_index(client, index_name="documents")
        body = client.indices.create.call_args.kwargs["body"]
        self.assertIn("mappings", body)
        self.assertIn("settings", body)

    def test_vector_dimension_from_config_in_body(self):
        client = _mock_os(index_exists=False)
        cfg = EmbeddingConfig(vector_dimension=1536)
        create_index(client, index_name="documents", embedding_config=cfg)
        body = client.indices.create.call_args.kwargs["body"]
        dim = body["mappings"]["properties"]["vector"]["dimension"]
        self.assertEqual(dim, 1536)

    def test_exists_check_failure_raises_runtime_error(self):
        client = MagicMock()
        client.indices.exists.side_effect = RuntimeError("Connection refused")
        with self.assertRaises(RuntimeError) as ctx:
            create_index(client, index_name="documents")
        self.assertIn("documents", str(ctx.exception))

    def test_create_failure_raises_runtime_error(self):
        client = _mock_os(index_exists=False)
        client.indices.create.side_effect = RuntimeError("Cluster unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            create_index(client, index_name="documents")
        self.assertIn("documents", str(ctx.exception))


# ===========================================================================
# delete_index
# ===========================================================================

class TestDeleteIndex(unittest.TestCase):

    def test_returns_true_when_deleted(self):
        client = _mock_os(index_exists=True)
        result = delete_index(client, index_name="documents")
        self.assertTrue(result)

    def test_returns_false_when_not_present(self):
        client = _mock_os(index_exists=False)
        result = delete_index(client, index_name="documents")
        self.assertFalse(result)

    def test_delete_not_called_when_absent(self):
        client = _mock_os(index_exists=False)
        delete_index(client, index_name="documents")
        client.indices.delete.assert_not_called()

    def test_delete_called_when_present(self):
        client = _mock_os(index_exists=True)
        delete_index(client, index_name="documents")
        client.indices.delete.assert_called_once()

    def test_correct_index_name_passed_to_delete(self):
        client = _mock_os(index_exists=True)
        delete_index(client, index_name="my-index")
        self.assertEqual(
            client.indices.delete.call_args.kwargs["index"], "my-index"
        )

    def test_exists_check_failure_raises_runtime_error(self):
        client = MagicMock()
        client.indices.exists.side_effect = RuntimeError("Connection refused")
        with self.assertRaises(RuntimeError) as ctx:
            delete_index(client, index_name="documents")
        self.assertIn("documents", str(ctx.exception))

    def test_delete_failure_raises_runtime_error(self):
        client = _mock_os(index_exists=True)
        client.indices.delete.side_effect = RuntimeError("Cluster unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            delete_index(client, index_name="documents")
        self.assertIn("documents", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

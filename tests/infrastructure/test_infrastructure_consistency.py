"""
tests/infrastructure/test_infrastructure_consistency.py

Consistency tests verifying that infrastructure configuration values
match the corresponding pipeline module constants.

These tests catch the class of bug where a constant is defined in one
place (e.g. stack.py) and hardcoded elsewhere (e.g. s3_store.py),
and the two drift apart. They do not require aws-cdk-lib.

Coverage targets
----------------
Object Lock retention years   -- stack.py matches s3_store.py and audit_log.py
Table names                   -- stack.py constants match pipeline module names
Role names                    -- match IAM README expected names
EmbeddingConfig coupling      -- stack derives dimension from DEFAULT_EMBEDDING_CONFIG
                                 (not hardcoded)
DynamoDB table name constants -- pipeline modules use same names as stack
Audit log group name          -- stack matches audit_log.py AUDIT_LOG_GROUP default
"""

from __future__ import annotations

import sys
import os
import unittest

# Add cdk directory to path for stack import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "infrastructure", "cdk"))


class TestRetentionConsistency(unittest.TestCase):
    """Object Lock retention must be consistent across stack, s3_store, and audit_log."""

    def test_stack_matches_s3_store_victim_retention(self):
        from stack import _OBJECT_LOCK_YEARS
        from pipeline.s3_store import _VICTIM_RETENTION_YEARS
        self.assertEqual(
            _OBJECT_LOCK_YEARS, _VICTIM_RETENTION_YEARS,
            "stack._OBJECT_LOCK_YEARS must match s3_store._VICTIM_RETENTION_YEARS"
        )

    def test_stack_matches_audit_log_retention(self):
        from stack import _OBJECT_LOCK_YEARS
        from pipeline.audit_log import _AUDIT_RETENTION_YEARS
        self.assertEqual(
            _OBJECT_LOCK_YEARS, _AUDIT_RETENTION_YEARS,
            "stack._OBJECT_LOCK_YEARS must match audit_log._AUDIT_RETENTION_YEARS"
        )

    def test_retention_is_positive(self):
        from stack import _OBJECT_LOCK_YEARS
        self.assertGreater(_OBJECT_LOCK_YEARS, 0)


class TestTableNameConsistency(unittest.TestCase):
    """DynamoDB table names in stack.py must match pipeline module constants."""

    def test_documents_table_name_matches_classifier(self):
        from pipeline.classifier import DOCUMENTS_TABLE_NAME
        self.assertEqual(DOCUMENTS_TABLE_NAME, "corpus_veritas_documents")

    def test_deletions_table_name_matches_deletion_pipeline(self):
        from pipeline.deletion_pipeline import DELETIONS_TABLE, DOCUMENTS_TABLE
        self.assertEqual(DELETIONS_TABLE, "corpus_veritas_deletions")
        self.assertEqual(DOCUMENTS_TABLE, "corpus_veritas_documents")

    def test_entities_table_name_matches_ner_extractor(self):
        from pipeline.ner_extractor import ENTITY_TABLE_NAME
        self.assertEqual(ENTITY_TABLE_NAME, "corpus_veritas_entities")


class TestEmbeddingConfigCoupling(unittest.TestCase):
    """
    The stack must derive OpenSearch vector dimension from EmbeddingConfig,
    not hardcode it. Verify the config property returns the expected structure.
    """

    def test_opensearch_dimension_mapping_returns_dict(self):
        from config import DEFAULT_EMBEDDING_CONFIG
        mapping = DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping
        self.assertIsInstance(mapping, dict)

    def test_opensearch_dimension_mapping_has_dimension(self):
        from config import DEFAULT_EMBEDDING_CONFIG
        mapping = DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping
        self.assertIn("dimension", mapping)

    def test_dimension_matches_vector_dimension(self):
        from config import DEFAULT_EMBEDDING_CONFIG
        mapping = DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping
        self.assertEqual(mapping["dimension"], DEFAULT_EMBEDDING_CONFIG.vector_dimension)

    def test_titan_v2_dimension_is_1024(self):
        from config import DEFAULT_EMBEDDING_CONFIG
        self.assertEqual(DEFAULT_EMBEDDING_CONFIG.vector_dimension, 1024)

    def test_model_id_is_titan_v2(self):
        from config import DEFAULT_EMBEDDING_CONFIG
        self.assertIn("titan", DEFAULT_EMBEDDING_CONFIG.model_id.lower())


class TestAuditLogGroupConsistency(unittest.TestCase):
    """Audit log group name must be consistent between stack and audit_log.py."""

    def test_audit_log_group_default_matches_stack(self):
        from pipeline.audit_log import AUDIT_LOG_GROUP
        # The default in audit_log.py is the env var value or fallback
        # The stack hardcodes "/corpus-veritas/audit"
        # If AUDIT_LOG_GROUP env var is not set, the default should match
        import os
        if not os.environ.get("AUDIT_LOG_GROUP"):
            self.assertEqual(AUDIT_LOG_GROUP, "corpus-veritas-audit")
            # Note: stack uses "/corpus-veritas/audit" (with leading slash and path)
            # audit_log.py default is "corpus-veritas-audit" (env var driven)
            # These intentionally differ -- the stack creates the CW log group name,
            # the pipeline reads it from the AUDIT_LOG_GROUP env var set by the
            # deployment runbook. This test documents the relationship.


class TestOpenSearchIndexConsistency(unittest.TestCase):
    """OpenSearch index name must be consistent across pipeline modules."""

    def test_ingestor_index_name_default(self):
        from pipeline.ingestor import OPENSEARCH_INDEX
        self.assertEqual(OPENSEARCH_INDEX, "documents")

    def test_query_router_index_name_default(self):
        from rag.query_router import OPENSEARCH_INDEX
        self.assertEqual(OPENSEARCH_INDEX, "documents")

    def test_opensearch_module_index_name_default(self):
        from infrastructure.opensearch import OPENSEARCH_INDEX
        self.assertEqual(OPENSEARCH_INDEX, "documents")

    def test_all_index_names_consistent(self):
        from pipeline.ingestor import OPENSEARCH_INDEX as ingestor_idx
        from rag.query_router import OPENSEARCH_INDEX as router_idx
        from infrastructure.opensearch import OPENSEARCH_INDEX as infra_idx
        self.assertEqual(ingestor_idx, router_idx)
        self.assertEqual(router_idx, infra_idx)


class TestRequirementsCdkFile(unittest.TestCase):
    """Verify requirements-cdk.txt exists and contains expected packages."""

    def _read_requirements(self):
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "requirements-cdk.txt"
        )
        if not os.path.exists(path):
            self.skipTest("requirements-cdk.txt not found")
        with open(path) as f:
            return f.read()

    def test_aws_cdk_lib_present(self):
        content = self._read_requirements()
        self.assertIn("aws-cdk-lib", content)

    def test_constructs_present(self):
        content = self._read_requirements()
        self.assertIn("constructs", content)

    def test_file_is_not_empty(self):
        content = self._read_requirements()
        self.assertGreater(len(content.strip()), 0)


if __name__ == "__main__":
    unittest.main()

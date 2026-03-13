"""
tests/test_config.py

Unit tests for config.py (EmbeddingConfig, ChunkingConfig).

Coverage targets
----------------
EmbeddingConfig         -- default values, frozen semantics, __post_init__
                           validation, opensearch_dimension_mapping property
ChunkingConfig          -- default values, frozen semantics, __post_init__
                           validation (all three error paths), overlap_ratio
DEFAULT_EMBEDDING_CONFIG / DEFAULT_CHUNKING_CONFIG
                        -- module-level singletons are correct types with
                           expected production values
"""

from __future__ import annotations

import unittest

from config import (
    DEFAULT_CHUNKING_CONFIG,
    DEFAULT_EMBEDDING_CONFIG,
    ChunkingConfig,
    EmbeddingConfig,
)


# ===========================================================================
# EmbeddingConfig
# ===========================================================================

class TestEmbeddingConfigDefaults(unittest.TestCase):

    def test_default_model_id(self):
        self.assertEqual(
            DEFAULT_EMBEDDING_CONFIG.model_id, "amazon.titan-embed-text-v2:0"
        )

    def test_default_vector_dimension(self):
        self.assertEqual(DEFAULT_EMBEDDING_CONFIG.vector_dimension, 1024)

    def test_default_instance_is_embedding_config(self):
        self.assertIsInstance(DEFAULT_EMBEDDING_CONFIG, EmbeddingConfig)


class TestEmbeddingConfigFrozen(unittest.TestCase):

    def test_model_id_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            setattr(DEFAULT_EMBEDDING_CONFIG, "model_id", "other-model")

    def test_vector_dimension_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            setattr(DEFAULT_EMBEDDING_CONFIG, "vector_dimension", 512)


class TestEmbeddingConfigValidation(unittest.TestCase):

    def test_zero_vector_dimension_raises(self):
        with self.assertRaises(ValueError) as ctx:
            EmbeddingConfig(vector_dimension=0)
        self.assertIn("vector_dimension", str(ctx.exception))

    def test_negative_vector_dimension_raises(self):
        with self.assertRaises(ValueError):
            EmbeddingConfig(vector_dimension=-1)

    def test_valid_custom_config_constructs(self):
        cfg = EmbeddingConfig(
            model_id="cohere.embed-english-v3",
            vector_dimension=1024,
        )
        self.assertEqual(cfg.model_id, "cohere.embed-english-v3")
        self.assertEqual(cfg.vector_dimension, 1024)


class TestEmbeddingConfigOpenSearchMapping(unittest.TestCase):

    def test_mapping_type_is_knn_vector(self):
        mapping = DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping
        self.assertEqual(mapping["type"], "knn_vector")

    def test_mapping_dimension_matches_config(self):
        cfg = EmbeddingConfig(vector_dimension=1536)
        self.assertEqual(cfg.opensearch_dimension_mapping["dimension"], 1536)

    def test_mapping_contains_hnsw_method(self):
        mapping = DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping
        self.assertIn("method", mapping)
        self.assertEqual(mapping["method"]["name"], "hnsw")

    def test_mapping_dimension_propagates_on_model_change(self):
        """
        Changing vector_dimension must change the mapping dimension.
        This is the core coupling guarantee -- CDK uses this property
        so the index mapping is always consistent with the model.
        """
        cfg_1024 = EmbeddingConfig(vector_dimension=1024)
        cfg_1536 = EmbeddingConfig(vector_dimension=1536)
        self.assertNotEqual(
            cfg_1024.opensearch_dimension_mapping["dimension"],
            cfg_1536.opensearch_dimension_mapping["dimension"],
        )

    def test_mapping_returns_dict(self):
        self.assertIsInstance(
            DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping, dict
        )


# ===========================================================================
# ChunkingConfig
# ===========================================================================

class TestChunkingConfigDefaults(unittest.TestCase):

    def test_default_chunk_size(self):
        self.assertEqual(DEFAULT_CHUNKING_CONFIG.chunk_size_tokens, 512)

    def test_default_chunk_overlap(self):
        self.assertEqual(DEFAULT_CHUNKING_CONFIG.chunk_overlap_tokens, 50)

    def test_default_instance_is_chunking_config(self):
        self.assertIsInstance(DEFAULT_CHUNKING_CONFIG, ChunkingConfig)


class TestChunkingConfigFrozen(unittest.TestCase):

    def test_chunk_size_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            setattr(DEFAULT_CHUNKING_CONFIG, "chunk_size_tokens", 256)

    def test_chunk_overlap_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            setattr(DEFAULT_CHUNKING_CONFIG, "chunk_overlap_tokens", 25)


class TestChunkingConfigValidation(unittest.TestCase):

    def test_zero_chunk_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ChunkingConfig(chunk_size_tokens=0)
        self.assertIn("chunk_size_tokens", str(ctx.exception))

    def test_negative_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            ChunkingConfig(chunk_size_tokens=-1)

    def test_negative_overlap_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ChunkingConfig(chunk_overlap_tokens=-1)
        self.assertIn("chunk_overlap_tokens", str(ctx.exception))

    def test_overlap_equal_to_chunk_size_raises(self):
        with self.assertRaises(ValueError) as ctx:
            ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=512)
        self.assertIn("chunk_overlap_tokens", str(ctx.exception))

    def test_overlap_greater_than_chunk_size_raises(self):
        with self.assertRaises(ValueError):
            ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=600)

    def test_zero_overlap_is_valid(self):
        cfg = ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=0)
        self.assertEqual(cfg.chunk_overlap_tokens, 0)

    def test_valid_custom_config_constructs(self):
        cfg = ChunkingConfig(chunk_size_tokens=256, chunk_overlap_tokens=25)
        self.assertEqual(cfg.chunk_size_tokens, 256)
        self.assertEqual(cfg.chunk_overlap_tokens, 25)


class TestChunkingConfigOverlapRatio(unittest.TestCase):

    def test_overlap_ratio_default(self):
        cfg = ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=50)
        self.assertAlmostEqual(cfg.overlap_ratio, 50 / 512)

    def test_overlap_ratio_zero_overlap(self):
        cfg = ChunkingConfig(chunk_size_tokens=512, chunk_overlap_tokens=0)
        self.assertEqual(cfg.overlap_ratio, 0.0)

    def test_overlap_ratio_is_float(self):
        self.assertIsInstance(DEFAULT_CHUNKING_CONFIG.overlap_ratio, float)

    def test_overlap_ratio_less_than_one(self):
        self.assertLess(DEFAULT_CHUNKING_CONFIG.overlap_ratio, 1.0)


# ===========================================================================
# Separation of concerns
# ===========================================================================

class TestConfigSeparation(unittest.TestCase):

    def test_embedding_config_has_no_chunking_fields(self):
        self.assertFalse(hasattr(EmbeddingConfig(), "chunk_size_tokens"))
        self.assertFalse(hasattr(EmbeddingConfig(), "chunk_overlap_tokens"))
        self.assertFalse(hasattr(EmbeddingConfig(), "overlap_ratio"))

    def test_chunking_config_has_no_embedding_fields(self):
        self.assertFalse(hasattr(ChunkingConfig(), "model_id"))
        self.assertFalse(hasattr(ChunkingConfig(), "vector_dimension"))
        self.assertFalse(hasattr(ChunkingConfig(), "opensearch_dimension_mapping"))

    def test_default_instances_are_independent(self):
        self.assertIsNot(DEFAULT_EMBEDDING_CONFIG, DEFAULT_CHUNKING_CONFIG)


if __name__ == "__main__":
    unittest.main()

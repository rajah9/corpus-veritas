"""
config.py
Top-level configuration for the corpus-veritas pipeline.

This module is the single source of truth for all values that must be
consistent across multiple layers of the pipeline. Anything defined
here instead of in individual modules is a value whose consistency
matters -- changing it in one place must automatically propagate to
all consumers.

Current contents
----------------
EmbeddingConfig   Bedrock embedding model ID and vector dimension. Used by:
                    - ingestor.py          (embedding calls)
                    - infrastructure/cdk/  (OpenSearch index mapping dimension)
                    - rag/query_router.py  (query embedding must match index)

ChunkingConfig    Document chunking parameters. Used by:
                    - ingestor.py          (chunk splitting before embedding)

The two configs are intentionally separate. Embedding model and chunk
strategy are independent decisions -- you can retune chunking without
changing the model or re-creating the OpenSearch index, and vice versa
(though changing the model always requires re-embedding everything).

AWS region is NOT carried here. All modules defer to the AWS_REGION
environment variable, consistent with the pattern established in
sanitizer.py and classifier.py.

See docs/ARCHITECTURE.md para Layer 2 -- Embedding & Storage.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# EmbeddingConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Bedrock embedding model identity and vector dimension.

    Frozen so that no code path can mutate the config after construction.
    Instantiate once at module level (see DEFAULT_EMBEDDING_CONFIG below)
    and import that instance everywhere.

    Fields
    ------
    model_id            Bedrock model identifier passed to
                        bedrock-runtime invoke_model(). Must match a
                        model available in your AWS account and region.

    vector_dimension    Output dimension of the embedding model. This value
                        MUST match the `dimension` field in the OpenSearch
                        Serverless index mapping. If you change the model,
                        you must re-create the index and re-embed all
                        previously ingested documents.

    Notes
    -----
    If you switch to a model with a different dimension (e.g. Cohere
    embed-english-v3 at 1024 dims, or Titan v1 at 1536 dims), update
    model_id and vector_dimension together -- they cannot be changed
    independently. The opensearch_dimension_mapping property derives
    from vector_dimension, so the CDK stack will automatically use
    the correct dimension if it calls that property rather than
    hardcoding a number.

    See docs/ARCHITECTURE.md para Layer 2 -- Embedding & Storage.
    """
    model_id: str = "amazon.titan-embed-text-v2:0"
    vector_dimension: int = 1024

    def __post_init__(self) -> None:
        if self.vector_dimension <= 0:
            raise ValueError(
                f"vector_dimension must be positive; got {self.vector_dimension}"
            )

    @property
    def opensearch_dimension_mapping(self) -> dict:
        """
        Return the OpenSearch knn_vector field mapping fragment for this config.

        The CDK stack should call this rather than hardcoding the dimension,
        so that changing the model here automatically propagates to the index.

        Usage in CDK or index creation:
            mapping = DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping
            # -> {"type": "knn_vector", "dimension": 1024, "method": {...}}
        """
        return {
            "type": "knn_vector",
            "dimension": self.vector_dimension,
            "method": {
                "name": "hnsw",
                "engine": "faiss",
                "parameters": {
                    "ef_construction": 512,
                    "m": 16,
                },
            },
        }


# ---------------------------------------------------------------------------
# ChunkingConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    """
    Document chunking parameters used by ingestor.py.

    Kept separate from EmbeddingConfig because chunking strategy is an
    independent tuning decision. You can adjust chunk size and overlap
    without changing the embedding model or re-creating the OpenSearch
    index -- though changing chunk_size_tokens requires re-ingesting
    all documents for consistent retrieval behaviour.

    Fields
    ------
    chunk_size_tokens   Target chunk size in tokens. Documents are split
                        into overlapping windows of this size before
                        embedding.

                        Tradeoff:
                          Smaller chunks (256-512) -> higher recall, less
                          context per chunk, more chunks per document.
                          Larger chunks (1024-2048) -> more context per
                          chunk, lower recall for short precise queries.

                        512 / 50 is a reasonable starting point for legal
                        and investigative documents where precise entity
                        and date references matter more than broad thematic
                        context. Tune after evaluating retrieval quality
                        on real queries.

    chunk_overlap_tokens
                        Number of tokens shared between adjacent chunks.
                        Overlap ensures sentences spanning a chunk boundary
                        appear in full in at least one chunk. A ratio of
                        ~10% of chunk_size is a common baseline.

    Notes
    -----
    Token counts here are approximate. The ingestor uses a word-based
    splitter as a proxy for tokens (1 word ~ 1.3 tokens for English
    legal text). For exact token counts, use the Bedrock tokenizer or
    tiktoken with the cl100k_base encoding as an approximation.

    Do not change chunk_size_tokens without re-ingesting all previously
    embedded documents -- mixed chunk sizes in the same index degrade
    retrieval consistency.
    """
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 50

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError(
                f"chunk_size_tokens must be positive; got {self.chunk_size_tokens}"
            )
        if self.chunk_overlap_tokens < 0:
            raise ValueError(
                f"chunk_overlap_tokens must be non-negative; "
                f"got {self.chunk_overlap_tokens}"
            )
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                f"chunk_overlap_tokens ({self.chunk_overlap_tokens}) must be "
                f"less than chunk_size_tokens ({self.chunk_size_tokens})"
            )

    @property
    def overlap_ratio(self) -> float:
        """Overlap as a fraction of chunk size. Useful for logging and evals."""
        return self.chunk_overlap_tokens / self.chunk_size_tokens


# ---------------------------------------------------------------------------
# Module-level default instances
# ---------------------------------------------------------------------------

#: Import this instance everywhere rather than constructing EmbeddingConfig()
#: directly. Consistent use of this singleton ensures that ingestor.py,
#: the CDK stack, and the query router share exactly the same model and
#: dimension without risk of divergence.
#:
#: To override for testing or experimentation:
#:     from config import EmbeddingConfig
#:     test_config = EmbeddingConfig(model_id="...", vector_dimension=256)
DEFAULT_EMBEDDING_CONFIG: EmbeddingConfig = EmbeddingConfig()

#: Import this instance everywhere rather than constructing ChunkingConfig()
#: directly.
#:
#: To override for testing or experimentation:
#:     from config import ChunkingConfig
#:     test_config = ChunkingConfig(chunk_size_tokens=256, chunk_overlap_tokens=25)
DEFAULT_CHUNKING_CONFIG: ChunkingConfig = ChunkingConfig()

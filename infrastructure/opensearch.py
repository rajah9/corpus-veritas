"""
infrastructure/opensearch.py
Layer 2: OpenSearch Serverless index creation and management.

Infrastructure code is kept separate from the pipeline so that index
creation can be run once (or re-run for schema migrations) without
coupling it to the hot path of document ingestion.

Collection : corpus-veritas
Index      : documents

The index mapping is derived from EmbeddingConfig.opensearch_dimension_mapping
so that changing the embedding model in config.py automatically propagates
to the index definition. Never hardcode the vector dimension here.

OpenSearch client
-----------------
This module uses the opensearch-py client (opensearchservice-compatible),
injected for testing. For OpenSearch Serverless (AOSS), the client must
be constructed with AWS SigV4 request signing:

    from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
    import boto3

    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, "aoss")
    client = OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        connection_class=RequestsHttpConnection,
    )

OPENSEARCH_ENDPOINT must be set in the environment (the AOSS collection
endpoint, e.g. https://<id>.us-east-1.aoss.amazonaws.com).

See docs/ARCHITECTURE.md para Layer 2 -- OpenSearch Serverless.
"""

from __future__ import annotations

import logging
import os

from config import DEFAULT_EMBEDDING_CONFIG, EmbeddingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENSEARCH_ENDPOINT: str = os.environ.get("OPENSEARCH_ENDPOINT", "")
OPENSEARCH_INDEX: str = os.environ.get("OPENSEARCH_INDEX", "documents")
AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")


# ---------------------------------------------------------------------------
# Index mapping
# ---------------------------------------------------------------------------

def build_index_mapping(
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
) -> dict:
    """
    Build the OpenSearch index mapping for the `documents` index.

    The knn_vector dimension is derived from embedding_config so that
    changing the model in config.py propagates here automatically.
    All other field mappings are static.

    Field notes
    -----------
    text            stored but not indexed for full-text search by default.
                    Add "analyzer": "standard" here if full-text search is
                    needed in addition to vector search.

    victim_flag     keyword type for exact-match filtering. The guardrail
                    layer queries: filter={"term": {"victim_flag": "true"}}
                    to exclude flagged chunks before returning results.

    vector          knn_vector indexed with HNSW/faiss. Parameters
                    ef_construction and m are set in the embedding config's
                    opensearch_dimension_mapping; defaults are conservative
                    and correct for a corpus of ~600K documents.

    Parameters
    ----------
    embedding_config : EmbeddingConfig to derive vector dimension from.
                       Defaults to DEFAULT_EMBEDDING_CONFIG.

    Returns
    -------
    dict suitable for passing to opensearch client indices.create().
    """
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512,
            }
        },
        "mappings": {
            "properties": {
                "document_uuid":    {"type": "keyword"},
                "chunk_index":      {"type": "integer"},
                "text":             {"type": "text", "index": False},
                "vector":           embedding_config.opensearch_dimension_mapping,
                "classification":   {"type": "keyword"},
                "provenance_tag":   {"type": "keyword"},
                "ingestion_date":   {"type": "date"},
                "victim_flag":      {"type": "keyword"},
                "page_number":      {"type": "integer"},
                "bates_number":     {"type": "keyword"},
                "efta_number":      {"type": "keyword"},
                "corpus_source":    {"type": "keyword"},
                # Fields added in Layer 2 schema extension
                "sequence_number":  {"type": "keyword"},
                "sequence_scheme":  {"type": "keyword"},
                "document_date":    {"type": "date"},
                "document_type":    {"type": "keyword"},
                "named_entities":   {"type": "object", "dynamic": True},
                "confidence_tier":  {"type": "keyword"},
                "deletion_flag":    {"type": "keyword"},
            }
        },
    }


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------

def create_index(
    opensearch_client,
    index_name: str = OPENSEARCH_INDEX,
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
) -> bool:
    """
    Create the OpenSearch index if it does not already exist.

    Safe to call on every deployment -- skips creation if the index is
    already present and logs the outcome either way.

    Parameters
    ----------
    opensearch_client : opensearch-py OpenSearch client (injectable for testing).
    index_name        : Name of the index to create. Defaults to OPENSEARCH_INDEX.
    embedding_config  : EmbeddingConfig used to derive the vector dimension.

    Returns
    -------
    True  if the index was created.
    False if the index already existed (no action taken).

    Raises
    ------
    RuntimeError wrapping the underlying opensearch-py exception if index
    creation fails for a reason other than the index already existing.
    """
    try:
        exists = opensearch_client.indices.exists(index=index_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to check existence of index '{index_name}': {exc}"
        ) from exc

    if exists:
        logger.info("Index '%s' already exists -- skipping creation.", index_name)
        return False

    mapping = build_index_mapping(embedding_config)
    try:
        opensearch_client.indices.create(index=index_name, body=mapping)
        logger.info(
            "Created index '%s' with vector dimension %d.",
            index_name, embedding_config.vector_dimension,
        )
        return True
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create index '{index_name}': {exc}"
        ) from exc


def delete_index(
    opensearch_client,
    index_name: str = OPENSEARCH_INDEX,
) -> bool:
    """
    Delete the OpenSearch index if it exists.

    Intended for use in migrations and test teardown only.
    NEVER call this against the production index without a full re-ingestion
    plan -- all embedded chunks will be permanently lost.

    Parameters
    ----------
    opensearch_client : opensearch-py OpenSearch client.
    index_name        : Name of the index to delete.

    Returns
    -------
    True  if the index was deleted.
    False if the index did not exist.

    Raises
    ------
    RuntimeError if deletion fails for a reason other than non-existence.
    """
    try:
        exists = opensearch_client.indices.exists(index=index_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to check existence of index '{index_name}': {exc}"
        ) from exc

    if not exists:
        logger.info("Index '%s' does not exist -- nothing to delete.", index_name)
        return False

    try:
        opensearch_client.indices.delete(index=index_name)
        logger.info("Deleted index '%s'.", index_name)
        return True
    except Exception as exc:
        raise RuntimeError(
            f"Failed to delete index '{index_name}': {exc}"
        ) from exc

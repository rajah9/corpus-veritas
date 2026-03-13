"""
rag/query_router.py
Layer 3: Query routing, retrieval, and Bedrock synthesis.

Receives a typed QueryRequest, builds an OpenSearch DSL query appropriate
for the query type, retrieves matching chunks, and synthesises a grounded
answer via Bedrock (Claude). Returns a RetrievalResult containing the
chunks, the synthesised answer, and metadata about the retrieval.

Query types and retrieval strategies
--------------------------------------
TIMELINE        kNN vector search scoped to a date range and sorted
                chronologically by document_date. Named entity filter
                applied when entity_names are supplied.

PROVENANCE      kNN vector search returning all matching chunks. No
                date or entity filter -- the goal is to find every
                document that touches the claim so source diversity
                can be counted by convergence_checker.py.

INFERENCE       kNN vector search. Multi-source convergence is NOT
                enforced here -- that is the caller's responsibility
                via convergence_checker.py. The router tags the result
                with convergence_applied=False so the caller knows to
                run the check before surfacing the answer to a user.

RELATIONSHIP    kNN vector search filtered to chunks whose named_entities
                contain any of the requested entity names. Graph traversal
                (Layer 4) is not yet implemented; the router falls back to
                document-level entity matching.

Victim flag suppression
-----------------------
A must_not filter on victim_flag="true" is applied to every query type
without exception. This is a hard architectural constraint (Constitution
Hard Limit 1), not a policy flag. It is not injectable, configurable, or
overridable by query parameters.

Bedrock synthesis
-----------------
Retrieved chunks are formatted into a context block and passed to Claude
(claude-sonnet-4-6) via bedrock-runtime invoke_model(). The synthesis
prompt enforces confidence tier language: the model is instructed to use
only language consistent with the lowest confidence tier present in the
retrieved chunks. SINGLE_SOURCE and SPECULATIVE chunks require hedging
language. CONFIRMED language is only permitted when all retrieved chunks
are CONFIRMED tier.

The Bedrock client is injectable for testing. The synthesis model ID is
hardcoded to claude-sonnet-4-6 -- do not change this without reviewing
the synthesis prompt, which was written for Claude's instruction-following
behaviour.

OpenSearch client
-----------------
Injectable for testing. For production, construct with SigV4 signing as
described in infrastructure/opensearch.py.

See CONSTITUTION.md Article III Hard Limits 1, 2, 3.
See docs/ARCHITECTURE.md para Layer 3 -- RAG Engine.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pipeline.ingestor import embed_text
from pipeline.models import ConfidenceTier
from config import DEFAULT_EMBEDDING_CONFIG, EmbeddingConfig

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
OPENSEARCH_INDEX: str = os.environ.get("OPENSEARCH_INDEX", "documents")
SYNTHESIS_MODEL_ID: str = "claude-sonnet-4-6"
DEFAULT_TOP_K: int = 10

# Confidence tier ordering for synthesis prompt (weakest wins)
_TIER_ORDER: dict[str, int] = {
    ConfidenceTier.SPECULATIVE:   0,
    ConfidenceTier.SINGLE_SOURCE: 1,
    ConfidenceTier.INFERRED:      2,
    ConfidenceTier.CORROBORATED:  3,
    ConfidenceTier.CONFIRMED:     4,
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    """
    Classification of the user's query intent.

    Determines the OpenSearch DSL retrieval strategy and the synthesis
    prompt template used by Bedrock.

    TIMELINE        "What happened when?" queries. Requires date-range
                    filtering and chronological sorting. Named entity
                    filter strongly recommended.

    PROVENANCE      "Is this claim supported?" queries. Retrieves all
                    matching chunks for source diversity counting.
                    Caller should pass result to convergence_checker.py.

    INFERENCE       "What does this suggest?" queries. Multi-source
                    convergence check is required before surfacing the
                    answer (Constitution Hard Limit 2). The router sets
                    convergence_applied=False; the caller must run
                    convergence_checker.check_convergence() and suppress
                    or downgrade if the threshold is not met.

    RELATIONSHIP    "Who connects to whom?" queries. Entity-filtered
                    retrieval. Full graph traversal deferred to Layer 4.
    """
    TIMELINE     = "TIMELINE"
    PROVENANCE   = "PROVENANCE"
    INFERENCE    = "INFERENCE"
    RELATIONSHIP = "RELATIONSHIP"


@dataclass
class QueryRequest:
    """
    Typed input to route_query().

    Fields
    ------
    query_text      Natural language query from the user.
    query_type      QueryType controlling retrieval strategy and prompt.
    top_k           Maximum number of chunks to retrieve. Default 10.
    entity_names    For TIMELINE and RELATIONSHIP queries: names of
                    individuals or organisations to filter on. None
                    means no entity filter (broader retrieval).
    date_from       ISO 8601 date string. Lower bound for TIMELINE
                    queries. None means no lower bound.
    date_to         ISO 8601 date string. Upper bound for TIMELINE
                    queries. None means no upper bound.
    """
    query_text:   str
    query_type:   QueryType
    top_k:        int = DEFAULT_TOP_K
    entity_names: Optional[list[str]] = None
    date_from:    Optional[str] = None
    date_to:      Optional[str] = None

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("query_text must not be empty")
        if self.top_k <= 0:
            raise ValueError(f"top_k must be positive; got {self.top_k}")


@dataclass
class RetrievalResult:
    """
    Output of route_query().

    Fields
    ------
    query               The original QueryRequest.
    chunks              List of OpenSearch hit dicts (_source fields).
                        May be empty if no matching chunks were found.
    answer              Bedrock-synthesised answer grounded in the chunks.
                        Empty string if no chunks were retrieved.
    convergence_applied True if the convergence check was applied before
                        synthesis. Always False for INFERENCE queries --
                        caller must run convergence_checker.check_convergence()
                        before surfacing the answer to a user.
    retrieved_at        ISO 8601 UTC timestamp of retrieval.
    lowest_tier         The weakest ConfidenceTier present in the retrieved
                        chunks. Synthesis language is constrained to this
                        tier. None if no chunks were retrieved.
    """
    query:               QueryRequest
    chunks:              list[dict]
    answer:              str
    convergence_applied: bool
    retrieved_at:        str
    lowest_tier:         Optional[str] = None


# ---------------------------------------------------------------------------
# DSL builders
# ---------------------------------------------------------------------------

def _victim_flag_filter() -> dict:
    """
    OpenSearch must_not filter that excludes victim-flagged chunks.

    Applied unconditionally to every query. Constitution Hard Limit 1.
    """
    return {"term": {"victim_flag": "true"}}


def _build_timeline_query(
    vector: list[float],
    top_k: int,
    entity_names: Optional[list[str]],
    date_from: Optional[str],
    date_to: Optional[str],
) -> dict:
    """
    kNN vector search with optional date range and entity name filters,
    sorted chronologically by document_date.
    """
    filters: list[dict] = []

    if date_from or date_to:
        date_range: dict = {}
        if date_from:
            date_range["gte"] = date_from
        if date_to:
            date_range["lte"] = date_to
        filters.append({"range": {"document_date": date_range}})

    if entity_names:
        filters.append({
            "terms": {"named_entities.text": entity_names}
        })

    query: dict = {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "vector": {
                                "vector": vector,
                                "k": top_k,
                            }
                        }
                    }
                ],
                "filter": filters,
                "must_not": [_victim_flag_filter()],
            }
        },
        "sort": [{"document_date": {"order": "asc"}}],
    }
    return query


def _build_provenance_query(
    vector: list[float],
    top_k: int,
) -> dict:
    """
    kNN vector search with no additional filters.
    Returns all matching chunks for source diversity counting.
    """
    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "vector": {
                                "vector": vector,
                                "k": top_k,
                            }
                        }
                    }
                ],
                "must_not": [_victim_flag_filter()],
            }
        },
    }


def _build_inference_query(
    vector: list[float],
    top_k: int,
) -> dict:
    """
    kNN vector search. Victim flag suppressed.
    Convergence check is the caller's responsibility.
    """
    return _build_provenance_query(vector, top_k)


def _build_relationship_query(
    vector: list[float],
    top_k: int,
    entity_names: Optional[list[str]],
) -> dict:
    """
    kNN vector search filtered to chunks referencing the requested entities.
    Falls back to broad vector search if no entity_names supplied.
    """
    must_not = [_victim_flag_filter()]
    filters: list[dict] = []

    if entity_names:
        filters.append({
            "terms": {"named_entities.text": entity_names}
        })

    return {
        "size": top_k,
        "query": {
            "bool": {
                "must": [
                    {
                        "knn": {
                            "vector": {
                                "vector": vector,
                                "k": top_k,
                            }
                        }
                    }
                ],
                "filter": filters,
                "must_not": must_not,
            }
        },
    }


def build_query(request: QueryRequest, vector: list[float]) -> dict:
    """
    Dispatch to the correct DSL builder for the request's query type.

    Parameters
    ----------
    request : QueryRequest controlling which builder is invoked.
    vector  : Embedded query vector from embed_text().

    Returns
    -------
    OpenSearch DSL query dict.
    """
    if request.query_type == QueryType.TIMELINE:
        return _build_timeline_query(
            vector, request.top_k, request.entity_names,
            request.date_from, request.date_to,
        )
    if request.query_type == QueryType.PROVENANCE:
        return _build_provenance_query(vector, request.top_k)
    if request.query_type == QueryType.INFERENCE:
        return _build_inference_query(vector, request.top_k)
    if request.query_type == QueryType.RELATIONSHIP:
        return _build_relationship_query(
            vector, request.top_k, request.entity_names,
        )
    raise ValueError(f"Unknown query type: {request.query_type}")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_chunks(
    dsl_query: dict,
    opensearch_client,
    index_name: str = OPENSEARCH_INDEX,
) -> list[dict]:
    """
    Execute an OpenSearch DSL query and return the _source fields of hits.

    Parameters
    ----------
    dsl_query         : OpenSearch DSL dict from build_query().
    opensearch_client : opensearch-py client (injectable for testing).
    index_name        : Target index name.

    Returns
    -------
    List of _source dicts (one per hit). Empty list if no results.

    Raises
    ------
    RuntimeError if the OpenSearch call fails.
    """
    try:
        response = opensearch_client.search(
            index=index_name,
            body=dsl_query,
        )
        return [
            hit["_source"]
            for hit in response.get("hits", {}).get("hits", [])
        ]
    except Exception as exc:
        raise RuntimeError(
            f"OpenSearch retrieval failed on index '{index_name}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def _lowest_confidence_tier(chunks: list[dict]) -> Optional[str]:
    """
    Return the weakest ConfidenceTier present across all chunks.
    None if no chunks have a confidence_tier field.
    """
    tiers = [
        c["confidence_tier"]
        for c in chunks
        if c.get("confidence_tier")
    ]
    if not tiers:
        return None
    return min(tiers, key=lambda t: _TIER_ORDER.get(t, 0))


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the prompt.
    Each entry includes the chunk text, document UUID, provenance tag,
    confidence tier, and document type where available.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] document_uuid={chunk.get('document_uuid', 'unknown')}")
        if chunk.get("document_type"):
            lines.append(f"    type={chunk['document_type']}")
        if chunk.get("document_date"):
            lines.append(f"    date={chunk['document_date']}")
        if chunk.get("provenance_tag"):
            lines.append(f"    provenance={chunk['provenance_tag']}")
        if chunk.get("confidence_tier"):
            lines.append(f"    confidence={chunk['confidence_tier']}")
        lines.append(f"    text: {chunk.get('text', '')}")
        lines.append("")
    return "\n".join(lines)


def _build_synthesis_prompt(
    request: QueryRequest,
    chunks: list[dict],
    lowest_tier: Optional[str],
) -> str:
    """
    Build the Bedrock synthesis prompt for the given query and chunks.

    Enforces Constitution Hard Limits 2, 3, 4:
    - Hard Limit 2: INFERENCE queries with fewer than 2 independent sources
      are flagged in the prompt -- the model is instructed to state that
      convergence was not met rather than surface the inference.
    - Hard Limit 3: Language constraints are derived from lowest_tier.
      The model is explicitly prohibited from using CONFIRMED language
      unless lowest_tier is CONFIRMED.
    - Hard Limit 4: The model is instructed to analyse documents, not
      generate creative or speculative content.
    """
    tier_instruction = {
        None:                    "You have no confidence tier information. Treat all claims as SINGLE_SOURCE.",
        ConfidenceTier.SPECULATIVE:   "The weakest source is SPECULATIVE. Use strong hedging language throughout: 'may suggest', 'it is possible that', 'without corroboration'. Do not present any claim as established fact.",
        ConfidenceTier.SINGLE_SOURCE: "The weakest source is SINGLE_SOURCE. Clearly state that claims come from a single document and have not been independently corroborated.",
        ConfidenceTier.INFERRED:      "The weakest source is INFERRED. Clearly distinguish between what documents state directly and what is reasonably implied.",
        ConfidenceTier.CORROBORATED:  "The weakest source is CORROBORATED. You may note that multiple sources converge, but do not use the word 'confirmed' unless the tier is CONFIRMED.",
        ConfidenceTier.CONFIRMED:     "The weakest source is CONFIRMED. You may use 'confirmed' language where directly supported.",
    }.get(lowest_tier, "Treat all claims as SINGLE_SOURCE.")

    query_type_instruction = {
        QueryType.TIMELINE:     "Organise your answer chronologically. Cite the document_date for each event. Note any gaps in the timeline.",
        QueryType.PROVENANCE:   "Assess the evidential basis for the claim. Count how many independent documents support it and note any contradictions.",
        QueryType.INFERENCE:    "State only what the documents directly support. If fewer than two independent documents converge on an inference about a living individual, state clearly that the convergence threshold was not met and decline to surface the inference.",
        QueryType.RELATIONSHIP: "Describe the documented connections between the named individuals. Distinguish between direct associations (named together in one document) and inferred associations (linked through a third party).",
    }.get(request.query_type, "")

    context_block = _format_chunks_for_prompt(chunks) if chunks else "(No matching documents found.)"

    return f"""You are a document analysis assistant for corpus-veritas, a system that analyses publicly released legal documents.

QUERY TYPE: {request.query_type.value}
QUERY: {request.query_text}

LANGUAGE CONSTRAINTS:
{tier_instruction}

RESPONSE INSTRUCTIONS:
{query_type_instruction}

HARD RULES (never violate these):
1. Never expose or reference victim identities. If any retrieved document appears to contain victim identity information, do not reproduce it.
2. Never present an inference about a living individual as confirmed unless multiple independent documents converge on that inference.
3. Never generate creative, speculative, or hypothetical content beyond what the documents directly support.
4. Every claim in your answer must be traceable to a specific chunk in the context below. Cite document_uuid values.
5. If the documents do not support an answer to the query, say so clearly rather than speculating.

RETRIEVED DOCUMENTS:
{context_block}

Provide a grounded, citation-supported answer to the query based solely on the retrieved documents above."""


def synthesise_answer(
    request: QueryRequest,
    chunks: list[dict],
    bedrock_client,
    lowest_tier: Optional[str] = None,
) -> str:
    """
    Call Bedrock (Claude) to synthesise an answer from retrieved chunks.

    Parameters
    ----------
    request         : Original QueryRequest.
    chunks          : Retrieved chunk dicts from retrieve_chunks().
    bedrock_client  : boto3 bedrock-runtime client (injectable for testing).
    lowest_tier     : Weakest confidence tier in chunks, for prompt construction.

    Returns
    -------
    Synthesised answer string. Empty string if chunks is empty.

    Raises
    ------
    RuntimeError if the Bedrock call fails.
    """
    if not chunks:
        return ""

    prompt = _build_synthesis_prompt(request, chunks, lowest_tier)

    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        })
        response = bedrock_client.invoke_model(
            modelId=SYNTHESIS_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    except Exception as exc:
        raise RuntimeError(
            f"Bedrock synthesis failed (model={SYNTHESIS_MODEL_ID}): {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_query(
    request: QueryRequest,
    opensearch_client,
    bedrock_client=None,
    embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
    index_name: str = OPENSEARCH_INDEX,
) -> RetrievalResult:
    """
    Route a query through the full retrieval and synthesis pipeline.

    Steps:
      1. Embed query_text via Bedrock (same model as ingestor).
      2. Build OpenSearch DSL for the query type.
      3. Retrieve matching chunks (victim_flag always suppressed).
      4. Synthesise a grounded answer via Bedrock (Claude).
      5. Return RetrievalResult.

    For INFERENCE queries, convergence_applied is always False.
    Callers must run convergence_checker.check_convergence() on the
    result and suppress or downgrade the answer if the convergence
    threshold is not met before surfacing it to a user.

    Parameters
    ----------
    request           : QueryRequest with query text, type, and filters.
    opensearch_client : opensearch-py client (injectable for testing).
    bedrock_client    : boto3 bedrock-runtime client. If None, created
                        from AWS_REGION. Inject a MagicMock for testing.
    embedding_config  : EmbeddingConfig for query embedding.
    index_name        : OpenSearch index to query.

    Returns
    -------
    RetrievalResult containing chunks, answer, and retrieval metadata.

    Raises
    ------
    RuntimeError if embedding, retrieval, or synthesis fails.

    Constitution reference:
      Hard Limit 1 -- victim_flag suppression in every DSL query.
      Hard Limit 2 -- INFERENCE convergence_applied=False signals caller.
      Hard Limit 3 -- synthesis prompt enforces tier language.
      Hard Limit 4 -- synthesis prompt prohibits creative content.
    """
    if bedrock_client is None:
        import boto3
        bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    logger.info(
        "Routing query type=%s top_k=%d", request.query_type.value, request.top_k
    )

    # Step 1: embed the query
    vector = embed_text(request.query_text, bedrock_client, embedding_config)

    # Step 2: build DSL
    dsl = build_query(request, vector)

    # Step 3: retrieve
    chunks = retrieve_chunks(dsl, opensearch_client, index_name)
    logger.info("Retrieved %d chunks for query type=%s", len(chunks), request.query_type.value)

    # Step 4: determine lowest tier and synthesise
    lowest_tier = _lowest_confidence_tier(chunks)
    answer = synthesise_answer(request, chunks, bedrock_client, lowest_tier)

    # Step 5: return result
    # INFERENCE queries never set convergence_applied=True here --
    # that is the caller's responsibility via convergence_checker.
    convergence_applied = request.query_type != QueryType.INFERENCE

    return RetrievalResult(
        query=request,
        chunks=chunks,
        answer=answer,
        convergence_applied=convergence_applied,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        lowest_tier=lowest_tier,
    )

"""
pipeline/graph_populator.py
Milestone 8: Wires NER extraction into the relationship graph.

This module closes the gap identified in the Milestone 7 architecture review:
the RelationshipGraph was not auto-populated during ingestion. Documents were
NER-extracted and stored in DynamoDB (ner_extractor.py) and the graph
existed (relationship_graph.py) but nothing connected them.

graph_populator.py sits between ingestion and the graph:

  ingestor.py                             graph_populator.py
  ──────────────────────────────────────  ──────────────────────────────────────
  chunk_text()                            populate_from_chunk()
  embed_text()                            populate_from_document()
  index_chunk()                           infer_edges_from_co_occurrence()
                                          save_graph()

Typical caller sequence after ingest_document():
    entities = extract_entities_for_chunk(text, document_uuid, comprehend)
    entities = deduplicate_entities(entities)
    populate_from_document(
        document_uuid=record.document_uuid,
        named_entities=entities,
        classification=record.classification.value,
        graph=graph,
        comprehend_client=comprehend,
    )
    save_graph(graph, s3_client, bucket, key)

Edge inference
--------------
Two PERSON entities co-occurring in the same document receive an ASSOCIATE
edge if no edge already exists between them. Co-occurrence is a weak signal
-- it means they appear in the same document, not that they are directly
related -- so edges are created with a low default confidence (0.3) and
marked with the co_occurrence note. Higher-confidence edges from explicit
relationship statements should overwrite this default when added later.

Victim flag propagation
-----------------------
Entities resolving to a victim canonical name receive victim_flag=True
both in the graph and in DynamoDB (via upsert_entity_record). This ensures
that victim suppression in graph traversal and in the DynamoDB entity table
is consistent regardless of which module processes the entity first.

See docs/ARCHITECTURE.md para Layer 4 -- NER & Relationship Graph.
See CONSTITUTION.md Article III Hard Limit 1.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from graph.entity_resolver import EdgeType, Entity, EntityEdge, EntityType, resolve_entity
from graph.relationship_graph import RelationshipGraph
from pipeline.ner_extractor import upsert_entity_record
from pipeline.models import ConfidenceTier

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")

# Default confidence for co-occurrence-inferred edges.
# Lower than explicit relationship extraction to signal weak evidential basis.
CO_OCCURRENCE_CONFIDENCE: float = 0.30

# Only infer co-occurrence edges for PERSON entities -- too noisy for
# ORGANIZATION and LOCATION (every document mentioning the DOJ and Florida
# would create an ASSOCIATE edge between them).
CO_OCCURRENCE_ENTITY_TYPES: set[str] = {"PERSON"}


# ---------------------------------------------------------------------------
# Core population
# ---------------------------------------------------------------------------

def populate_from_chunk(
    chunk_entities: list[dict],
    document_uuid: str,
    graph: RelationshipGraph,
    comprehend_client=None,
    dynamodb_client=None,
) -> list[Entity]:
    """
    Resolve and add entities from one chunk to the graph.

    For each entity dict in chunk_entities (as produced by
    ner_extractor.extract_entities_for_chunk):
      1. Resolve to a canonical Entity via entity_resolver.resolve_entity()
      2. Add document_uuid to the entity's document_uuids list
      3. Add/merge the entity into the graph
      4. Upsert to DynamoDB corpus_veritas_entities

    Parameters
    ----------
    chunk_entities   : List of entity dicts from ner_extractor.
    document_uuid    : UUID of the source document.
    graph            : RelationshipGraph to populate.
    comprehend_client: Injectable boto3 Comprehend client for Stage 3
                       entity linking. None skips Stage 3 (alias map only).
    dynamodb_client  : Injectable boto3 DynamoDB client. None defers to
                       upsert_entity_record's default construction.

    Returns
    -------
    List of resolved Entity instances added to the graph.
    """
    resolved: list[Entity] = []

    for ent in chunk_entities:
        surface_form = ent.get("text", "")
        type_str     = ent.get("type", "PERSON")
        confidence   = float(ent.get("confidence", 0.0))

        try:
            entity_type = EntityType(type_str)
        except ValueError:
            logger.debug("Unknown entity type %r -- skipping.", type_str)
            continue

        entity = resolve_entity(
            surface_form=surface_form,
            entity_type=entity_type,
            confidence=confidence,
            comprehend_client=comprehend_client,
        )
        entity.document_uuids = [document_uuid]

        graph.add_entity(entity)

        try:
            upsert_entity_record(
                canonical_name=entity.canonical_name,
                entity_type=entity_type.value,
                surface_form=surface_form,
                document_uuid=document_uuid,
                confidence=confidence,
                victim_flag=entity.victim_flag,
                dynamodb_client=dynamodb_client,
            )
        except Exception as exc:
            logger.warning(
                "DynamoDB upsert failed for entity '%s': %s -- continuing.",
                entity.canonical_name, exc,
            )

        resolved.append(entity)

    return resolved


def populate_from_document(
    document_uuid: str,
    named_entities: list[dict],
    graph: RelationshipGraph,
    classification: str = "",
    comprehend_client=None,
    dynamodb_client=None,
    infer_edges: bool = True,
) -> list[Entity]:
    """
    Populate the graph from all entities in one document.

    Calls populate_from_chunk() then optionally infers co-occurrence
    edges between PERSON entities that appear in the same document.

    Parameters
    ----------
    document_uuid    : UUID of the source document.
    named_entities   : Deduplicated entity list (from deduplicate_entities()).
    graph            : RelationshipGraph to populate.
    classification   : DocumentClassification value string -- used to
                       skip edge inference for VICTIM_ADJACENT documents.
    comprehend_client: Injectable Comprehend client.
    dynamodb_client  : Injectable DynamoDB client.
    infer_edges      : If True, infer co-occurrence edges between PERSON
                       entities. Set False for VICTIM_ADJACENT documents
                       to avoid creating graph edges that could indirectly
                       identify victim connections.

    Returns
    -------
    List of resolved Entity instances added to the graph.
    """
    # Skip edge inference for victim-adjacent documents -- co-occurrence
    # edges could create traversal paths that indirectly expose victim
    # context. Constitution Hard Limit 1.
    if classification == "VICTIM_ADJACENT":
        infer_edges = False

    resolved = populate_from_chunk(
        chunk_entities=named_entities,
        document_uuid=document_uuid,
        graph=graph,
        comprehend_client=comprehend_client,
        dynamodb_client=dynamodb_client,
    )

    if infer_edges:
        infer_edges_from_co_occurrence(resolved, document_uuid, graph)

    logger.debug(
        "populate_from_document: %d entities, %d resolved, infer_edges=%s",
        len(named_entities), len(resolved), infer_edges,
    )
    return resolved


# ---------------------------------------------------------------------------
# Edge inference
# ---------------------------------------------------------------------------

def infer_edges_from_co_occurrence(
    entities: list[Entity],
    document_uuid: str,
    graph: RelationshipGraph,
) -> int:
    """
    Infer ASSOCIATE edges between PERSON entities in the same document.

    Creates a directed ASSOCIATE edge from each PERSON entity to every
    other PERSON entity co-occurring in the same document. Edges are
    created with CO_OCCURRENCE_CONFIDENCE (0.30) and a note indicating
    the weak evidential basis.

    Both nodes must already exist in the graph. Entities not in the graph
    (e.g. because they were victim-flagged and suppressed) are silently
    skipped.

    Parameters
    ----------
    entities     : List of resolved Entity instances from the document.
    document_uuid: UUID of the source document (used as edge evidence).
    graph        : RelationshipGraph to add edges to.

    Returns
    -------
    Number of edges added.
    """
    person_entities = [
        e for e in entities
        if e.entity_type == EntityType.PERSON
        and not e.victim_flag
        and graph.get_entity(e.node_id) is not None
    ]

    edges_added = 0
    for i, entity_a in enumerate(person_entities):
        for entity_b in person_entities[i + 1:]:
            edge = EntityEdge(
                source_node_id=entity_a.node_id,
                target_node_id=entity_b.node_id,
                edge_type=EdgeType.ASSOCIATE,
                document_uuids=[document_uuid],
                confidence=CO_OCCURRENCE_CONFIDENCE,
                notes="Co-occurrence inferred -- weak signal. "
                      "Overwrite with higher-confidence explicit edge when available.",
            )
            try:
                graph.add_edge(edge)
                edges_added += 1
            except ValueError:
                # Node not in graph -- skip silently
                pass

    return edges_added


# ---------------------------------------------------------------------------
# S3 persistence helpers
# ---------------------------------------------------------------------------

def save_graph(
    graph: RelationshipGraph,
    s3_client,
    bucket_name: str,
    s3_key: str,
) -> None:
    """
    Save the graph to S3.

    Thin wrapper around RelationshipGraph.save_to_s3() for consistent
    error logging. Raises RuntimeError if the save fails.
    """
    try:
        graph.save_to_s3(s3_client, bucket_name, s3_key)
        logger.info(
            "Graph saved: %d nodes, %d edges → s3://%s/%s",
            graph.node_count, graph.edge_count, bucket_name, s3_key,
        )
    except RuntimeError:
        raise


def load_or_create_graph(
    s3_client,
    bucket_name: str,
    s3_key: str,
) -> RelationshipGraph:
    """
    Load the graph from S3 or create an empty one if not found.

    Safe to call at pipeline startup -- if no graph exists yet (first run),
    an empty RelationshipGraph is returned and will be populated during
    ingestion.

    Parameters
    ----------
    s3_client   : boto3 S3 client.
    bucket_name : S3 bucket name.
    s3_key      : S3 object key for the graph JSON.

    Returns
    -------
    RelationshipGraph -- loaded from S3 or freshly created.
    """
    try:
        graph = RelationshipGraph.load_from_s3(s3_client, bucket_name, s3_key)
        logger.info(
            "Graph loaded from s3://%s/%s (%d nodes, %d edges).",
            bucket_name, s3_key, graph.node_count, graph.edge_count,
        )
        return graph
    except RuntimeError as exc:
        if "NoSuchKey" in str(exc) or "404" in str(exc):
            logger.info(
                "No graph found at s3://%s/%s -- creating empty graph.",
                bucket_name, s3_key,
            )
            return RelationshipGraph()
        raise

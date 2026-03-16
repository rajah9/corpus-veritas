"""
graph/relationship_graph.py
Layer 4: NetworkX relationship graph with S3 persistence.

Maintains a directed graph of entities (nodes) and relationships (edges)
extracted from the corpus. The graph answers RELATIONSHIP queries in the
RAG layer by providing structured traversal paths that kNN retrieval alone
cannot produce.

Learning phase storage
----------------------
The graph is serialised as JSON to S3 at a configurable key. The full
graph is loaded into memory for traversal. This is appropriate for the
current corpus size (~600K documents, estimated <100K distinct entities).
Migration to AWS Neptune is recommended when the in-memory graph exceeds
available Lambda memory or traversal latency degrades.

JSON format
-----------
{
    "nodes": [
        {
            "node_id": "PERSON::jeffrey epstein",
            "canonical_name": "jeffrey epstein",
            "entity_type": "PERSON",
            "surface_forms": ["Jeffrey Epstein", "Epstein"],
            "document_uuids": ["uuid-001", "uuid-002"],
            "victim_flag": false,
            "confidence": 0.99,
            "notes": null
        },
        ...
    ],
    "edges": [
        {
            "source_node_id": "PERSON::ghislaine maxwell",
            "target_node_id": "PERSON::jeffrey epstein",
            "edge_type": "ASSOCIATE",
            "document_uuids": ["uuid-001"],
            "confidence": 0.95,
            "victim_flag": false,
            "notes": null
        },
        ...
    ]
}

Victim flag suppression
-----------------------
All three traversal methods (shortest_path, all_paths, neighbours) filter
out nodes and edges where victim_flag=True before returning results.
This is a hard constraint -- the filter is applied inside the traversal
methods and cannot be disabled by the caller. Constitution Hard Limit 1.

See docs/ARCHITECTURE.md para Layer 4 -- NER & Relationship Graph.
See CONSTITUTION.md Article III Hard Limit 1.
"""

from __future__ import annotations

import io
import json
import logging
import os
from dataclasses import asdict
from typing import Optional

from graph.entity_resolver import EdgeType, Entity, EntityEdge, EntityType

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")
GRAPH_S3_KEY: str = os.environ.get("GRAPH_S3_KEY", "graph/relationship_graph.json")

# Maximum path depth for all_paths() to prevent combinatorial explosion
# on densely connected nodes.
MAX_PATH_DEPTH: int = 4


# ---------------------------------------------------------------------------
# RelationshipGraph
# ---------------------------------------------------------------------------

class RelationshipGraph:
    """
    Directed relationship graph backed by NetworkX.

    Nodes are Entity instances keyed by Entity.node_id.
    Edges are EntityEdge instances stored as NetworkX edge attributes.

    Usage
    -----
        graph = RelationshipGraph()
        graph.add_entity(entity)
        graph.add_edge(edge)
        path = graph.shortest_path("PERSON::ghislaine maxwell",
                                    "PERSON::prince andrew")
        graph.save_to_s3(s3_client, "my-bucket")
        graph = RelationshipGraph.load_from_s3(s3_client, "my-bucket")
    """

    def __init__(self) -> None:
        try:
            import networkx as nx
            self._G = nx.DiGraph()
        except ImportError as exc:
            raise ImportError(
                "networkx is required for RelationshipGraph. "
                "Add it to requirements.txt and install: pip install networkx"
            ) from exc
        self._entities: dict[str, Entity] = {}

    # -----------------------------------------------------------------------
    # Graph population
    # -----------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        """
        Add or update an entity node in the graph.

        If the node_id already exists, merges the incoming entity into the
        existing one (union of surface_forms and document_uuids, max
        confidence, conservative victim_flag). If not, adds as new node.

        Parameters
        ----------
        entity : Resolved Entity from entity_resolver.resolve_entity().
        """
        node_id = entity.node_id
        if node_id in self._entities:
            from graph.entity_resolver import merge_entity
            merge_entity(self._entities[node_id], entity)
        else:
            self._entities[node_id] = entity
            self._G.add_node(node_id, **self._entity_attrs(entity))
        # Sync victim_flag on the NetworkX node (may have changed after merge)
        self._G.nodes[node_id]["victim_flag"] = self._entities[node_id].victim_flag

    def add_edge(self, edge: EntityEdge) -> None:
        """
        Add a directed relationship edge between two entities.

        Both source and target nodes must already exist in the graph
        (call add_entity() first). If an edge with the same
        (source, target, edge_type) already exists, merges document_uuids
        and takes max confidence.

        Parameters
        ----------
        edge : EntityEdge to add.

        Raises
        ------
        ValueError if source_node_id or target_node_id is not in the graph.
        """
        if edge.source_node_id not in self._entities:
            raise ValueError(
                f"Source node '{edge.source_node_id}' not in graph. "
                "Call add_entity() before add_edge()."
            )
        if edge.target_node_id not in self._entities:
            raise ValueError(
                f"Target node '{edge.target_node_id}' not in graph. "
                "Call add_entity() before add_edge()."
            )

        key = edge.edge_type.value

        if self._G.has_edge(edge.source_node_id, edge.target_node_id):
            existing = self._G[edge.source_node_id][edge.target_node_id]
            if existing.get("edge_type") == key:
                # Merge document_uuids, take max confidence
                existing_uuids = set(existing.get("document_uuids", []))
                existing_uuids.update(edge.document_uuids)
                existing["document_uuids"] = list(existing_uuids)
                existing["confidence"] = max(
                    existing.get("confidence", 0.0), edge.confidence
                )
                if edge.victim_flag:
                    existing["victim_flag"] = True
                return

        self._G.add_edge(
            edge.source_node_id,
            edge.target_node_id,
            edge_type=key,
            document_uuids=list(edge.document_uuids),
            confidence=edge.confidence,
            victim_flag=edge.victim_flag,
            notes=edge.notes,
        )

    @staticmethod
    def _entity_attrs(entity: Entity) -> dict:
        return {
            "canonical_name": entity.canonical_name,
            "entity_type":    entity.entity_type.value,
            "surface_forms":  entity.surface_forms,
            "document_uuids": entity.document_uuids,
            "victim_flag":    entity.victim_flag,
            "confidence":     entity.confidence,
            "notes":          entity.notes,
        }

    # -----------------------------------------------------------------------
    # Traversal -- victim_flag suppression applied in all three methods
    # -----------------------------------------------------------------------

    def _safe_graph(self):
        """
        Return a view of the graph with victim-flagged nodes removed.

        Used by all traversal methods. Victim-flagged nodes are excluded
        from the subgraph so they cannot appear in any returned path or
        neighbour list. Constitution Hard Limit 1.
        """
        import networkx as nx
        safe_nodes = [
            n for n, attrs in self._G.nodes(data=True)
            if not attrs.get("victim_flag", False)
        ]
        return self._G.subgraph(safe_nodes)

    def shortest_path(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> Optional[list[str]]:
        """
        Return the shortest directed path between two entities.

        Parameters
        ----------
        source_node_id : node_id of the source entity.
        target_node_id : node_id of the target entity.

        Returns
        -------
        List of node_ids from source to target (inclusive), or None if
        no path exists or either node is victim-flagged.

        Victim-flagged nodes are excluded from the traversal graph.
        Constitution Hard Limit 1.
        """
        import networkx as nx

        G = self._safe_graph()
        if source_node_id not in G or target_node_id not in G:
            logger.debug(
                "shortest_path: one or both nodes absent or victim-flagged: "
                "%s → %s", source_node_id, target_node_id,
            )
            return None

        try:
            path = nx.shortest_path(G, source=source_node_id, target=target_node_id)
            return path
        except nx.NetworkXNoPath:
            return None
        except nx.NodeNotFound:
            return None

    def all_paths(
        self,
        source_node_id: str,
        target_node_id: str,
        max_depth: int = MAX_PATH_DEPTH,
    ) -> list[list[str]]:
        """
        Return all simple directed paths between two entities up to max_depth.

        Parameters
        ----------
        source_node_id : node_id of the source entity.
        target_node_id : node_id of the target entity.
        max_depth      : Maximum path length (number of edges).

        Returns
        -------
        List of paths, each a list of node_ids. Empty list if no paths exist,
        either node is absent, or either node is victim-flagged.

        Victim-flagged nodes are excluded. Constitution Hard Limit 1.
        """
        import networkx as nx

        G = self._safe_graph()
        if source_node_id not in G or target_node_id not in G:
            return []

        try:
            paths = list(nx.all_simple_paths(
                G,
                source=source_node_id,
                target=target_node_id,
                cutoff=max_depth,
            ))
            return paths
        except (nx.NodeNotFound, nx.NetworkXError):
            return []

    def neighbours(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
    ) -> list[dict]:
        """
        Return direct neighbours of a node, optionally filtered by edge type.

        Parameters
        ----------
        node_id   : node_id of the entity whose neighbours to retrieve.
        edge_type : If provided, only return neighbours connected by this
                    edge type. If None, return all neighbours.

        Returns
        -------
        List of dicts, each with:
            node_id    : Neighbour node_id.
            edge_type  : Edge type string.
            confidence : Edge confidence.
            document_uuids : Evidence documents for the edge.

        Victim-flagged nodes and edges are excluded. Constitution Hard Limit 1.
        """
        G = self._safe_graph()
        if node_id not in G:
            return []

        results: list[dict] = []
        for neighbour in G.successors(node_id):
            edge_attrs = G[node_id][neighbour]
            if edge_attrs.get("victim_flag", False):
                continue
            if edge_type is not None and edge_attrs.get("edge_type") != edge_type.value:
                continue
            results.append({
                "node_id":       neighbour,
                "edge_type":     edge_attrs.get("edge_type"),
                "confidence":    edge_attrs.get("confidence", 0.0),
                "document_uuids": edge_attrs.get("document_uuids", []),
            })
        return results

    # -----------------------------------------------------------------------
    # Graph properties
    # -----------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Total number of entity nodes (including victim-flagged)."""
        return self._G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of relationship edges (including victim-flagged)."""
        return self._G.number_of_edges()

    def get_entity(self, node_id: str) -> Optional[Entity]:
        """Return the Entity for a node_id, or None if absent."""
        return self._entities.get(node_id)

    # -----------------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the graph to a JSON-serialisable dict.

        Format:
            {"nodes": [...], "edges": [...]}

        Both victim-flagged and non-flagged nodes/edges are serialised.
        Suppression is applied at traversal time, not at storage time --
        this preserves the ability to audit and correct victim flags.
        """
        nodes = []
        for node_id, attrs in self._G.nodes(data=True):
            nodes.append({
                "node_id":       node_id,
                "canonical_name": attrs.get("canonical_name"),
                "entity_type":   attrs.get("entity_type"),
                "surface_forms": attrs.get("surface_forms", []),
                "document_uuids": attrs.get("document_uuids", []),
                "victim_flag":   attrs.get("victim_flag", False),
                "confidence":    attrs.get("confidence", 0.0),
                "notes":         attrs.get("notes"),
            })

        edges = []
        for src, tgt, attrs in self._G.edges(data=True):
            edges.append({
                "source_node_id": src,
                "target_node_id": tgt,
                "edge_type":      attrs.get("edge_type"),
                "document_uuids": attrs.get("document_uuids", []),
                "confidence":     attrs.get("confidence", 0.0),
                "victim_flag":    attrs.get("victim_flag", False),
                "notes":          attrs.get("notes"),
            })

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "RelationshipGraph":
        """
        Reconstruct a RelationshipGraph from a to_dict() output.

        Parameters
        ----------
        data : Dict with "nodes" and "edges" lists.

        Returns
        -------
        RelationshipGraph with all nodes and edges restored.
        """
        graph = cls()

        for node_data in data.get("nodes", []):
            entity = Entity(
                canonical_name=node_data["canonical_name"],
                entity_type=EntityType(node_data["entity_type"]),
                surface_forms=node_data.get("surface_forms", []),
                document_uuids=node_data.get("document_uuids", []),
                victim_flag=node_data.get("victim_flag", False),
                confidence=node_data.get("confidence", 0.0),
                notes=node_data.get("notes"),
            )
            graph.add_entity(entity)

        for edge_data in data.get("edges", []):
            edge = EntityEdge(
                source_node_id=edge_data["source_node_id"],
                target_node_id=edge_data["target_node_id"],
                edge_type=EdgeType(edge_data["edge_type"]),
                document_uuids=edge_data.get("document_uuids", []),
                confidence=edge_data.get("confidence", 0.0),
                victim_flag=edge_data.get("victim_flag", False),
                notes=edge_data.get("notes"),
            )
            graph.add_edge(edge)

        return graph

    # -----------------------------------------------------------------------
    # S3 persistence
    # -----------------------------------------------------------------------

    def save_to_s3(
        self,
        s3_client,
        bucket_name: str,
        s3_key: str = GRAPH_S3_KEY,
    ) -> None:
        """
        Serialise the graph to JSON and write to S3.

        Parameters
        ----------
        s3_client   : boto3 S3 client (injectable for testing).
        bucket_name : S3 bucket name.
        s3_key      : Object key. Defaults to GRAPH_S3_KEY env var.

        Raises
        ------
        RuntimeError if the S3 write fails.
        """
        body = json.dumps(self.to_dict(), indent=2).encode("utf-8")
        try:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=body,
                ContentType="application/json",
            )
            logger.info(
                "Graph saved to s3://%s/%s (%d nodes, %d edges).",
                bucket_name, s3_key, self.node_count, self.edge_count,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save graph to s3://{bucket_name}/{s3_key}: {exc}"
            ) from exc

    @classmethod
    def load_from_s3(
        cls,
        s3_client,
        bucket_name: str,
        s3_key: str = GRAPH_S3_KEY,
    ) -> "RelationshipGraph":
        """
        Load a RelationshipGraph from S3.

        Parameters
        ----------
        s3_client   : boto3 S3 client (injectable for testing).
        bucket_name : S3 bucket name.
        s3_key      : Object key. Defaults to GRAPH_S3_KEY env var.

        Returns
        -------
        RelationshipGraph reconstructed from the stored JSON.

        Raises
        ------
        RuntimeError if the S3 read or JSON parse fails.
        """
        try:
            response = s3_client.get_object(Bucket=bucket_name, Key=s3_key)
            data = json.loads(response["Body"].read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load graph from s3://{bucket_name}/{s3_key}: {exc}"
            ) from exc

        graph = cls.from_dict(data)
        logger.info(
            "Graph loaded from s3://%s/%s (%d nodes, %d edges).",
            bucket_name, s3_key, graph.node_count, graph.edge_count,
        )
        return graph

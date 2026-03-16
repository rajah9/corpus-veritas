"""
tests/graph/test_relationship_graph.py

Unit tests for graph/relationship_graph.py.

Coverage targets
----------------
RelationshipGraph.add_entity()  -- adds node, merges on duplicate node_id,
                                   victim_flag synced on NetworkX node
add_edge()                      -- adds edge, merges document_uuids on
                                   duplicate (source, target, type),
                                   raises ValueError for unknown nodes
shortest_path()                 -- connected nodes, no path, victim-flagged
                                   node excluded, absent node returns None
all_paths()                     -- multiple paths found, depth limit
                                   respected, victim-flagged excluded,
                                   no path returns empty list
neighbours()                    -- direct neighbours returned, edge_type
                                   filter works, victim-flagged neighbour
                                   excluded, victim-flagged edge excluded,
                                   absent node returns empty list
to_dict() / from_dict()         -- round-trip, nodes and edges preserved,
                                   victim_flag preserved, empty graph
save_to_s3() / load_from_s3()   -- put_object/get_object called, round-trip
                                   through mock S3, RuntimeError on failure
node_count / edge_count         -- correct values after adds
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock

from graph.entity_resolver import EdgeType, Entity, EntityEdge, EntityType
from graph.relationship_graph import RelationshipGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(
    name: str,
    etype: EntityType = EntityType.PERSON,
    victim: bool = False,
    uuids: list = None,
) -> Entity:
    return Entity(
        canonical_name=name,
        entity_type=etype,
        surface_forms=[name.title()],
        document_uuids=uuids or ["uuid-001"],
        victim_flag=victim,
        confidence=0.95,
    )


def _edge(
    src: str,
    tgt: str,
    etype: EdgeType = EdgeType.ASSOCIATE,
    victim: bool = False,
    uuids: list = None,
) -> EntityEdge:
    return EntityEdge(
        source_node_id=src,
        target_node_id=tgt,
        edge_type=etype,
        document_uuids=uuids or ["uuid-001"],
        confidence=0.90,
        victim_flag=victim,
    )


def _graph_with_two_nodes() -> RelationshipGraph:
    g = RelationshipGraph()
    g.add_entity(_entity("alice"))
    g.add_entity(_entity("bob"))
    return g


def _connected_graph() -> RelationshipGraph:
    g = RelationshipGraph()
    g.add_entity(_entity("alice"))
    g.add_entity(_entity("bob"))
    g.add_entity(_entity("carol"))
    g.add_edge(_edge("PERSON::alice", "PERSON::bob"))
    g.add_edge(_edge("PERSON::bob", "PERSON::carol"))
    return g


def _mock_s3(graph: RelationshipGraph) -> MagicMock:
    body = json.dumps(graph.to_dict()).encode()
    client = MagicMock()
    client.put_object.return_value = {}
    client.get_object.return_value = {"Body": io.BytesIO(body)}
    return client


# ===========================================================================
# add_entity
# ===========================================================================

class TestAddEntity(unittest.TestCase):

    def test_node_added(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        self.assertEqual(g.node_count, 1)

    def test_get_entity_returns_entity(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        self.assertIsNotNone(g.get_entity("PERSON::alice"))

    def test_duplicate_node_merged_not_duplicated(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        g.add_entity(_entity("alice"))
        self.assertEqual(g.node_count, 1)

    def test_victim_flag_synced_on_merge(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice", victim=False))
        g.add_entity(_entity("alice", victim=True))
        entity = g.get_entity("PERSON::alice")
        self.assertTrue(entity.victim_flag)

    def test_surface_forms_merged(self):
        g = RelationshipGraph()
        e1 = _entity("alice")
        e1.surface_forms = ["Alice"]
        e2 = _entity("alice")
        e2.surface_forms = ["Alice Smith"]
        g.add_entity(e1)
        g.add_entity(e2)
        entity = g.get_entity("PERSON::alice")
        self.assertIn("Alice Smith", entity.surface_forms)


# ===========================================================================
# add_edge
# ===========================================================================

class TestAddEdge(unittest.TestCase):

    def test_edge_added(self):
        g = _graph_with_two_nodes()
        g.add_edge(_edge("PERSON::alice", "PERSON::bob"))
        self.assertEqual(g.edge_count, 1)

    def test_unknown_source_raises(self):
        g = RelationshipGraph()
        g.add_entity(_entity("bob"))
        with self.assertRaises(ValueError) as ctx:
            g.add_edge(_edge("PERSON::unknown", "PERSON::bob"))
        self.assertIn("Source node", str(ctx.exception))

    def test_unknown_target_raises(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        with self.assertRaises(ValueError) as ctx:
            g.add_edge(_edge("PERSON::alice", "PERSON::unknown"))
        self.assertIn("Target node", str(ctx.exception))

    def test_duplicate_edge_merged(self):
        g = _graph_with_two_nodes()
        g.add_edge(_edge("PERSON::alice", "PERSON::bob", uuids=["uuid-001"]))
        g.add_edge(_edge("PERSON::alice", "PERSON::bob", uuids=["uuid-002"]))
        self.assertEqual(g.edge_count, 1)


# ===========================================================================
# shortest_path
# ===========================================================================

class TestShortestPath(unittest.TestCase):

    def test_direct_path_found(self):
        g = _connected_graph()
        path = g.shortest_path("PERSON::alice", "PERSON::bob")
        self.assertEqual(path, ["PERSON::alice", "PERSON::bob"])

    def test_indirect_path_found(self):
        g = _connected_graph()
        path = g.shortest_path("PERSON::alice", "PERSON::carol")
        self.assertIsNotNone(path)
        self.assertEqual(path[0], "PERSON::alice")
        self.assertEqual(path[-1], "PERSON::carol")

    def test_no_path_returns_none(self):
        g = _graph_with_two_nodes()  # no edges
        result = g.shortest_path("PERSON::alice", "PERSON::bob")
        self.assertIsNone(result)

    def test_absent_node_returns_none(self):
        g = _connected_graph()
        result = g.shortest_path("PERSON::alice", "PERSON::unknown")
        self.assertIsNone(result)

    def test_victim_flagged_node_excluded(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        g.add_entity(_entity("victim", victim=True))
        g.add_entity(_entity("carol"))
        g.add_edge(_edge("PERSON::alice", "PERSON::victim"))
        g.add_edge(_edge("PERSON::victim", "PERSON::carol"))
        # Only path goes through victim node -- should return None
        result = g.shortest_path("PERSON::alice", "PERSON::carol")
        self.assertIsNone(result)


# ===========================================================================
# all_paths
# ===========================================================================

class TestAllPaths(unittest.TestCase):

    def test_path_found(self):
        g = _connected_graph()
        paths = g.all_paths("PERSON::alice", "PERSON::carol")
        self.assertGreater(len(paths), 0)

    def test_no_path_returns_empty(self):
        g = _graph_with_two_nodes()
        self.assertEqual(g.all_paths("PERSON::alice", "PERSON::bob"), [])

    def test_absent_node_returns_empty(self):
        g = _connected_graph()
        self.assertEqual(g.all_paths("PERSON::alice", "PERSON::unknown"), [])

    def test_depth_limit_respected(self):
        g = RelationshipGraph()
        for name in ["a", "b", "c", "d", "e"]:
            g.add_entity(_entity(name))
        for src, tgt in [("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")]:
            g.add_edge(_edge(f"PERSON::{src}", f"PERSON::{tgt}"))
        # Path a→e requires depth 4; with max_depth=2 should return empty
        paths = g.all_paths("PERSON::a", "PERSON::e", max_depth=2)
        self.assertEqual(paths, [])

    def test_victim_flagged_node_excluded(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        g.add_entity(_entity("victim", victim=True))
        g.add_entity(_entity("carol"))
        g.add_edge(_edge("PERSON::alice", "PERSON::victim"))
        g.add_edge(_edge("PERSON::victim", "PERSON::carol"))
        paths = g.all_paths("PERSON::alice", "PERSON::carol")
        self.assertEqual(paths, [])


# ===========================================================================
# neighbours
# ===========================================================================

class TestNeighbours(unittest.TestCase):

    def test_direct_neighbour_returned(self):
        g = _connected_graph()
        neighbours = g.neighbours("PERSON::alice")
        node_ids = [n["node_id"] for n in neighbours]
        self.assertIn("PERSON::bob", node_ids)

    def test_absent_node_returns_empty(self):
        g = _connected_graph()
        self.assertEqual(g.neighbours("PERSON::unknown"), [])

    def test_edge_type_filter_works(self):
        g = _graph_with_two_nodes()
        g.add_edge(_edge("PERSON::alice", "PERSON::bob", EdgeType.ASSOCIATE))
        result = g.neighbours("PERSON::alice", edge_type=EdgeType.EMPLOYEE)
        self.assertEqual(result, [])

    def test_edge_type_filter_returns_match(self):
        g = _graph_with_two_nodes()
        g.add_edge(_edge("PERSON::alice", "PERSON::bob", EdgeType.ASSOCIATE))
        result = g.neighbours("PERSON::alice", edge_type=EdgeType.ASSOCIATE)
        self.assertEqual(len(result), 1)

    def test_victim_flagged_neighbour_excluded(self):
        g = RelationshipGraph()
        g.add_entity(_entity("alice"))
        g.add_entity(_entity("victim", victim=True))
        g.add_edge(_edge("PERSON::alice", "PERSON::victim"))
        result = g.neighbours("PERSON::alice")
        self.assertEqual(result, [])

    def test_victim_flagged_edge_excluded(self):
        g = _graph_with_two_nodes()
        g.add_edge(_edge("PERSON::alice", "PERSON::bob", victim=True))
        result = g.neighbours("PERSON::alice")
        self.assertEqual(result, [])

    def test_neighbour_dict_has_required_fields(self):
        g = _connected_graph()
        neighbours = g.neighbours("PERSON::alice")
        self.assertGreater(len(neighbours), 0)
        n = neighbours[0]
        self.assertIn("node_id", n)
        self.assertIn("edge_type", n)
        self.assertIn("confidence", n)
        self.assertIn("document_uuids", n)


# ===========================================================================
# to_dict / from_dict round-trip
# ===========================================================================

class TestSerialisation(unittest.TestCase):

    def test_empty_graph_round_trips(self):
        g = RelationshipGraph()
        g2 = RelationshipGraph.from_dict(g.to_dict())
        self.assertEqual(g2.node_count, 0)
        self.assertEqual(g2.edge_count, 0)

    def test_nodes_preserved(self):
        g = _connected_graph()
        g2 = RelationshipGraph.from_dict(g.to_dict())
        self.assertEqual(g2.node_count, g.node_count)

    def test_edges_preserved(self):
        g = _connected_graph()
        g2 = RelationshipGraph.from_dict(g.to_dict())
        self.assertEqual(g2.edge_count, g.edge_count)

    def test_victim_flag_preserved(self):
        g = RelationshipGraph()
        g.add_entity(_entity("victim", victim=True))
        g2 = RelationshipGraph.from_dict(g.to_dict())
        entity = g2.get_entity("PERSON::victim")
        self.assertTrue(entity.victim_flag)

    def test_path_survives_round_trip(self):
        g = _connected_graph()
        g2 = RelationshipGraph.from_dict(g.to_dict())
        path = g2.shortest_path("PERSON::alice", "PERSON::carol")
        self.assertIsNotNone(path)


# ===========================================================================
# S3 persistence
# ===========================================================================

class TestS3Persistence(unittest.TestCase):

    def test_save_calls_put_object(self):
        g = _connected_graph()
        s3 = _mock_s3(g)
        g.save_to_s3(s3, "my-bucket", "graph/test.json")
        s3.put_object.assert_called_once()

    def test_save_uses_correct_bucket_and_key(self):
        g = _connected_graph()
        s3 = _mock_s3(g)
        g.save_to_s3(s3, "my-bucket", "graph/test.json")
        self.assertEqual(s3.put_object.call_args.kwargs["Bucket"], "my-bucket")
        self.assertEqual(s3.put_object.call_args.kwargs["Key"], "graph/test.json")

    def test_load_calls_get_object(self):
        g = _connected_graph()
        s3 = _mock_s3(g)
        RelationshipGraph.load_from_s3(s3, "my-bucket", "graph/test.json")
        s3.get_object.assert_called_once()

    def test_round_trip_through_mock_s3(self):
        g = _connected_graph()
        body = json.dumps(g.to_dict()).encode()
        s3 = MagicMock()
        s3.put_object.return_value = {}
        s3.get_object.return_value = {"Body": io.BytesIO(body)}
        g.save_to_s3(s3, "bucket", "key")
        g2 = RelationshipGraph.load_from_s3(s3, "bucket", "key")
        self.assertEqual(g2.node_count, g.node_count)
        self.assertEqual(g2.edge_count, g.edge_count)

    def test_save_failure_raises_runtime_error(self):
        g = _connected_graph()
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(RuntimeError) as ctx:
            g.save_to_s3(s3, "bucket", "key")
        self.assertIn("Failed to save", str(ctx.exception))

    def test_load_failure_raises_runtime_error(self):
        s3 = MagicMock()
        s3.get_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(RuntimeError) as ctx:
            RelationshipGraph.load_from_s3(s3, "bucket", "key")
        self.assertIn("Failed to load", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

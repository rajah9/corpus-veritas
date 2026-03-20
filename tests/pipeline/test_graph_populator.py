"""
tests/pipeline/test_graph_populator.py

Unit tests for pipeline/graph_populator.py.

Coverage targets
----------------
populate_from_chunk()       -- entities added to graph, DynamoDB upserted,
                               victim_flag propagated, unknown type skipped,
                               DynamoDB failure continues (non-fatal),
                               returns resolved entity list
infer_edges_from_co_occurrence()
                            -- PERSON-PERSON edge added, non-PERSON skipped,
                               victim entity skipped, returns edge count
populate_from_document()    -- delegates to populate_from_chunk + infer_edges,
                               VICTIM_ADJACENT skips edge inference,
                               infer_edges=False skips edges
load_or_create_graph()      -- loads from S3 when exists, creates empty on
                               NoSuchKey, reraises on other S3 errors
save_graph()                -- delegates to RelationshipGraph.save_to_s3
"""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from graph.entity_resolver import Entity, EntityType
from graph.relationship_graph import RelationshipGraph
from pipeline.graph_populator import (
    CO_OCCURRENCE_CONFIDENCE,
    infer_edges_from_co_occurrence,
    load_or_create_graph,
    populate_from_chunk,
    populate_from_document,
    save_graph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity_dict(text="Jeffrey Epstein", etype="PERSON", conf=0.98) -> dict:
    return {"text": text, "type": etype, "confidence": conf}


def _mock_db() -> MagicMock:
    db = MagicMock()
    db.update_item.return_value = {}
    db.exceptions.ConditionalCheckFailedException = type(
        "ConditionalCheckFailedException", (Exception,), {}
    )
    return db


def _mock_s3_with_graph(graph: RelationshipGraph) -> MagicMock:
    body = json.dumps(graph.to_dict()).encode()
    client = MagicMock()
    client.get_object.return_value = {"Body": io.BytesIO(body)}
    client.put_object.return_value = {}
    return client


# ===========================================================================
# populate_from_chunk
# ===========================================================================

class TestPopulateFromChunk(unittest.TestCase):

    def setUp(self):
        self.graph = RelationshipGraph()
        self.db = _mock_db()

    def test_entity_added_to_graph(self):
        populate_from_chunk(
            [_entity_dict("Jeffrey Epstein")], "uuid-001", self.graph,
            dynamodb_client=self.db,
        )
        self.assertGreater(self.graph.node_count, 0)

    def test_returns_resolved_entities(self):
        result = populate_from_chunk(
            [_entity_dict("Jeffrey Epstein")], "uuid-001", self.graph,
            dynamodb_client=self.db,
        )
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Entity)

    def test_dynamo_upserted(self):
        populate_from_chunk(
            [_entity_dict("Jeffrey Epstein")], "uuid-001", self.graph,
            dynamodb_client=self.db,
        )
        self.db.update_item.assert_called()

    def test_victim_flag_propagated(self):
        result = populate_from_chunk(
            [_entity_dict("Virginia Giuffre")], "uuid-001", self.graph,
            dynamodb_client=self.db,
        )
        self.assertTrue(result[0].victim_flag)

    def test_unknown_entity_type_skipped(self):
        result = populate_from_chunk(
            [_entity_dict("42", etype="QUANTITY")], "uuid-001", self.graph,
            dynamodb_client=self.db,
        )
        self.assertEqual(result, [])
        self.assertEqual(self.graph.node_count, 0)

    def test_dynamo_failure_continues(self):
        self.db.update_item.side_effect = RuntimeError("DynamoDB down")
        # Must not raise -- non-fatal
        result = populate_from_chunk(
            [_entity_dict("Jeffrey Epstein")], "uuid-001", self.graph,
            dynamodb_client=self.db,
        )
        self.assertEqual(len(result), 1)

    def test_multiple_entities_processed(self):
        entities = [
            _entity_dict("Jeffrey Epstein"),
            _entity_dict("Ghislaine Maxwell"),
        ]
        result = populate_from_chunk(entities, "uuid-001", self.graph,
                                     dynamodb_client=self.db)
        self.assertEqual(len(result), 2)
        self.assertEqual(self.graph.node_count, 2)


# ===========================================================================
# infer_edges_from_co_occurrence
# ===========================================================================

class TestInferEdgesFromCoOccurrence(unittest.TestCase):

    def _two_person_graph(self) -> tuple[RelationshipGraph, list[Entity]]:
        g = RelationshipGraph()
        e1 = Entity("jeffrey epstein", EntityType.PERSON, surface_forms=["Jeffrey Epstein"])
        e2 = Entity("ghislaine maxwell", EntityType.PERSON, surface_forms=["Ghislaine Maxwell"])
        g.add_entity(e1)
        g.add_entity(e2)
        return g, [e1, e2]

    def test_person_person_edge_added(self):
        g, entities = self._two_person_graph()
        count = infer_edges_from_co_occurrence(entities, "uuid-001", g)
        self.assertGreater(count, 0)

    def test_edge_count_returned(self):
        g, entities = self._two_person_graph()
        count = infer_edges_from_co_occurrence(entities, "uuid-001", g)
        self.assertEqual(count, 1)

    def test_non_person_entity_skipped(self):
        g = RelationshipGraph()
        p = Entity("jeffrey epstein", EntityType.PERSON, surface_forms=["J.E."])
        o = Entity("fbi", EntityType.ORGANIZATION, surface_forms=["FBI"])
        g.add_entity(p)
        g.add_entity(o)
        count = infer_edges_from_co_occurrence([p, o], "uuid-001", g)
        self.assertEqual(count, 0)

    def test_victim_entity_skipped(self):
        g = RelationshipGraph()
        p = Entity("jeffrey epstein", EntityType.PERSON, surface_forms=["J.E."])
        v = Entity("virginia giuffre", EntityType.PERSON,
                   surface_forms=["V.G."], victim_flag=True)
        g.add_entity(p)
        g.add_entity(v)
        count = infer_edges_from_co_occurrence([p, v], "uuid-001", g)
        self.assertEqual(count, 0)

    def test_edge_confidence_is_low(self):
        g, entities = self._two_person_graph()
        infer_edges_from_co_occurrence(entities, "uuid-001", g)
        neighbours = g.neighbours("PERSON::jeffrey epstein")
        self.assertGreater(len(neighbours), 0)
        self.assertAlmostEqual(neighbours[0]["confidence"], CO_OCCURRENCE_CONFIDENCE)


# ===========================================================================
# populate_from_document
# ===========================================================================

class TestPopulateFromDocument(unittest.TestCase):

    def test_victim_adjacent_skips_edges(self):
        g = RelationshipGraph()
        entities = [
            _entity_dict("Jeffrey Epstein"),
            _entity_dict("Ghislaine Maxwell"),
        ]
        result = populate_from_document(
            "uuid-001", entities, g,
            classification="VICTIM_ADJACENT",
            dynamodb_client=_mock_db(),
        )
        # Entities added but no edges inferred
        self.assertEqual(g.edge_count, 0)

    def test_normal_classification_infers_edges(self):
        g = RelationshipGraph()
        entities = [
            _entity_dict("Jeffrey Epstein"),
            _entity_dict("Ghislaine Maxwell"),
        ]
        populate_from_document(
            "uuid-001", entities, g,
            classification="PROCEDURAL",
            dynamodb_client=_mock_db(),
            infer_edges=True,
        )
        self.assertGreater(g.edge_count, 0)

    def test_infer_edges_false_skips_edges(self):
        g = RelationshipGraph()
        entities = [
            _entity_dict("Jeffrey Epstein"),
            _entity_dict("Ghislaine Maxwell"),
        ]
        populate_from_document(
            "uuid-001", entities, g,
            dynamodb_client=_mock_db(),
            infer_edges=False,
        )
        self.assertEqual(g.edge_count, 0)


# ===========================================================================
# load_or_create_graph / save_graph
# ===========================================================================

class TestLoadOrCreateGraph(unittest.TestCase):

    def test_loads_from_s3_when_exists(self):
        g = RelationshipGraph()
        g.add_entity(Entity("test", EntityType.PERSON, surface_forms=["Test"]))
        s3 = _mock_s3_with_graph(g)
        loaded = load_or_create_graph(s3, "bucket", "graph/graph.json")
        self.assertEqual(loaded.node_count, 1)

    def test_creates_empty_on_no_such_key(self):
        s3 = MagicMock()
        s3.get_object.side_effect = RuntimeError("NoSuchKey: key not found")
        graph = load_or_create_graph(s3, "bucket", "graph/graph.json")
        self.assertIsInstance(graph, RelationshipGraph)
        self.assertEqual(graph.node_count, 0)

    def test_reraises_on_other_s3_errors(self):
        s3 = MagicMock()
        s3.get_object.side_effect = RuntimeError("AccessDenied")
        with self.assertRaises(RuntimeError):
            load_or_create_graph(s3, "bucket", "graph/graph.json")


class TestSaveGraph(unittest.TestCase):

    def test_save_calls_put_object(self):
        g = RelationshipGraph()
        s3 = MagicMock()
        s3.put_object.return_value = {}
        save_graph(g, s3, "bucket", "graph/graph.json")
        s3.put_object.assert_called_once()

    def test_save_failure_propagates(self):
        g = RelationshipGraph()
        s3 = MagicMock()
        s3.put_object.side_effect = RuntimeError("S3 down")
        with self.assertRaises(RuntimeError):
            save_graph(g, s3, "bucket", "graph/graph.json")


if __name__ == "__main__":
    unittest.main()

"""
tests/red_team/test_rl1_victim_reidentification.py
Red Team: Hard Limit 1 — Victim identity must never be exposed.

Adversarial tests attempting to extract suppressed victim identities
through direct queries, indirect inference, multi-step reasoning,
and API-level probing. All must fail to expose victim identities.

A system that fails any test here must not be deployed.
Constitution Hard Limit 1.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from rag.guardrail import check_victim_identity
from pipeline.audit_log import AuditLogEntry


class TestDirectVictimNameQuery(unittest.TestCase):
    """Attempt to elicit victim names directly."""

    def test_victim_name_suppressed_in_guardrail(self):
        answer = "virginia giuffre testified that epstein abused her."
        safe, triggered = check_victim_identity(answer)
        self.assertTrue(triggered)
        self.assertNotIn("virginia giuffre", safe.lower())
        self.assertIn("[protected identity]", safe)

    def test_victim_surface_form_suppressed(self):
        answer = "Virginia Roberts described the abuse in detail."
        safe, triggered = check_victim_identity(
            answer, victim_names=["virginia giuffre", "virginia roberts"]
        )
        self.assertTrue(triggered)
        self.assertNotIn("Virginia Roberts", safe)

    def test_victim_name_case_insensitive_suppression(self):
        answer = "VIRGINIA GIUFFRE provided testimony about the island."
        safe, triggered = check_victim_identity(answer)
        self.assertTrue(triggered)
        self.assertNotIn("VIRGINIA", safe)

    def test_partial_name_not_suppressed(self):
        """'Virginia' alone should not be suppressed -- too broad."""
        answer = "Virginia is a state on the east coast."
        safe, triggered = check_victim_identity(answer)
        # 'Virginia' alone is not in the victim canonical names set
        self.assertFalse(triggered)
        self.assertIn("Virginia", safe)

    def test_multiple_victim_references_all_suppressed(self):
        answer = (
            "Virginia Giuffre met with the subject. "
            "Later, virginia giuffre filed a lawsuit."
        )
        safe, triggered = check_victim_identity(answer)
        self.assertTrue(triggered)
        self.assertEqual(safe.count("[protected identity]"), 2)


class TestIndirectVictimInference(unittest.TestCase):
    """Attempt to infer victim identity through indirect paths."""

    def test_victim_flag_suppresses_entity_api_response(self):
        """
        A victim-flagged entity record must return 403 from the API,
        not 200 with the entity data.
        """
        import json
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "api"))
        from handler import handle_entity_lookup, inject_clients

        db = MagicMock()
        db.get_item.return_value = {
            "Item": {
                "canonical_name": {"S": "virginia giuffre"},
                "entity_type":    {"S": "PERSON"},
                "victim_flag":    {"S": "true"},
            }
        }
        inject_clients(dynamodb=db)

        event = {
            "pathParameters": {"canonical_name": "virginia giuffre"},
            "queryStringParameters": {"entity_type": "PERSON"},
        }
        result = handle_entity_lookup(event)
        self.assertEqual(result["statusCode"], 403)
        body = json.loads(result["body"])
        self.assertEqual(body["error"], "forbidden")
        self.assertNotIn("virginia giuffre", result["body"])

    def test_victim_chunk_not_returned_in_query(self):
        """
        Query results must not contain victim-flagged chunks.
        Verified by checking that victim_flag=true is in must_not of every DSL.
        """
        from rag.query_router import build_query, QueryRequest, QueryType
        request = QueryRequest(
            query_text="who were the victims?",
            query_type=QueryType.PROVENANCE,
        )
        dsl = build_query(request, [0.1] * 1024)
        must_not = dsl["query"]["bool"].get("must_not", [])
        terms = [f.get("term", {}) for f in must_not]
        self.assertTrue(
            any("victim_flag" in t for t in terms),
            "victim_flag must_not filter missing from DSL"
        )

    def test_audit_entry_does_not_log_victim_name_in_safe_answer(self):
        """
        The audit log records both original_answer and safe_answer.
        safe_answer must have victim names suppressed.
        """
        from rag.guardrail import apply_guardrail
        from rag.query_router import QueryRequest, QueryType, RetrievalResult

        result = RetrievalResult(
            query=QueryRequest(query_text="test", query_type=QueryType.PROVENANCE),
            chunks=[],
            answer="virginia giuffre testified.",
            convergence_applied=True,
            retrieved_at="2026-01-01T00:00:00+00:00",
        )

        with patch("rag.guardrail.write_audit_log") as mock_write:
            gr = apply_guardrail(result, audit_bucket="bucket")
            entry: AuditLogEntry = mock_write.call_args[0][0]

        self.assertNotIn("virginia giuffre", gr.safe_answer.lower())
        # original_answer preserved in audit (needed for accountability)
        self.assertIn("virginia giuffre", entry.original_answer.lower())
        # safe_answer in audit also suppressed
        self.assertNotIn("virginia giuffre", entry.safe_answer.lower())


class TestVictimFlagPropagation(unittest.TestCase):
    """Verify victim_flag propagates consistently through the system."""

    def test_known_victim_resolved_with_flag(self):
        from graph.entity_resolver import resolve_entity, EntityType
        entity = resolve_entity("Virginia Giuffre", EntityType.PERSON, 0.99)
        self.assertTrue(entity.victim_flag)

    def test_victim_entity_excluded_from_graph_traversal(self):
        from graph.entity_resolver import Entity, EntityType, EntityEdge, EdgeType
        from graph.relationship_graph import RelationshipGraph

        g = RelationshipGraph()
        g.add_entity(Entity("alice", EntityType.PERSON, surface_forms=["Alice"]))
        g.add_entity(Entity("virginia giuffre", EntityType.PERSON,
                            surface_forms=["Virginia Giuffre"], victim_flag=True))

        # No edge possible through victim node
        result = g.shortest_path("PERSON::alice", "PERSON::virginia giuffre")
        self.assertIsNone(result)

    def test_victim_not_in_neighbours(self):
        from graph.entity_resolver import Entity, EntityType, EntityEdge, EdgeType
        from graph.relationship_graph import RelationshipGraph

        g = RelationshipGraph()
        g.add_entity(Entity("alice", EntityType.PERSON, surface_forms=["Alice"]))
        g.add_entity(Entity("victim", EntityType.PERSON,
                            surface_forms=["Victim"], victim_flag=True))
        g.add_edge(EntityEdge(
            source_node_id="PERSON::alice",
            target_node_id="PERSON::victim",
            edge_type=EdgeType.ASSOCIATE,
        ))
        neighbours = g.neighbours("PERSON::alice")
        node_ids = [n["node_id"] for n in neighbours]
        self.assertNotIn("PERSON::victim", node_ids)


if __name__ == "__main__":
    unittest.main()

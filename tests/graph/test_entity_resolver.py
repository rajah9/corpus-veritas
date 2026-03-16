"""
tests/graph/test_entity_resolver.py

Unit tests for graph/entity_resolver.py.

Coverage targets
----------------
normalise_name()    -- lowercase, whitespace strip, title removal (multiple
                       titles), punctuation after title (Dr.), no title
                       unchanged, empty string
resolve_alias()     -- known alias resolved, unknown returns original,
                       case sensitive (alias map uses lowercase keys)
is_victim_flagged() -- known victim returns True, unknown returns False
resolve_entity()    -- canonical_name set, victim_flag set for known
                       victims, surface_forms populated, entity_type stored,
                       confidence stored, comprehend_client=None skips
                       stage 3, comprehend stage 3 failure falls back
                       gracefully
merge_entity()      -- surface_forms unioned, document_uuids unioned,
                       confidence takes max, victim_flag conservative
                       (True if either), notes concatenated
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from graph.entity_resolver import (
    ALIAS_MAP,
    Entity,
    EntityEdge,
    EntityType,
    EdgeType,
    is_victim_flagged,
    merge_entity,
    normalise_name,
    resolve_alias,
    resolve_entity,
)


# ===========================================================================
# normalise_name
# ===========================================================================

class TestNormaliseName(unittest.TestCase):

    def test_lowercase(self):
        self.assertEqual(normalise_name("JOHN DOE"), "john doe")

    def test_strips_whitespace(self):
        self.assertEqual(normalise_name("  John Doe  "), "john doe")

    def test_strips_mr(self):
        self.assertEqual(normalise_name("Mr John Doe"), "john doe")

    def test_strips_prince(self):
        self.assertEqual(normalise_name("Prince Andrew"), "andrew")

    def test_strips_dr_with_period(self):
        self.assertEqual(normalise_name("Dr. Ghislaine Maxwell"), "ghislaine maxwell")

    def test_strips_sir(self):
        self.assertEqual(normalise_name("Sir Richard Branson"), "richard branson")

    def test_no_title_unchanged(self):
        self.assertEqual(normalise_name("Ghislaine Maxwell"), "ghislaine maxwell")

    def test_empty_string(self):
        self.assertEqual(normalise_name(""), "")

    def test_collapses_multiple_spaces(self):
        self.assertEqual(normalise_name("John   Doe"), "john doe")


# ===========================================================================
# resolve_alias
# ===========================================================================

class TestResolveAlias(unittest.TestCase):

    def test_epstein_resolved(self):
        self.assertEqual(resolve_alias("epstein"), "jeffrey epstein")

    def test_maxwell_resolved(self):
        self.assertEqual(resolve_alias("maxwell"), "ghislaine maxwell")

    def test_andrew_resolved(self):
        self.assertEqual(resolve_alias("andrew"), "prince andrew")

    def test_unknown_returns_original(self):
        self.assertEqual(resolve_alias("some unknown person"), "some unknown person")

    def test_already_canonical_unchanged(self):
        self.assertEqual(resolve_alias("jeffrey epstein"), "jeffrey epstein")


# ===========================================================================
# is_victim_flagged
# ===========================================================================

class TestIsVictimFlagged(unittest.TestCase):

    def test_virginia_giuffre_flagged(self):
        self.assertTrue(is_victim_flagged("virginia giuffre"))

    def test_unknown_not_flagged(self):
        self.assertFalse(is_victim_flagged("jeffrey epstein"))

    def test_empty_string_not_flagged(self):
        self.assertFalse(is_victim_flagged(""))


# ===========================================================================
# resolve_entity
# ===========================================================================

class TestResolveEntity(unittest.TestCase):

    def test_canonical_name_set(self):
        entity = resolve_entity("Jeffrey Epstein", EntityType.PERSON, 0.99)
        self.assertEqual(entity.canonical_name, "jeffrey epstein")

    def test_alias_applied(self):
        entity = resolve_entity("Epstein", EntityType.PERSON, 0.99)
        self.assertEqual(entity.canonical_name, "jeffrey epstein")

    def test_title_stripped(self):
        entity = resolve_entity("Prince Andrew", EntityType.PERSON, 0.95)
        self.assertEqual(entity.canonical_name, "prince andrew")

    def test_entity_type_stored(self):
        entity = resolve_entity("FBI", EntityType.ORGANIZATION, 0.99)
        self.assertEqual(entity.entity_type, EntityType.ORGANIZATION)

    def test_confidence_stored(self):
        entity = resolve_entity("Jane Doe", EntityType.PERSON, 0.87)
        self.assertAlmostEqual(entity.confidence, 0.87)

    def test_surface_form_stored(self):
        entity = resolve_entity("Epstein", EntityType.PERSON, 0.99)
        self.assertIn("Epstein", entity.surface_forms)

    def test_victim_flag_set_for_known_victim(self):
        entity = resolve_entity("Virginia Giuffre", EntityType.PERSON, 0.95)
        self.assertTrue(entity.victim_flag)

    def test_victim_flag_false_for_non_victim(self):
        entity = resolve_entity("Jeffrey Epstein", EntityType.PERSON, 0.99)
        self.assertFalse(entity.victim_flag)

    def test_no_comprehend_client_skips_stage_3(self):
        entity = resolve_entity("John Doe", EntityType.PERSON, 0.90, comprehend_client=None)
        self.assertIsInstance(entity, Entity)

    def test_comprehend_stage_3_failure_falls_back(self):
        client = MagicMock()
        client.detect_entities.side_effect = RuntimeError("Comprehend down")
        # Must not raise -- falls back gracefully to alias map result
        entity = resolve_entity("Epstein", EntityType.PERSON, 0.90, comprehend_client=client)
        self.assertEqual(entity.canonical_name, "jeffrey epstein")

    def test_node_id_format(self):
        entity = resolve_entity("Jeffrey Epstein", EntityType.PERSON, 0.99)
        self.assertEqual(entity.node_id, "PERSON::jeffrey epstein")


# ===========================================================================
# merge_entity
# ===========================================================================

class TestMergeEntity(unittest.TestCase):

    def _entity(self, name="jeffrey epstein", etype=EntityType.PERSON,
                surfaces=None, uuids=None, confidence=0.90, victim=False):
        return Entity(
            canonical_name=name,
            entity_type=etype,
            surface_forms=surfaces or ["Jeffrey Epstein"],
            document_uuids=uuids or ["uuid-001"],
            confidence=confidence,
            victim_flag=victim,
        )

    def test_surface_forms_unioned(self):
        existing = self._entity(surfaces=["Jeffrey Epstein"])
        incoming = self._entity(surfaces=["Epstein"])
        merge_entity(existing, incoming)
        self.assertIn("Epstein", existing.surface_forms)
        self.assertIn("Jeffrey Epstein", existing.surface_forms)

    def test_document_uuids_unioned(self):
        existing = self._entity(uuids=["uuid-001"])
        incoming = self._entity(uuids=["uuid-002"])
        merge_entity(existing, incoming)
        self.assertIn("uuid-001", existing.document_uuids)
        self.assertIn("uuid-002", existing.document_uuids)

    def test_duplicate_uuids_not_added(self):
        existing = self._entity(uuids=["uuid-001"])
        incoming = self._entity(uuids=["uuid-001"])
        merge_entity(existing, incoming)
        self.assertEqual(existing.document_uuids.count("uuid-001"), 1)

    def test_confidence_takes_max(self):
        existing = self._entity(confidence=0.90)
        incoming = self._entity(confidence=0.97)
        merge_entity(existing, incoming)
        self.assertAlmostEqual(existing.confidence, 0.97)

    def test_victim_flag_conservative(self):
        existing = self._entity(victim=False)
        incoming = self._entity(victim=True)
        merge_entity(existing, incoming)
        self.assertTrue(existing.victim_flag)

    def test_victim_flag_stays_true(self):
        existing = self._entity(victim=True)
        incoming = self._entity(victim=False)
        merge_entity(existing, incoming)
        self.assertTrue(existing.victim_flag)

    def test_notes_concatenated(self):
        existing = self._entity()
        existing.notes = "note A"
        incoming = self._entity()
        incoming.notes = "note B"
        merge_entity(existing, incoming)
        self.assertIn("note A", existing.notes)
        self.assertIn("note B", existing.notes)

    def test_notes_set_when_existing_none(self):
        existing = self._entity()
        existing.notes = None
        incoming = self._entity()
        incoming.notes = "new note"
        merge_entity(existing, incoming)
        self.assertEqual(existing.notes, "new note")


if __name__ == "__main__":
    unittest.main()

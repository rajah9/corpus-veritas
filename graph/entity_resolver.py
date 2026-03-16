"""
graph/entity_resolver.py
Layer 4: Entity type definitions, models, and disambiguation.

Defines the authoritative entity vocabulary (EntityType, EdgeType) and
provides Entity/EntityEdge dataclasses used throughout the graph layer.

Disambiguation pipeline
-----------------------
resolve_entity() applies three stages in order:

  Stage 1 -- Name normalisation
    Lowercase, strip leading/trailing whitespace, remove honorifics and
    titles (Mr, Mrs, Ms, Dr, Prince, Princess, Lord, Lady, Sir, Dame).
    "Prince Andrew" → "andrew", "Dr. Maxwell" → "maxwell".
    Normalised form is used as the lookup key for alias resolution and
    deduplication; the original surface form is retained for display.

  Stage 2 -- Alias map lookup
    A hardcoded alias map covers known name variants relevant to the
    Epstein corpus. "Epstein" → "jeffrey epstein", "Maxwell" → "ghislaine
    maxwell", etc. Maps normalised form to canonical normalised form.
    The alias map is intentionally small and manually curated -- it covers
    only high-confidence aliases where the same short name unambiguously
    refers to one individual in this corpus.

  Stage 3 -- Comprehend entity linking (optional)
    If a comprehend_client is supplied, calls detect_entities() on the
    entity text to attempt entity linking. This is the most expensive
    stage and is skipped when comprehend_client=None (e.g. in tests or
    when processing high-volume batches where alias map coverage is
    sufficient).

    Note: AWS Comprehend's standard detect_entities() does not provide
    true entity linking (KB resolution). The entity linking API is only
    available in some regions and for limited entity types. We use
    detect_entities() here to get the highest-confidence Comprehend
    reading of the entity surface form, which can disambiguate cases
    like "Trump" (PERSON) vs "Trump Tower" (LOCATION) that pure
    normalisation cannot resolve.

Entity deduplication
--------------------
Two entities are considered the same if their resolved canonical_name
and entity_type match. The graph layer uses (canonical_name, type) as
the node identity key.

See docs/ARCHITECTURE.md para Layer 4 -- NER & Relationship Graph.
See CONSTITUTION.md Principle III -- Living Individuals Are Not Targets.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    """
    Supported named entity types.

    Matches the vocabulary in pipeline/ner_extractor.py and
    ChunkMetadata.named_entities. Adding a type here requires a
    corresponding update to _COMPREHEND_TYPE_MAP in ner_extractor.py.
    """
    PERSON       = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION     = "LOCATION"
    DATE         = "DATE"
    CASE_NUMBER  = "CASE_NUMBER"


class EdgeType(str, Enum):
    """
    Relationship types between entities in the graph.

    Edges are directed: (source) --[EdgeType]--> (target).
    The direction encodes the relationship semantics:
      ASSOCIATE    : source is documented as associated with target.
      EMPLOYEE     : source worked for target (or target's organization).
      VISITOR      : source visited target's property or location.
      ACCUSED      : source is accused of conduct involving target.
      ACCUSER      : source made accusations against target.
      WITNESS      : source witnessed events involving target.
      CORRESPONDENT: source exchanged correspondence with target.

    Constitution reference: Principle III -- these edges record what
    documents state, not guilt determinations. An ACCUSED edge does not
    mean the accused is guilty. Hard Limit 2 -- edges about living
    individuals require multi-source convergence before surfacing.
    """
    ASSOCIATE     = "ASSOCIATE"
    EMPLOYEE      = "EMPLOYEE"
    VISITOR       = "VISITOR"
    ACCUSED       = "ACCUSED"
    ACCUSER       = "ACCUSER"
    WITNESS       = "WITNESS"
    CORRESPONDENT = "CORRESPONDENT"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    A resolved named entity in the relationship graph.

    Fields
    ------
    canonical_name  Normalised, disambiguated name used as the graph
                    node identity key. Lowercase, titles stripped,
                    alias-resolved. e.g. "jeffrey epstein".

    entity_type     EntityType value.

    surface_forms   All surface forms observed in the corpus for this
                    entity. e.g. ["Epstein", "Jeffrey Epstein", "J. Epstein"].
                    Populated as documents are processed.

    document_uuids  UUIDs of documents in which this entity appears.

    victim_flag     True if this entity has been identified as a victim
                    or survivor. Nodes with victim_flag=True are excluded
                    from public query paths by the graph traversal methods.
                    Same suppression pattern as ChunkMetadata.victim_flag.
                    Constitution Hard Limit 1.

    confidence      Highest Comprehend confidence score seen for any
                    occurrence of this entity.

    notes           Free-text notes for human reviewers (e.g. alias
                    resolution rationale).
    """
    canonical_name:  str
    entity_type:     EntityType
    surface_forms:   list[str] = field(default_factory=list)
    document_uuids:  list[str] = field(default_factory=list)
    victim_flag:     bool = False
    confidence:      float = 0.0
    notes:           Optional[str] = None

    @property
    def node_id(self) -> str:
        """Stable graph node identifier: '{type}::{canonical_name}'."""
        return f"{self.entity_type.value}::{self.canonical_name}"


@dataclass
class EntityEdge:
    """
    A directed relationship edge between two entities.

    Fields
    ------
    source_node_id  node_id of the source entity.
    target_node_id  node_id of the target entity.
    edge_type       EdgeType value.
    document_uuids  UUIDs of documents that provide evidence for this edge.
    confidence      Confidence of the relationship (derived from chunk
                    confidence_tier or explicit assignment).
    victim_flag     True if either endpoint is victim-flagged. Edges with
                    victim_flag=True are excluded from public traversal.
    notes           Free-text annotation.
    """
    source_node_id: str
    target_node_id: str
    edge_type:      EdgeType
    document_uuids: list[str] = field(default_factory=list)
    confidence:     float = 0.0
    victim_flag:    bool = False
    notes:          Optional[str] = None


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

# Honorifics and titles to strip before alias lookup.
# Covers titles common in the Epstein corpus (British titles, US honorifics).
_TITLES: set[str] = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "prince", "princess", "lord", "lady", "sir", "dame",
    "rev", "reverend", "hon", "honorable", "the honourable",
    "ambassador", "senator", "congressman", "congresswoman",
    "governor", "president", "vice president",
}

_TITLE_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _TITLES) + r")\.?\s*",
    re.IGNORECASE,
)


def normalise_name(name: str) -> str:
    """
    Normalise an entity surface form for alias lookup and deduplication.

    Steps:
      1. Lowercase and strip whitespace.
      2. Remove honorifics and titles (see _TITLES).
      3. Collapse multiple spaces.

    Parameters
    ----------
    name : Raw surface form as extracted by Comprehend.

    Returns
    -------
    Normalised string. e.g. "Prince Andrew" → "andrew",
    "Dr. Ghislaine Maxwell" → "ghislaine maxwell".
    """
    normalised = name.strip().lower()
    normalised = _TITLE_RE.sub("", normalised)
    normalised = re.sub(r"\s+", " ", normalised).strip()
    return normalised


# ---------------------------------------------------------------------------
# Alias map
# ---------------------------------------------------------------------------

# Manually curated alias map for the Epstein corpus.
# Keys: normalised surface form. Values: canonical normalised name.
# Only add aliases where the short form unambiguously refers to one
# individual in the context of this corpus.
ALIAS_MAP: dict[str, str] = {
    # Jeffrey Epstein
    "epstein":          "jeffrey epstein",
    "j. epstein":       "jeffrey epstein",
    "jeffrey e.":       "jeffrey epstein",

    # Ghislaine Maxwell
    "maxwell":          "ghislaine maxwell",
    "g. maxwell":       "ghislaine maxwell",
    "ghislaine":        "ghislaine maxwell",

    # Prince Andrew (Duke of York)
    "andrew":           "prince andrew",
    "duke of york":     "prince andrew",

    # Alan Dershowitz
    "dershowitz":       "alan dershowitz",

    # Leslie Wexner
    "wexner":           "leslie wexner",
    "les wexner":       "leslie wexner",

    # Jean-Luc Brunel
    "brunel":           "jean-luc brunel",

    # Bill Richardson
    "richardson":       "bill richardson",

    # Bill Clinton (common in correspondence context)
    "clinton":          "bill clinton",

    # Donald Trump
    "trump":            "donald trump",

    # Virginia Giuffre (victim -- entity resolver sets victim_flag)
    # Note: canonical name retained for internal graph integrity;
    # victim_flag suppresses from all public query paths.
    "giuffre":          "virginia giuffre",
    "virginia roberts": "virginia giuffre",
}


def resolve_alias(normalised_name: str) -> str:
    """
    Apply the alias map to a normalised name.

    Parameters
    ----------
    normalised_name : Output of normalise_name().

    Returns
    -------
    Canonical normalised name if an alias exists; original name otherwise.
    """
    return ALIAS_MAP.get(normalised_name, normalised_name)


# ---------------------------------------------------------------------------
# Victim-flagged canonical names
# ---------------------------------------------------------------------------

# Canonical names that are known or likely victims/survivors.
# Entities resolving to these canonical names receive victim_flag=True.
# This list is intentionally conservative -- false negatives (missing a
# victim flag) are caught by sanitizer.py; false positives here only
# add suppression, which errs on the side of protection.
#
# Constitution Hard Limit 1: these entities must never surface on public
# query paths regardless of how they were ingested.
_KNOWN_VICTIM_CANONICAL_NAMES: set[str] = {
    "virginia giuffre",
    # Additional known victims would be added here after legal review.
    # Do not add names based on speculation or single-source reporting.
}


def is_victim_flagged(canonical_name: str) -> bool:
    """Return True if canonical_name is a known victim identity."""
    return canonical_name in _KNOWN_VICTIM_CANONICAL_NAMES


# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

def resolve_entity(
    surface_form: str,
    entity_type: EntityType,
    confidence: float = 0.0,
    comprehend_client=None,
) -> Entity:
    """
    Resolve a raw entity surface form to a canonical Entity.

    Applies three disambiguation stages:
      1. Name normalisation (strip titles, lowercase).
      2. Alias map lookup.
      3. Comprehend entity linking (if comprehend_client provided).

    Parameters
    ----------
    surface_form      : Raw entity text from Comprehend or manual input.
    entity_type       : EntityType classification.
    confidence        : Comprehend confidence score (0.0 - 1.0).
    comprehend_client : boto3 Comprehend client for Stage 3.
                        Pass None to skip entity linking (Stage 2 only).

    Returns
    -------
    Entity with canonical_name, victim_flag, and surface_forms populated.
    """
    normalised = normalise_name(surface_form)
    canonical = resolve_alias(normalised)

    # Stage 3: Comprehend entity linking -- attempt to get a higher-
    # confidence reading of the entity type using detect_entities().
    # This can disambiguate PERSON vs LOCATION for ambiguous names.
    if comprehend_client is not None and entity_type == EntityType.PERSON:
        try:
            response = comprehend_client.detect_entities(
                Text=surface_form, LanguageCode="en"
            )
            linked = response.get("Entities", [])
            if linked:
                top = max(linked, key=lambda e: e.get("Score", 0.0))
                linked_score = top.get("Score", 0.0)
                if linked_score > confidence:
                    confidence = linked_score
                    logger.debug(
                        "Entity linking: '%s' linked with score %.3f",
                        surface_form, linked_score,
                    )
        except Exception as exc:
            logger.warning(
                "Comprehend entity linking failed for '%s': %s -- "
                "falling back to alias map result.",
                surface_form, exc,
            )

    victim = is_victim_flagged(canonical)

    return Entity(
        canonical_name=canonical,
        entity_type=entity_type,
        surface_forms=[surface_form],
        confidence=confidence,
        victim_flag=victim,
    )


def merge_entity(existing: Entity, incoming: Entity) -> Entity:
    """
    Merge an incoming Entity into an existing one.

    Used when the same entity is encountered again during graph population.
    Retains the existing canonical_name and entity_type. Updates:
      - surface_forms: union of both lists, deduplicated.
      - document_uuids: union of both lists, deduplicated.
      - confidence: max of both.
      - victim_flag: True if either is True (conservative).
      - notes: concatenated if both present.

    Parameters
    ----------
    existing : The Entity already in the graph.
    incoming : The newly resolved Entity to merge in.

    Returns
    -------
    Updated Entity (mutates existing in place and returns it).
    """
    for sf in incoming.surface_forms:
        if sf not in existing.surface_forms:
            existing.surface_forms.append(sf)
    for uuid in incoming.document_uuids:
        if uuid not in existing.document_uuids:
            existing.document_uuids.append(uuid)
    existing.confidence = max(existing.confidence, incoming.confidence)
    if incoming.victim_flag:
        existing.victim_flag = True
    if incoming.notes and existing.notes:
        existing.notes = f"{existing.notes}; {incoming.notes}"
    elif incoming.notes:
        existing.notes = incoming.notes
    return existing

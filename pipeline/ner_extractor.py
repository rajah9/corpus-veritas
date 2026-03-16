"""
pipeline/ner_extractor.py
Layer 4: Named entity recognition via AWS Comprehend.

Calls Comprehend detect_entities() on document text and maps the results
to EntityType values. Returns a list of named_entity dicts suitable for
storing directly in ChunkMetadata.named_entities and indexing in OpenSearch.

Each named_entity dict carries:
    text        : The entity surface form as it appears in the document.
    type        : EntityType value (PERSON, ORGANIZATION, etc.).
    confidence  : Comprehend confidence score (0.0 - 1.0).
    begin_offset: Character offset of the entity start in the source text.
    end_offset  : Character offset of the entity end in the source text.

Offsets are retained for two reasons:
  1. Entity resolution (graph/entity_resolver.py) can use them to
     disambiguate entities that appear in different contexts.
  2. Audit trail -- if a named_entity is later flagged as victim-adjacent,
     the offset lets the pipeline locate and suppress the specific span
     rather than suppressing the entire chunk.

Comprehend language
-------------------
Always English ("en"). The DOJ Epstein corpus is entirely English-language.

Confidence threshold
--------------------
ENTITY_CONFIDENCE_THRESHOLD = 0.90. Entities below this threshold are
discarded. This is intentionally conservative -- low-confidence extractions
generate noisy entity tables and degrade graph quality. Tune downward if
coverage is insufficient after evaluating against real corpus documents.

Victim flag
-----------
ner_extractor.py does NOT assign victim_flag. That determination is made
by sanitizer.py (Layer 1) and stored on ClassificationRecord. The NER
extractor simply extracts all entities above the confidence threshold.
The victim_flag assignment happens when entity resolver or sanitizer
identifies an entity as matching a known victim identity.

See CONSTITUTION.md Article III Hard Limit 1.
See docs/ARCHITECTURE.md para Layer 4 -- NER & Relationship Graph.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from pipeline.models import DocumentState

logger = logging.getLogger(__name__)

AWS_REGION: str = os.environ.get("AWS_REGION", "us-east-1")

# Comprehend entity types mapped to EntityType strings.
# Comprehend returns its own type strings; we map to our vocabulary.
# Types not in this map are discarded (e.g. QUANTITY, TITLE, OTHER).
_COMPREHEND_TYPE_MAP: dict[str, str] = {
    "PERSON":       "PERSON",
    "ORGANIZATION": "ORGANIZATION",
    "LOCATION":     "LOCATION",
    "DATE":         "DATE",
    "OTHER":        "CASE_NUMBER",   # "OTHER" in Comprehend often catches
                                      # case numbers and docket identifiers
                                      # in legal documents
}

# Minimum Comprehend confidence score to retain an entity.
ENTITY_CONFIDENCE_THRESHOLD: float = 0.90


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_entities(
    text: str,
    comprehend_client=None,
    confidence_threshold: float = ENTITY_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Extract named entities from text using AWS Comprehend.

    Calls detect_entities() and maps results to the corpus-veritas entity
    vocabulary. Entities below confidence_threshold are discarded. Entity
    types not in the supported vocabulary are discarded.

    Parameters
    ----------
    text                 : Document or chunk text to analyse.
    comprehend_client    : boto3 Comprehend client (injectable for testing).
                           If None, created from AWS_REGION.
    confidence_threshold : Minimum confidence score to retain an entity.

    Returns
    -------
    List of entity dicts, each with keys:
        text, type, confidence, begin_offset, end_offset.
    Empty list if no entities are found or text is empty.

    Raises
    ------
    RuntimeError if the Comprehend call fails.
    """
    if not text.strip():
        return []

    if comprehend_client is None:
        import boto3
        comprehend_client = boto3.client("comprehend", region_name=AWS_REGION)

    try:
        response = comprehend_client.detect_entities(
            Text=text,
            LanguageCode="en",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Comprehend detect_entities failed: {exc}"
        ) from exc

    entities: list[dict] = []
    for ent in response.get("Entities", []):
        score = ent.get("Score", 0.0)
        comp_type = ent.get("Type", "")
        mapped_type = _COMPREHEND_TYPE_MAP.get(comp_type)

        if mapped_type is None:
            continue
        if score < confidence_threshold:
            continue

        entities.append({
            "text":         ent["Text"],
            "type":         mapped_type,
            "confidence":   round(score, 4),
            "begin_offset": ent.get("BeginOffset", 0),
            "end_offset":   ent.get("EndOffset", 0),
        })

    logger.debug(
        "extract_entities: %d entities retained (threshold=%.2f)",
        len(entities), confidence_threshold,
    )
    return entities


# ---------------------------------------------------------------------------
# Chunk-level extraction
# ---------------------------------------------------------------------------

def extract_entities_for_chunk(
    chunk_text: str,
    document_uuid: str,
    comprehend_client=None,
    confidence_threshold: float = ENTITY_CONFIDENCE_THRESHOLD,
) -> list[dict]:
    """
    Extract entities for a single chunk and annotate with document_uuid.

    Thin wrapper around extract_entities() that adds document_uuid to each
    entity dict so the origin document can be traced when entities from
    multiple chunks are merged in the entity resolver.

    Parameters
    ----------
    chunk_text        : Text of one chunk.
    document_uuid     : UUID of the source document.
    comprehend_client : Injectable boto3 Comprehend client.
    confidence_threshold : Minimum confidence score.

    Returns
    -------
    List of entity dicts with an additional document_uuid field.
    """
    entities = extract_entities(chunk_text, comprehend_client, confidence_threshold)
    for ent in entities:
        ent["document_uuid"] = document_uuid
    return entities


# ---------------------------------------------------------------------------
# Document-level deduplication
# ---------------------------------------------------------------------------

def deduplicate_entities(entities: list[dict]) -> list[dict]:
    """
    Deduplicate entities by (normalised text, type).

    When the same entity surface form appears multiple times in a document
    (e.g. "Epstein" appearing in 12 chunks), only the highest-confidence
    occurrence is retained. This keeps named_entities lean for storage.

    Normalisation: lowercase, strip leading/trailing whitespace.
    The original (un-normalised) text of the highest-confidence occurrence
    is retained in the output.

    Parameters
    ----------
    entities : List of entity dicts from extract_entities_for_chunk().

    Returns
    -------
    Deduplicated list, one entry per (normalised text, type) pair.
    """
    best: dict[tuple, dict] = {}
    for ent in entities:
        key = (ent["text"].strip().lower(), ent["type"])
        existing = best.get(key)
        if existing is None or ent["confidence"] > existing["confidence"]:
            best[key] = ent
    return list(best.values())


# ---------------------------------------------------------------------------
# DynamoDB entity table
# ---------------------------------------------------------------------------
#
# Table name : corpus_veritas_entities
# PK         : canonical_name (S)  -- resolved canonical name from entity_resolver
# SK         : entity_type    (S)  -- EntityType value e.g. "PERSON"
#
# Attributes
# ----------
# surface_forms         SS  All observed surface forms for this entity.
#                           e.g. {"Jeffrey Epstein", "Epstein", "J. Epstein"}
#                           Grows via ADD on every upsert.
#
# document_uuids        SS  Full set of document UUIDs this entity appears in.
#                           Not indexed. Grows via ADD on every upsert.
#
# first_document_uuid   S   The first document UUID this entity was seen in.
#                           Scalar -- used as PK for gsi-document-uuid.
#                           Set once via if_not_exists; never overwritten.
#
#                           ⚠ Limitation: gsi-document-uuid indexes only the
#                           first document for each entity. It answers "which
#                           entities were first seen in document X?" not "which
#                           entities appear anywhere in document X?". For full
#                           entity-document coverage a separate join table would
#                           be required. This is documented and accepted for the
#                           learning-phase corpus size.
#
# victim_flag           S   "true" when the entity is victim-flagged.
#                           SPARSE -- only written when True. Never set to
#                           "false". Once flagged, always flagged (conservative).
#                           PK for gsi-victim-flag.
#
# confidence            N   Highest Comprehend confidence score seen.
#                           Updated via MAX logic (conditional expression).
#
# occurrence_count      N   Number of times this entity has been upserted.
#                           Incremented via ADD :one on every upsert.
#
# updated_at            S   ISO 8601 UTC timestamp of last upsert.
#
# GSIs
# ----
# gsi-entity-type   PK=entity_type    SK=canonical_name
#                   Query all entities of a given type.
#                   Example: "show me all PERSON entities"
#
# gsi-victim-flag   PK=victim_flag    (sparse -- only items where victim_flag="true")
#                   Fast lookup of all victim-flagged entities.
#                   Example: pre-flight check before a query session.
#
# gsi-document-uuid PK=first_document_uuid
#                   Entities first seen in a given document.
#                   See ⚠ limitation note above.

ENTITY_TABLE_NAME: str = "corpus_veritas_entities"


def upsert_entity_record(
    canonical_name: str,
    entity_type: str,
    surface_form: str,
    document_uuid: str,
    confidence: float,
    victim_flag: bool = False,
    dynamodb_client=None,
    table_name: str = ENTITY_TABLE_NAME,
) -> None:
    """
    Write or update an entity record in DynamoDB.

    Uses UpdateItem so every call is safe to repeat -- re-processing the
    same document adds its UUID to the set rather than creating a duplicate
    record. All set-type attributes (surface_forms, document_uuids) grow
    monotonically; they are never shrunk.

    Confidence is updated only when the incoming value is higher than the
    stored value (MAX logic via conditional expression + two-phase update).
    victim_flag is set to "true" when True and is never unset (conservative).

    Parameters
    ----------
    canonical_name   : Resolved canonical name from entity_resolver.
    entity_type      : EntityType value string e.g. "PERSON".
    surface_form     : Raw surface form observed in this document.
    document_uuid    : UUID of the document this entity was extracted from.
    confidence       : Comprehend confidence score for this occurrence.
    victim_flag      : True if this entity is victim-flagged.
    dynamodb_client  : boto3 DynamoDB client (injectable for testing).
                       If None, created from AWS_REGION.
    table_name       : DynamoDB table name. Defaults to ENTITY_TABLE_NAME.

    Raises
    ------
    RuntimeError if the DynamoDB call fails.
    """
    if dynamodb_client is None:
        import boto3
        dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    # Build UpdateExpression. victim_flag is only added to the expression
    # when True -- it is never written as "false".
    update_parts = [
        "SET updated_at = :now",
        "    first_document_uuid = if_not_exists(first_document_uuid, :duuid)",
    ]
    expression_values: dict = {
        ":now":    {"S": now},
        ":duuid":  {"S": document_uuid},
        ":sform":  {"SS": [surface_form]},
        ":duuids": {"SS": [document_uuid]},
        ":conf":   {"N": str(round(confidence, 4))},
        ":one":    {"N": "1"},
    }

    if victim_flag:
        update_parts.append("    victim_flag = :vflag")
        expression_values[":vflag"] = {"S": "true"}

    update_parts.append("ADD surface_forms :sform,")
    update_parts.append("    document_uuids :duuids,")
    update_parts.append("    occurrence_count :one")

    update_expression = "\n".join(update_parts)

    try:
        # Phase 1: upsert all fields except confidence
        dynamodb_client.update_item(
            TableName=table_name,
            Key={
                "canonical_name": {"S": canonical_name},
                "entity_type":    {"S": entity_type},
            },
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
        )

        # Phase 2: update confidence only if incoming value is higher.
        # Separate call because DynamoDB does not support MAX() natively
        # in UpdateExpression -- we use a ConditionExpression instead.
        dynamodb_client.update_item(
            TableName=table_name,
            Key={
                "canonical_name": {"S": canonical_name},
                "entity_type":    {"S": entity_type},
            },
            UpdateExpression="SET confidence = :conf",
            ConditionExpression=(
                "attribute_not_exists(confidence) OR confidence < :conf"
            ),
            ExpressionAttributeValues={
                ":conf": {"N": str(round(confidence, 4))},
            },
        )
    except dynamodb_client.exceptions.ConditionalCheckFailedException:
        # Existing confidence is already >= incoming -- no update needed.
        pass
    except Exception as exc:
        raise RuntimeError(
            f"Failed to upsert entity record "
            f"({canonical_name}, {entity_type}): {exc}"
        ) from exc

    logger.debug(
        "Upserted entity (%s, %s) from document %s.",
        canonical_name, entity_type, document_uuid,
    )


def get_entity_record(
    canonical_name: str,
    entity_type: str,
    dynamodb_client=None,
    table_name: str = ENTITY_TABLE_NAME,
) -> Optional[dict]:
    """
    Retrieve an entity record by primary key.

    Parameters
    ----------
    canonical_name  : Resolved canonical name (PK).
    entity_type     : EntityType value string (SK).
    dynamodb_client : Injectable boto3 DynamoDB client.
    table_name      : DynamoDB table name.

    Returns
    -------
    Dict of attribute values (DynamoDB typed format deserialised to Python
    native types), or None if the record does not exist.

    Raises
    ------
    RuntimeError if the DynamoDB call fails.
    """
    if dynamodb_client is None:
        import boto3
        dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)

    try:
        response = dynamodb_client.get_item(
            TableName=table_name,
            Key={
                "canonical_name": {"S": canonical_name},
                "entity_type":    {"S": entity_type},
            },
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to get entity record ({canonical_name}, {entity_type}): {exc}"
        ) from exc

    item = response.get("Item")
    if item is None:
        return None
    return _deserialise_item(item)


def query_entities_by_type(
    entity_type: str,
    dynamodb_client=None,
    table_name: str = ENTITY_TABLE_NAME,
) -> list[dict]:
    """
    Query all entities of a given type via gsi-entity-type.

    Parameters
    ----------
    entity_type     : EntityType value string e.g. "PERSON".
    dynamodb_client : Injectable boto3 DynamoDB client.
    table_name      : DynamoDB table name.

    Returns
    -------
    List of deserialised entity record dicts.

    Raises
    ------
    RuntimeError if the DynamoDB call fails.
    """
    if dynamodb_client is None:
        import boto3
        dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)

    try:
        response = dynamodb_client.query(
            TableName=table_name,
            IndexName="gsi-entity-type",
            KeyConditionExpression="entity_type = :etype",
            ExpressionAttributeValues={":etype": {"S": entity_type}},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query entities by type '{entity_type}': {exc}"
        ) from exc

    return [_deserialise_item(item) for item in response.get("Items", [])]


def query_entities_by_document(
    document_uuid: str,
    dynamodb_client=None,
    table_name: str = ENTITY_TABLE_NAME,
) -> list[dict]:
    """
    Query entities first seen in a given document via gsi-document-uuid.

    See ⚠ limitation in module-level docstring: this returns entities whose
    first_document_uuid matches, not all entities that appear in the document.

    Parameters
    ----------
    document_uuid   : UUID of the document.
    dynamodb_client : Injectable boto3 DynamoDB client.
    table_name      : DynamoDB table name.

    Returns
    -------
    List of deserialised entity record dicts.

    Raises
    ------
    RuntimeError if the DynamoDB call fails.
    """
    if dynamodb_client is None:
        import boto3
        dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)

    try:
        response = dynamodb_client.query(
            TableName=table_name,
            IndexName="gsi-document-uuid",
            KeyConditionExpression="first_document_uuid = :duuid",
            ExpressionAttributeValues={":duuid": {"S": document_uuid}},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to query entities by document '{document_uuid}': {exc}"
        ) from exc

    return [_deserialise_item(item) for item in response.get("Items", [])]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _deserialise_item(item: dict) -> dict:
    """
    Convert a DynamoDB typed attribute dict to Python native types.

    Handles S (string), N (number → float), SS (string set → list),
    BOOL (bool). Unknown types are passed through as-is.
    """
    result: dict = {}
    for key, value in item.items():
        if "S" in value:
            result[key] = value["S"]
        elif "N" in value:
            result[key] = float(value["N"])
        elif "SS" in value:
            result[key] = sorted(value["SS"])  # sorted for determinism
        elif "BOOL" in value:
            result[key] = value["BOOL"]
        else:
            result[key] = value
    return result

"""
api/handler.py
Milestone 8: AWS Lambda handler for corpus-veritas API.

Serves five routes via API Gateway:

  POST /query
    Body: { query_text, query_type, top_k?, entity_names?, date_from?, date_to? }
    Returns: GuardrailResult as JSON with audit_entry_id, safe_answer,
             check flags, and chunk metadata.

  GET /gap-report
    Query params: version? (release version label), public? (true/false)
    Returns: Latest gap report markdown from S3, or 404 if none found.

  GET /entity/{canonical_name}
    Returns: DynamoDB entity record from corpus_veritas_entities.

  GET /document/{uuid}
    Returns: DynamoDB chain-of-custody record from corpus_veritas_documents.

  GET /health
    Returns: { status: "ok", timestamp } -- liveness check.

Request routing
---------------
lambda_handler() dispatches to one of five route handler functions based
on (httpMethod, path). Each route handler is responsible for:
  1. Parsing and validating its own input
  2. Calling the appropriate pipeline function
  3. Returning a well-formed API Gateway response dict

Error handling
--------------
All route handlers return structured JSON error responses rather than
raising exceptions to Lambda. HTTP 400 for invalid input, 404 for not
found, 500 for internal errors. AuditLogFailure from the guardrail
layer returns 503 (Service Unavailable) -- the response cannot be
delivered when the audit log write fails (Hard Limit 5).

Client construction
-------------------
AWS clients (boto3 Bedrock, OpenSearch, DynamoDB, S3) are constructed
once at module load time and reused across Lambda invocations (warm
start optimisation). They are injectable via module-level variables for
testing: inject_clients() replaces them with mocks.

OpenSearch client requires SigV4 signing -- see infrastructure/DEPLOYMENT.md
for the construction pattern. The OPENSEARCH_ENDPOINT environment variable
must be set from the CDK stack output before Lambda deployment.

See infrastructure/DEPLOYMENT.md for environment variable configuration.
See CONSTITUTION.md for ethical constraints enforced by this handler.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

AWS_REGION:           str = os.environ.get("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT:  str = os.environ.get("OPENSEARCH_ENDPOINT", "")
OPENSEARCH_INDEX:     str = os.environ.get("OPENSEARCH_INDEX", "documents")
CORPUS_S3_BUCKET:     str = os.environ.get("CORPUS_S3_BUCKET", "")
AUDIT_S3_BUCKET:      str = os.environ.get("AUDIT_S3_BUCKET", "")
AUDIT_LOG_GROUP:      str = os.environ.get("AUDIT_LOG_GROUP", "/corpus-veritas/audit")
DOCUMENTS_TABLE:      str = os.environ.get("DOCUMENTS_TABLE", "corpus_veritas_documents")
ENTITIES_TABLE:       str = os.environ.get("ENTITIES_TABLE", "corpus_veritas_entities")

# ---------------------------------------------------------------------------
# Module-level clients (constructed once for warm-start reuse)
# ---------------------------------------------------------------------------

_bedrock_client     = None
_opensearch_client  = None
_dynamodb_client    = None
_s3_client          = None
_cloudwatch_client  = None


def _get_bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        import boto3
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client


def _get_dynamodb():
    global _dynamodb_client
    if _dynamodb_client is None:
        import boto3
        _dynamodb_client = boto3.client("dynamodb", region_name=AWS_REGION)
    return _dynamodb_client


def _get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client("s3", region_name=AWS_REGION)
    return _s3_client


def _get_cloudwatch():
    global _cloudwatch_client
    if _cloudwatch_client is None:
        import boto3
        _cloudwatch_client = boto3.client("logs", region_name=AWS_REGION)
    return _cloudwatch_client


def _get_opensearch():
    global _opensearch_client
    if _opensearch_client is None:
        if not OPENSEARCH_ENDPOINT:
            raise RuntimeError(
                "OPENSEARCH_ENDPOINT is not set. "
                "Deploy the CDK stack and set the environment variable."
            )
        try:
            import boto3
            from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
            credentials = boto3.Session().get_credentials()
            auth = AWSV4SignerAuth(credentials, AWS_REGION, "aoss")
            host = OPENSEARCH_ENDPOINT.replace("https://", "").rstrip("/")
            _opensearch_client = OpenSearch(
                hosts=[{"host": host, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                connection_class=RequestsHttpConnection,
            )
        except ImportError as exc:
            raise RuntimeError(
                "opensearch-py is required for OpenSearch queries. "
                "Add it to the Lambda layer or deployment package."
            ) from exc
    return _opensearch_client


def inject_clients(
    bedrock=None, opensearch=None, dynamodb=None,
    s3=None, cloudwatch=None,
) -> None:
    """
    Replace module-level clients with test mocks.
    Call before invoking lambda_handler in tests.
    """
    global _bedrock_client, _opensearch_client, _dynamodb_client
    global _s3_client, _cloudwatch_client
    if bedrock     is not None: _bedrock_client    = bedrock
    if opensearch  is not None: _opensearch_client = opensearch
    if dynamodb    is not None: _dynamodb_client   = dynamodb
    if s3          is not None: _s3_client         = s3
    if cloudwatch  is not None: _cloudwatch_client = cloudwatch


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _response(status: int, body: dict) -> dict:
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "X-Content-Type-Options": "nosniff",
        },
        "body": json.dumps(body),
    }


def _ok(body: dict) -> dict:
    return _response(200, body)


def _bad_request(message: str) -> dict:
    return _response(400, {"error": "bad_request", "message": message})


def _not_found(message: str) -> dict:
    return _response(404, {"error": "not_found", "message": message})


def _service_unavailable(message: str) -> dict:
    return _response(503, {"error": "service_unavailable", "message": message})


def _internal_error(message: str) -> dict:
    return _response(500, {"error": "internal_error", "message": message})


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

def handle_query(event: dict) -> dict:
    """
    POST /query

    Runs the full pipeline: route_query → check_convergence (INFERENCE) →
    apply_guardrail. Returns GuardrailResult as JSON.

    Body fields:
        query_text    (required) Natural language query.
        query_type    (required) TIMELINE|PROVENANCE|INFERENCE|RELATIONSHIP
        top_k         (optional, default 10) Number of chunks to retrieve.
        entity_names  (optional) List of entity names for TIMELINE/RELATIONSHIP.
        date_from     (optional) ISO 8601 date lower bound for TIMELINE.
        date_to       (optional) ISO 8601 date upper bound for TIMELINE.

    Returns 400 if query_text or query_type is missing or invalid.
    Returns 503 if audit log write fails (Hard Limit 5).
    Returns 500 for other pipeline errors.
    """
    from rag.query_router import QueryRequest, QueryType, route_query
    from rag.convergence_checker import check_convergence
    from rag.guardrail import apply_guardrail
    from pipeline.audit_log import AuditLogFailure

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _bad_request("Request body must be valid JSON.")

    query_text = body.get("query_text", "").strip()
    query_type_str = body.get("query_type", "").upper()

    if not query_text:
        return _bad_request("query_text is required.")
    if not query_type_str:
        return _bad_request("query_type is required.")

    try:
        query_type = QueryType(query_type_str)
    except ValueError:
        valid = [qt.value for qt in QueryType]
        return _bad_request(f"Invalid query_type '{query_type_str}'. Must be one of: {valid}")

    try:
        request = QueryRequest(
            query_text=query_text,
            query_type=query_type,
            top_k=int(body.get("top_k", 10)),
            entity_names=body.get("entity_names"),
            date_from=body.get("date_from"),
            date_to=body.get("date_to"),
        )
    except (ValueError, TypeError) as exc:
        return _bad_request(str(exc))

    try:
        result = route_query(
            request=request,
            opensearch_client=_get_opensearch(),
            bedrock_client=_get_bedrock(),
        )

        convergence_result = None
        if query_type == QueryType.INFERENCE:
            convergence_result = check_convergence(result)

        guardrail_result = apply_guardrail(
            result=result,
            convergence_result=convergence_result,
            cloudwatch_client=_get_cloudwatch(),
            s3_client=_get_s3(),
            audit_log_group=AUDIT_LOG_GROUP,
            audit_bucket=AUDIT_S3_BUCKET,
        )

    except AuditLogFailure as exc:
        logger.error("Audit log failure -- response withheld: %s", exc)
        return _service_unavailable(
            "The audit log write failed. This response cannot be delivered "
            "per corpus-veritas Constitution Hard Limit 5. "
            "The system administrator has been notified."
        )
    except RuntimeError as exc:
        logger.error("Pipeline error: %s", exc)
        return _internal_error(f"Pipeline error: {exc}")

    return _ok({
        "answer":                    guardrail_result.safe_answer,
        "audit_entry_id":            guardrail_result.audit_entry_id,
        "query_type":                query_type.value,
        "lowest_tier":               result.lowest_tier,
        "convergence_applied":       result.convergence_applied,
        "victim_scan_triggered":     guardrail_result.victim_scan_triggered,
        "inference_downgraded":      guardrail_result.inference_downgraded,
        "confidence_violation":      guardrail_result.confidence_violation,
        "creative_content_suppressed": guardrail_result.creative_content_suppressed,
        "chunk_count":               len(result.chunks),
        "retrieved_at":              result.retrieved_at,
    })


def handle_gap_report(event: dict) -> dict:
    """
    GET /gap-report

    Returns the latest gap report markdown from S3.

    Query params:
        version  (optional) Release version label. Defaults to "latest".
        public   (optional) "true" returns public (victim-suppressed) report.
                             Default is technical report.

    The report is stored at:
        reports/{version}/gap-report-public.md  (public=true)
        reports/{version}/gap-report.md         (public=false)

    Returns 404 if no report exists for the requested version.
    """
    params = event.get("queryStringParameters") or {}
    version = params.get("version", "latest")
    is_public = params.get("public", "false").lower() == "true"

    suffix = "gap-report-public.md" if is_public else "gap-report.md"
    s3_key = f"reports/{version}/{suffix}"

    try:
        response = _get_s3().get_object(
            Bucket=CORPUS_S3_BUCKET, Key=s3_key
        )
        markdown = response["Body"].read().decode("utf-8")
        return _ok({
            "version":   version,
            "public":    is_public,
            "s3_key":    s3_key,
            "report":    markdown,
        })
    except Exception as exc:
        error_str = str(exc)
        if "NoSuchKey" in error_str or "404" in error_str:
            return _not_found(
                f"No gap report found for version '{version}'. "
                f"Run the deletion pipeline to generate one."
            )
        logger.error("S3 error retrieving gap report: %s", exc)
        return _internal_error(f"Failed to retrieve gap report: {exc}")


def handle_entity_lookup(event: dict) -> dict:
    """
    GET /entity/{canonical_name}

    Query params:
        entity_type  (optional) Filter to specific EntityType e.g. PERSON.

    Returns the entity record from corpus_veritas_entities.
    Returns 404 if the entity is not found.
    Returns 403 if the entity is victim-flagged -- victim entities
    are not surfaced via the public API regardless of request.
    """
    path_params  = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}

    canonical_name = path_params.get("canonical_name", "").strip().lower()
    entity_type    = query_params.get("entity_type", "PERSON").upper()

    if not canonical_name:
        return _bad_request("canonical_name path parameter is required.")

    try:
        response = _get_dynamodb().get_item(
            TableName=ENTITIES_TABLE,
            Key={
                "canonical_name": {"S": canonical_name},
                "entity_type":    {"S": entity_type},
            },
        )
    except Exception as exc:
        logger.error("DynamoDB error in entity lookup: %s", exc)
        return _internal_error(f"Entity lookup failed: {exc}")

    item = response.get("Item")
    if not item:
        return _not_found(
            f"Entity '{canonical_name}' ({entity_type}) not found."
        )

    # Hard Limit 1: victim-flagged entities are not surfaced via API
    if item.get("victim_flag", {}).get("S") == "true":
        return _response(403, {
            "error": "forbidden",
            "message": (
                "This entity record is protected and cannot be returned "
                "via the public API. Constitution Hard Limit 1."
            ),
        })

    return _ok(_deserialise_dynamo_item(item))


def handle_document_lookup(event: dict) -> dict:
    """
    GET /document/{uuid}

    Returns the chain-of-custody record from corpus_veritas_documents.
    Returns 404 if the document is not found.
    Returns 403 if the document is victim-flagged.
    """
    path_params = event.get("pathParameters") or {}
    document_uuid = path_params.get("uuid", "").strip()

    if not document_uuid:
        return _bad_request("uuid path parameter is required.")

    try:
        response = _get_dynamodb().get_item(
            TableName=DOCUMENTS_TABLE,
            Key={"document_uuid": {"S": document_uuid}},
        )
    except Exception as exc:
        logger.error("DynamoDB error in document lookup: %s", exc)
        return _internal_error(f"Document lookup failed: {exc}")

    item = response.get("Item")
    if not item:
        return _not_found(f"Document '{document_uuid}' not found.")

    # Hard Limit 1: victim-flagged documents suppressed
    if item.get("victim_flag", {}).get("S") == "true":
        return _response(403, {
            "error": "forbidden",
            "message": (
                "This document record is protected and cannot be returned "
                "via the public API. Constitution Hard Limit 1."
            ),
        })

    return _ok(_deserialise_dynamo_item(item))


def handle_health(event: dict) -> dict:
    """
    GET /health

    Liveness check. Returns 200 with timestamp and environment summary.
    Does not check downstream service connectivity -- that would be a
    readiness check and is too expensive for every health ping.
    """
    return _ok({
        "status":    "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "region":    AWS_REGION,
        "index":     OPENSEARCH_INDEX,
    })


# ---------------------------------------------------------------------------
# DynamoDB deserialisation
# ---------------------------------------------------------------------------

def _deserialise_dynamo_item(item: dict) -> dict:
    """Convert DynamoDB typed attribute dict to Python native types."""
    result: dict = {}
    for key, value in item.items():
        if "S"    in value: result[key] = value["S"]
        elif "N"  in value: result[key] = float(value["N"])
        elif "SS" in value: result[key] = sorted(value["SS"])
        elif "NS" in value: result[key] = [float(v) for v in value["NS"]]
        elif "BOOL" in value: result[key] = value["BOOL"]
        elif "L"  in value: result[key] = [_deserialise_dynamo_item(v) if isinstance(v, dict) else v for v in value["L"]]
        else: result[key] = value
    return result


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

# Route table: (method, path_prefix) → handler function
_ROUTES: list[tuple[str, str, callable]] = [
    ("POST", "/query",      handle_query),
    ("GET",  "/gap-report", handle_gap_report),
    ("GET",  "/entity/",    handle_entity_lookup),
    ("GET",  "/document/",  handle_document_lookup),
    ("GET",  "/health",     handle_health),
]


def lambda_handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    Dispatches API Gateway proxy events to route handlers based on
    (httpMethod, path). Unknown routes return 404.

    All responses include Content-Type: application/json and
    X-Content-Type-Options: nosniff headers.

    Constitution reference: every /query response is audited before
    delivery per Hard Limit 5. AuditLogFailure returns 503.
    """
    method = event.get("httpMethod", "").upper()
    path   = event.get("path", "")

    logger.info("Request: %s %s", method, path)

    for route_method, route_path, handler in _ROUTES:
        if method == route_method and (
            path == route_path or path.startswith(route_path)
        ):
            try:
                return handler(event)
            except Exception as exc:
                logger.exception("Unhandled error in %s: %s", handler.__name__, exc)
                return _internal_error(f"Unexpected error: {exc}")

    return _not_found(f"Route not found: {method} {path}")

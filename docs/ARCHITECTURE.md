# Architecture Specification

*Epstein Files AI Analysis System — corpus-veritas*
*Version 0.3 — Milestone 7 — March 2026*

> This document is the technical companion to [`CONSTITUTION.md`](../CONSTITUTION.md).
> Ethical boundaries defined there govern all decisions here.
> When they conflict, the Constitution governs.

**Implementation status:** All eight milestones complete. 1032 tests passing. System is deployment-ready pending red team sign-off.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture: Five Layers](#2-architecture-five-layers)
3. [Deletion Detection Module](#3-deletion-detection-module)
4. [Infrastructure](#4-infrastructure)
5. [Repository Structure](#5-repository-structure)
6. [Red Team Audit Framework](#6-red-team-audit-framework)
7. [Production Roadmap](#7-production-roadmap)

---

## 1. Project Overview

corpus-veritas answers four classes of question about the publicly released Epstein documents, grading every response by confidence tier and provenance:

| Query Type | Example |
|---|---|
| **Timeline** | What did Person X know, and when? |
| **Provenance** | Is this claim confirmed, corroborated, or rumor? |
| **Inference** | Who might the unnamed individuals in this document be? |
| **Relationship** | Show all documents connecting Person A to Person B. |

A fifth capability — **deletion detection** — identifies pages and documents present in the DOJ index but absent from the public release.

---

## 2. Architecture: Five Layers

### Layer 1 — Ingestion and Sanitization Pipeline

Sub-Module 1B: `corpus_evaluator.py` runs three sequential checks:
1. Git history audit (forced pushes, suspicious modification gaps)
2. Sequence number reconciliation (`EFTANumber` or `BatesNumber`)
3. Community vetting against `trusted_endorsers.json`

Sub-Module 1C: `sanitizer.py` — Comprehend PII detection, four-rule entity analysis, SQS human review queue. Entity text offsets only in SQS messages.

Sub-Module 1D: `classifier.py` — UUID assignment, `ClassificationRecord` written to DynamoDB `corpus_veritas_documents`. Priority: `VICTIM_ADJACENT > PROCEDURAL > PERPETRATOR_ADJACENT > UNKNOWN`.

#### Sequence Numbering

```
SequenceNumber (ABC)
├── BatesNumber   — all gaps suspicious
└── EFTANumber    — DS9 692,473 documented gaps are expected_gap_numbers, not deletion_candidates
```

#### DynamoDB: corpus_veritas_documents

PK: `document_uuid`. GSIs: `gsi-classification-date`, `gsi-corpus-source`, `gsi-victim-flag` (sparse).

---

### Layer 2 — Storage and Metadata

| System | Role |
|---|---|
| S3 `corpus-veritas-corpus` | Raw documents. Object Lock COMPLIANCE 7yr on victim-flagged. |
| S3 `corpus-veritas-audit` | Audit logs. Object Lock COMPLIANCE 7yr on all entries. |
| DynamoDB `corpus_veritas_documents` | Chain of custody |
| DynamoDB `corpus_veritas_entities` | Named entity registry |
| DynamoDB `corpus_veritas_deletions` | Gap and withholding findings |
| OpenSearch `corpus-veritas` / `documents` | kNN vector search, 1024-dim Titan v2 |

#### ChunkMetadata (19 fields)

`vector` (kNN, 1024-dim), `victim_flag` (keyword, guardrail fast-path), `confidence_tier`, `sequence_number`, `sequence_scheme`, `document_type`, `named_entities`, `deletion_flag`, `provenance_tag`.

`EmbeddingConfig.opensearch_dimension_mapping` is the single source of truth for vector dimension. CDK stack, `ingestor.py`, and `query_router.py` all derive dimension from this property.

---

### Layer 3 — RAG Engine

Direct OpenSearch DSL. Four query types:

| QueryType | Retrieval Strategy |
|---|---|
| TIMELINE | kNN + date range + entity filter + chronological sort |
| PROVENANCE | kNN only — maximises coverage for source counting |
| INFERENCE | kNN — caller must run `convergence_checker.check_convergence()` |
| RELATIONSHIP | Graph traversal first, kNN fallback |

`victim_flag must_not` filter on every query. Unconditional. Constitution Hard Limit 1.

**Convergence rule:** count=1 → SINGLE_SOURCE; count≥2 → CORROBORATED; count≥3 + type diversity → CONFIRMED. Independence test: different `document_uuid` AND sequence numbers differ by >100.

**Bedrock synthesis:** `claude-sonnet-4-6`. Prompt enforces Hard Limits 1–4 inline. Lowest tier across retrieved chunks constrains language instruction.

---

### Layer 4 — NER and Relationship Graph

`ner_extractor.py`: Comprehend, threshold 0.90. DynamoDB entity table: PK=canonical_name, SK=entity_type. GSIs: `gsi-entity-type`, `gsi-victim-flag` (sparse), `gsi-document-uuid`.

`entity_resolver.py`: three-stage disambiguation (normalisation → alias map → Comprehend linking). Known victim canonical names receive `victim_flag=True`.

`relationship_graph.py`: NetworkX DiGraph. Seven edge types. Three traversal methods all apply `_safe_graph()` victim suppression. JSON to S3. Neptune migration path documented.

---

### Layer 5 — Ethical Guardrail

Four checks in order, then audit write:

1. **Victim scan** — regex against known victim names + caller-supplied names → `[protected identity]`
2. **Inference threshold** — INFERENCE queries: convergence backstop, suppression message if below threshold
3. **Confidence calibration** — 10 language patterns, hedged replacements for sub-CONFIRMED responses
4. **Creative content (HL4)** — 13 lexical patterns for hypothetical/speculative framing → full suppression

**Audit log:** CloudWatch + S3 Object Lock COMPLIANCE 7yr. Write is delivery prerequisite. `AuditLogFailure` prevents `GuardrailResult` from being returned. Constitution Hard Limit 5.

---

## 3. Deletion Detection Module

Three signals grade each finding: EFTA gap, discovery log entry, document stamp gap.

| Flag | Signals | Confidence |
|---|---|---|
| `REFERENCE_UNRESOLVED` | Internal reference, no entry | SPECULATIVE |
| `DELETION_POSSIBLE` | 1 | SINGLE_SOURCE |
| `DELETION_SUSPECTED` | 2 | CORROBORATED |
| `DELETION_CONFIRMED` | 3 | CONFIRMED |
| `WITHHELD_SELECTIVELY` | Gov't confirmed; siblings released | CONFIRMED |
| `WITHHELD_ACKNOWLEDGED` | Gov't confirmed | CONFIRMED |

`manifest_loader.py`: CSV ingest, tolerant column detection, EFTA normalisation.

`version_comparator.py`: cross-release comparison. Disappeared documents receive `DELETION_CONFIRMED` — all three signals present by construction.

`deletion_pipeline.py`: end-to-end orchestrator. `FBI302SeriesDescriptor` for partial delivery detection. Selective release → `WITHHELD_SELECTIVELY` with `sibling_document_ids`. Fully withheld → `WITHHELD_ACKNOWLEDGED`.

`gap_reporter.py`: markdown reports, public (victim suppression) and technical modes.

---

## 4. Infrastructure

CDK stack (`infrastructure/cdk/stack.py`) provisions all resources:

- S3 (two buckets, Object Lock, `RemovalPolicy.RETAIN`)
- DynamoDB (three tables, all GSIs, PAY_PER_REQUEST, PITR)
- OpenSearch Serverless (encryption + network + access policies)
- CloudWatch Logs (10-year retention)
- IAM (pipeline, query, admin roles with least-privilege permissions)

`cdk deploy` from `infrastructure/cdk/`. See `infrastructure/DEPLOYMENT.md`.

**Cost target: $50–200/month.** OpenSearch Serverless is the largest cost — use smallest collection tier.

**Object Lock buckets cannot be deleted by `cdk destroy`.** This is intentional.

---

## 5. Repository Structure

See [README.md](../README.md) for the full directory listing.

---

## 6. Red Team Audit Framework

`tests/red_team/` — **complete, 64 tests passing.** Six files covering all six Hard Limits:

| File | Hard Limit | Tests |
|---|---|---|
| `test_rl1_victim_reidentification.py` | HL1: victim identity never exposed | 12 |
| `test_rl2_inference_bypass.py` | HL2: no single-source inference about living individuals | 9 |
| `test_rl3_confidence_manipulation.py` | HL3: no CONFIRMED language for sub-CONFIRMED tier | 12 |
| `test_rl4_audit_circumvention.py` | HL5: audit log is delivery prerequisite | 9 |
| `test_rl5_deletion_suppression.py` | Principle IV: deletion flags surface and cannot be downgraded | 10 |
| `test_rl6_creative_content.py` | HL4: no creative/speculative content about real individuals | 12 |

**A system that fails a Hard Limit test must not be deployed.**

Run before every deployment: `python -m pytest tests/red_team/ -v`

---

## 7. Production Roadmap

**All eight milestones are complete.** The system is deployment-ready pending red team sign-off.

**Milestone 8 delivered:**
- `api/handler.py` — Lambda handler, five routes, full guardrail pipeline per request
- `pipeline/graph_populator.py` — NER → entity_resolver → RelationshipGraph wired
- `ui/app.py` — Streamlit prototype (Chat + Structured View)
- `tests/red_team/` — six adversarial test files, all six Hard Limits covered, 1032 tests green

**Pre-deployment gate:** run `python -m pytest tests/red_team/ -v` before every release.
A system that fails a Hard Limit test must not be deployed.

**Neptune migration:** when the in-memory graph exceeds Lambda memory, migrate to AWS Neptune.
The `RelationshipGraph.to_dict()` / `from_dict()` JSON format is the migration data contract.

**API Gateway:** the CDK stack provisions the Lambda role but not the API Gateway resource.
Add `AWS::ApiGateway::RestApi` to `infrastructure/cdk/stack.py` for production deployment.

**Streamlit deployment:** `ui/app.py` is a prototype. For production, deploy via AWS Amplify
or ECS and set `API_ENDPOINT` to the deployed API Gateway URL.

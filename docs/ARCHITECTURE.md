# Architecture Specification

*Epstein Files AI Analysis System — corpus-veritas*
*Version 0.2 — March 2026*

> This document is the technical companion to [`CONSTITUTION.md`](../../../Downloads/corpus-veritas-deletion-detector/corpus-veritas/CONSTITUTION.md). Ethical boundaries defined there govern all decisions here. When they conflict, the Constitution governs.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture: Five Layers](#2-architecture-five-layers)
3. [Deletion Detection Module](#3-deletion-detection-module)
4. [Learning Path](#4-learning-path)
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

A fifth capability — **deletion detection** — identifies pages and documents present in the DOJ's own index but absent from the public release, following methodology established by NPR's February 2026 investigation.

---

## 2. Architecture: Five Layers

### Layer 1 — Ingestion & Sanitization Pipeline

Runs before any data touches the vector store. Evaluates, verifies, and sanitizes every document. **This is the most ethically critical layer.**

#### Sub-Module 1A: Corpus Discovery (Manual Phase)

Rather than processing all DOJ PDFs through OCR from scratch, the pipeline first evaluates pre-processed corpora from GitHub and document journalism platforms.

**Workflow:** Research → evaluate → record in `corpus_registry.json` → proceed to 1B.

**Search targets:**
- GitHub: `epstein files text`, `epstein documents parsed`, `epstein 302`, `epstein DOJ corpus`
- [MuckRock](https://www.muckrock.com)
- [DocumentCloud](https://www.documentcloud.org)
- Internet Archive

**Each `corpus_registry.json` entry records:**
- GitHub URL and commit hash at time of evaluation
- Document types covered
- Presence/absence of Bates stamps
- Page count vs. DOJ index
- Star/fork count and known journalistic endorsements
- `GIT_INTEGRITY` score: `CLEAN`, `REVIEW_RECOMMENDED`, or `REJECTED`

#### Sub-Module 1B: Corpus Verification Pipeline

`pipeline/corpus_evaluator.py` runs three sequential checks. Each check gates the next.

**Check 1 — Git History Audit**

Using the GitHub API, pull full commit history. Flag repos where:
- Initial and most recent commits differ by >5% of total bytes with no descriptive commit message
- Forced pushes (`--force`) exist in history
- Modification gap is suspicious relative to the DOJ release date

Output: `GIT_INTEGRITY` score.

**Check 2 — Sequence Number Reconciliation**

Extract sequence numbers from the corpus using the appropriate `SequenceNumber` scheme and reconcile against the DOJ index manifest. The DOJ Epstein release uses **EFTA numbers** (per-page, sequential across all datasets) — not traditional Bates stamps. Pass an `EFTANumber` instance for Epstein corpora; `BatesNumber` remains supported for other legal corpora.

Produces four counts/lists:
- `PRESENT` — in both index and corpus
- `MISSING_FROM_CORPUS` — in index, not in corpus
- `UNINDEXED` — in corpus, not in index (flag for review)
- `DELETION_CANDIDATES` — missing numbers *excluding* documented expected gaps (see EFTA note below)

Corpora covering <60% of the index are tagged `PARTIAL_COVERAGE`.

**Check 3 — Community Vetting**

Cross-reference against `trusted_endorsers.json`. Endorsement upgrades `PROVENANCE_UNVERIFIED` to `PROVENANCE_COMMUNITY_VOUCHED`.

#### Sequence Numbering Schemes (`pipeline/sequence_numbers.py`)

The DOJ Epstein release does not use traditional Bates stamps. It uses **EFTA numbers** — assigned per-page (not per-document), sequential across all 12 datasets with no resets at dataset boundaries. This required a scheme-agnostic abstraction.

```
SequenceNumber (ABC)
├── BatesNumber   — traditional legal stamping; all gaps are suspicious
└── EFTANumber    — EFTA per-page numbering; DS9 gaps are documented/expected
```

The shared reconciliation algorithm (`SequenceNumber.reconcile()`) lives on the base class. Subclasses override only five methods that capture scheme-specific behaviour:

| Method | BatesNumber | EFTANumber |
|---|---|---|
| `scheme_name` | `"BATES"` | `"EFTA"` |
| `validate()` | Matches `PREFIX-DIGITS` or 6+ digit strings | Positive integer strings only |
| `extract_from_text()` | Regex for Bates stamp patterns | Regex for `EFTA-NNNN` variants |
| `sort_key()` | `(prefix_str, int)` tuple | `int` |
| `gap_is_expected()` | Always `False` | `True` for DS9 documented gaps |

**The DS9 gap distinction is critical.** The rhowardstone analysis identified 692,473 EFTA numbers in DS9's range that have no corresponding document in the release. Their status is unknown (withheld, unimaged, or unused tracking slots) but their absence is documented. They must appear in `ReconciliationResult.expected_gap_numbers`, not `deletion_candidates` — otherwise the deletion detector generates thousands of false alarms.

`ReconciliationResult` replaces the former `BatesReconciliationResult` and adds:
- `expected_gap_numbers` — documented absences, not escalated to deletion detection
- `deletion_candidates` — true gaps that feed `deletion_detector.py` as `DELETION_SUSPECTED`

The invariant `expected_gap_numbers ∪ deletion_candidates = missing_numbers` is enforced by contract tests in `TestSequenceSchemeContract`.

#### Sub-Module 1C: PII Sanitization

Every document passes through AWS Comprehend PII detection before chunking. Potential victim identities enter a human-review queue. Documents do not proceed until classified.

> ⚠️ **This step is not optional and must be built before any document is embedded — even in prototype mode.** Embedding victim identities into the vector store is very difficult to reverse.

#### Sub-Module 1D: Document Classification & Chain of Custody

Every document receives a UUID, classification tag, and chain-of-custody record in DynamoDB.

| Classification | Meaning |
|---|---|
| `VICTIM_ADJACENT` | Contains or references victim identities |
| `PERPETRATOR_ADJACENT` | Names individuals in connection with alleged conduct |
| `PROCEDURAL` | Court filings, administrative records |
| `UNKNOWN` | Requires human classification |

---

### Layer 2 — Storage & Metadata

Three systems in concert:

| System | Purpose |
|---|---|
| **Amazon S3** | Raw document storage. Versioned. Object Lock on victim-flagged content. |
| **DynamoDB** | Document registry, entity table, corpus registry, audit log, deletion manifest. Stores `WithholdingRecord` and `DeletionRecord` objects from `pipeline/deletion_detector.py`. |
| **OpenSearch Serverless** | Vector store. Every chunk carries the full metadata schema below. |

#### Chunk Metadata Schema

Every chunk embedded into OpenSearch carries:

| Field | Values |
|---|---|
| `source_document_uuid` | UUID from DynamoDB registry |
| `sequence_number` | Extracted EFTA number (primary) or Bates stamp; `NULL` if absent |
| `sequence_scheme` | `"EFTA"` \| `"BATES"` \| `NULL` |
| `document_date` | ISO 8601 |
| `document_type` | `FBI_302` \| `CORRESPONDENCE` \| `COURT_FILING` \| `EXHIBIT` \| `OTHER` |
| `named_entities` | JSON array of NER extractions with type and confidence |
| `confidence_tier` | See table below |
| `provenance_tag` | See table below |
| `victim_flag` | `BOOLEAN` — suppresses chunk from public query paths if `TRUE` |
| `deletion_flag` | `DELETION_CONFIRMED` \| `DELETION_SUSPECTED` \| `DELETION_POSSIBLE` \| `NULL` |
| `corpus_source` | `corpus_registry` UUID if from external corpus; `NULL` if `DOJ_DIRECT` |

#### Confidence Tiers

| Tier | Definition |
|---|---|
| `CONFIRMED` | Stated explicitly in a primary source with known provenance |
| `CORROBORATED` | Multiple independent sources converge on the same claim |
| `INFERRED` | Reasonably implied but not directly stated |
| `SINGLE_SOURCE` | One document only; cannot be independently verified |
| `SPECULATIVE` | Possible but evidence is thin or circumstantial |

#### Provenance Tag Hierarchy

| Tag | Level | Meaning |
|---|---|---|
| `PROVENANCE_DOJ_DIRECT` | Highest | OCR'd by this pipeline from original DOJ PDF |
| `PROVENANCE_HASH_VERIFIED` | High | Third-party corpus, SHA-256 matches DOJ original |
| `PROVENANCE_COMMUNITY_VOUCHED` | Medium-High | Endorsed by known journalists, Bates-reconciled |
| `PROVENANCE_UNVERIFIED` | Medium | Passed git + Bates checks, no endorsement |
| `PROVENANCE_FLAGGED` | Low | Failed one or more checks, human review required |
| `PROVENANCE_REJECTED` | None | Failed critical checks, will not ingest |

---

### Layer 3 — RAG Engine

AWS Bedrock (Claude) with Bedrock Knowledge Bases for retrieval management. Per-query billing — no running server cost during development.

#### Multi-Source Convergence Rule

The system will not surface an inference about a living individual unless multiple **independent** documents converge on the same conclusion. Independent means different `source_document_uuid` values with different Bates ranges.

- Count = 1 → `SINGLE_SOURCE` language required
- Count ≥ 2 → `CORROBORATED`
- Count ≥ 3 with `document_type` diversity → `CONFIRMED` possible

#### Query Routing

| Query Type | Retrieval Strategy |
|---|---|
| Timeline | Named entity + date range, sorted chronologically |
| Provenance | All chunks matching claim, count source diversity |
| Inference | Multi-source convergence rule strictly enforced |
| Relationship | Graph query first, then supporting document chunks |

---

### Layer 4 — NER & Relationship Graph

AWS Comprehend for baseline NER. Custom entity resolution for disambiguation — Person vs. Organization vs. Location sharing a name (e.g., "Trump" the person vs. "Trump Tower" the building).

Entity types tracked: `PERSON`, `ORGANIZATION`, `LOCATION`, `DATE`, `CASE_NUMBER`

Edge types in relationship graph: `ASSOCIATE` | `EMPLOYEE` | `VISITOR` | `ACCUSED` | `ACCUSER` | `WITNESS` | `CORRESPONDENT`

**Learning phase:** NetworkX (Python) serialized to S3.
**Production:** Migrate to AWS Neptune when graph outgrows in-memory processing.

---

### Layer 5 — Ethical Guardrail Layer

Every response passes four checks before delivery:

1. **Victim identity check** — Does the response reference a victim-flagged entity? Suppress and substitute with `[protected identity]` language.
2. **Inference threshold check** — Was multi-source convergence satisfied for any inference about a living individual? If not, downgrade or suppress.
3. **Confidence calibration check** — Does response language match the assigned confidence tier? `CONFIRMED` language may only appear for `CONFIRMED` tier claims.
4. **Audit log write** — Query, response, retrieved chunk UUIDs, provenance tags, and confidence tiers are written to the immutable audit log **before** response delivery. If the write fails, the response is not delivered.

**Audit log storage:** CloudWatch Logs exported to S3 with Object Lock (Compliance mode, 7-year retention).

---

## 3. Deletion Detection Module

Implements methodology from NPR's February 2026 investigation. Three independent signals grade each deletion finding:

| Signal | Description |
|---|---|
| FBI Serial Number gap | Assigned in Sentinel case management. Gaps = unreleased documents. |
| Discovery log entry | Prosecution's own catalogue. Entry with no matching release = documented withholding. |
| Bates / document stamp gap | Sequential numbering. Gap is citable structural fact. |

#### Deletion Flag Taxonomy

Flags live in `pipeline/models.py` as `DeletionFlag(str, Enum)`. They support ordering operators (`<`, `>`, `<=`, `>=`) — government-acknowledged flags rank above evidence-graded flags.

| Flag | Signals / Basis | Confidence Tier | Human Review? |
|---|---|---|---|
| `REFERENCE_UNRESOLVED` | Internal reference, no index entry | `SPECULATIVE` | Yes |
| `DELETION_POSSIBLE` | 1 signal | `SINGLE_SOURCE` | Yes |
| `DELETION_SUSPECTED` | 2 signals | `CORROBORATED` | No |
| `DELETION_CONFIRMED` | 3 signals | `CONFIRMED` | No |
| `WITHHELD_SELECTIVELY` | Gov't confirmed; sibling docs released | `CONFIRMED` | No |
| `WITHHELD_ACKNOWLEDGED` | Gov't confirmed; no siblings required | `CONFIRMED` | No |

`WITHHELD_SELECTIVELY` and `WITHHELD_ACKNOWLEDGED` were added following the WSJ's March 2026 reporting that the DOJ confirmed 47,635 files were held offline pending review, and that three FBI 302s from a Trump-related interview series were withheld while a fourth was released.

The government's characterisation of withheld documents (e.g. claims described as "baseless") is stored as `WithholdingRecord.stated_reason` — the DOJ's stated position, not a system finding.

#### FBI 302 Partial Delivery

FBI 302s have rigid structure: interview date, subject, agent, file number. The pipeline detects partially delivered 302s (header present, pages missing) as distinct from completely absent 302s. Both are flagged, with different deletion types.

#### Version Comparison

Each new DOJ release is compared against prior releases for content that appeared before and is now absent. Retroactive deletions receive `DELETION_CONFIRMED` if prior release provides documentary evidence.

---

## 4. Learning Path

Eight milestones. Each builds on the last. Each teaches specific AWS services.

> ⚠️ **Do not skip Milestone 1's victim sanitization step.** Build IAM and PII detection before embedding anything.

| # | Milestone | AWS Services | Deliverable |
|---|---|---|---|
| 1 | Ingest one document, query with Bedrock | S3, IAM, Bedrock, Lambda | Working prompt → response chain |
| 2 | Metadata schema + audit log | DynamoDB, CloudWatch | Every chunk has UUID, provenance tag, date |
| 3 | Real RAG pipeline | OpenSearch Serverless, Bedrock Knowledge Bases | Retrieval-grounded vs. naive prompt comparison |
| 4 | NER + entity table | Comprehend, DynamoDB | Prince Andrew query returns structured results |
| 5 | Confidence tier + multi-source logic | Lambda orchestration, LangChain | Inference requires convergence to surface |
| 6 | Deletion detection module | S3, DynamoDB, Lambda | Bates gap report, `DELETION_SUSPECTED` flags |
| 7 | Ethical guardrail layer | Bedrock Guardrails, CloudWatch + S3 Object Lock | Every output auditable; victim flag suppresses chunks |
| 8 | Chat + structured UI | Amplify or ECS, API Gateway | Streamlit prototype → production frontend |

### Cost Management ($50–200/month target)

| Service | Risk | Mitigation |
|---|---|---|
| OpenSearch Serverless | Largest recurring cost | Use smallest collection tier; exploit 730-hour free tier |
| Textract | Surprise bills on large batches | Only invoke for docs not in verified corpus; batch deliberately |
| Bedrock | Per-token billing | Set `max_tokens` limits; use smaller model for iteration |
| DynamoDB | Low risk | Use on-demand pricing only |

---

## 5. Repository Structure

```
corpus-veritas/
├── CONSTITUTION.md               # Ethical framework — governs all decisions
├── ARCHITECTURE.md               # This document
├── corpus_registry.json          # Authoritative external corpus source list
├── trusted_endorsers.json        # Community vetting endorser list
├── requirements.txt
├── requirements-dev.txt
├── infrastructure/
│   ├── cdk/                      # AWS CDK stack definitions
│   └── iam/                      # IAM role and policy definitions
├── pipeline/
│   ├── corpus_evaluator.py       # Sub-module 1B: corpus verification
│   ├── sanitizer.py              # Sub-module 1C: PII detection
│   ├── ingestor.py               # Chunk, embed, store
│   ├── deletion_detector.py      # Deletion detection module
│   └── entity_resolver.py        # NER + entity disambiguation
├── rag/
│   ├── query_router.py           # Route queries to retrieval strategy
│   ├── convergence_checker.py    # Multi-source convergence rule
│   └── guardrail.py              # Layer 5 ethical output filter
├── graph/
│   └── relationship_graph.py     # NetworkX graph, serialized to S3
├── api/
│   └── handler.py                # Lambda handler for API Gateway
├── ui/
│   └── app.py                    # Streamlit prototype
└── tests/
    ├── integration/              # Integration tests per milestone
    ├── red_team/                 # Adversarial ethical boundary tests
    └── fixtures/                 # Synthetic test documents (no real corpus content)
```

---

## 6. Red Team Audit Framework

Adversarial tests run after each milestone and before any production deployment. Stored in `tests/red_team/`. All Hard Limits from `CONSTITUTION.md` Article III must be tested explicitly.

| Test Category | What It Attempts |
|---|---|
| Victim re-identification | Extract suppressed victim identities via indirect queries or multi-step inference |
| Confidence manipulation | Get `INFERRED` claims surfaced with `CONFIRMED` language |
| Living individual inference bypass | Name living individuals from single-source evidence via rephrasing |
| Audit log circumvention | Query in ways that might bypass the audit log write |
| Deletion evidence suppression | Verify deletion flags surface and are not overridden |

A system that fails a Hard Limit test during red teaming **must not be deployed** until remediated and re-tested.

---

## 7. Production Roadmap

Target users: small newsrooms, independent journalists, researchers, and members of the public seeking to separate fact from rumor.

**Two interaction modes:**

- **Chat mode** — Plain-English exploration. Every response includes provenance tags and confidence tiers inline. Sources cited with document UUIDs and Bates numbers.
- **Structured view** — Publication-ready output. Timeline visualizations, relationship graphs, deletion manifests.

**Third mode (commercial):** REST API via API Gateway with per-key rate limiting and full audit logging. Newsrooms integrate into their own tools.

Legal review recommended before commercial deployment.

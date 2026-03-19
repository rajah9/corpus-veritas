# corpus-veritas

**AI-powered analysis pipeline for the DOJ Epstein document release.**

corpus-veritas ingests, sanitises, classifies, embeds, and queries the publicly
released Epstein documents. It grades every response by confidence tier and
provenance, enforces constitutional ethical constraints at every layer, and
maintains an immutable audit trail of every query and response.

This system exists to support accountability journalism and public understanding.
It does not exist to generate content about individuals for shock value or
entertainment. Read [CONSTITUTION.md](CONSTITUTION.md) before contributing.

---

## Status

| Milestone | Description | Status |
|---|---|---|
| Layer 1 | Ingestion & Sanitization Pipeline | ✅ Complete |
| Layer 2 | Storage & Metadata (S3, DynamoDB, OpenSearch) | ✅ Complete |
| Layer 3 | RAG Engine (query routing, convergence checking) | ✅ Complete |
| Layer 4 | NER & Relationship Graph | ✅ Complete |
| Layer 5 | Ethical Guardrail & Audit Log | ✅ Complete |
| Milestone 6 | Deletion Detection Pipeline | ✅ Complete |
| Milestone 7 | Infrastructure as Code (CDK) | ✅ Complete |
| Milestone 8 | API & UI | 🔲 Pending |

**855 tests passing across all layers.**

---

## Architecture overview

```
                         ┌─────────────────────────────────┐
                         │         CONSTITUTION.md          │
                         │  Six Hard Limits — governs all   │
                         └────────────────┬────────────────┘
                                          │
          ┌───────────────────────────────▼──────────────────────────────┐
          │                     LAYER 1 — Ingestion                       │
          │  corpus_evaluator → sanitizer → classifier → sequence_numbers │
          │  Comprehend PII • SQS human review • DynamoDB chain-of-custody│
          └───────────────────────────────┬──────────────────────────────┘
                                          │
          ┌───────────────────────────────▼──────────────────────────────┐
          │                  LAYER 2 — Storage & Metadata                 │
          │   S3 (Object Lock) • DynamoDB (3 tables) • OpenSearch kNN     │
          │   chunk_schema • ingestor • s3_store • EmbeddingConfig        │
          └───────────────────────────────┬──────────────────────────────┘
                                          │
          ┌───────────────────────────────▼──────────────────────────────┐
          │                    LAYER 3 — RAG Engine                       │
          │   query_router (4 types) • convergence_checker                │
          │   Bedrock synthesis • victim suppression in every DSL query   │
          └───────────────────────────────┬──────────────────────────────┘
                                          │
          ┌───────────────────────────────▼──────────────────────────────┐
          │               LAYER 4 — NER & Relationship Graph              │
          │   ner_extractor • entity_resolver • relationship_graph        │
          │   NetworkX → S3 • DynamoDB entity table • graph-backed RELATIONSHIP │
          └───────────────────────────────┬──────────────────────────────┘
                                          │
          ┌───────────────────────────────▼──────────────────────────────┐
          │               LAYER 5 — Ethical Guardrail                     │
          │   4 checks: victim scan • inference threshold •               │
          │   confidence calibration • HL4 creative content               │
          │   Audit log → CloudWatch + S3 Object Lock (7-year retention)  │
          └───────────────────────────────┬──────────────────────────────┘
                                          │
          ┌───────────────────────────────▼──────────────────────────────┐
          │             MILESTONE 6 — Deletion Detection                  │
          │   manifest_loader • version_comparator • gap_reporter         │
          │   deletion_pipeline • FBI 302 partial delivery detection      │
          │   corpus_veritas_deletions DynamoDB table                     │
          └──────────────────────────────────────────────────────────────┘
```

---

## Ethical constraints

This system enforces six Hard Limits defined in [CONSTITUTION.md](CONSTITUTION.md):

| Hard Limit | Enforcement |
|---|---|
| HL1: Never expose victim identities | OpenSearch filter + synthesis prompt + guardrail scan |
| HL2: No single-source inference about living individuals | convergence_checker + guardrail backstop |
| HL3: No CONFIRMED language for sub-CONFIRMED tier | synthesis prompt + confidence calibration check |
| HL4: No creative/speculative content about real individuals | synthesis prompt + lexical guardrail check |
| HL5: Never operate without active audit log | write_audit_log() blocks delivery on failure |
| HL6: Never ingest PROVENANCE_REJECTED corpus | corpus_evaluator checks before ingestion |

Every constraint is enforced structurally (code that cannot be bypassed by
query parameters or operator flags) rather than as policy.

---

## Repository structure

```
corpus-veritas/
├── CONSTITUTION.md               Ethical framework — governs all decisions
├── CHANGELOG.md
├── CONTRIBUTING.md
├── README.md                     This file
├── config.py                     EmbeddingConfig, ChunkingConfig
├── corpus_registry.json          External corpus source registry
├── trusted_endorsers.json        Community vetting endorser list
├── requirements.txt              Runtime dependencies
├── requirements-dev.txt          Development/test dependencies
├── requirements-cdk.txt          CDK infrastructure dependencies
│
├── docs/
│   └── ARCHITECTURE.md           Technical specification (v0.2)
│
├── infrastructure/
│   ├── DEPLOYMENT.md             AWS deployment runbook
│   ├── cdk/
│   │   ├── app.py                CDK application entry point
│   │   ├── stack.py              CorpusVeritasStack (all AWS resources)
│   │   └── cdk.json              CDK configuration
│   ├── opensearch.py             OpenSearch index lifecycle management
│   ├── s3.py                     S3 bucket provisioning
│   └── iam/
│       └── README.md             IAM setup guide
│
├── pipeline/
│   ├── models.py                 DeletionFlag, ConfidenceTier, DocumentState, WithholdingRecord
│   ├── sequence_numbers.py       BatesNumber, EFTANumber, ReconciliationResult
│   ├── corpus_evaluator.py       Sub-module 1B: corpus verification
│   ├── deletion_detector.py      DetectionSignals, DeletionRecord, FBI 302 checks
│   ├── sanitizer.py              Sub-module 1C: PII detection (Comprehend + SQS)
│   ├── classifier.py             Sub-module 1D: classification + DynamoDB
│   ├── chunk_schema.py           ChunkMetadata Pydantic model
│   ├── ingestor.py               Chunking, embedding, OpenSearch write
│   ├── s3_store.py               Raw document S3 read/write
│   ├── ner_extractor.py          Comprehend NER + DynamoDB entity table
│   ├── audit_log.py              AuditLogEntry, write_audit_log (CloudWatch + S3)
│   ├── manifest_loader.py        DOJ manifest CSV ingest
│   ├── version_comparator.py     Cross-release retroactive deletion detection
│   ├── gap_reporter.py           Markdown gap reports (public + technical)
│   └── deletion_pipeline.py      End-to-end deletion detection orchestrator
│
├── graph/
│   ├── entity_resolver.py        EntityType, EdgeType, disambiguation
│   └── relationship_graph.py     NetworkX graph, S3 persistence
│
├── rag/
│   ├── query_router.py           QueryType routing, DSL builders, Bedrock synthesis
│   ├── convergence_checker.py    Multi-source convergence rule
│   └── guardrail.py              4-check ethical guardrail, audit log integration
│
├── api/
│   └── handler.py                Lambda handler stub (Milestone 8)
│
├── ui/
│   └── app.py                    Streamlit prototype stub (Milestone 8)
│
└── tests/
    ├── pipeline/                 Unit tests for all pipeline modules
    ├── graph/                    Unit tests for entity resolver and graph
    ├── rag/                      Unit tests for query router, convergence, guardrail
    ├── infrastructure/           Unit tests for opensearch and s3 provisioning
    ├── integration/              Integration tests (sequence numbers)
    └── red_team/                 Adversarial Hard Limit tests (Milestone 8)
```

---

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Run tests

```bash
python -m pytest tests/                          # all tests
python -m pytest tests/pipeline/                 # pipeline only
python -m pytest tests/rag/                      # RAG layer
```

### 3. Deploy to AWS

See [infrastructure/DEPLOYMENT.md](infrastructure/DEPLOYMENT.md) for the
full AWS deployment runbook including CDK setup, Object Lock requirements,
and environment variable configuration.

### 4. Create the OpenSearch index

After deploying the CDK stack, create the index:

```bash
python -c "
from infrastructure.opensearch import create_index
# construct client with SigV4 signing -- see infrastructure/DEPLOYMENT.md
create_index(client)
"
```

---

## Corpus registry

External corpora are evaluated and recorded in `corpus_registry.json` before
any ingestion. Provenance tags determine how content from each corpus is
weighted and labelled in query responses.

| Corpus | Provenance | Status |
|---|---|---|
| rhowardstone/Epstein-research-data | PROVENANCE_COMMUNITY_VOUCHED | Approved |
| DOJ direct release | PROVENANCE_DOJ_DIRECT | Primary source |
| yung-megafone/Epstein-Files | PROVENANCE_FLAGGED | Overlay redaction issues |
| s0fskr1p/epsteinfiles | PROVENANCE_FLAGGED | Overlay redaction reveals underlying text |
| epstein-docs/epstein-docs.github.io | PROVENANCE_UNVERIFIED | Pending evaluation |

---

## Key design decisions

**Confidence tiers are mandatory.** Every response carries a tier from
SPECULATIVE through CONFIRMED. Users cannot receive a response without
knowing its evidential basis.

**Victim suppression is structural.** Victim-flagged entities are filtered
at three independent layers: OpenSearch DSL (retrieval), synthesis prompt
(generation), and guardrail regex scan (output). No query parameter or
operator flag can disable this.

**The audit log is a delivery prerequisite.** `apply_guardrail()` writes the
audit log before returning a response. If the write fails, the response is
not returned. Audit integrity is enforced by the call stack, not by policy.

**Chunking and embedding are decoupled.** `ChunkingConfig` and `EmbeddingConfig`
are separate frozen dataclasses in `config.py`. Chunking parameters can be
retuned without changing the embedding model or rebuilding the OpenSearch index.

**EmbeddingConfig is the single source of truth.** The CDK stack derives the
OpenSearch vector dimension from `DEFAULT_EMBEDDING_CONFIG.opensearch_dimension_mapping`.
Changing the model in `config.py` automatically propagates to the index definition.

---

## License

Apache 2.0. See LICENSE.

---

## Constitution

This project is governed by a written ethical constitution.
[Read it](CONSTITUTION.md) before contributing, querying, or deploying.

> *Accountability without protection of victims is exploitation.
> Protection of victims without accountability is complicity.
> This system is the attempt to hold both at once.*

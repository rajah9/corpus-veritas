# corpus-veritas

> *Provenance-aware document analysis for investigative journalism.*

An AI-powered research tool for navigating large, sensitive document corpora. The reference implementation uses the publicly released DOJ Epstein files. Built on AWS Bedrock, OpenSearch Serverless, and Python.

**This is simultaneously a learning project in Generative AI and a production-grade tool built to responsible standards.** Ethical constraints are architectural, not cosmetic.

---

## What It Does

- **Answers questions graded by confidence** — every response carries a tier: `CONFIRMED`, `CORROBORATED`, `INFERRED`, `SINGLE_SOURCE`, or `SPECULATIVE`
- **Protects victims by design** — victim-flagged entities are suppressed at the storage layer, before any query can reach them
- **Detects document deletions** — identifies pages and documents present in the government's own index but absent from the public release
- **Traces provenance** — every claim cites its source document, Bates number, and provenance tag
- **Maintains an immutable audit trail** — every query and response is logged to S3 with Object Lock

## What It Does Not Do

- Name living individuals based on single-source evidence
- Present inferences as confirmed facts
- Surface victim identities
- Generate creative or speculative content about real people
- Operate without an active audit log

See [`CONSTITUTION.md`](CONSTITUTION.md) for the full ethical framework that governs this project.

---

## Architecture

```
Layer 1 — Ingestion & Sanitization Pipeline
Layer 2 — Storage & Metadata (S3, DynamoDB, OpenSearch)
Layer 3 — RAG Engine (AWS Bedrock + LangChain)
Layer 4 — NER & Relationship Graph (Comprehend + NetworkX)
Layer 5 — Ethical Guardrail Layer + Audit Trail
```

Full specification: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

---

## Getting Started

### Prerequisites

- Python 3.11+
- AWS account with appropriate IAM permissions (see [`infrastructure/iam/`](./infrastructure/iam/))
- AWS CLI configured locally

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/corpus-veritas.git
cd corpus-veritas
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### First Steps

Before running anything, read [`CONSTITUTION.md`](CONSTITUTION.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

Then follow the [Learning Path](docs/ARCHITECTURE.md#4-learning-path) — Milestone 1 starts with IAM setup and a single document ingestion. **Do not skip the victim sanitization step.**

---

## Branch Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Stable, reviewed code only |
| `layer-1` | Ingestion & sanitization pipeline |
| `layer-2` | Storage and metadata schema |
| `layer-3` | RAG engine |
| `layer-4` | NER and relationship graph |
| `layer-5` | Guardrail layer and audit trail |

---

## Project Status

🟡 **Pre-alpha** — Architecture and constitution complete. Layer 1 in active development.

---

## License

Apache 2.0 — see [`LICENSE`](./LICENSE).

### Why Apache 2.0?
Chosen for this "Responsible AI" learning project because it:
* **Protects Contributors:** Includes an explicit grant of patent rights from contributors to users.
* **Ensures Transparency:** Requires clear labeling of modified files.
* **Limits Liability:** Provides a "no warranty" clause, essential when analyzing sensitive legal documents like the Epstein files.

> **Note on Responsible AI:** While this license allows for broad use, this tool was built to empower investigative journalism and factual analysis. Users are encouraged to adhere to the [Model AI Governance Framework](https://www.pdpc.gov.sg/help-and-resources/2020/01/model-ai-governance-framework) principles of transparency and explainability.
---

## Contributing

Read [`CONSTITUTION.md`](CONSTITUTION.md) before opening a pull request. All contributions must comply with the ethical constraints defined there. Pull requests that weaken victim protections or Hard Limits will not be merged.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidelines.

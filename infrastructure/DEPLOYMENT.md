# corpus-veritas Deployment Runbook

*Version 0.1 — Milestone 7*

This document covers initial AWS environment setup and CDK stack deployment.
Read it in full before running any `cdk` commands.

---

## Prerequisites

### 1. AWS account setup

- Create an AWS account if you don't have one
- Enable MFA on the root account immediately
- Create an IAM user for local development — **never use root credentials**
- Store credentials in `~/.aws/credentials`, not in `.env` files
- Confirm your account ID: `aws sts get-caller-identity`

### 2. Tool installation

```bash
# Node.js 18+ (required by CDK CLI)
# https://nodejs.org/

# CDK CLI
npm install -g aws-cdk

# Python CDK dependencies (separate from pipeline deps)
pip install -r requirements-cdk.txt

# Verify
cdk --version   # should be 2.x
```

### 3. Bootstrap CDK (once per account/region)

CDK requires a bootstrap stack in each account/region before first deploy:

```bash
export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

cdk bootstrap aws://$AWS_ACCOUNT/$AWS_REGION
```

---

## First deployment

```bash
cd infrastructure/cdk

export AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# Review what will be created (no changes made)
cdk diff

# Deploy
cdk deploy

# Save the outputs -- you'll need these as environment variables
# CorpusBucketName, AuditBucketName, OpenSearchEndpoint, etc.
```

---

## Environment variables

After deployment, set these environment variables for the pipeline:

```bash
# From CDK outputs
export CORPUS_S3_BUCKET=corpus-veritas-corpus-<account-id>
export AUDIT_S3_BUCKET=corpus-veritas-audit-<account-id>
export OPENSEARCH_ENDPOINT=<collection-endpoint-from-output>
export AUDIT_LOG_GROUP=/corpus-veritas/audit
export AWS_REGION=us-east-1

# Table names (fixed, not from outputs)
export DOCUMENTS_TABLE=corpus_veritas_documents
export ENTITIES_TABLE=corpus_veritas_entities
export DELETIONS_TABLE=corpus_veritas_deletions
export OPENSEARCH_INDEX=documents
```

---

## Post-deployment: create the OpenSearch index

The CDK stack creates the OpenSearch Serverless *collection* but not the
*index*. The index must be created separately using `infrastructure/opensearch.py`:

```python
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from infrastructure.opensearch import create_index

credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, "us-east-1", "aoss")

client = OpenSearch(
    hosts=[{"host": "<endpoint-without-https>", "port": 443}],
    http_auth=auth,
    use_ssl=True,
    connection_class=RequestsHttpConnection,
)

create_index(client)  # idempotent -- safe to re-run
```

---

## Critical: Object Lock buckets

The corpus and audit S3 buckets are created with **Object Lock enabled**.
This has irreversible consequences:

- `cdk destroy` will **not** delete these buckets
- Victim-flagged documents and audit records cannot be deleted until
  the 7-year retention period expires
- If you need to recreate buckets (e.g. wrong region), you must create
  them with new names — existing Object Lock buckets cannot be repurposed

To destroy the non-Object-Lock resources only:

```bash
cdk destroy --exclude CorpusBucket --exclude AuditBucket
```

---

## Updating the stack

```bash
cdk diff   # review changes
cdk deploy # apply
```

CDK is safe to re-run — it only changes what has drifted from the template.
DynamoDB tables, S3 buckets, and IAM roles are all set to `RemovalPolicy.RETAIN`
so they are never deleted by CDK, even on `cdk destroy`.

---

## Cost estimate

| Service | Estimated monthly cost |
|---|---|
| OpenSearch Serverless | $50–100 (2 OCUs minimum, 730-hour free tier first month) |
| S3 (corpus + audit) | $5–20 depending on corpus size |
| DynamoDB (on-demand) | $1–5 for typical query volumes |
| Bedrock (per-query) | Variable — set `max_tokens=1024` in query_router.py |
| Comprehend (NER) | $1–5 per 1M characters |
| CloudWatch Logs | $1–3 |

**Target: $50–200/month** (see ARCHITECTURE.md § Cost Management).

OpenSearch Serverless is the largest recurring cost. Use the smallest
collection tier and stay within the 730-hour free tier for initial
development.

---

## IAM roles

Three roles are created:

| Role | Usage |
|---|---|
| `corpus-veritas-pipeline` | Ingestion, sanitization, embedding, deletion detection |
| `corpus-veritas-query` | RAG queries, guardrail, audit log writes |
| `corpus-veritas-admin` | CDK deploy, infrastructure management only |

Assign the pipeline role to Lambda functions or EC2 instances running
the ingestion pipeline. Assign the query role to the API handler Lambda.
Never use the admin role for runtime workloads.

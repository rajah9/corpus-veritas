# IAM Setup

**Complete this before writing any other code. IAM is the hardest AWS beginner hurdle and the most important security foundation.**

## Principles

- Never use root AWS credentials for this project
- Enable MFA on your AWS account before doing anything else
- Use least-privilege permissions — each Lambda, each pipeline script, each service gets only the permissions it needs
- Rotate credentials regularly

## Required Roles

| Role | Purpose | Key Permissions |
|---|---|---|
| `corpus-veritas-pipeline` | Ingestion pipeline (local + Lambda) | S3 read/write, Comprehend, Bedrock, DynamoDB write |
| `corpus-veritas-query` | RAG query handler | OpenSearch read, DynamoDB read, Bedrock, CloudWatch write |
| `corpus-veritas-admin` | Infrastructure management only | CDK deploy permissions |

## Setup Steps

1. Create an IAM user for local development (do not use root)
2. Attach the `corpus-veritas-pipeline` policy to that user
3. Store credentials in `~/.aws/credentials` — never in `.env` files committed to git
4. For Lambda functions, use IAM roles (not user credentials)

## CDK Deployment

IAM policies are defined in `infrastructure/cdk/` as code. Do not create policies manually in the console — define them in CDK so they are version-controlled and reproducible.

## Reminder

The `.gitignore` excludes credential files. Do not override this.

"""
infrastructure/cdk/stack.py
CDK stack for corpus-veritas.

Defines every AWS resource required to run the pipeline:

  S3
    corpus-veritas-corpus   Raw document storage. Object Lock COMPLIANCE,
                            7-year retention on victim-flagged objects.
                            Versioning enabled. Public access blocked.
                            See pipeline/s3_store.py, infrastructure/s3.py.

    corpus-veritas-audit    Immutable audit log archive. Object Lock COMPLIANCE,
                            7-year retention on all audit entries.
                            See pipeline/audit_log.py.

  DynamoDB
    corpus_veritas_documents
                            Chain-of-custody for every ingested document.
                            PK: document_uuid
                            GSIs: gsi-classification-date,
                                  gsi-corpus-source,
                                  gsi-victim-flag (sparse)
                            See pipeline/classifier.py.

    corpus_veritas_entities
                            Named entity registry.
                            PK: canonical_name  SK: entity_type
                            GSIs: gsi-entity-type,
                                  gsi-victim-flag (sparse),
                                  gsi-document-uuid
                            See pipeline/ner_extractor.py.

    corpus_veritas_deletions
                            Gap and withholding findings.
                            PK: record_id
                            GSIs: gsi-flag-date,
                                  gsi-efta-number
                            See pipeline/deletion_pipeline.py.

  OpenSearch Serverless
    Collection: corpus-veritas
    Index: documents (kNN vector search, 1024-dim Titan v2)
    Access policy: pipeline role read/write, query role read-only.
    See infrastructure/opensearch.py, config.py (EmbeddingConfig).

  CloudWatch Logs
    /corpus-veritas/audit   Audit log stream. Retention: 7 years.
                            See pipeline/audit_log.py.

  IAM
    corpus-veritas-pipeline Pipeline execution role.
                            Permissions: S3 rw, Comprehend, Bedrock,
                            DynamoDB write, OpenSearch write, CloudWatch Logs.

    corpus-veritas-query    Query handler role.
                            Permissions: S3 r, DynamoDB read,
                            Bedrock, OpenSearch read, CloudWatch Logs write.

Resource naming
---------------
All resource names are prefixed with the stack name so that multiple
environments (dev, staging, prod) can coexist in the same account.
The prefix is derived from the CDK stack ID.

Critical: Object Lock buckets
------------------------------
S3 buckets with Object Lock enabled CANNOT be deleted by CDK destroy.
They must be manually emptied and deleted after the Object Lock retention
period expires. This is intentional -- the audit trail and victim-flagged
documents are immutable by design (Constitution Hard Limit 5).

See docs/ARCHITECTURE.md for full specification.
See CONSTITUTION.md Article III for ethical constraints.
"""

from __future__ import annotations

import sys
import os

# Add parent dirs to path so we can import config.py for EmbeddingConfig
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_cloudwatch as cloudwatch,
    aws_dynamodb as dynamodb,
    aws_iam as iam,
    aws_logs as logs,
    aws_opensearchserverless as opensearch,
    aws_s3 as s3,
)
from constructs import Construct

from config import DEFAULT_EMBEDDING_CONFIG

# Retention period matching s3_store.py and audit_log.py
_OBJECT_LOCK_YEARS = 7


class CorpusVeritasStack(Stack):
    """Main infrastructure stack for corpus-veritas."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------------------------------------------------------------
        # S3 -- corpus document storage
        # ---------------------------------------------------------------

        self.corpus_bucket = s3.Bucket(
            self,
            "CorpusBucket",
            bucket_name=f"corpus-veritas-corpus-{self.account}",
            versioned=True,
            object_lock_enabled=True,
            # Default retention: COMPLIANCE, 7 years
            # Individual objects (victim-flagged) get explicit COMPLIANCE lock
            # via pipeline/s3_store.store_document(victim_flag=True)
            object_lock_default_retention=s3.ObjectLockRetention.compliance(
                Duration.days(_OBJECT_LOCK_YEARS * 365)
            ),
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            # Object Lock buckets cannot be auto-deleted by CDK destroy
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="flagged-corpus-to-glacier",
                    enabled=True,
                    prefix="PROVENANCE_FLAGGED/",
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ],
                ),
                s3.LifecycleRule(
                    id="abort-incomplete-multipart",
                    enabled=True,
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                ),
            ],
        )

        # ---------------------------------------------------------------
        # S3 -- audit log archive
        # ---------------------------------------------------------------

        self.audit_bucket = s3.Bucket(
            self,
            "AuditBucket",
            bucket_name=f"corpus-veritas-audit-{self.account}",
            versioned=True,
            object_lock_enabled=True,
            object_lock_default_retention=s3.ObjectLockRetention.compliance(
                Duration.days(_OBJECT_LOCK_YEARS * 365)
            ),
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ---------------------------------------------------------------
        # CloudWatch Logs -- audit log stream
        # ---------------------------------------------------------------

        self.audit_log_group = logs.LogGroup(
            self,
            "AuditLogGroup",
            log_group_name="/corpus-veritas/audit",
            retention=logs.RetentionDays.TEN_YEARS,
            removal_policy=RemovalPolicy.RETAIN,
        )

        # ---------------------------------------------------------------
        # DynamoDB -- corpus_veritas_documents
        # ---------------------------------------------------------------

        self.documents_table = dynamodb.Table(
            self,
            "DocumentsTable",
            table_name="corpus_veritas_documents",
            partition_key=dynamodb.Attribute(
                name="document_uuid",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        # GSI: gsi-classification-date
        self.documents_table.add_global_secondary_index(
            index_name="gsi-classification-date",
            partition_key=dynamodb.Attribute(
                name="classification", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="ingestion_date", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # GSI: gsi-corpus-source
        self.documents_table.add_global_secondary_index(
            index_name="gsi-corpus-source",
            partition_key=dynamodb.Attribute(
                name="corpus_source", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="document_uuid", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # GSI: gsi-victim-flag (sparse -- only items where victim_flag exists)
        self.documents_table.add_global_secondary_index(
            index_name="gsi-victim-flag",
            partition_key=dynamodb.Attribute(
                name="victim_flag", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.KEYS_ONLY,
        )

        # ---------------------------------------------------------------
        # DynamoDB -- corpus_veritas_entities
        # ---------------------------------------------------------------

        self.entities_table = dynamodb.Table(
            self,
            "EntitiesTable",
            table_name="corpus_veritas_entities",
            partition_key=dynamodb.Attribute(
                name="canonical_name",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="entity_type",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        # GSI: gsi-entity-type
        self.entities_table.add_global_secondary_index(
            index_name="gsi-entity-type",
            partition_key=dynamodb.Attribute(
                name="entity_type", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="canonical_name", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # GSI: gsi-victim-flag (sparse)
        self.entities_table.add_global_secondary_index(
            index_name="gsi-victim-flag",
            partition_key=dynamodb.Attribute(
                name="victim_flag", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # GSI: gsi-document-uuid
        self.entities_table.add_global_secondary_index(
            index_name="gsi-document-uuid",
            partition_key=dynamodb.Attribute(
                name="first_document_uuid", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # ---------------------------------------------------------------
        # DynamoDB -- corpus_veritas_deletions
        # ---------------------------------------------------------------

        self.deletions_table = dynamodb.Table(
            self,
            "DeletionsTable",
            table_name="corpus_veritas_deletions",
            partition_key=dynamodb.Attribute(
                name="record_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            removal_policy=RemovalPolicy.RETAIN,
        )

        # GSI: gsi-flag-date
        self.deletions_table.add_global_secondary_index(
            index_name="gsi-flag-date",
            partition_key=dynamodb.Attribute(
                name="deletion_flag", type=dynamodb.AttributeType.STRING
            ),
            sort_key=dynamodb.Attribute(
                name="acknowledgment_date", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # GSI: gsi-efta-number
        self.deletions_table.add_global_secondary_index(
            index_name="gsi-efta-number",
            partition_key=dynamodb.Attribute(
                name="efta_number", type=dynamodb.AttributeType.STRING
            ),
            projection_type=dynamodb.ProjectionType.ALL,
        )

        # ---------------------------------------------------------------
        # OpenSearch Serverless
        # ---------------------------------------------------------------

        # Encryption policy -- required before collection can be created
        encryption_policy = opensearch.CfnSecurityPolicy(
            self,
            "OpenSearchEncryptionPolicy",
            name="corpus-veritas-encryption",
            type="encryption",
            policy=cdk.Fn.to_json_string({
                "Rules": [{"Resource": ["collection/corpus-veritas"], "ResourceType": "collection"}],
                "AWSOwnedKey": True,
            }),
        )

        # Network policy -- VPC or public access
        # Default: public access for initial deployment.
        # For production, replace with VPC-based policy.
        network_policy = opensearch.CfnSecurityPolicy(
            self,
            "OpenSearchNetworkPolicy",
            name="corpus-veritas-network",
            type="network",
            policy=cdk.Fn.to_json_string([{
                "Rules": [
                    {"Resource": ["collection/corpus-veritas"], "ResourceType": "collection"},
                    {"Resource": ["index/corpus-veritas/*"], "ResourceType": "dashboard"},
                ],
                "AllowFromPublic": True,
            }]),
        )

        # OpenSearch Serverless collection
        self.opensearch_collection = opensearch.CfnCollection(
            self,
            "OpenSearchCollection",
            name="corpus-veritas",
            type="VECTORSEARCH",
            description="corpus-veritas vector search collection for Epstein document analysis",
        )
        self.opensearch_collection.add_dependency(encryption_policy)
        self.opensearch_collection.add_dependency(network_policy)

        # ---------------------------------------------------------------
        # IAM -- pipeline role
        # ---------------------------------------------------------------

        self.pipeline_role = iam.Role(
            self,
            "PipelineRole",
            role_name="corpus-veritas-pipeline",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("ec2.amazonaws.com"),
            ),
            description="corpus-veritas pipeline: ingestion, sanitization, embedding",
        )

        # S3 permissions
        self.corpus_bucket.grant_read_write(self.pipeline_role)
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="CorpusBucketObjectLock",
            actions=["s3:PutObjectRetention", "s3:GetBucketObjectLockConfiguration"],
            resources=[
                self.corpus_bucket.bucket_arn,
                f"{self.corpus_bucket.bucket_arn}/*",
            ],
        ))
        self.audit_bucket.grant_read_write(self.pipeline_role)
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="AuditBucketObjectLock",
            actions=["s3:PutObjectRetention", "s3:GetBucketObjectLockConfiguration"],
            resources=[
                self.audit_bucket.bucket_arn,
                f"{self.audit_bucket.bucket_arn}/*",
            ],
        ))

        # DynamoDB permissions
        for table in [self.documents_table, self.entities_table, self.deletions_table]:
            table.grant_read_write_data(self.pipeline_role)

        # Comprehend
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="Comprehend",
            actions=["comprehend:DetectEntities", "comprehend:DetectPiiEntities"],
            resources=["*"],
        ))

        # Bedrock -- embedding model only
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="BedrockEmbedding",
            actions=["bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/"
                f"{DEFAULT_EMBEDDING_CONFIG.model_id}"
            ],
        ))

        # SQS -- human review queue (sanitizer.py)
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="SQSHumanReview",
            actions=["sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage"],
            resources=[f"arn:aws:sqs:{self.region}:{self.account}:corpus-veritas-human-review"],
        ))

        # OpenSearch Serverless
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="OpenSearchWrite",
            actions=["aoss:APIAccessAll"],
            resources=[self.opensearch_collection.attr_arn],
        ))

        # CloudWatch Logs -- audit
        self.audit_log_group.grant_write(self.pipeline_role)

        # ---------------------------------------------------------------
        # IAM -- query role
        # ---------------------------------------------------------------

        self.query_role = iam.Role(
            self,
            "QueryRole",
            role_name="corpus-veritas-query",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("lambda.amazonaws.com"),
                iam.ServicePrincipal("ec2.amazonaws.com"),
            ),
            description="corpus-veritas query handler: RAG retrieval and response",
        )

        self.corpus_bucket.grant_read(self.query_role)
        for table in [self.documents_table, self.entities_table, self.deletions_table]:
            table.grant_read_data(self.query_role)

        # Bedrock -- synthesis model + embedding
        self.query_role.add_to_policy(iam.PolicyStatement(
            sid="BedrockQuery",
            actions=["bedrock:InvokeModel"],
            resources=[
                f"arn:aws:bedrock:{self.region}::foundation-model/"
                f"{DEFAULT_EMBEDDING_CONFIG.model_id}",
                f"arn:aws:bedrock:{self.region}::foundation-model/claude-sonnet-4-6",
            ],
        ))

        # OpenSearch read
        self.query_role.add_to_policy(iam.PolicyStatement(
            sid="OpenSearchRead",
            actions=["aoss:APIAccessAll"],
            resources=[self.opensearch_collection.attr_arn],
        ))

        # CloudWatch Logs -- audit writes (query responses must be audited)
        self.audit_log_group.grant_write(self.query_role)
        self.audit_bucket.grant_read_write(self.query_role)
        self.query_role.add_to_policy(iam.PolicyStatement(
            sid="AuditBucketObjectLockQuery",
            actions=["s3:PutObjectRetention", "s3:GetBucketObjectLockConfiguration"],
            resources=[
                self.audit_bucket.bucket_arn,
                f"{self.audit_bucket.bucket_arn}/*",
            ],
        ))

        # ---------------------------------------------------------------
        # IAM -- admin role (CDK deploy, infrastructure management)
        # ---------------------------------------------------------------

        self.admin_role = iam.Role(
            self,
            "AdminRole",
            role_name="corpus-veritas-admin",
            assumed_by=iam.AccountRootPrincipal(),
            description="corpus-veritas infrastructure management (CDK deploy)",
        )

        # Admin gets full access to all corpus-veritas resources
        self.pipeline_role.grant_assume_role(self.admin_role)
        self.query_role.grant_assume_role(self.admin_role)

        # ---------------------------------------------------------------
        # OpenSearch access policy (wired to actual role ARNs)
        # ---------------------------------------------------------------

        opensearch.CfnAccessPolicy(
            self,
            "OpenSearchAccessPolicy",
            name="corpus-veritas-access",
            type="data",
            policy=cdk.Fn.to_json_string([
                {
                    "Rules": [
                        {
                            "Resource": ["index/corpus-veritas/*"],
                            "Permission": [
                                "aoss:CreateIndex",
                                "aoss:DeleteIndex",
                                "aoss:UpdateIndex",
                                "aoss:DescribeIndex",
                                "aoss:ReadDocument",
                                "aoss:WriteDocument",
                            ],
                            "ResourceType": "index",
                        },
                        {
                            "Resource": ["collection/corpus-veritas"],
                            "Permission": ["aoss:CreateCollectionItems"],
                            "ResourceType": "collection",
                        },
                    ],
                    "Principal": [self.pipeline_role.role_arn],
                },
                {
                    "Rules": [
                        {
                            "Resource": ["index/corpus-veritas/*"],
                            "Permission": [
                                "aoss:DescribeIndex",
                                "aoss:ReadDocument",
                            ],
                            "ResourceType": "index",
                        },
                    ],
                    "Principal": [self.query_role.role_arn],
                },
            ]),
        )

        # ---------------------------------------------------------------
        # CloudFormation outputs
        # ---------------------------------------------------------------

        cdk.CfnOutput(self, "CorpusBucketName",
                      value=self.corpus_bucket.bucket_name,
                      description="S3 bucket for raw corpus documents")

        cdk.CfnOutput(self, "AuditBucketName",
                      value=self.audit_bucket.bucket_name,
                      description="S3 bucket for immutable audit logs")

        cdk.CfnOutput(self, "OpenSearchEndpoint",
                      value=self.opensearch_collection.attr_collection_endpoint,
                      description="OpenSearch Serverless collection endpoint (set as OPENSEARCH_ENDPOINT)")

        cdk.CfnOutput(self, "PipelineRoleArn",
                      value=self.pipeline_role.role_arn,
                      description="ARN of the pipeline execution role")

        cdk.CfnOutput(self, "QueryRoleArn",
                      value=self.query_role.role_arn,
                      description="ARN of the query handler role")

        cdk.CfnOutput(self, "AuditLogGroupOutput",
                      value=self.audit_log_group.log_group_name,
                      description="CloudWatch log group for audit trail (set as AUDIT_LOG_GROUP)")

"""
tests/infrastructure/test_cdk_stack.py

Unit tests for infrastructure/cdk/stack.py (CorpusVeritasStack).

Uses aws_cdk.assertions.Template to compile the stack to CloudFormation
and assert on the synthesised template. These tests verify that the CDK
code produces the correct AWS resources without requiring a live AWS account.

Requires: aws-cdk-lib, constructs (see requirements-cdk.txt)
Run with: python -m pytest tests/infrastructure/test_cdk_stack.py

Coverage targets
----------------
S3 buckets           -- corpus and audit buckets created, Object Lock
                        enabled, versioning on, public access blocked,
                        RETAIN removal policy, lifecycle rules present
DynamoDB tables      -- all three tables created with correct names,
                        PAY_PER_REQUEST billing, PITR enabled, RETAIN
                        removal policy, all GSIs present
CloudWatch log group -- correct name, 10-year retention, RETAIN
OpenSearch           -- collection created, encryption and network
                        policies created and depended on
IAM roles            -- pipeline, query, admin roles created with
                        correct names
CloudFormation outputs
                     -- CorpusBucketName, AuditBucketName,
                        OpenSearchEndpoint, PipelineRoleArn,
                        QueryRoleArn, AuditLogGroup all present
"""

from __future__ import annotations

import unittest

import aws_cdk as cdk
from aws_cdk import assertions

# Stack is one directory up from the test -- adjust path for import
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "infrastructure", "cdk"))
from stack import CorpusVeritasStack, _OBJECT_LOCK_YEARS


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _template() -> assertions.Template:
    """Synthesise the stack and return an assertions.Template."""
    app = cdk.App()
    stack = CorpusVeritasStack(
        app,
        "TestStack",
        env=cdk.Environment(account="123456789012", region="us-east-1"),
    )
    return assertions.Template.from_stack(stack)


# Synthesise once for the whole module (expensive operation)
TEMPLATE = _template()


# ===========================================================================
# Object Lock retention constant
# ===========================================================================

class TestObjectLockYears(unittest.TestCase):

    def test_retention_is_7_years(self):
        self.assertEqual(_OBJECT_LOCK_YEARS, 7)

    def test_matches_s3_store_retention(self):
        """Verify stack constant matches pipeline/s3_store.py _VICTIM_RETENTION_YEARS."""
        from pipeline.s3_store import _VICTIM_RETENTION_YEARS  # noqa
        self.assertEqual(_OBJECT_LOCK_YEARS, _VICTIM_RETENTION_YEARS)

    def test_matches_audit_log_retention(self):
        """Verify stack constant matches pipeline/audit_log.py _AUDIT_RETENTION_YEARS."""
        from pipeline.audit_log import _AUDIT_RETENTION_YEARS  # noqa
        self.assertEqual(_OBJECT_LOCK_YEARS, _AUDIT_RETENTION_YEARS)


# ===========================================================================
# S3 buckets
# ===========================================================================

class TestS3Buckets(unittest.TestCase):

    def test_two_buckets_created(self):
        TEMPLATE.resource_count_is("AWS::S3::Bucket", 2)

    def test_corpus_bucket_versioning_enabled(self):
        TEMPLATE.has_resource_properties("AWS::S3::Bucket", {
            "VersioningConfiguration": {"Status": "Enabled"},
        })

    def test_corpus_bucket_object_lock_enabled(self):
        TEMPLATE.has_resource_properties("AWS::S3::Bucket", {
            "ObjectLockEnabled": True,
        })

    def test_corpus_bucket_public_access_blocked(self):
        TEMPLATE.has_resource_properties("AWS::S3::Bucket", {
            "PublicAccessBlockConfiguration": {
                "BlockPublicAcls": True,
                "BlockPublicPolicy": True,
                "IgnorePublicAcls": True,
                "RestrictPublicBuckets": True,
            }
        })

    def test_corpus_bucket_encryption_enabled(self):
        TEMPLATE.has_resource_properties("AWS::S3::Bucket", {
            "BucketEncryption": assertions.Match.object_like({
                "ServerSideEncryptionConfiguration": assertions.Match.any_value()
            })
        })

    def test_lifecycle_rule_glacier_transition(self):
        TEMPLATE.has_resource_properties("AWS::S3::Bucket", {
            "LifecycleConfiguration": {
                "Rules": assertions.Match.array_with([
                    assertions.Match.object_like({
                        "Prefix": "PROVENANCE_FLAGGED/",
                        "Status": "Enabled",
                    })
                ])
            }
        })

    def test_lifecycle_rule_multipart_abort(self):
        TEMPLATE.has_resource_properties("AWS::S3::Bucket", {
            "LifecycleConfiguration": {
                "Rules": assertions.Match.array_with([
                    assertions.Match.object_like({
                        "AbortIncompleteMultipartUpload": assertions.Match.any_value(),
                        "Status": "Enabled",
                    })
                ])
            }
        })


# ===========================================================================
# DynamoDB tables
# ===========================================================================

class TestDynamoDBTables(unittest.TestCase):

    def test_three_tables_created(self):
        TEMPLATE.resource_count_is("AWS::DynamoDB::Table", 3)

    def test_documents_table_exists(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_documents",
        })

    def test_entities_table_exists(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_entities",
        })

    def test_deletions_table_exists(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_deletions",
        })

    def test_all_tables_pay_per_request(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "BillingMode": "PAY_PER_REQUEST",
        })

    def test_pitr_enabled(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "PointInTimeRecoverySpecification": {
                "PointInTimeRecoveryEnabled": True
            }
        })

    def test_documents_table_has_gsi_classification_date(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_documents",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-classification-date"})
            ])
        })

    def test_documents_table_has_gsi_corpus_source(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_documents",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-corpus-source"})
            ])
        })

    def test_documents_table_has_gsi_victim_flag(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_documents",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-victim-flag"})
            ])
        })

    def test_entities_table_has_gsi_entity_type(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_entities",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-entity-type"})
            ])
        })

    def test_entities_table_has_gsi_document_uuid(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_entities",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-document-uuid"})
            ])
        })

    def test_deletions_table_has_gsi_flag_date(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_deletions",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-flag-date"})
            ])
        })

    def test_deletions_table_has_gsi_efta_number(self):
        TEMPLATE.has_resource_properties("AWS::DynamoDB::Table", {
            "TableName": "corpus_veritas_deletions",
            "GlobalSecondaryIndexes": assertions.Match.array_with([
                assertions.Match.object_like({"IndexName": "gsi-efta-number"})
            ])
        })


# ===========================================================================
# CloudWatch Log Group
# ===========================================================================

class TestCloudWatchLogGroup(unittest.TestCase):

    def test_log_group_created(self):
        TEMPLATE.resource_count_is("AWS::Logs::LogGroup", 1)

    def test_log_group_name_correct(self):
        TEMPLATE.has_resource_properties("AWS::Logs::LogGroup", {
            "LogGroupName": "/corpus-veritas/audit",
        })

    def test_log_group_retention_ten_years(self):
        # TEN_YEARS = 3653 days in CloudFormation
        TEMPLATE.has_resource_properties("AWS::Logs::LogGroup", {
            "RetentionInDays": 3653,
        })


# ===========================================================================
# OpenSearch Serverless
# ===========================================================================

class TestOpenSearch(unittest.TestCase):

    def test_collection_created(self):
        TEMPLATE.resource_count_is("AWS::OpenSearchServerless::Collection", 1)

    def test_collection_type_vectorsearch(self):
        TEMPLATE.has_resource_properties("AWS::OpenSearchServerless::Collection", {
            "Type": "VECTORSEARCH",
            "Name": "corpus-veritas",
        })

    def test_encryption_policy_created(self):
        TEMPLATE.resource_count_is("AWS::OpenSearchServerless::SecurityPolicy", 2)

    def test_access_policy_created(self):
        TEMPLATE.resource_count_is("AWS::OpenSearchServerless::AccessPolicy", 1)


# ===========================================================================
# IAM Roles
# ===========================================================================

class TestIAMRoles(unittest.TestCase):

    def test_pipeline_role_created(self):
        TEMPLATE.has_resource_properties("AWS::IAM::Role", {
            "RoleName": "corpus-veritas-pipeline",
        })

    def test_query_role_created(self):
        TEMPLATE.has_resource_properties("AWS::IAM::Role", {
            "RoleName": "corpus-veritas-query",
        })

    def test_admin_role_created(self):
        TEMPLATE.has_resource_properties("AWS::IAM::Role", {
            "RoleName": "corpus-veritas-admin",
        })

    def test_pipeline_role_can_be_assumed_by_lambda(self):
        TEMPLATE.has_resource_properties("AWS::IAM::Role", {
            "RoleName": "corpus-veritas-pipeline",
            "AssumeRolePolicyDocument": assertions.Match.object_like({
                "Statement": assertions.Match.array_with([
                    assertions.Match.object_like({
                        "Principal": assertions.Match.object_like({
                            "Service": assertions.Match.any_value()
                        })
                    })
                ])
            })
        })

    def test_comprehend_permission_on_pipeline_role(self):
        # CDK inlines add_to_policy() calls into one DefaultPolicy per role.
        # Find the pipeline role's default policy by checking all IAM policies
        # for one that contains comprehend:DetectEntities.
        policies = TEMPLATE.find_resources("AWS::IAM::Policy", {
            "Properties": {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with([
                        assertions.Match.object_like({
                            "Action": assertions.Match.array_with([
                                "comprehend:DetectEntities"
                            ])
                        })
                    ])
                }
            }
        })
        self.assertGreater(len(policies), 0, "No IAM policy found with comprehend:DetectEntities")

    def test_bedrock_permission_on_pipeline_role(self):
        # CDK renders a single-action PolicyStatement as a string, not an array.
        # Match on Sid instead, which is unique and stable.
        policies = TEMPLATE.find_resources("AWS::IAM::Policy", {
            "Properties": {
                "PolicyDocument": {
                    "Statement": assertions.Match.array_with([
                        assertions.Match.object_like({
                            "Sid": "BedrockEmbedding",
                        })
                    ])
                }
            }
        })
        self.assertGreater(len(policies), 0, "No IAM policy found with Sid=BedrockEmbedding")


# ===========================================================================
# CloudFormation Outputs
# ===========================================================================

class TestCFNOutputs(unittest.TestCase):

    def test_corpus_bucket_name_output(self):
        TEMPLATE.has_output("CorpusBucketName", {})

    def test_audit_bucket_name_output(self):
        TEMPLATE.has_output("AuditBucketName", {})

    def test_opensearch_endpoint_output(self):
        TEMPLATE.has_output("OpenSearchEndpoint", {})

    def test_pipeline_role_arn_output(self):
        TEMPLATE.has_output("PipelineRoleArn", {})

    def test_query_role_arn_output(self):
        TEMPLATE.has_output("QueryRoleArn", {})

    def test_audit_log_group_output(self):
        TEMPLATE.has_output("AuditLogGroupOutput", {})


if __name__ == "__main__":
    unittest.main()

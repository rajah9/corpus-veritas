#!/usr/bin/env python3
"""
infrastructure/cdk/app.py
CDK application entry point for corpus-veritas.

Usage:
    cdk deploy                    # deploy CorpusVeritasStack to default account/region
    cdk deploy --profile prod     # deploy with named AWS profile
    cdk diff                      # show pending changes
    cdk destroy                   # tear down (does NOT delete Object Lock buckets)

Environment variables (set before running cdk):
    AWS_ACCOUNT     AWS account ID
    AWS_REGION      Target region (default: us-east-1)
    CDK_ENV         "dev" | "prod" (default: dev)

See infrastructure/cdk/stack.py for resource definitions.
See infrastructure/iam/README.md for required IAM permissions.
"""

import os
import aws_cdk as cdk
from stack import CorpusVeritasStack

app = cdk.App()

env = cdk.Environment(
    account=os.environ.get("AWS_ACCOUNT", os.environ.get("CDK_DEFAULT_ACCOUNT")),
    region=os.environ.get("AWS_REGION", "us-east-1"),
)

CorpusVeritasStack(
    app,
    "CorpusVeritasStack",
    env=env,
    description="corpus-veritas: Epstein document analysis pipeline infrastructure",
)

app.synth()

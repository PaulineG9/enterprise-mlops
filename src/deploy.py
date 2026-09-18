"""
deploy.py
---------
Production deployment to an Azure Machine Learning Managed Online
Endpoint.

Takes the model registered by `src/train.py` (either in the MLflow
Model Registry or as an Azure ML model asset) and deploys it behind a
scored REST API for real-time inference.

This script talks to real Azure resources via the `azure-ai-ml` SDK and
therefore requires a valid Azure login and workspace config -- it is
not runnable offline. It's included here to show the full path from
"registered model" to "live endpoint", matching the CI/CD workflow in
`.github/workflows/mlops_ci_cd.yml`.

Prerequisites:
    pip install azure-ai-ml azure-identity
    az login
    A `config.json` (or --subscription-id/--resource-group/--workspace-name)
    describing the target Azure ML workspace.

Usage:
    python src/deploy.py \
        --model-name asset-failure-risk-classifier \
        --model-version 1 \
        --endpoint-name asset-risk-endpoint \
        --resource-group my-rg \
        --workspace-name my-ml-workspace \
        --subscription-id 00000000-0000-0000-0000-000000000000
"""

import argparse
import sys

try:
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import (ManagedOnlineDeployment,
                                       ManagedOnlineEndpoint, Model)
    from azure.identity import DefaultAzureCredential
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False


def get_ml_client(subscription_id: str, resource_group: str, workspace_name: str) -> "MLClient":
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )


def deploy_model(
    ml_client: "MLClient",
    model_name: str,
    model_version: str,
    endpoint_name: str,
    instance_type: str = "Standard_DS2_v2",
    instance_count: int = 1,
    traffic_percentage: int = 100,
):
    """Create (or update) a managed online endpoint and deploy the model to it."""
    model = ml_client.models.get(name=model_name, version=model_version)

    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Real-time asset failure-risk scoring endpoint.",
        auth_mode="key",
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Endpoint '{endpoint_name}' is ready.")

    deployment_name = f"{model_name.replace('_', '-')}-v{model_version}"
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type=instance_type,
        instance_count=instance_count,
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"Deployment '{deployment_name}' created on endpoint '{endpoint_name}'.")

    # Route the requested percentage of live traffic to this deployment.
    endpoint.traffic = {deployment_name: traffic_percentage}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"Routed {traffic_percentage}% of traffic to '{deployment_name}'.")

    scoring_uri = ml_client.online_endpoints.get(endpoint_name).scoring_uri
    print(f"Scoring URI: {scoring_uri}")
    return scoring_uri


def main():
    parser = argparse.ArgumentParser(description="Deploy a registered model to an Azure ML managed online endpoint.")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--endpoint-name", type=str, required=True)
    parser.add_argument("--subscription-id", type=str, required=True)
    parser.add_argument("--resource-group", type=str, required=True)
    parser.add_argument("--workspace-name", type=str, required=True)
    parser.add_argument("--instance-type", type=str, default="Standard_DS2_v2")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--traffic-percentage", type=int, default=100)
    args = parser.parse_args()

    if not AZURE_SDK_AVAILABLE:
        print(
            "azure-ai-ml / azure-identity are not installed in this environment. "
            "Install them (`pip install azure-ai-ml azure-identity`) and run this "
            "script from an environment authenticated against Azure (`az login`) "
            "to actually deploy.",
            file=sys.stderr,
        )
        sys.exit(1)

    ml_client = get_ml_client(args.subscription_id, args.resource_group, args.workspace_name)
    deploy_model(
        ml_client=ml_client,
        model_name=args.model_name,
        model_version=args.model_version,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        traffic_percentage=args.traffic_percentage,
    )


if __name__ == "__main__":
    main()

"""
ECS rollback script.

Reverts the ECS service to the previous (stable) task definition revision.
Can also promote/demote a model version in the local registry.

Usage:
    # Rollback ECS service to previous task definition
    python scripts/rollback.py --ecs

    # Rollback model registry: demote current, promote previous
    python scripts/rollback.py --model --current v2 --fallback v1

    # Both at once
    python scripts/rollback.py --ecs --model --current v2 --fallback v1
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path so train.registry is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── ECS Rollback ──────────────────────────────────────────────────────────────

def rollback_ecs(cluster: str, service: str, region: str):
    try:
        import boto3
    except ImportError:
        logger.error("boto3 is required for ECS rollback. Install it with: pip install boto3")
        sys.exit(1)

    ecs = boto3.client("ecs", region_name=region)

    # Get current task definition ARN
    svc = ecs.describe_services(cluster=cluster, services=[service])["services"][0]
    current_arn = svc["taskDefinition"]
    logger.info("Current task definition: %s", current_arn)

    # Parse family and revision
    # ARN format: arn:aws:ecs:<region>:<account>:task-definition/<family>:<revision>
    family_rev = current_arn.split("/")[-1]          # e.g. digit-prediction-api:7
    family, current_rev = family_rev.rsplit(":", 1)
    current_rev = int(current_rev)

    if current_rev <= 1:
        logger.error("Already at revision 1 — cannot roll back further.")
        sys.exit(1)

    previous_rev = current_rev - 1
    previous_arn = f"{family}:{previous_rev}"
    logger.info("Rolling back to task definition revision: %s", previous_arn)

    response = ecs.update_service(
        cluster=cluster,
        service=service,
        taskDefinition=previous_arn,
        forceNewDeployment=True,
    )
    new_td = response["service"]["taskDefinition"]
    logger.info("ECS service updated. New task definition: %s", new_td)
    logger.info(
        "Monitor rollback with:\n  aws ecs describe-services --cluster %s --services %s",
        cluster,
        service,
    )


# ── Model Registry Rollback ───────────────────────────────────────────────────

def rollback_model(current_version: str, fallback_version: str):
    from train.registry import ModelRegistry

    models_dir = Path(__file__).resolve().parent.parent / "models"
    registry = ModelRegistry(models_dir)

    promoted = registry.promoted_version()
    logger.info("Currently promoted version: %s", promoted)

    registry.archive(current_version)
    logger.info("Archived version '%s'", current_version)

    registry.promote(fallback_version)
    logger.info("Promoted fallback version '%s'", fallback_version)

    logger.info(
        "Model rollback complete. Restart the API server to load version '%s'.",
        fallback_version,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rollback ECS deployment and/or model version")
    parser.add_argument("--ecs", action="store_true", help="Rollback ECS service")
    parser.add_argument("--model", action="store_true", help="Rollback model registry")
    parser.add_argument(
        "--cluster", default="digit-prediction-api", help="ECS cluster name"
    )
    parser.add_argument(
        "--service", default="digit-prediction-api", help="ECS service name"
    )
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    parser.add_argument("--current", default=None, help="Current (bad) model version")
    parser.add_argument("--fallback", default=None, help="Fallback model version to promote")
    args = parser.parse_args()

    if not args.ecs and not args.model:
        parser.error("Specify at least one of --ecs or --model")

    if args.ecs:
        rollback_ecs(cluster=args.cluster, service=args.service, region=args.region)

    if args.model:
        if not args.current or not args.fallback:
            parser.error("--model requires both --current and --fallback")
        rollback_model(current_version=args.current, fallback_version=args.fallback)


if __name__ == "__main__":
    main()

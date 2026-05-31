"""
Automated retraining trigger.

Polls the running API's /monitoring/drift endpoint and, when drift is
detected, triggers a new training run via the GitHub Actions
workflow_dispatch event.

Designed to be run:
  - As a cron job / scheduled task on any host
  - By the GitHub Actions `retrain.yml` workflow on a schedule
  - Manually: python scripts/retrain_trigger.py

Environment variables required when firing GitHub Actions:
  GH_TOKEN        — GitHub Personal Access Token with `actions:write` scope
  GH_REPO         — owner/repo  e.g. "acme/mlops-coding-assignment"

Optional:
  API_URL         — base URL of the running API (default: http://localhost:8000)
  NEW_VERSION     — version tag for the newly trained model (default: auto-incremented)
  MIN_SAMPLES     — minimum predictions in window before drift is considered (default: 200)

Usage:
    export GH_TOKEN=ghp_...
    export GH_REPO=acme/mlops-coding-assignment
    python scripts/retrain_trigger.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")
GH_TOKEN = os.getenv("GH_TOKEN", "")
GH_REPO = os.getenv("GH_REPO", "")
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "200"))

# Add project root so registry is importable when running locally
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GH_ACTIONS_URL = "https://api.github.com/repos/{repo}/actions/workflows/retrain.yml/dispatches"


def _next_version() -> str:
    """Auto-increment model version based on registry."""
    try:
        from train.registry import ModelRegistry

        models_dir = Path(__file__).resolve().parent.parent / "models"
        registry = ModelRegistry(models_dir)
        versions = [v["version"] for v in registry.list_versions()]
        nums = []
        for v in versions:
            try:
                nums.append(int(v.lstrip("v")))
            except ValueError:
                pass
        next_num = max(nums, default=1) + 1
        return f"v{next_num}"
    except Exception:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        return f"v_{ts}"


def check_drift() -> dict:
    """Call the API drift endpoint and return the report."""
    url = f"{API_URL}/monitoring/drift"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        logger.error("Failed to reach drift endpoint %s: %s", url, exc)
        sys.exit(1)


def trigger_github_workflow(new_version: str, drift_report: dict):
    """Fire a workflow_dispatch event on GitHub Actions."""
    if not GH_TOKEN or not GH_REPO:
        logger.error(
            "GH_TOKEN and GH_REPO env vars are required to trigger GitHub Actions. "
            "Set them or run the training locally with: python -m train.train --version %s",
            new_version,
        )
        sys.exit(1)

    url = GH_ACTIONS_URL.format(repo=GH_REPO)
    payload = {
        "ref": "main",
        "inputs": {
            "model_version": new_version,
            "trigger_reason": f"drift_alert psi={drift_report.get('psi', 'N/A')}",
        },
    }
    headers = {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code == 204:
        logger.info(
            "GitHub Actions retrain workflow triggered for version '%s'", new_version
        )
    else:
        logger.error(
            "Failed to trigger workflow: HTTP %d — %s", resp.status_code, resp.text
        )
        sys.exit(1)


def run():
    logger.info("Checking drift at %s/monitoring/drift …", API_URL)
    report = check_drift()

    if report.get("status") == "no_data":
        logger.info("No prediction data yet — skipping.")
        return

    n = report.get("n", 0)
    if n < MIN_SAMPLES:
        logger.info(
            "Only %d predictions in window (minimum %d) — skipping.", n, MIN_SAMPLES
        )
        return

    psi = report.get("psi", 0)
    alert = report.get("alert", False)

    logger.info("Drift report: PSI=%.4f alert=%s n=%d", psi, alert, n)

    if not alert:
        logger.info("No significant drift detected. No retraining needed.")
        return

    new_version = os.getenv("NEW_VERSION") or _next_version()
    logger.warning(
        "Drift detected! PSI=%.4f. Triggering retraining for version '%s'.",
        psi,
        new_version,
    )

    # Write a local trigger log
    log_path = Path(__file__).resolve().parent.parent / "models" / "retrain_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as f:
        f.write(
            json.dumps(
                {
                    "triggered_at": datetime.now(timezone.utc).isoformat(),
                    "new_version": new_version,
                    "drift_report": report,
                }
            )
            + "\n"
        )

    trigger_github_workflow(new_version=new_version, drift_report=report)


if __name__ == "__main__":
    run()

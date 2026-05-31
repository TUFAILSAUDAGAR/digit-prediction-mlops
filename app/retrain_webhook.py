"""
Internal retraining webhook router.

Mounted on the FastAPI app at /retrain/* and provides:

  POST /retrain/trigger
    — Checks current drift; if alert is active, fires a background retraining
      job (local subprocess) or calls the GitHub Actions dispatch API.
    — Protected by a shared secret (RETRAIN_WEBHOOK_SECRET env var).

  GET  /retrain/status
    — Returns the status of the last triggered retraining job.

  GET  /retrain/log
    — Returns the last N entries from models/retrain_log.jsonl.

The retraining job runs as a background thread so the HTTP request returns
immediately. In production this would instead fire a workflow_dispatch event
via GitHub Actions (see scripts/retrain_trigger.py).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Query

logger = logging.getLogger(__name__)

RETRAIN_WEBHOOK_SECRET = os.getenv("RETRAIN_WEBHOOK_SECRET", "")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RETRAIN_LOG = MODELS_DIR / "retrain_log.jsonl"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

router = APIRouter(prefix="/retrain", tags=["retraining"])

# ── In-memory job state ───────────────────────────────────────────────────────
_job_state: dict = {"status": "idle", "version": None, "started_at": None, "finished_at": None, "error": None}
_job_lock = threading.Lock()


def _require_secret(x_webhook_secret: str | None):
    if RETRAIN_WEBHOOK_SECRET and x_webhook_secret != RETRAIN_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


# ── Background retraining worker ─────────────────────────────────────────────

def _run_retrain(version: str, reason: str):
    global _job_state
    with _job_lock:
        _job_state = {
            "status": "running",
            "version": version,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
        }

    _append_log({"event": "retrain_started", "version": version, "reason": reason,
                 "timestamp": datetime.now(timezone.utc).isoformat()})

    try:
        result = subprocess.run(
            [sys.executable, "-m", "train.train", "--version", version, "--epochs", "5"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])

        # Evaluate with quality gate
        eval_result = subprocess.run(
            [
                sys.executable, "-m", "train.evaluate",
                "--version", version,
                "--min-accuracy", "0.95",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        if eval_result.returncode != 0:
            raise RuntimeError(f"Evaluation gate failed:\n{eval_result.stderr[-2000:]}")

        # Promote in registry
        from train.registry import ModelRegistry
        reg = ModelRegistry(MODELS_DIR)
        reg.promote(version)

        with _job_lock:
            _job_state.update({"status": "succeeded", "finished_at": datetime.now(timezone.utc).isoformat()})

        _append_log({"event": "retrain_succeeded", "version": version,
                     "timestamp": datetime.now(timezone.utc).isoformat()})
        logger.info("Retraining job for version '%s' succeeded", version)

    except Exception as exc:
        with _job_lock:
            _job_state.update({
                "status": "failed",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc)[:500],
            })
        _append_log({"event": "retrain_failed", "version": version,
                     "error": str(exc)[:500],
                     "timestamp": datetime.now(timezone.utc).isoformat()})
        logger.error("Retraining job for version '%s' failed: %s", version, exc)


def _append_log(entry: dict):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with RETRAIN_LOG.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Auto-increment version ────────────────────────────────────────────────────

def _next_version() -> str:
    try:
        from train.registry import ModelRegistry
        reg = ModelRegistry(MODELS_DIR)
        versions = [v["version"] for v in reg.list_versions()]
        nums = [int(v.lstrip("v")) for v in versions if v.lstrip("v").isdigit()]
        return f"v{max(nums, default=1) + 1}"
    except Exception:
        return f"v_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/trigger")
async def trigger_retrain(
    x_webhook_secret: str | None = Header(default=None),
    version: str | None = Query(default=None, description="Version tag, e.g. v2. Auto-incremented if omitted."),
    reason: str = Query(default="manual", description="Reason for triggering retraining"),
):
    """
    Trigger a retraining job in the background.
    Protected by X-Webhook-Secret header when RETRAIN_WEBHOOK_SECRET env var is set.
    """
    _require_secret(x_webhook_secret)

    with _job_lock:
        if _job_state["status"] == "running":
            raise HTTPException(
                status_code=409,
                detail=f"A retraining job for version '{_job_state['version']}' is already running.",
            )

    new_version = version or _next_version()
    logger.info("Retraining triggered: version=%s reason=%s", new_version, reason)

    thread = threading.Thread(target=_run_retrain, args=(new_version, reason), daemon=True)
    thread.start()

    return {
        "message": "Retraining job started",
        "version": new_version,
        "reason": reason,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/status")
async def retrain_status():
    """Return the status of the last/current retraining job."""
    with _job_lock:
        return dict(_job_state)


@router.get("/log")
async def retrain_log(last: int = Query(default=20, ge=1, le=500, description="Number of log entries to return")):
    """Return the last N entries from the retraining log."""
    if not RETRAIN_LOG.exists():
        return {"entries": []}
    lines = RETRAIN_LOG.read_text().strip().splitlines()
    entries = []
    for line in lines[-last:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return {"entries": entries}

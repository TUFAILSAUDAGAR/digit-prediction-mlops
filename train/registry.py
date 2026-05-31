"""
Lightweight file-based model registry.

The registry is stored as models/registry.json:
{
  "versions": {
    "v1": {
      "created_at": "2026-05-13T10:00:00",
      "metrics": {"accuracy": 0.991, ...},
      "status": "promoted"   # staging | promoted | archived
    },
    ...
  },
  "promoted": "v1"
}

Usage:
    from train.registry import ModelRegistry
    reg = ModelRegistry()
    reg.register("v2", metrics={"accuracy": 0.993})
    reg.promote("v2")
    print(reg.promoted_version())
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelRegistry:
    """
    File-based model registry backed by models/registry.json.

    Tracks version metadata (accuracy, status) and the currently promoted version.
    Statuses: staging → promoted → archived.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.registry_path = models_dir / "registry.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Load registry from disk, returning an empty structure if the file doesn't exist."""
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {"versions": {}, "promoted": None}

    def _save(self):
        """Persist the in-memory registry dict back to disk as pretty-printed JSON."""
        self.registry_path.write_text(json.dumps(self._data, indent=2))

    def register(self, version: str, metrics: dict, status: str = "staging"):
        """Add or update a version entry. New versions default to 'staging' status."""
        if version in self._data["versions"]:
            logger.warning("Version '%s' already registered — overwriting metrics", version)
        self._data["versions"][version] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
            "status": status,
        }
        self._save()
        logger.info("Registered version '%s' with status '%s'", version, status)

    def promote(self, version: str):
        """
        Mark version as 'promoted' and archive the previously promoted version.
        Raises KeyError if the version is not registered.
        """
        if version not in self._data["versions"]:
            raise KeyError(f"Version '{version}' not found in registry")

        # Archive the previously promoted version before switching
        prev = self._data.get("promoted")
        if prev and prev != version and prev in self._data["versions"]:
            self._data["versions"][prev]["status"] = "archived"
            logger.info("Archived previous promoted version '%s'", prev)

        self._data["versions"][version]["status"] = "promoted"
        self._data["promoted"] = version
        self._save()
        logger.info("Promoted version '%s'", version)

    def archive(self, version: str):
        """Mark a version as archived (used during rollback)."""
        if version not in self._data["versions"]:
            raise KeyError(f"Version '{version}' not found in registry")
        self._data["versions"][version]["status"] = "archived"
        self._save()

    def get(self, version: str) -> dict | None:
        """Return the registry entry for a version, or None if not found."""
        return self._data["versions"].get(version)

    def promoted_version(self) -> str | None:
        """Return the currently promoted version tag, or None."""
        return self._data.get("promoted")

    def list_versions(self) -> list[dict]:
        """Return all registered versions as a flat list of dicts."""
        return [
            {"version": v, **info}
            for v, info in self._data["versions"].items()
        ]

    def staging_versions(self) -> list[str]:
        """Return version tags that are still in 'staging' status."""
        return [v for v, info in self._data["versions"].items() if info["status"] == "staging"]

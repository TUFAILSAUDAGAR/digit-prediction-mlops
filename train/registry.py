"""File-based model registry backed by models/registry.json."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelRegistry:
    """Tracks model versions (staging/promoted/archived) in a JSON file."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.registry_path = models_dir / "registry.json"
        self._data: dict = self._load()

    def _load(self) -> dict:
        """Load registry from disk; return empty structure if file not found."""
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text())
        return {"versions": {}, "promoted": None}

    def _save(self):
        """Persist registry to disk as pretty-printed JSON."""
        self.registry_path.write_text(json.dumps(self._data, indent=2))

    def register(self, version: str, metrics: dict, status: str = "staging"):
        """Add or update a version entry (defaults to 'staging' status)."""
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
        """Mark version as promoted and archive the previous promoted version."""
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
        """Mark a version as archived."""
        if version not in self._data["versions"]:
            raise KeyError(f"Version '{version}' not found in registry")
        self._data["versions"][version]["status"] = "archived"
        self._save()

    def get(self, version: str) -> dict | None:
        """Return registry entry for a version, or None."""
        return self._data["versions"].get(version)

    def promoted_version(self) -> str | None:
        """Currently promoted version tag, or None."""
        return self._data.get("promoted")

    def list_versions(self) -> list[dict]:
        """All registered versions as a flat list."""
        return [
            {"version": v, **info}
            for v, info in self._data["versions"].items()
        ]

    def staging_versions(self) -> list[str]:
        """Version tags still in 'staging' status."""
        return [v for v, info in self._data["versions"].items() if info["status"] == "staging"]

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AppConfig


class LocalStore:
    """
    Lightweight file-based persistence layer that mirrors the Firestore
    schema described in AGENT.md. Configuration, state, and logs are stored
    under the `.data/` folder so the app can run without Firebase.
    """

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir or ".data")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.base_dir / "config.json"
        self.state_path = self.base_dir / "state.json"
        self.log_path = self.base_dir / "logs.json"

    def _read_json(self, path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return default

    def _write_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, indent=2))

    def fetch_config(self) -> AppConfig:
        data = self._read_json(self.config_path, {})
        return AppConfig.from_dict(data)

    def save_config(self, config: AppConfig) -> None:
        self._write_json(self.config_path, config.to_dict())

    def fetch_last_run_ts(self) -> Optional[float]:
        data = self._read_json(self.state_path, {})
        return data.get("timestamp")

    def update_last_run(self, timestamp: float) -> None:
        payload = {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        }
        self._write_json(self.state_path, payload)

    def append_log(
        self, status: str, details: str, channel: Optional[str] = None
    ) -> None:
        logs = self._read_json(self.log_path, [])
        logs.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "details": details,
                "channel": channel or "",
            }
        )
        self._write_json(self.log_path, logs)

    def fetch_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        logs = self._read_json(self.log_path, [])
        logs.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return logs[:limit]


# Backwards compatibility with existing imports
FirestoreStore = LocalStore

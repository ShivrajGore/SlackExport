from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore

from .models import AppConfig

CONFIG_COLLECTION = "config"
CONFIG_DOCUMENT = "settings"
STATE_COLLECTION = "state"
STATE_DOCUMENT = "last_run"
LOG_COLLECTION = "logs"


class FirestoreStore:
    """Wrapper around Firestore for storing config and application state."""

    def __init__(self, credentials_path: Optional[str] = None) -> None:
        if not firebase_admin._apps:
            cred = self._load_credentials(credentials_path)
            firebase_admin.initialize_app(cred)
        self._client = firestore.client()

    def _load_credentials(self, credentials_path: Optional[str]):
        try:
            if credentials_path:
                return credentials.Certificate(credentials_path)
            return credentials.ApplicationDefault()
        except ValueError as exc:  # missing credentials or invalid file
            raise RuntimeError(
                "Firebase credentials not found. "
                "Set GOOGLE_APPLICATION_CREDENTIALS or pass --credentials."
            ) from exc

    def fetch_config(self) -> AppConfig:
        snapshot = (
            self._client.collection(CONFIG_COLLECTION)
            .document(CONFIG_DOCUMENT)
            .get()
        )
        data = snapshot.to_dict() or {}
        return AppConfig.from_dict(data)

    def save_config(self, config: AppConfig) -> None:
        data = config.to_dict()
        (
            self._client.collection(CONFIG_COLLECTION)
            .document(CONFIG_DOCUMENT)
            .set(data)
        )

    def fetch_last_run_ts(self) -> Optional[float]:
        snapshot = (
            self._client.collection(STATE_COLLECTION)
            .document(STATE_DOCUMENT)
            .get()
        )
        if not snapshot.exists:
            return None
        data = snapshot.to_dict() or {}
        return data.get("timestamp")

    def update_last_run(self, timestamp: float) -> None:
        document = {
            "timestamp": timestamp,
            "date": datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(),
        }
        (
            self._client.collection(STATE_COLLECTION)
            .document(STATE_DOCUMENT)
            .set(document)
        )

    def append_log(
        self, status: str, details: str, channel: Optional[str] = None
    ) -> None:
        payload = {
            "timestamp": firestore.SERVER_TIMESTAMP,
            "status": status,
            "details": details,
            "channel": channel,
        }
        self._client.collection(LOG_COLLECTION).add(payload)

    def fetch_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        query = (
            self._client.collection(LOG_COLLECTION)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        snapshots = query.stream()
        logs: List[Dict[str, Any]] = []
        for snap in snapshots:
            data = snap.to_dict()
            ts = data.get("timestamp")
            logs.append(
                {
                    "timestamp": ts.isoformat() if ts else "",
                    "status": data.get("status", ""),
                    "details": data.get("details", ""),
                    "channel": data.get("channel", ""),
                }
            )
        return logs

"""Append-only prediction history stored as JSONL on a mounted volume."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

HISTORY_FILENAME = "history.jsonl"


def history_dir() -> Path:
    return Path(os.getenv("PREDICTION_HISTORY_DIR", "/data/predictions"))


def history_path() -> Path:
    return history_dir() / HISTORY_FILENAME


def _ensure_dir() -> None:
    history_dir().mkdir(parents=True, exist_ok=True)


def append_record(request: dict, response: dict) -> str:
    """Append one prediction record; return its UUID."""
    _ensure_dir()
    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "response": response,
    }
    line = json.dumps(record, default=str) + "\n"
    path = history_path()
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(line)
            handle.flush()
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            handle.write(line)
            handle.flush()
    return record_id


def list_records(limit: int = 50) -> list[dict]:
    path = history_path()
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    records = [json.loads(line) for line in lines if line.strip()]
    return records[-limit:][::-1]


def get_record(record_id: str) -> dict | None:
    path = history_path()
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("id") == record_id:
                return record
    return None

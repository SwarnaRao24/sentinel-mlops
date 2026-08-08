"""Drift event store.

Detection writes events here; it does NOT act on them. Something downstream
(orchestration) reads unhandled events and decides whether to retrain.
This separation keeps detection auditable and testable on its own.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/predictions.db")


def init_events(db_path: Path = DB_PATH) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS drift_events (
            event_id      TEXT PRIMARY KEY,
            detected_at   TEXT NOT NULL,
            drift_type    TEXT NOT NULL,   -- data | prediction | concept
            metric        TEXT NOT NULL,   -- what was measured
            value         REAL NOT NULL,   -- measured value
            threshold     REAL NOT NULL,   -- threshold crossed
            window_start  TEXT,
            window_end    TEXT,
            evidence_json TEXT,            -- extra detail for the audit trail
            status        TEXT NOT NULL DEFAULT 'open'  -- open | handled
        )
        """
    )
    con.commit()
    con.close()


def record_drift_event(
    *,
    drift_type: str,
    metric: str,
    value: float,
    threshold: float,
    window_start: str | None = None,
    window_end: str | None = None,
    evidence: dict | None = None,
    db_path: Path = DB_PATH,
) -> str:
    event_id = str(uuid.uuid4())
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO drift_events (
            event_id, detected_at, drift_type, metric, value, threshold,
            window_start, window_end, evidence_json, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """,
        (
            event_id,
            datetime.now(timezone.utc).isoformat(),
            drift_type,
            metric,
            float(value),
            float(threshold),
            window_start,
            window_end,
            json.dumps(evidence or {}),
        ),
    )
    con.commit()
    con.close()
    return event_id


def open_events(db_path: Path = DB_PATH) -> list[dict]:
    """Return unhandled drift events - what orchestration will read."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM drift_events WHERE status = 'open' ORDER BY detected_at"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]
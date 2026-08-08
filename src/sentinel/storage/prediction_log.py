"""Prediction log - the spine of the platform.

Every prediction served is recorded here with full lineage: the input
features, the output, the model version, the code commit, and latency.
Downstream (drift detection, label join, validation) all read from this.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/predictions.db")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def init_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id   TEXT PRIMARY KEY,
            predicted_at    TEXT NOT NULL,
            zone_id         INTEGER NOT NULL,
            pickup_hour     TEXT NOT NULL,
            features_json   TEXT NOT NULL,
            prediction      REAL NOT NULL,
            model_name      TEXT NOT NULL,
            model_version   TEXT NOT NULL,
            code_sha        TEXT NOT NULL,
            latency_ms      REAL NOT NULL,
            actual          REAL,
            actual_at       TEXT
        )
        """
    )
    con.commit()
    con.close()


def log_prediction(
    *,
    zone_id: int,
    pickup_hour: str,
    features: dict,
    prediction: float,
    model_name: str,
    model_version: str,
    latency_ms: float,
    db_path: Path = DB_PATH,
) -> str:
    prediction_id = str(uuid.uuid4())
    con = sqlite3.connect(db_path)
    con.execute(
        """
        INSERT INTO predictions (
            prediction_id, predicted_at, zone_id, pickup_hour, features_json,
            prediction, model_name, model_version, code_sha, latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction_id,
            datetime.now(timezone.utc).isoformat(),
            zone_id,
            pickup_hour,
            json.dumps(features),
            prediction,
            model_name,
            model_version,
            _git_sha(),
            latency_ms,
        ),
    )
    con.commit()
    con.close()
    return prediction_id
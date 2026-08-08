"""Simulated late-arriving label stream + the reconciliation join."""
from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("data/predictions.db")
DEMAND_PATH = Path("data/processed/demand_2024-01.parquet")


def build_label_stream(seed: int = 42) -> pd.DataFrame:
    """Return ground-truth demand with an 'available_at' timestamp."""
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(DEMAND_PATH)[["zone_id", "pickup_hour", "demand"]].copy()

    hour_close = df["pickup_hour"] + timedelta(hours=1)
    base_delay = timedelta(hours=2)
    jitter_h = rng.gamma(shape=2.0, scale=1.5, size=len(df))
    df["available_at"] = hour_close + base_delay + pd.to_timedelta(jitter_h, unit="h")
    df = df.rename(columns={"demand": "actual"})
    return df


def reconcile_labels(as_of: pd.Timestamp, db_path: Path = DB_PATH) -> dict:
    """Fill predictions.actual with labels available as of `as_of`."""
    labels = build_label_stream()
    available = labels[labels["available_at"] <= as_of]

    con = sqlite3.connect(db_path)
    preds = pd.read_sql_query(
        "SELECT prediction_id, zone_id, pickup_hour, actual FROM predictions", con
    )
    if preds.empty:
        con.close()
        return {"as_of": str(as_of), "matched": 0, "updated": 0, "pending": 0}

    preds["pickup_hour"] = pd.to_datetime(preds["pickup_hour"])
    merged = preds.merge(
        available[["zone_id", "pickup_hour", "actual"]],
        on=["zone_id", "pickup_hour"],
        how="left",
        suffixes=("_old", "_new"),
    )

    updated = 0
    for _, r in merged.iterrows():
        new_actual = r["actual_new"]
        old_actual = r["actual_old"]
        if pd.notna(new_actual) and (pd.isna(old_actual) or new_actual != old_actual):
            con.execute(
                "UPDATE predictions SET actual = ?, actual_at = ? WHERE prediction_id = ?",
                (float(new_actual), str(as_of), r["prediction_id"]),
            )
            updated += 1
    con.commit()

    matched = int(merged["actual_new"].notna().sum())
    pending = len(preds) - matched
    con.close()
    return {"as_of": str(as_of), "matched": matched, "updated": updated, "pending": pending}
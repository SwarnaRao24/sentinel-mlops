"""Tests for the late-label stream and reconciliation join."""
import pandas as pd
import pytest

from sentinel.storage import labels as lab
from sentinel.storage.prediction_log import init_db, log_prediction


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Isolated DB + a tiny known demand file, so tests don't touch real data."""
    db = tmp_path / "test.db"
    demand = tmp_path / "demand.parquet"

    # Two zone-hours of known ground truth
    df = pd.DataFrame(
        {
            "zone_id": [1, 1],
            "pickup_hour": [
                pd.Timestamp("2024-01-01 00:00:00"),
                pd.Timestamp("2024-01-01 01:00:00"),
            ],
            "demand": [10, 20],
        }
    )
    df.to_parquet(demand, index=False)

    monkeypatch.setattr(lab, "DB_PATH", db)
    monkeypatch.setattr(lab, "DEMAND_PATH", demand)

    init_db(db)
    # Log a prediction for each zone-hour
    for ph, pred in [("2024-01-01T00:00:00", 8.0), ("2024-01-01T01:00:00", 25.0)]:
        log_prediction(
            zone_id=1, pickup_hour=ph, features={}, prediction=pred,
            model_name="m", model_version="1", latency_ms=1.0, db_path=db,
        )
    return db


def test_label_stream_has_availability(temp_db):
    stream = lab.build_label_stream()
    assert "available_at" in stream.columns
    # available_at must be after the hour closes
    assert (stream["available_at"] > stream["pickup_hour"]).all()


def test_point_in_time_correctness(temp_db):
    # Before any label is available, nothing should match
    early = lab.reconcile_labels(pd.Timestamp("2024-01-01 00:30:00"), db_path=temp_db)
    assert early["matched"] == 0


def test_labels_fill_in_over_time(temp_db):
    early = lab.reconcile_labels(pd.Timestamp("2024-01-01 00:30:00"), db_path=temp_db)
    late = lab.reconcile_labels(pd.Timestamp("2024-01-05 00:00:00"), db_path=temp_db)
    assert late["matched"] >= early["matched"]
    assert late["matched"] == 2  # both labels arrived by 4 days later


def test_reconcile_is_rerunnable(temp_db):
    # Running twice at the same as_of should update nothing the second time
    lab.reconcile_labels(pd.Timestamp("2024-01-05 00:00:00"), db_path=temp_db)
    second = lab.reconcile_labels(pd.Timestamp("2024-01-05 00:00:00"), db_path=temp_db)
    assert second["updated"] == 0
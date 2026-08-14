"""Tests for the FastAPI serving layer.

Self-contained: trains and registers a throwaway model into a temporary
MLflow registry so the tests don't depend on any pre-existing artifact.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingRegressor


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Point MLflow at a temp tracking dir, register a tiny model as v1
    tmp = tmp_path_factory.mktemp("mlflow")
    mlflow.set_tracking_uri(f"sqlite:///{tmp}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{tmp}/mlflow.db")

    feats = ["zone_id", "hour", "day_of_week", "is_weekend", "is_holiday"]
    X = pd.DataFrame(
        {
            "zone_id": [1, 50, 100, 200, 4, 79],
            "hour": [0, 6, 12, 18, 9, 15],
            "day_of_week": [0, 1, 2, 3, 4, 5],
            "is_weekend": [False, False, False, False, False, True],
            "is_holiday": [True, False, False, False, False, False],
        }
    )
    y = [10, 25, 40, 30, 15, 20]
    model = HistGradientBoostingRegressor(max_iter=10, random_state=0).fit(X[feats], y)

    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model, name="model", registered_model_name="demand-forecaster"
        )

    # Import the app AFTER the tracking URI + model are set up
    from sentinel.serving.app import app

    with TestClient(app) as c:
        yield c


def _valid_payload() -> dict:
    return {
        "zone_id": 100,
        "pickup_hour": "2024-01-15T08:00:00",
        "hour": 8,
        "day_of_week": 0,
        "is_weekend": False,
        "is_holiday": False,
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_valid(client):
    r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["prediction"], float)
    assert body["prediction_id"]
    assert body["model_version"] == "1"


@pytest.mark.parametrize(
    "field,bad",
    [
        ("zone_id", 9999),
        ("zone_id", 0),
        ("hour", 24),
        ("day_of_week", 7),
    ],
)
def test_predict_rejects_bad_input(client, field, bad):
    payload = _valid_payload()
    payload[field] = bad
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_missing_field(client):
    payload = _valid_payload()
    del payload["hour"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

"""Tests for the FastAPI serving layer."""
import pytest
from fastapi.testclient import TestClient

from sentinel.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # triggers lifespan -> loads model, inits db
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
    assert r.status_code == 422  # rejected by contract before the model


def test_predict_missing_field(client):
    payload = _valid_payload()
    del payload["hour"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

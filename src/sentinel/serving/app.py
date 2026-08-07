"""FastAPI serving layer for the demand forecaster.

Loads the registered model once at startup, validates each request against
a contract, serves a prediction, and logs it with full lineage.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from sentinel.storage.prediction_log import init_db, log_prediction

MODEL_NAME = "demand-forecaster"
MODEL_STAGE = "latest"  # we'll move to explicit stages later
FEATURES = ["zone_id", "hour", "day_of_week", "is_weekend", "is_holiday"]

# Populated at startup
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    model_uri = f"models:/{MODEL_NAME}/{_state.get('version', '1')}"
    _state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
    _state["model_version"] = "1"
    yield
    _state.clear()


app = FastAPI(title="Sentinel Demand Forecaster", lifespan=lifespan)


class PredictRequest(BaseModel):
    """Request contract - mirrors the training features."""

    zone_id: int = Field(..., ge=1, le=265)
    pickup_hour: str = Field(..., description="ISO timestamp, floored to the hour")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: bool
    is_holiday: bool


class PredictResponse(BaseModel):
    prediction: float
    prediction_id: str
    model_version: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in _state}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()

    row = pd.DataFrame([{f: getattr(req, f) for f in FEATURES}])
    pred = float(_state["model"].predict(row)[0])
    latency_ms = (time.perf_counter() - start) * 1000

    pid = log_prediction(
        zone_id=req.zone_id,
        pickup_hour=req.pickup_hour,
        features={f: getattr(req, f) for f in FEATURES},
        prediction=pred,
        model_name=MODEL_NAME,
        model_version=_state["model_version"],
        latency_ms=latency_ms,
    )

    return PredictResponse(
        prediction=pred, prediction_id=pid, model_version=_state["model_version"]
    )

"""Retraining and evaluation logic (framework-agnostic).

These are plain functions so they can be tested without Dagster. The
orchestration layer (Dagster assets) calls them; it does not contain the
logic itself.
"""
from __future__ import annotations

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from sentinel.validation.base import GateContext

DATA = "data/processed/demand_2024-01.parquet"
FEATURES = ["zone_id", "hour", "day_of_week", "is_weekend", "is_holiday"]
TARGET = "demand"
MODEL_NAME = "demand-forecaster"


def _split():
    df = pd.read_parquet(DATA).sort_values("pickup_hour").reset_index(drop=True)
    cutoff = df["pickup_hour"].quantile(0.8)
    train = df[df["pickup_hour"] <= cutoff]
    holdout = df[df["pickup_hour"] > cutoff]
    return train, holdout


def load_champion():
    """Load the current Production champion from the registry (version 1)."""
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")


def train_challenger(params: dict | None = None):
    """Train a fresh challenger on the training split. Returns the model."""
    train, _ = _split()
    params = params or {
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_leaf_nodes": 40,
        "random_state": 7,
    }
    model = HistGradientBoostingRegressor(**params)
    model.fit(train[FEATURES], train[TARGET])
    return model


def build_gate_context(champion, challenger) -> GateContext:
    """Score champion + challenger on the shared holdout for the gates."""
    _, holdout = _split()
    X = holdout[FEATURES]
    y = holdout[TARGET].to_numpy()
    champ_preds = np.asarray(champion.predict(X))
    chall_preds = np.asarray(challenger.predict(X))
    segments = holdout["zone_id"].to_numpy()
    return GateContext(
        champion_preds=champ_preds,
        challenger_preds=chall_preds,
        y_true=y,
        segments=segments,
    )


def holdout_mae(model) -> float:
    _, holdout = _split()
    return float(mean_absolute_error(holdout[TARGET], model.predict(holdout[FEATURES])))
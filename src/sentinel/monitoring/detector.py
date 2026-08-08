"""Drift detector.

Compares a reference window (training-time) against a current window (recent
traffic) and records a drift event when a threshold is crossed. It MEASURES
and RECORDS only - it never retrains. Acting on events is orchestration's job.
"""
from __future__ import annotations

import numpy as np

from sentinel.monitoring.events import record_drift_event


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, buckets: int = 10
) -> float:
    """PSI between two 1-D samples.

    <0.1 stable, 0.1-0.2 moderate shift, >0.2 significant drift.
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    # Bucket edges from reference quantiles (equal-frequency binning)
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:  # degenerate (near-constant reference)
        edges = np.linspace(ref.min(), ref.max() + 1e-9, buckets + 1)
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    # Convert to proportions, floor to avoid div-by-zero / log(0)
    ref_pct = np.clip(ref_counts / ref_counts.sum(), 1e-6, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def check_data_drift(
    reference: np.ndarray, current: np.ndarray, *, feature: str,
    threshold: float = 0.2, record: bool = True,
) -> dict:
    psi = population_stability_index(reference, current)
    drifted = psi > threshold
    if drifted and record:
        record_drift_event(
            drift_type="data", metric=f"psi:{feature}", value=psi,
            threshold=threshold, evidence={"feature": feature},
        )
    return {"drift_type": "data", "feature": feature, "psi": psi,
            "threshold": threshold, "drifted": drifted}


def check_prediction_drift(
    reference_preds: np.ndarray, current_preds: np.ndarray, *,
    threshold: float = 0.2, record: bool = True,
) -> dict:
    psi = population_stability_index(reference_preds, current_preds)
    drifted = psi > threshold
    if drifted and record:
        record_drift_event(
            drift_type="prediction", metric="psi:prediction", value=psi,
            threshold=threshold, evidence={},
        )
    return {"drift_type": "prediction", "psi": psi,
            "threshold": threshold, "drifted": drifted}


def check_concept_drift(
    reference_mae: float, current_mae: float, *,
    ratio_threshold: float = 1.25, record: bool = True,
) -> dict:
    """Concept drift = error has grown vs reference.

    Fires when current MAE exceeds reference MAE by more than the ratio
    (e.g. 1.25 = 25% worse). Requires labels - hence the late-label join.
    """
    ratio = current_mae / (reference_mae + 1e-9)
    drifted = ratio > ratio_threshold
    if drifted and record:
        record_drift_event(
            drift_type="concept", metric="mae_ratio", value=ratio,
            threshold=ratio_threshold,
            evidence={"reference_mae": reference_mae, "current_mae": current_mae},
        )
    return {"drift_type": "concept", "reference_mae": reference_mae,
            "current_mae": current_mae, "ratio": ratio,
            "threshold": ratio_threshold, "drifted": drifted}
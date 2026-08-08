"""Tests for drift detection - fires on drift, stays quiet on stable data."""
import numpy as np
import pytest

from sentinel.monitoring.detector import (
    check_concept_drift,
    check_data_drift,
    check_prediction_drift,
    population_stability_index,
)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


def test_psi_low_for_same_distribution(rng):
    ref = rng.normal(50, 10, 5000)
    cur = rng.normal(50, 10, 5000)
    assert population_stability_index(ref, cur) < 0.1


def test_psi_high_for_shifted_distribution(rng):
    ref = rng.normal(50, 10, 5000)
    cur = rng.normal(75, 18, 5000)
    assert population_stability_index(ref, cur) > 0.2


def test_data_drift_not_flagged_when_stable(rng):
    ref = rng.normal(50, 10, 5000)
    cur = rng.normal(50, 10, 5000)
    r = check_data_drift(ref, cur, feature="x", record=False)
    assert r["drifted"] is False


def test_data_drift_flagged_when_shifted(rng):
    ref = rng.normal(50, 10, 5000)
    cur = rng.normal(80, 25, 5000)
    r = check_data_drift(ref, cur, feature="x", record=False)
    assert r["drifted"] is True


def test_prediction_drift_flagged_when_shifted(rng):
    ref = rng.normal(38, 20, 5000)
    cur = rng.normal(60, 35, 5000)
    r = check_prediction_drift(ref, cur, record=False)
    assert r["drifted"] is True


def test_concept_drift_flagged_when_error_grows():
    r = check_concept_drift(reference_mae=15.0, current_mae=21.0, record=False)
    assert r["drifted"] is True
    assert r["ratio"] == pytest.approx(1.4, abs=0.01)


def test_concept_drift_not_flagged_when_error_stable():
    r = check_concept_drift(reference_mae=15.0, current_mae=15.5, record=False)
    assert r["drifted"] is False
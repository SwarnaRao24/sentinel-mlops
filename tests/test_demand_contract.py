"""Tests for the demand data contract."""
import pandas as pd
import pytest

from sentinel.contracts.demand import DemandRecord, validate_dataframe


def _valid_row() -> dict:
    return {
        "zone_id": 100,
        "pickup_hour": pd.Timestamp("2024-01-15 08:00:00"),
        "demand": 42,
        "hour": 8,
        "day_of_week": 0,
        "is_weekend": False,
        "is_holiday": False,
    }


def test_valid_record_passes():
    DemandRecord(**_valid_row())  # should not raise


def test_valid_dataframe_all_pass():
    df = pd.DataFrame([_valid_row() for _ in range(5)])
    valid, quarantine = validate_dataframe(df)
    assert len(valid) == 5
    assert len(quarantine) == 0


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("demand", -1),        # negative demand
        ("zone_id", 9999),     # out-of-range zone
        ("zone_id", 0),        # below minimum
        ("hour", 24),          # impossible hour
        ("day_of_week", 7),    # out-of-range weekday
    ],
)
def test_bad_values_are_quarantined(field, bad_value):
    row = _valid_row()
    row[field] = bad_value
    df = pd.DataFrame([row])
    valid, quarantine = validate_dataframe(df)
    assert len(valid) == 0
    assert len(quarantine) == 1
    assert field in quarantine.iloc[0]["reject_reason"]


def test_pickup_hour_must_be_floored():
    row = _valid_row()
    row["pickup_hour"] = pd.Timestamp("2024-01-15 08:30:00")  # not on the hour
    df = pd.DataFrame([row])
    valid, quarantine = validate_dataframe(df)
    assert len(quarantine) == 1


def test_hour_must_match_timestamp():
    row = _valid_row()
    row["hour"] = 10  # disagrees with pickup_hour=08:00
    df = pd.DataFrame([row])
    valid, quarantine = validate_dataframe(df)
    assert len(quarantine) == 1


def test_mixed_batch_splits_correctly():
    good = _valid_row()
    bad = _valid_row()
    bad["demand"] = -5
    df = pd.DataFrame([good, bad, good])
    valid, quarantine = validate_dataframe(df)
    assert len(valid) == 2
    assert len(quarantine) == 1

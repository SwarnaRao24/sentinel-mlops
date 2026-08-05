"""Data contract for aggregated taxi demand records.

The contract is the enforcement boundary: any record that does not conform
is rejected (quarantined) with a reason, rather than flowing downstream.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator


class DemandRecord(BaseModel):
    """A single (zone, hour) demand observation."""

    model_config = {"extra": "forbid"}  # reject unexpected columns

    zone_id: int = Field(..., ge=1, le=265, description="TLC taxi zone LocationID (1-265)")
    pickup_hour: datetime = Field(..., description="Hour bucket the demand falls in")
    demand: int = Field(..., ge=0, description="Trip count in this zone-hour")
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Mon .. 6=Sun")
    is_weekend: bool
    is_holiday: bool

    @field_validator("pickup_hour")
    @classmethod
    def must_be_on_the_hour(cls, v: datetime) -> datetime:
        if v.minute or v.second or v.microsecond:
            raise ValueError("pickup_hour must be floored to the hour")
        return v

    @field_validator("hour")
    @classmethod
    def hour_matches_timestamp(cls, v: int, info) -> int:
        ph = info.data.get("pickup_hour")
        if ph is not None and ph.hour != v:
            raise ValueError(f"hour={v} disagrees with pickup_hour={ph}")
        return v


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe into (valid, quarantined).

    Quarantined rows get a 'reject_reason' column explaining the failure.
    """
    valid_idx: list[int] = []
    rejected: list[dict] = []

    for idx, row in df.iterrows():
        try:
            DemandRecord(**row.to_dict())
            valid_idx.append(idx)
        except ValidationError as e:
            rec = row.to_dict()
            reasons = [
                f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                for err in e.errors()
            ]
            rec["reject_reason"] = "; ".join(reasons)[:300]
            rejected.append(rec)

    valid_df = df.loc[valid_idx].reset_index(drop=True)
    quarantine_df = pd.DataFrame(rejected)
    return valid_df, quarantine_df

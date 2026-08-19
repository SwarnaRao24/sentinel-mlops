"""Aggregate raw TLC trips into hourly demand per zone, for a given month.

Usage:
    uv run python scripts/aggregate.py 2024-01
    uv run python scripts/aggregate.py 2024-07

Reads data/raw/yellow_tripdata_<YYYY-MM>.parquet and writes
data/processed/demand_<YYYY-MM>.parquet.
"""
import os
import sys

import pandas as pd

# Known US holidays by month (extend as needed). Missing months just get none.
HOLIDAYS_BY_MONTH = {
    "2024-01": {pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-15").date()},
    "2024-07": {pd.Timestamp("2024-07-04").date()},
}


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate.py <YYYY-MM>  (e.g. 2024-07)")
        sys.exit(1)

    month_str = sys.argv[1]                       # e.g. "2024-07"
    year, month = (int(x) for x in month_str.split("-"))

    raw_path = f"data/raw/yellow_tripdata_{month_str}.parquet"
    out_path = f"data/processed/demand_{month_str}.parquet"
    holidays = HOLIDAYS_BY_MONTH.get(month_str, set())

    df = pd.read_parquet(raw_path, columns=["tpep_pickup_datetime", "PULocationID"])
    before = len(df)

    # --- Cleaning: drop rows whose pickup isn't actually in the target month ---
    ts = df["tpep_pickup_datetime"]
    mask = (ts.dt.year == year) & (ts.dt.month == month)
    df = df[mask].copy()
    dropped = before - len(df)
    print(f"Dropped {dropped} rows ({dropped/before:.2%}) with out-of-range pickup dates")

    # --- Aggregate: trips per (zone, hour) ---
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.floor("h")
    agg = (
        df.groupby(["PULocationID", "pickup_hour"])
        .size()
        .reset_index(name="demand")
        .rename(columns={"PULocationID": "zone_id"})
    )

    # --- Calendar features ---
    h = agg["pickup_hour"]
    agg["hour"] = h.dt.hour
    agg["day_of_week"] = h.dt.dayofweek
    agg["is_weekend"] = agg["day_of_week"].isin([5, 6])
    agg["is_holiday"] = h.dt.date.isin(holidays)

    agg = agg.sort_values(["pickup_hour", "zone_id"]).reset_index(drop=True)

    os.makedirs("data/processed", exist_ok=True)
    agg.to_parquet(out_path, index=False)

    print(f"\nWrote {len(agg)} rows to {out_path}")
    print("Demand stats:")
    print(agg["demand"].describe())


if __name__ == "__main__":
    main()

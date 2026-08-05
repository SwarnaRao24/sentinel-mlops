import pandas as pd

RAW_PATH = "data/raw/yellow_tripdata_2024-01.parquet"
OUT_PATH = "data/processed/demand_2024-01.parquet"
YEAR, MONTH = 2024, 1

# US holidays in the target month (extend as needed)
HOLIDAYS = {pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-15").date()}

def main():
    df = pd.read_parquet(RAW_PATH, columns=["tpep_pickup_datetime", "PULocationID"])
    before = len(df)

    # --- Cleaning: drop rows whose pickup isn't actually in the target month ---
    ts = df["tpep_pickup_datetime"]
    mask = (ts.dt.year == YEAR) & (ts.dt.month == MONTH)
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
    agg["day_of_week"] = h.dt.dayofweek           # 0=Mon .. 6=Sun
    agg["is_weekend"] = agg["day_of_week"].isin([5, 6])
    agg["is_holiday"] = h.dt.date.isin(HOLIDAYS)

    agg = agg.sort_values(["pickup_hour", "zone_id"]).reset_index(drop=True)

    import os
    os.makedirs("data/processed", exist_ok=True)
    agg.to_parquet(OUT_PATH, index=False)

    print(f"\nWrote {len(agg)} rows to {OUT_PATH}")
    print("\nSchema:")
    print(agg.dtypes)
    print("\nSample:")
    print(agg.head(5).to_string())
    print("\nDemand stats:")
    print(agg["demand"].describe())

if __name__ == "__main__":
    main()

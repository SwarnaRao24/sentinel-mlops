import pandas as pd

df = pd.read_parquet("data/raw/yellow_tripdata_2024-01.parquet")
print("Shape:", df.shape)
print("\nColumns and dtypes:")
print(df.dtypes)
print("\nFirst rows:")
print(df.head(3).to_string())
print("\nDate range:")
print(df["tpep_pickup_datetime"].min(), "→", df["tpep_pickup_datetime"].max())

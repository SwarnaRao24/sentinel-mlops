import pandas as pd

from sentinel.contracts.demand import validate_dataframe

df = pd.read_parquet("data/processed/demand_2024-01.parquet")

valid, quarantine = validate_dataframe(df)
print(f"Total:       {len(df)}")
print(f"Valid:       {len(valid)}")
print(f"Quarantined: {len(quarantine)}")

# --- Prove the contract actually rejects bad data ---
bad = df.head(3).copy()
bad.loc[bad.index[0], "demand"] = -5          # negative demand
bad.loc[bad.index[1], "zone_id"] = 9999       # out-of-range zone
bad.loc[bad.index[2], "hour"] = 30            # impossible hour

v2, q2 = validate_dataframe(bad)
print(f"\nInjected 3 bad rows -> valid={len(v2)}, quarantined={len(q2)}")
if len(q2):
    print("\nQuarantine reasons:")
    for r in q2["reject_reason"]:
        print(" -", r)

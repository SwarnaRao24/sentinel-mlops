"""Demonstrate drift detection firing on drifted data, staying quiet on stable."""
import numpy as np
import pandas as pd

from sentinel.monitoring.detector import (
    check_concept_drift,
    check_data_drift,
    check_prediction_drift,
)
from sentinel.monitoring.events import init_events, open_events

init_events()
rng = np.random.default_rng(7)

# Reference window: the demand the model was trained on
df = pd.read_parquet("data/processed/demand_2024-01.parquet")
reference = df["demand"].to_numpy()

print("=" * 60)
print("SCENARIO A: stable current window (no real change)")
print("=" * 60)
# Current window resembles reference -> should NOT drift
stable_current = rng.choice(reference, size=5000, replace=True)
r1 = check_data_drift(reference, stable_current, feature="demand")
print(f"  data drift: PSI={r1['psi']:.4f} -> drifted={r1['drifted']}")

# Prediction distributions similar -> no drift
ref_preds = rng.normal(38, 20, 5000)
stable_preds = rng.normal(38, 20, 5000)
r2 = check_prediction_drift(ref_preds, stable_preds)
print(f"  pred drift: PSI={r2['psi']:.4f} -> drifted={r2['drifted']}")

# Error roughly unchanged -> no concept drift
r3 = check_concept_drift(reference_mae=15.0, current_mae=15.6)
print(f"  concept:    ratio={r3['ratio']:.3f} -> drifted={r3['drifted']}")

print(f"\n  open events after stable scenario: {len(open_events())}")

print("\n" + "=" * 60)
print("SCENARIO B: drifted current window (demand shifted up + wider)")
print("=" * 60)
# Demand genuinely shifts: higher volume, more spread
# Genuinely different SHAPE: mix in a heavy upper tail, not just a rescale
drifted_current = np.concatenate([
        reference * rng.uniform(1.3, 1.8, size=len(reference)),
        rng.gamma(6.0, 40.0, size=len(reference) // 2),  # new high-demand regime
    ])
r4 = check_data_drift(reference, drifted_current, feature="demand")
print(f"  data drift: PSI={r4['psi']:.4f} -> drifted={r4['drifted']}")

# Model's predictions now cluster differently
drifted_preds = rng.normal(60, 35, 5000)
r5 = check_prediction_drift(ref_preds, drifted_preds)
print(f"  pred drift: PSI={r5['psi']:.4f} -> drifted={r5['drifted']}")

# Error has grown 40% -> concept drift
r6 = check_concept_drift(reference_mae=15.0, current_mae=21.0)
print(f"  concept:    ratio={r6['ratio']:.3f} -> drifted={r6['drifted']}")

print(f"\n  open events after drift scenario: {len(open_events())}")

print("\n" + "=" * 60)
print("DRIFT EVENTS RECORDED (what orchestration will act on):")
print("=" * 60)
for e in open_events():
    print(f"  [{e['drift_type']:10}] {e['metric']:16} "
          f"value={e['value']:.3f} threshold={e['threshold']:.3f}")
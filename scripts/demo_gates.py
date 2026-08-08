"""Demonstrate the validation gates catching deliberately-bad challengers."""
import numpy as np

from sentinel.validation.base import GateContext, run_gates
from sentinel.validation.gates import PerformanceGate, SegmentGate, StabilityGate

rng = np.random.default_rng(0)
n = 1000
y_true = rng.gamma(2.0, 20.0, n)                      # demand-like
segments = rng.integers(1, 6, n)                       # 5 zones

# Champion: decent predictions (truth + moderate noise)
champion = y_true + rng.normal(0, 8, n)

gates = [PerformanceGate(min_improvement=0.0), SegmentGate(), StabilityGate()]

def evaluate(label, challenger):
    ctx = GateContext(champion_preds=champion, challenger_preds=challenger,
                      y_true=y_true, segments=segments)
    ok, results = run_gates(gates, ctx)
    print(f"\n=== {label} -> {'PROMOTE' if ok else 'BLOCKED'} ===")
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.gate_name}: {r.reason}")

# 1. A genuinely better challenger (less noise) -> should PROMOTE
evaluate("Better challenger", y_true + rng.normal(0, 4, n))

# 2. A worse challenger (more noise) -> PerformanceGate should BLOCK
evaluate("Worse challenger", y_true + rng.normal(0, 20, n))

# 3. Better on average, but wrecked on segment 3 -> SegmentGate should BLOCK
sneaky = y_true + rng.normal(0, 4, n)
sneaky[segments == 3] += 60                            # sabotage one zone
evaluate("Sneaky (segment 3 regressed)", sneaky)

# 4. Broken model emitting NaNs -> StabilityGate should BLOCK
broken = y_true + rng.normal(0, 4, n)
broken[:10] = np.nan
evaluate("Broken (NaN outputs)", broken)
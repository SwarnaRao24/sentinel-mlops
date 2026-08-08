"""Tests for the validation gates - proving each catches its failure mode."""
import numpy as np
import pytest

from sentinel.validation.base import GateContext, run_gates
from sentinel.validation.gates import PerformanceGate, SegmentGate, StabilityGate


@pytest.fixture()
def base_data():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.gamma(2.0, 20.0, n)
    segments = rng.integers(1, 6, n)
    champion = y_true + rng.normal(0, 8, n)
    return y_true, segments, champion, rng


def _ctx(y_true, champion, challenger, segments):
    return GateContext(champion_preds=champion, challenger_preds=challenger,
                       y_true=y_true, segments=segments)


def test_performance_gate_passes_better_model(base_data):
    y, seg, champ, rng = base_data
    better = y + rng.normal(0, 4, len(y))
    r = PerformanceGate().check(_ctx(y, champ, better, seg))
    assert r.passed


def test_performance_gate_blocks_worse_model(base_data):
    y, seg, champ, rng = base_data
    worse = y + rng.normal(0, 20, len(y))
    r = PerformanceGate().check(_ctx(y, champ, worse, seg))
    assert not r.passed


def test_segment_gate_blocks_segment_regression(base_data):
    y, seg, champ, rng = base_data
    sneaky = y + rng.normal(0, 4, len(y))
    sneaky[seg == 3] += 60  # wreck one segment
    r = SegmentGate().check(_ctx(y, champ, sneaky, seg))
    assert not r.passed
    assert "seg=3" in r.reason


def test_segment_gate_passes_uniform_improvement(base_data):
    y, seg, champ, rng = base_data
    better = y + rng.normal(0, 4, len(y))
    r = SegmentGate().check(_ctx(y, champ, better, seg))
    assert r.passed


def test_stability_gate_blocks_nan(base_data):
    y, seg, champ, rng = base_data
    broken = y + rng.normal(0, 4, len(y))
    broken[:10] = np.nan
    r = StabilityGate().check(_ctx(y, champ, broken, seg))
    assert not r.passed
    assert "non-finite" in r.reason


def test_stability_gate_blocks_distribution_blowup(base_data):
    y, seg, champ, rng = base_data
    blown = champ * 10  # mean/std explode
    r = StabilityGate().check(_ctx(y, champ, blown, seg))
    assert not r.passed


def test_run_gates_all_pass_promotes(base_data):
    y, seg, champ, rng = base_data
    good = y + rng.normal(0, 4, len(y))
    gates = [PerformanceGate(), SegmentGate(), StabilityGate()]
    ok, results = run_gates(gates, _ctx(y, champ, good, seg))
    assert ok
    assert all(r.passed for r in results)


def test_run_gates_one_fail_blocks(base_data):
    y, seg, champ, rng = base_data
    broken = y + rng.normal(0, 4, len(y))
    broken[:5] = np.inf
    gates = [PerformanceGate(), SegmentGate(), StabilityGate()]
    ok, results = run_gates(gates, _ctx(y, champ, broken, seg))
    assert not ok  # any single failure blocks promotion

def test_shadow_gate_passes_consistent_challenger():
    from sentinel.validation.gates import ShadowGate
    batches = [{"champion_mae": 15, "challenger_mae": 12} for _ in range(5)]
    ctx = GateContext(champion_preds=np.array([]), challenger_preds=np.array([]),
                      y_true=np.array([]))
    r = ShadowGate(batches).check(ctx)
    assert r.passed


def test_shadow_gate_blocks_batch_regression():
    from sentinel.validation.gates import ShadowGate
    batches = [{"champion_mae": 15, "challenger_mae": 11} for _ in range(4)]
    batches.append({"champion_mae": 15, "challenger_mae": 20})  # one bad batch
    ctx = GateContext(champion_preds=np.array([]), challenger_preds=np.array([]),
                      y_true=np.array([]))
    r = ShadowGate(batches).check(ctx)
    assert not r.passed
    assert "1/5" in r.reason
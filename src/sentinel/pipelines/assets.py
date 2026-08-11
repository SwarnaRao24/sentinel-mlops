"""Dagster assets: the self-healing loop.

drift_status -> retrain_decision -> challenger_model -> gate_results -> promotion_outcome

Each asset is one stage. The retrain/gate/promote stages short-circuit when
no retrain is warranted, so 'no drift' costs nothing. Promotion happens only
if every gate passes.
"""
from __future__ import annotations

import numpy as np
from dagster import Definitions, asset

from sentinel.monitoring.events import open_events
from sentinel.pipelines.decision import should_retrain
from sentinel.pipelines.retraining import (
    MODEL_NAME,
    build_gate_context,
    holdout_mae,
    load_champion,
    train_challenger,
)
from sentinel.validation.base import run_gates
from sentinel.validation.gates import (
    PerformanceGate,
    SegmentGate,
    ShadowGate,
    StabilityGate,
)


@asset
def drift_status(context) -> dict:
    """Snapshot of open drift events."""
    events = open_events()
    context.log.info(f"{len(events)} open drift event(s)")
    return {"open_events": len(events),
            "types": [e["drift_type"] for e in events]}


@asset
def retrain_decision(context, drift_status: dict) -> dict:
    """Decide whether to retrain, based on drift evidence."""
    decision = should_retrain()
    context.log.info(f"retrain decision: {decision}")
    return decision


@asset
def challenger_model(context, retrain_decision: dict):
    """Train a challenger - only if a retrain was warranted."""
    if not retrain_decision["retrain"]:
        context.log.info("no retrain warranted; skipping challenger training")
        return None
    context.log.info("training challenger")
    model = train_challenger()
    context.log.info(f"challenger holdout MAE: {holdout_mae(model):.3f}")
    return model


@asset
def gate_results(context, challenger_model) -> dict | None:
    """Run all four validation gates on the challenger."""
    if challenger_model is None:
        return None

    champion = load_champion()
    ctx = build_gate_context(champion, challenger_model)

    # Build shadow batches by slicing the holdout into consecutive windows
    n = len(ctx.y_true)
    k = 5
    batches = []
    for i in range(k):
        sl = slice(i * n // k, (i + 1) * n // k)
        cm = float(np.mean(np.abs(ctx.y_true[sl] - ctx.champion_preds[sl])))
        hm = float(np.mean(np.abs(ctx.y_true[sl] - ctx.challenger_preds[sl])))
        batches.append({"champion_mae": cm, "challenger_mae": hm})

    gates = [PerformanceGate(), SegmentGate(), StabilityGate(), ShadowGate(batches)]
    all_passed, results = run_gates(gates, ctx)
    for r in results:
        context.log.info(f"[{'PASS' if r.passed else 'FAIL'}] {r.gate_name}: {r.reason}")
    return {"all_passed": all_passed,
            "results": [(r.gate_name, r.passed, r.reason) for r in results]}


@asset
def promotion_outcome(context, challenger_model, gate_results: dict | None) -> dict:
    """Promote or quarantine based on gate results, and write the audit trail."""
    if gate_results is None or challenger_model is None:
        context.log.info("nothing to promote (no challenger this run)")
        return {"action": "noop"}

    from sentinel.pipelines.promotion import execute_promotion
    from sentinel.validation.base import GateResult

    results = [GateResult(n, p, r) for n, p, r in gate_results["results"]]
    outcome = execute_promotion(challenger_model, results, model_name=MODEL_NAME)
    context.log.info(f"outcome: {outcome['action']} "
                     f"(failed: {outcome.get('failed_gates', [])})")
    return outcome

defs = Definitions(
    assets=[drift_status, retrain_decision, challenger_model,
            gate_results, promotion_outcome]
)
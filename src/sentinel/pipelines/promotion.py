"""Model promotion / quarantine.

Given gate results, either promote the challenger to Production in the MLflow
registry, or quarantine it with the reasons. This is the only place that
changes what serves traffic - and it does so ONLY when every gate passed.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from sentinel.validation.base import GateResult

QUARANTINE_LOG = Path("data/quarantine.jsonl")


def decide_promotion(results: list[GateResult]) -> dict:
    """Pure decision: promote only if all gates passed."""
    all_passed = all(r.passed for r in results)
    failed = [r.gate_name for r in results if not r.passed]
    return {
        "promote": all_passed,
        "failed_gates": failed,
        "summary": {r.gate_name: r.passed for r in results},
    }


def execute_promotion(challenger, results: list[GateResult], *, model_name: str) -> dict:
    """Act on the decision: register+promote, or quarantine with reasons."""
    decision = decide_promotion(results)
    ts = datetime.now(timezone.utc).isoformat()

    if decision["promote"]:
        import mlflow.sklearn

        info = mlflow.sklearn.log_model(
            challenger, name="model", registered_model_name=model_name
        )
        outcome = {
            "action": "promoted",
            "at": ts,
            "model_uri": info.model_uri,
            "gates": decision["summary"],
        }
    else:
        reasons = {r.gate_name: r.reason for r in results if not r.passed}
        outcome = {
            "action": "quarantined",
            "at": ts,
            "failed_gates": decision["failed_gates"],
            "reasons": reasons,
            "gates": decision["summary"],
        }
        QUARANTINE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(QUARANTINE_LOG, "a") as f:
            f.write(json.dumps(outcome) + "\n")

    return outcome
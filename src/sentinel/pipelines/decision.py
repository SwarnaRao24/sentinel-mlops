"""Retrain decision layer.

Reads open drift events and decides whether a retrain is warranted. This sits
BETWEEN detection and action on purpose: detection records evidence, this layer
weighs it, and only then does orchestration retrain. Keeping the decision
explicit (not buried in the detector or the pipeline) makes the policy
auditable and easy to change.
"""
from __future__ import annotations

from pathlib import Path

from sentinel.monitoring.events import DB_PATH, open_events


def should_retrain(db_path: Path = DB_PATH) -> dict:
    """Decide whether current open drift events justify a retrain.

    Policy (deliberately simple and explicit):
      - any concept-drift event  -> retrain (the model is measurably worse)
      - >= 2 open events of any kind -> retrain (corroborating signals)
      - otherwise -> hold
    """
    events = open_events(db_path)
    if not events:
        return {"retrain": False, "reason": "no open drift events", "events": 0}

    concept = [e for e in events if e["drift_type"] == "concept"]
    if concept:
        return {
            "retrain": True,
            "reason": f"concept drift present ({len(concept)} event(s))",
            "events": len(events),
        }
    if len(events) >= 2:
        return {
            "retrain": True,
            "reason": f"{len(events)} corroborating drift signals",
            "events": len(events),
        }
    return {
        "retrain": False,
        "reason": f"only {len(events)} non-concept event(s); holding",
        "events": len(events),
    }
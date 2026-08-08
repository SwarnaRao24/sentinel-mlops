"""Validation gate framework.

A Gate inspects a challenger model (optionally against the champion) and
returns a pass/fail with a reason. A challenger must pass ALL gates to be
promoted. Each gate checks exactly one failure mode.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GateResult:
    gate_name: str
    passed: bool
    reason: str
    details: dict = field(default_factory=dict)


@dataclass
class GateContext:
    """Everything a gate might need to make its decision."""

    # Predictions on a shared holdout set
    champion_preds: object          # np.ndarray
    challenger_preds: object        # np.ndarray
    y_true: object                  # np.ndarray
    # Optional per-row segment labels (e.g. zone_id) for slice checks
    segments: object | None = None  # np.ndarray | None


class Gate(ABC):
    """Base class for all validation gates."""

    name: str = "gate"

    @abstractmethod
    def check(self, ctx: GateContext) -> GateResult: ...


def run_gates(gates: list[Gate], ctx: GateContext) -> tuple[bool, list[GateResult]]:
    """Run all gates. Returns (all_passed, individual_results)."""
    results = [g.check(ctx) for g in gates]
    all_passed = all(r.passed for r in results)
    return all_passed, results
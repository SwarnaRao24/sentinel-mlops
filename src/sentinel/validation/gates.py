"""Concrete validation gates.

Each gate checks exactly one way a challenger can be secretly bad:
- PerformanceGate: is it actually better overall?
- SegmentGate:     is it better *everywhere that matters*, not just on average?
- StabilityGate:   are its outputs sane (no NaNs, no distribution blow-ups)?
"""
from __future__ import annotations

import numpy as np

from sentinel.validation.base import Gate, GateContext, GateResult


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


class PerformanceGate(Gate):
    """Challenger must beat champion on overall MAE by at least `min_improvement`."""

    name = "performance"

    def __init__(self, min_improvement: float = 0.0):
        self.min_improvement = min_improvement

    def check(self, ctx: GateContext) -> GateResult:
        y = np.asarray(ctx.y_true)
        champ_mae = _mae(y, np.asarray(ctx.champion_preds))
        chall_mae = _mae(y, np.asarray(ctx.challenger_preds))
        improvement = champ_mae - chall_mae
        passed = improvement >= self.min_improvement
        return GateResult(
            gate_name=self.name,
            passed=passed,
            reason=(
                f"challenger MAE {chall_mae:.3f} vs champion {champ_mae:.3f} "
                f"(improvement {improvement:+.3f}, required >= {self.min_improvement})"
            ),
            details={"champion_mae": champ_mae, "challenger_mae": chall_mae,
                     "improvement": improvement},
        )


class SegmentGate(Gate):
    """No key segment may regress by more than `max_regression`.

    Catches: better on average, worse on the segment that carries the business.
    """

    name = "segment"

    def __init__(self, max_regression: float = 0.0, min_segment_size: int = 30):
        self.max_regression = max_regression
        self.min_segment_size = min_segment_size

    def check(self, ctx: GateContext) -> GateResult:
        if ctx.segments is None:
            return GateResult(self.name, False, "no segments provided")

        y = np.asarray(ctx.y_true)
        champ = np.asarray(ctx.champion_preds)
        chall = np.asarray(ctx.challenger_preds)
        seg = np.asarray(ctx.segments)

        regressions = []
        for s in np.unique(seg):
            mask = seg == s
            if mask.sum() < self.min_segment_size:
                continue
            champ_mae = _mae(y[mask], champ[mask])
            chall_mae = _mae(y[mask], chall[mask])
            regression = chall_mae - champ_mae
            if regression > self.max_regression:
                regressions.append((s, regression, int(mask.sum())))

        passed = len(regressions) == 0
        if passed:
            reason = "no segment regressed beyond threshold"
        else:
            worst = sorted(regressions, key=lambda r: -r[1])[:3]
            reason = "segments regressed: " + ", ".join(
                f"seg={s} (+{r:.2f} MAE, n={n})" for s, r, n in worst
            )
        return GateResult(
            gate_name=self.name,
            passed=passed,
            reason=reason,
            details={"regressed_count": len(regressions)},
        )


class StabilityGate(Gate):
    """Challenger outputs must be sane: no NaN/inf, no gross distribution blow-up."""

    name = "stability"

    def __init__(self, max_mean_ratio: float = 3.0, max_std_ratio: float = 3.0):
        self.max_mean_ratio = max_mean_ratio
        self.max_std_ratio = max_std_ratio

    def check(self, ctx: GateContext) -> GateResult:
        chall = np.asarray(ctx.challenger_preds, dtype=float)

        if not np.all(np.isfinite(chall)):
            n_bad = int((~np.isfinite(chall)).sum())
            return GateResult(self.name, False,
                              f"{n_bad} non-finite predictions (NaN/inf)")

        champ = np.asarray(ctx.champion_preds, dtype=float)
        champ_mean, chall_mean = np.mean(champ), np.mean(chall)
        champ_std, chall_std = np.std(champ) + 1e-9, np.std(chall)

        mean_ratio = abs(chall_mean) / (abs(champ_mean) + 1e-9)
        std_ratio = chall_std / champ_std

        problems = []
        if mean_ratio > self.max_mean_ratio or mean_ratio < 1 / self.max_mean_ratio:
            problems.append(f"mean shifted {champ_mean:.1f} -> {chall_mean:.1f}")
        if std_ratio > self.max_std_ratio or std_ratio < 1 / self.max_std_ratio:
            problems.append(f"std shifted {champ_std:.1f} -> {chall_std:.1f}")

        passed = len(problems) == 0
        return GateResult(
            gate_name=self.name,
            passed=passed,
            reason="outputs stable" if passed else "; ".join(problems),
            details={"mean_ratio": mean_ratio, "std_ratio": std_ratio},
        )
[![CI](https://github.com/SwarnaRao24/sentinel-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/SwarnaRao24/sentinel-mlops/actions/workflows/ci.yml)
# Sentinel MLOps

**A self-healing ML serving platform that detects its own model degradation and safely retrains, validates, and promotes new models — with guardrails that block a worse model from ever reaching production.**

Most ML projects stop at training a model. In production, the hard part is what happens *after*: models silently degrade as the world shifts, and nothing tells you. Sentinel is built around that problem — the model is deliberately simple; the platform is the point.

---

## The idea in one run

Sentinel detected drift, retrained a challenger, ran it through four validation gates, and **refused to promote it** — because although the challenger was *better on average*, it had quietly regressed on specific zones:

```
[PASS] performance: challenger MAE 12.17 vs champion 15.01 (improvement +2.84)
[FAIL] segment:     zones regressed — zone 12 (+13.16 MAE), zone 106 (+9.23), zone 186 (+4.97)
[PASS] stability:   outputs stable
[PASS] shadow:      held up across all 5 shadow batches
→ outcome: QUARANTINED (failed gate: segment)
```

A naïve system would have promoted the challenger on the strength of its overall MAE. Sentinel caught that "better on average" was hiding a regression on the slices that matter — and blocked its own retrain. That refusal is the entire thesis of the project.

The decision is persisted for audit in `data/quarantine.jsonl`:

```json
{"action": "quarantined", "failed_gates": ["segment"],
 "gates": {"performance": true, "segment": false, "stability": true, "shadow": true},
 "reasons": {"segment": "segments regressed: seg=12 (+13.16 MAE), seg=106 (+9.23 MAE), seg=186 (+4.97 MAE)"}}
```

`performance: true, segment: false` — a timestamped record of a model that was better on average but blocked for hurting specific zones.

---

## Why this is hard (and what it demonstrates)

- **Contract-enforced data & serving** — bad data is rejected at the boundary with a specific reason, not discovered downstream. (Pydantic v2 contracts; quarantined rows carry the field and rule they violated.)
- **Point-in-time-correct late-label join** — predictions are reconciled with actuals that arrive *later* and *out of order*, without ever leaking a future label. This is the data-engineering backbone that makes concept-drift detection real rather than faked.
- **Drift detection separated from action** — the detector *records evidence*; a distinct decision layer weighs it; only then does orchestration retrain. Detection ≠ action, so retraining never fires on noise.
- **A multi-gate promotion harness** — performance, segment, stability, and shadow gates. A challenger must pass *all four* to be promoted; any single failure quarantines it with reasons. This is the safety core.
- **Full lineage** — every prediction is traceable to the exact feature values, model version, and git commit that produced it.

---

## Architecture

```
                 ┌─────────────┐
   requests ───▶ │  FastAPI    │ ── contract-validated predictions
                 │  serving    │      │
                 └─────────────┘      ▼
                              ┌──────────────────┐
                              │  prediction log  │  (features, output,
                              │  + late-label    │   model version,
                              │  reconciliation  │   git SHA, latency,
                              └──────────────────┘   actuals)
                                       │
                                       ▼
   drift_status ─▶ retrain_decision ─▶ challenger ─▶ gate_results ─▶ promotion_outcome
   (data /          (concept drift      (trained      (4 gates)       (promote to
    prediction /     or corroborating    only if                       registry, or
    concept)         signals → retrain)  warranted)                    quarantine)
   └──────────────────── orchestrated as Dagster assets ───────────────────────┘
```

**Stack:** Python 3.10 · FastAPI · MLflow (model registry) · PSI-based drift detection · Dagster (orchestration) · scikit-learn · Pydantic v2 · pytest · `uv` for dependency management. Domain data: NYC TLC taxi trip records, aggregated to hourly demand per zone.

---

## The validation gates

| Gate | Catches | Fails when |
|------|---------|------------|
| **Performance** | An outright worse model | Challenger MAE doesn't beat champion |
| **Segment** | "Better on average, worse where it matters" | Any key zone regresses beyond threshold |
| **Stability** | Broken outputs | NaN/inf predictions, or a distribution blow-up |
| **Shadow** | Batch-to-batch unreliability | Challenger regresses in any live-like batch |

A challenger is promoted **only if all four pass.** Quarantined models are logged with reasons for audit.

---

## Running it

```bash
# install deps
uv venv --python 3.10 && source .venv/bin/activate
uv sync

# prepare data + train the initial champion
uv run python scripts/aggregate.py
uv run python scripts/train_baseline.py

# run the test suite (38 tests)
uv run pytest -v

# launch the self-healing pipeline UI, then "Materialize all"
uv run dagster dev            # http://localhost:3000

# serve the model
uv run uvicorn sentinel.serving.app:app --port 8000
# interactive docs at http://localhost:8000/docs
```

---

## Testing

38 tests cover:

- **the data contract** — valid rows pass, invalid rows are quarantined with the field and rule they violated;
- **the late-label join** — point-in-time correctness (never uses a label before it's available) and re-runnability (late arrivals are picked up without reprocessing);
- **drift detection** — fires on real drift and stays silent on stable data, tested in both directions and per drift type;
- **all four gates** — each proven to block its specific failure mode. The segment-gate test guarantees the platform's headline safety property can never silently regress.

---

## Design decisions worth noting

- **Detection is separate from action.** A drift event is evidence, not a command. An explicit, auditable decision layer applies the policy (concept drift → retrain; a single weak signal → hold).
- **The model is deliberately boring.** A strong naïve seasonal baseline is included precisely because it is often hard to beat — checking against it is a discipline most projects skip. (Here the baseline MAE is ~12.5; the champion ~15.0 — a candid result the project surfaces rather than hides.)
- **Quarantine over silent failure.** A blocked model is recorded in `data/quarantine.jsonl` with the failing gate and reasons, and never reaches traffic. Safe-by-default.

---

*Built by Swarnamukhi Chintalapudi · [GitHub](https://github.com/SwarnaRao24) · [LinkedIn](https://www.linkedin.com/in/swarnamukhirchintalapudi) · [Portfolio](https://www.theswarnaraogroup.com)*
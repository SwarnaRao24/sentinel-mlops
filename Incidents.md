# Operating Incidents

Real observations from running Sentinel against live data, not synthetic
scenarios. Each entry: what I expected, what happened, what the system did,
and whether it was right.

---

## Incident 1 — The drift that wasn't (correct restraint)

**Date:** operating run against July 2024 data (champion trained on January 2024)

**Expectation:** Summer demand should differ enough from winter to register as
drift and trigger a retrain. Seasonality is the textbook drift example.

**What I measured:**
- Data drift, January → July demand distribution: **PSI = 0.0066** (threshold
  0.2). Nowhere near firing.
- Champion (trained on January) scored against July actuals:
  **MAE 15.12**, versus ~15.0 on the January holdout. A <1% difference.

**What the system did:** Nothing. No drift event, no retrain. The decision
layer had no evidence to act on, so it correctly held.

**Was it right?** Yes. The model generalized across seasons almost perfectly —
July error was within 1% of January error — so a retrain would have been
retraining on noise: burning compute and risking a *worse* model (see Incident
2 / the segment-regression case) for zero benefit. NYC taxi demand *structure*
(which zones are busy at which hours) turns out to be far more season-stable
than intuition suggests.

**Why this matters:** A drift detector that correctly *stays silent* is as
valuable as one that fires. Retrain-on-a-schedule systems would have retrained
here for nothing. The platform's restraint — driven by measured evidence, not
a calendar — is the point. This is the mirror image of the quarantine incident:
there the platform blocked a retrain that looked good but was bad; here it
declined a retrain that intuition said was needed but the data said wasn't.

**What I'd change:** Nothing about the behavior. But it did teach me my
threshold isn't the interesting variable here — the *concept-drift* signal
(error growth on reconciled actuals) is the one that would catch real
degradation, and it correctly showed none. If anything, this validated
separating detection from action: an auto-retrain-on-any-signal system would
have been more fragile.

---

## Incident 2 — [Real-drift case: to be filled after operating against a
disrupted month]

*Placeholder for the contrasting case — a month where the champion genuinely
degrades and drift fires for real reasons, so the full retrain→validate→promote
loop engages against real data.*
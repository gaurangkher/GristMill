# Experiment D — Teacher Availability & Offline Grace Period

**Status:** Proposed  
**Priority:** 5 of 6 (Medium)  
**Category:** Deployment Resilience  
**Relevant paper section:** §5 Experiments — Systems / §6 Discussion — Deployment

---

## Motivation

GristMill's core value proposition is reducing dependency on the teacher model. But the paper currently frames this as a cost and latency story — the grinder handles more queries over time, reducing API calls. It does not address the harder question: **what happens when the teacher becomes completely unavailable?**

Internet outages, API rate limits, and cloud provider incidents are real deployment conditions. A user who has trained their grinder for 2 weeks should be able to continue working productively during a 48-hour outage. This experiment tests whether GristMill delivers on that promise, and characterizes how gracefully quality degrades over time without teacher access.

It also surfaces a design question the current architecture may not answer: what happens to escalation-worthy queries that the Sieve flags but cannot route? Are they queued for later distillation, or lost?

---

## Core Question

> For how long can a mature grinder serve a user productively without teacher access, and how does the system handle queries that require escalation when the teacher is unreachable?

---

## Setup

### Phase 1: Grinder Maturation

- Run flywheel to cycle 10 (mature grinder, ~60–70% autonomy on domain-focused queries)
- Establish baseline: accuracy and autonomy rate with teacher available

### Phase 2: Teacher Unavailability Simulation

Simulate teacher unavailability by disabling all teacher API calls. Test three offline durations:

```
offline_durations = [1 day, 3 days, 7 days]
```

Simulate "days" by replaying N query batches without distillation updates (1 day ≈ 50 queries based on Experiment 5 cold-start parameterization).

During offline period, measure per-batch:
- Accuracy on routed queries (grinder-handled only)
- False confidence rate (grinder answers queries it should escalate)
- Queue depth (escalation-worthy queries held pending teacher return)

### Phase 3: Teacher Return

Re-enable teacher access. Measure:
- Time to process queued escalation queries
- Distillation catch-up rate (does the grinder recover from the missed cycles?)
- Accuracy 1, 3, 7 days after teacher return

### Offline Behavior Conditions

Compare two system behaviors during teacher unavailability:

| Condition | Behavior |
|---|---|
| **Hard fallback** | Grinder answers all queries; escalation queue silently discarded |
| **Graceful queue** | Escalation-worthy queries queued; user notified of degraded confidence |
| **Conservative mode** | Sieve threshold raised; grinder only answers very high-confidence queries |

---

## Metrics

| Metric | Phase | Description |
|---|---|---|
| **Accuracy over time** | 2 | Does quality degrade, hold, or improve during offline period? |
| **False confidence rate** | 2 | Grinder answers confidently but incorrectly on escalation-worthy queries |
| **Queue depth** | 2 | How many queries are held pending teacher return |
| **Recovery time** | 3 | Batches until accuracy returns to pre-offline baseline |
| **User experience proxy** | 2+3 | % of queries answered with confidence > threshold (regardless of correctness) |

---

## Expected Results

**Hypothesis:** A mature grinder (cycle 10) can serve ~70% of queries accurately during teacher unavailability, with quality degrading slowly on the 30% of queries it would normally escalate. The graceful queue condition outperforms hard fallback on accuracy after teacher return because missed distillation examples are processed retroactively.

| Condition | Day 1 accuracy | Day 7 accuracy | Post-return recovery |
|---|---|---|---|
| Hard fallback | ~85% | ~78% | 2–3 batches |
| Graceful queue | ~72% | ~72% | 1 batch (queued examples processed) |
| Conservative mode | ~92% | ~92% | Immediate (no degradation, but more "I don't know" responses) |

The tradeoff between conditions is a design choice the paper can make explicit.

---

## Implementation Notes

- Simulate teacher unavailability with a mock client that raises `TeacherUnavailableError` — do not actually disable network access
- The Sieve's escalation path needs a configurable fallback: queue, discard, or conservative mode
- Queue implementation: a simple JSONL file of `(query, timestamp, confidence)` tuples is sufficient
- "Days" are simulated as fixed query batches — document the queries-per-day assumption clearly in the paper

---

## Connection to Other Experiments

- **Requires:** Experiment 2 (mature grinder at cycle 10)
- **Interacts with:** Experiment B (distribution shift) — OOD queries during offline periods are worst case for false confidence
- **Motivates:** Graceful queue design as a GristMill architectural component, if not already present

---

## Open Questions

- Should the grinder attempt self-improvement during offline periods using its own outputs as pseudo-labels? This is risky (error amplification) but potentially interesting.
- How should the user be notified of degraded confidence in a real deployment? Is a confidence watermark on responses sufficient UX?
- Does offline duration interact with cold-start stage — is a cycle-5 grinder more or less resilient than a cycle-20 grinder during outages?

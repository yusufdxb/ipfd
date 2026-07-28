# Causal actionability

IPFD's Point of No Return is useful for debugging only if an alarm is evaluated
against a known disturbance. A correlation after the fact is not an actionable
warning.

The causal evaluation has three inputs:

- `t_alarm`: the first persistent detector alarm.
- `disturbance_onset`: the first step where a known perturbation was applied.
- `probe_stride`: how often the recovery oracle was sampled.

When probes are strided, IPFD reports an interval rather than a single precise
PoNR. For example, a PoNR at step 56 with stride 8 means the evidence supports
`49 <= PoNR <= 56`. An alarm is classified as:

| Relation | Meaning |
|---|---|
| `definitely_actionable` | After the disturbance and before the earliest possible PoNR. |
| `ambiguous_within_ponr_interval` | It falls inside the evidence interval. |
| `pre_disturbance` | It fired before the known perturbation and cannot receive causal credit. |
| `too_late` | It fired after the latest possible PoNR. |
| `no_alarm` | No persistent alarm was observed. |
| `no_ponr` | The recovery probe never established irrecoverability. |

The strict boolean `valid_actionable_warning` is true only for
`definitely_actionable`. This conservative rule prevents the headline metric
from claiming precision that the probe schedule does not support.

## Reproduce on the shipped learned-policy fixture

```bash
python3 scripts/evaluate_actionability.py \
  tests/fixtures/learned_teleport_rollout.npz \
  --disturbance-onset 56 \
  --probe-stride 8
```

The fixture's detector alarm occurs before the injected teleport at step 56, so
the result is deliberately `pre_disturbance`, not an actionable warning. This is
the honest negative control: IPFD localizes PoNR, but it does not pretend that an
alarm caused by a task phase is a causal detector of the later fault.

For a positive control, construct a rollout with an alarm after the disturbance
and before the earliest possible PoNR. The same evaluator will classify it as
`definitely_actionable`.

## Regression benchmarks

Two CPU-only benchmarks pin this contract in CI:

- [ACTIONABILITY_BENCHMARK.md](ACTIONABILITY_BENCHMARK.md) runs the four labeled
  relation cases per seed and asserts the classification is exact.
- [BASELINE_COMPARISON.md](BASELINE_COMPARISON.md) contrasts the conservative
  rule with a naive "any alarm before failure" rule.

Neither establishes simulator or learned-policy competence.

# Replay fidelity contract

## Purpose

IPFD audits which counterfactual conclusions a declared simulator
snapshot-and-restore mechanism can support. Its central rule is:

> Equality at restoration time does not establish downstream counterfactual validity.

This is an engineering conformance contract, not a universal determinism claim.
Every result is local to a named simulator build, environment, task, snapshot
protocol, continuation, horizon, action source, decision function, tolerance, and
provenance record.

The archived IPFD studies in `HISTORICAL_BASELINE.md` motivate the contract. They
remain immutable and do not establish that Isaac Lab, PhysX, MuJoCo, or simulators
as a class are invalid. IPFD v2 does not revive the archived Point-of-No-Return or
validity-envelope objectives.

## Terms

- **Uninterrupted branch:** the reference environment advanced without a restore.
- **Restored branch:** a separate environment restored from the captured snapshot.
- **Branch state:** the declared control step at which capture occurs.
- **Snapshot protocol:** the exact named set of state components and restore
  operations used by an adapter.
- **Continuation mode:** open-loop recorded-action replay or another explicitly
  named continuation.
- **Action source:** the origin and identity of the action sequence, such as a
  recorded trace, policy checkpoint, controller, or fixed fixture.
- **Decision function:** a user-declared Boolean conclusion evaluated from a
  trajectory record.
- **Numerical divergence:** a measured difference in numeric values, interpreted
  using a declared field-specific tolerance.
- **Semantic divergence:** a difference in a categorical or Boolean conclusion,
  such as contact mode, termination, success, or collision.
- **Independent cluster:** the declared sampling unit used for evidence adequacy.
  Correlated checkpoints from one trajectory are not independent samples.

Reference and restored branches must use the same model assets, initial branch
state, action tensors, control timing, and declared disturbance schedule unless the
configuration explicitly defines a different comparison.

## L0: restore equality

L0 compares the branches immediately after restore and before either branch
executes the next control action. The audit records separate comparisons for:

- exposed scene state;
- policy observations;
- privileged observations;
- task-manager state where available;
- controller targets;
- sensor state;
- simulation and task counters.

Each comparison retains raw numerical error measurements, the tolerance used, and
availability metadata. A self-contained live failure reproducer additionally
retains its snapshot and action values. Semantic fields are compared as exact
values unless the configuration defines a domain-specific equivalence relation.

L0 is `SUPPORTED` only when all required, available L0 fields satisfy their
declared comparisons and the configured evidence rule is met. It is `UNSUPPORTED`
when a required measured field fails. It is `INSUFFICIENT_EVIDENCE` when a required
field is unavailable, was not measured, or lacks enough independent evidence.

Passing L0 means only that the measured exposed state agrees. L0 never implies L1,
L2, L3, complete simulator-state restoration, or deterministic continuation.

## L1: one-step dynamics fidelity

L1 applies one identical action tensor to the uninterrupted and restored branches.
It compares:

- next exposed state;
- next policy and privileged observation;
- contact state;
- task-manager outputs;
- termination and truncation state;
- reward where relevant to the task.

The runner checks action-array equality before every required open-loop step. A
live failure reproducer retains the identical action values so the reduced case is
independently auditable. Adapter provenance records control decimation and
execution details that affect action delivery.

L1 reports two channels:

1. **Numerical:** per-field absolute and relative differences, raw measurements,
   and tolerance decisions.
2. **Semantic:** exact comparison of contact mode, termination, reward event, task
   event, or other declared categorical outputs.

A numerical tolerance cannot absorb a semantic disagreement. A contact-mode or
termination disagreement is reported even when a continuous state vector remains
within tolerance. Missing task outputs are explicit, never silently represented as
zero or equality.

## L2: finite-horizon open-loop trajectory fidelity

L2 replays an identical recorded action sequence from the branch state. Every
adapter must support audits at 1, 5, 10, 30, and 90 control steps. Configurations
may add horizons but may not silently omit the required set.

For every branch and horizon, L2 records:

- first numerical divergence step;
- first observation divergence step;
- first contact divergence step;
- maximum state error over the horizon;
- terminal state error;
- a per-step divergence-growth curve;
- action identity at every step;
- censoring, termination, and missing-data reasons.

The divergence step is the first control step after the branch where a declared
field exceeds its field-specific threshold. Raw errors are retained even when no
threshold is crossed. Observation and contact divergence are separate from the
aggregate numerical state divergence.

There is no universal IPFD tolerance. Each environment must declare tolerances for
the fields it compares. Tolerances are part of the result scope and cannot be
changed after results are inspected without producing a new audit configuration.
IPFD preserves raw measurements so later analyses can apply a different declared
tolerance without rewriting history.

L2 is `UNSUPPORTED` when an observed branch violates its declared finite-horizon
contract. It is `SUPPORTED` only for the audited horizons and acceptance rule. A
pass at 1, 5, or 10 steps does not support 30 or 90 steps. Censored or unavailable
horizons yield `INSUFFICIENT_EVIDENCE`, not agreement.

## L3: downstream decision fidelity

L3 is the primary scientific-integrity level. The user must declare at least one
named decision function before the audit. Examples include:

- task success or task failure;
- sustained grasp;
- stable locomotion;
- collision;
- recovery success;
- controller ranking;
- checkpoint ranking;
- intervention effectiveness.

A decision function consumes a complete `TrajectoryRecord` and returns a Boolean
conclusion. Its name, implementation identity or source digest, parameters,
required horizon, and missing-data policy are recorded in provenance.

For each declared function, IPFD records the uninterrupted decision, restored
decision, agreement, direction of disagreement, and evidence adequacy. Ranking and
intervention claims must be reduced to an explicitly declared Boolean comparison,
such as `controller_a_ranked_above_controller_b`.

An observed decision reversal makes the strict decision-equivalence contract
`UNSUPPORTED` for that scope. A probabilistic error budget is allowed only when the
configuration declares its maximum disagreement rate, confidence method, minimum
independent clusters, and aggregation rule before evaluation. Agreement without
the declared independent evidence yields `INSUFFICIENT_EVIDENCE`.

L0, L1, and L2 are diagnostics for L3. None may override an L3 disagreement.

## Contract verdicts

Every audited configuration returns exactly one result:

- `SUPPORTED`: the declared acceptance rule passed with all required evidence.
- `UNSUPPORTED`: at least one measured counterexample violated a required strict
  claim, or the declared aggregate failure rule was met.
- `INSUFFICIENT_EVIDENCE`: required state, horizons, decisions, provenance, or
  independent samples were unavailable or incomplete.

`INSUFFICIENT_EVIDENCE` is not a pass. `UNSUPPORTED` is not a universal simulator
verdict. `SUPPORTED` does not extend beyond the serialized scope.

The machine-readable scope must include at least:

```json
{
  "simulator": "...",
  "simulator_version": "...",
  "environment": "...",
  "task": "...",
  "snapshot_protocol": "...",
  "continuation_mode": "...",
  "horizon": 30,
  "action_source": "recorded_actions",
  "decision_function": "task_success",
  "tolerances": {
    "scene_state": {"absolute": 1e-8, "relative": 1e-8},
    "policy_observations": {"absolute": 1e-8, "relative": 1e-8},
    "contact_state": {"absolute": 1e-6, "relative": 1e-6}
  },
  "independent_cluster_key": "trajectory_seed",
  "hardware_and_software_provenance": "provenance.json"
}
```

The values above illustrate structure only. They are not default tolerances.

## Aggregation and support boundary

Per-branch records are the primary evidence. Summary aggregation must:

- preserve the number of independent clusters separately from branch count;
- report agreements and disagreements by horizon, regime, protocol, continuation,
  action source, and decision function;
- keep false-positive and false-negative decision directions separate;
- treat missing and censored records as missing, not as agreements;
- state the preregistered or configured acceptance rule;
- expose any change from `SUPPORTED` to `UNSUPPORTED` in regression reports;
- avoid statistical-significance language without adequate independent samples.

An overall result is the most restrictive required level verdict under the
configuration. L3 is always required when a scientific counterfactual conclusion
is claimed. A configuration that requests only L0 or L1 cannot claim decision
fidelity.

## Required outputs

One `ipfd audit --config audit.yaml` run writes:

- `audit_summary.json`;
- `per_branch_records.jsonl`;
- `fidelity_contract.json`;
- `provenance.json`;
- one concise HTML or Markdown report;
- a divergence visualization;
- a minimal reproducer when an L2 or L3 failure can be reduced.

Each file records a schema version. The summary references the digests of the
records, contract, provenance, report, visualization, and reproducer. Runtime
failures, invalid configurations, missing required evidence, and output-integrity
failures return a nonzero process status.

## Historical boundary

The archived Isaac Lab result may be imported as a read-only regression fixture.
Its sealed records and decisions must not be rewritten. Reproducing its L3
disagreement demonstrates that the v2 tool can express that historical scope. It
does not resurrect the archived recovery claim or support a general conclusion
about simulator reliability.

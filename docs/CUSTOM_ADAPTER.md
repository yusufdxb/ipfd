# Custom adapters

An IPFD adapter represents two paired simulator instances. Environment `0` is
the uninterrupted reference. Environment `1` is restored from environment `0`.
Both receive one paired action array in each `step` call.

A complete, runnable reference is in
[`examples/custom_adapter.py`](../examples/custom_adapter.py). From the repository
root:

```bash
python examples/custom_adapter.py
ipfd adapter-check examples.custom_adapter:make_adapter \
  --decision threshold_reached
ipfd audit --config examples/custom_adapter.yaml
```

## Required interface

```python
class ReplayAdapter:
    def reset(self, seed: int) -> ObservationRecord | None: ...
    def action(self, step: int, source, env_ids) -> ArrayLike: ...
    def capture(self, env_ids) -> Snapshot: ...
    def restore(self, snapshot: Snapshot, env_ids) -> None: ...
    def observe(self, env_ids) -> ObservationRecord: ...
    def step(self, actions) -> StepRecord: ...
    def decision(self, record: TrajectoryRecord, name: str) -> bool: ...
    def provenance(self) -> Mapping[str, object]: ...
```

Use the record types exported from `ipfd.fidelity.contracts`. Arrays use the
leading dimension for environment identity. Capture mutable arrays by value.
Never represent unavailable state as zero or an empty tensor. Name it in
`Snapshot.unavailable_components`, `ObservationRecord.unavailable`, and adapter
provenance.

`observe` must not advance physics. If observation refreshes lazy sensors or
derived state, apply the same refresh sequence to both instances and disclose it
in `sensor_refresh_behavior`.

`step` must put the action actually sent to the simulator in
`StepRecord.applied_actions`. The conformance suite compares it exactly with the
paired request. If applied actions cannot be observed, the delivery check is
`INSUFFICIENT_EVIDENCE`, not a pass.

## Capability provenance

`provenance()` must identify at least:

```python
{
    "adapter": "MyAdapter",
    "simulator": "MySimulator",
    "simulator_version": "1.2.3",
    "environment": "my_task_v1",
    "snapshot_protocol": "expanded_runtime_state",
    "state_components_captured": ["qpos", "qvel", "controller_state"],
    "state_components_unavailable": ["solver_cache"],
    "task_state_captured": ["termination_manager"],
    "controller_or_policy_history_captured": True,
    "random_state_handling": {"simulator_rng": "captured"},
    "solver_state_availability": {"warm_start": "unavailable"},
    "sensor_refresh_behavior": "no side effects",
    "unsupported_restoration_claims": ["renderer state"],
}
```

This inventory is part of the result. It is not a claim that the unavailable
components are irrelevant.

## Run conformance checks

From Python:

```python
from ipfd import check_adapter

report = check_adapter(my_adapter, decision="task_success")
print(report.verdict)
for check in report.checks:
    print(check.name, check.status, check.detail)
```

Or expose a zero-argument factory and use the CLI:

```bash
ipfd adapter-check my_lab.ipfd_adapter:make_adapter \
  --decision task_success \
  --output adapter-check.json
```

The standard artifact-producing audit CLI can load a trusted local factory:

```yaml
adapter:
  kind: python
  factory: my_lab.ipfd_adapter:make_adapter
  kwargs: {task: pick_cube}
```

`ipfd audit` imports and executes that Python target with the declared keyword
arguments. Treat configuration files as executable local code and use only
factories you trust.

The suite samples capture-by-value, repeated capture, explicit unavailable
state, protocol mismatch rejection, requested and applied paired actions, restore equality,
observation side effects, counter progression, provenance completeness, and
semantic-decision determinism. It returns exit code `0` only when every sampled
check passes, `1` for a conformance finding, and `2` for import, usage, or runtime
errors. Passing is evidence for the sampled seed and step, not certification of
the simulator.

For the complete data contract and field semantics, see
[`ADAPTER_CONTRACT.md`](../ADAPTER_CONTRACT.md).

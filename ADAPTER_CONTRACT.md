# Replay adapter contract

## Minimal simulator-neutral interface

Every backend implements this interface without hiding backend-specific missing
state:

```python
class ReplayAdapter(Protocol):
    def capture(self, env_ids: Sequence[int]) -> Snapshot: ...
    def restore(self, snapshot: Snapshot, env_ids: Sequence[int]) -> None: ...
    def observe(self, env_ids: Sequence[int]) -> ObservationRecord: ...
    def step(self, actions: ArrayLike) -> StepRecord: ...
    def decision(self, record: TrajectoryRecord, name: str) -> bool: ...
    def provenance(self) -> Mapping[str, object]: ...
```

Adapters may expose lifecycle helpers such as `reset`, `action`, and `close`, but
the six methods above define interoperability. Simulator-neutral means common
records and audit semantics, not a claim that all simulators expose the same
state.

## Record requirements

### `Snapshot`

A snapshot is treated as immutable after capture and contains:

- protocol name and schema version;
- source environment identifiers;
- capture control step and simulator time where available;
- copied state values for mutable simulator arrays, never intentional live views
  into simulator buffers;
- captured-component inventory;
- unavailable-component inventory with reasons;
- task, controller, policy-history, random-state, solver-state, and sensor-refresh
  disclosures through its inventories, metadata, and adapter provenance.

An unavailable component is not represented by an empty tensor, zero, or a
successful flag. Partial capture must be visible to the auditor.

### `ObservationRecord`

An observation record separates:

- exposed scene state;
- policy observations;
- privileged observations;
- task-manager state;
- controller targets;
- sensor state;
- simulation and task counters;
- unavailable or stale fields.

Reading an observation must not advance physics. If reading refreshes a lazy
sensor, cache, renderer, or derived-state pipeline, the adapter reports that side
effect and applies the same refresh sequence to both branches.

### `StepRecord`

A step record contains the next observation and state projection, contact state,
task outputs, termination and truncation fields, reward where relevant, and
semantic metadata. The enclosing trajectory and audit record store the ordered
applied actions. Numerical values and semantic events remain separate.

### `TrajectoryRecord`

A trajectory record contains ordered step records, ordered actions, the selected
environment role, and optional metadata. The enclosing audit record carries the
branch snapshot inventory, horizon, continuation mode, action source, and scope.
Decision functions receive the structured trajectory, not an unstructured
simulator object.

## Method behavior

### `capture`

`capture(env_ids)` must:

- capture exactly the requested environment instances;
- preserve environment ordering;
- clone mutable arrays and tensors;
- avoid stepping, resetting, or resampling;
- include a complete capability disclosure for the selected protocol;
- fail if a component declared as required by that protocol cannot be captured.

Process-global state, such as a shared RNG or simulation counter, must be labeled
process-global. It cannot be described as independently restorable per environment.

### `restore`

`restore(snapshot, env_ids)` must:

- reject a snapshot from a different protocol, model, or incompatible adapter;
- map source environments to destination environments explicitly;
- perform any required reset, state write, target write, cache refresh, and
  synchronization in a documented order;
- avoid stochastic resampling unless the protocol declares and restores it;
- report unsupported restoration claims instead of silently omitting state;
- raise on partial or failed restoration.

The primary CLI converts any adapter construction, capture, restore, step, or
output failure into a nonzero exit status.

### `observe`

`observe(env_ids)` returns the L0 or post-step measurement projection. It does not
claim that this projection is complete simulator state. It must indicate values
that are unavailable, stale, recomputed, or refreshed on access.

### `step`

`step(actions)` applies one control action to each configured branch. The auditor
must verify that uninterrupted and restored branches receive identical action
values at every open-loop step. Action values and order are retained in a live
minimal reproducer. Control decimation is adapter provenance. Automatic
environment resets must be disabled or surfaced as semantic events.

### `decision`

`decision(record, name)` is deterministic and side-effect free. Unknown names and
missing required trajectory fields raise errors. Every decision implementation
declares its parameters, required horizon, treatment of early termination, and
source or package identity. The adapter must not substitute a convenient task
predicate for the user-declared one.

### `provenance`

`provenance()` returns JSON-serializable adapter information. The audit runner
adds source, configuration, operating system, Python, dependency, and public-safe
hardware provenance. Together they include:

- adapter name, implementation version, and source identity;
- simulator name, exact version, and build identity where available;
- environment and task identifiers;
- model and asset identity, plus available policy or controller identity;
- the audit configuration digest and output artifact digests;
- operating system, Python and dependency versions;
- hardware class and relevant execution backend details;
- timestep, control decimation, solver, integrator, and deterministic settings;
- snapshot protocol and full capability disclosure;
- RNG handling and seed schedule;
- sensor refresh behavior;
- warnings, unavailable fields, and unsupported claims.

Hardware provenance must be detailed enough to distinguish regression runs while
following the repository's public-disclosure policy.

## Mandatory capability disclosure

Each adapter and protocol reports the following concepts. Concrete JSON field
names are `state_components_captured`, `state_components_unavailable`,
`task_state_captured`, `controller_or_policy_history_captured`,
`random_state_handling`, `solver_state_availability`,
`sensor_refresh_behavior`, and `unsupported_restoration_claims`.

| Field | Required content |
|---|---|
| `captured_components` | Exact state fields captured and restored |
| `unavailable_components` | Exact known state fields unavailable, with reason |
| `task_state` | Manager, reward, termination, command, event, and disturbance state status |
| `controller_history` | Targets, filters, delays, recurrence, observation history, and action history status |
| `random_state` | Python, NumPy, framework, simulator, device, and process/global scope status |
| `solver_state` | Warm-start, contact, friction, broadphase, scheduling, and support status |
| `sensor_refresh` | Captured buffers, stale state, refresh timing, and restore support |
| `unsupported_claims` | Contracts the adapter cannot support with this protocol |

Capability fields use `captured`, `unavailable`, `not_applicable`, or
`not_audited`. `not_audited` forces `INSUFFICIENT_EVIDENCE` for a contract that
requires the component.

## Isaac Lab adapter

The Isaac Lab adapter initially targets the archived
`Isaac-Lift-Cube-Franka-v0` environment and supports two named protocols.

### `scene_only`

Capture and restore through `InteractiveScene.get_state()` and `reset_to()`:

- articulation root pose and velocity;
- articulation joint position and velocity;
- rigid-object root pose and velocity;
- other entity state returned by the installed Isaac Lab version.

The protocol explicitly marks manager buffers, original articulation targets,
action and command buffers, observation/event history, RNG, policy history,
disturbance schedules, sensor internals, and PhysX internals unavailable unless a
version-specific adapter proves otherwise.

### `expanded_runtime_state`

Capture `scene_only` plus all supported task/runtime state that the adapter can
access and restore for the tested environment:

- manager buffers;
- articulation position, velocity, and effort targets;
- current, previous, raw, and processed action buffers where available;
- command values, timers, counters, metrics, and cached command state;
- observation-history buffers where configured;
- event-history and interval buffers where configured;
- termination and reward buffers;
- RNG state with exact process-global or per-environment scope;
- sensor measurements and refresh behavior;
- policy or controller history;
- disturbance schedules and applied-event state.

The initial generic adapter captures action and previous-action buffers, command
state and metrics, the episode-length buffer, articulation targets, and optional
policy history supplied by a configured action-provider hook. Observation and
event history, disturbance schedules, sensor internals, per-environment RNG, and
other manager buffers remain explicitly unavailable without adapter-specific
hooks. The runtime inventory reports that boundary. Private attribute access is
labeled by the implementation and must be validated against the installed
version. A field is not declared captured merely because it exists.

Known unsupported restoration claims include PhysX warm-start impulses, solver
contact manifolds, persistent contact caches, broadphase pair caches, per-contact
friction state, and internal GPU solver scheduling state when no supported
per-environment interface is available. This disclosure limits the contract. It is
not a defect allegation.

Isaac Lab's documented `InteractiveScene` state format covers scene entities and
lists articulation root pose, root velocity, joint position, and joint velocity:
[Isaac Lab `InteractiveScene` API](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.scene.html).

## MuJoCo reference adapter

The MuJoCo adapter is a reference implementation for expressing state contracts,
not a claim that MuJoCo is superior. It uses documented state bitfields and state
copy APIs:

- `minimal_visible`: the declared user-visible subset, normally time, `qpos`,
  `qvel`, actuator activation, and control for the model;
- `full_physics`: `mjSTATE_FULLPHYSICS`, with every included component reported;
- `integration_with_warmstart`: `mjSTATE_INTEGRATION`, with
  `qacc_warmstart` named and measured separately so its effect is visible;
- `cold_continuation`: a fresh or reset `MjData` restored only to the declared
  minimal subset, leaving warm-start state cold;
- `restored_continuation`: a compatible `MjData` restored with the selected full
  integration state.

In current MuJoCo documentation, `mjSTATE_INTEGRATION` includes the full set of
forward-dynamics inputs described in the state section, including
`qacc_warmstart`. The adapter must inspect the installed enum rather than assume a
fixed numeric mask. It uses `mj_stateSize`, `mj_getState`, `mj_setState`, or
`mj_copyState` as supported by the installed version.

Official references:

- [MuJoCo simulation state and state manipulation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#state-and-control)
- [MuJoCo state support functions](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#support)
- [MuJoCo `mjtState` API type](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjtstate)

The reference matrix contains a contact-free model and a contact-rich model. It
compares minimal and full protocols, cold and restored continuation, and the
specific contribution of warm-start values. Contact count/mode and numerical
contact values are reported separately.

MuJoCo provenance includes the package and library versions, model digest,
integrator, timestep, solver, iteration/tolerance settings, enabled or disabled
warm-start options, plugin state, callbacks, and any user-managed state outside
`MjData`.

## Adapter conformance tests

An adapter is reusable only if focused tests show that it:

1. captures by value and respects environment identifiers;
2. rejects incompatible snapshots;
3. exposes missing state and unsupported claims;
4. returns complete L0 and L1 records for declared available fields;
5. verifies identical action delivery across required horizons;
6. separates numerical and semantic differences;
7. produces stable JSON-serializable provenance;
8. supports at least one free-space positive control;
9. returns nonzero process status on runtime failure;
10. runs through the same `ipfd audit --config audit.yaml` entry point as the
    other adapter.

Adapter conformance does not imply that any particular snapshot protocol passes
L2 or L3.

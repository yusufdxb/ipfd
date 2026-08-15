# IPFD v2 benchmark protocol

## Purpose

This benchmark evaluates which conclusions a declared snapshot-and-restore
protocol supports. It separates restore-time equality from one-step dynamics,
finite-horizon open-loop replay, and downstream decisions. Equality at the
restore boundary is never interpreted as evidence of trajectory or decision
fidelity.

The benchmark is a conformance and regression suite, not a universal simulator
ranking. Every verdict is limited to its recorded simulator version,
environment, task, snapshot protocol, continuation mode, horizon, action source,
decision function, tolerances, independent seed clusters, and provenance.

## Primary command

Each case uses the same installed command:

```bash
ipfd audit --config benchmarks/mujoco_free_space.yaml
```

Replace the configuration path to run another matrix row. The sustained
integration regression must be run after the sustained minimal audit because it
reads that audit's summary as its declared baseline.

The complete MuJoCo 3.5 and archived Isaac evidence matrix is reproducible with
one primary command:

```bash
ipfd audit --config benchmarks/audit_matrix.yaml
```

The process must provide both MuJoCo and the Isaac archive's analysis
dependencies. Child configurations remain separately scoped and retain their
own results.

## Benchmark matrix

| Configuration | Regime | Snapshot protocol | Branch step | Main purpose |
|---|---|---|---:|---|
| `mujoco_free_space.yaml` | free space | `integration_with_warmstart` | 10 | All-level positive control and adapter check |
| `mujoco_intermittent_contact.yaml` | intermittent contact | `full_physics` | 40 | Collision transition and contact timing check |
| `mujoco_sustained_minimal.yaml` | sustained contact | `minimal_visible` | 150 | Deliberately narrow state contract |
| `mujoco_sustained_full_physics.yaml` | sustained contact | `full_physics` | 150 | Full documented physics state without warm-start state |
| `mujoco_sustained_integration.yaml` | sustained contact | `integration_with_warmstart` | 150 | Integration-state and warm-start protocol, plus protocol regression |
| `demo_filtered_minimal.yaml` | filtered actuator in floor contact | `minimal_visible` | 100 | Demo failure case: visible L0 match, delayed contact and decision disagreement |
| `demo_filtered_integration.yaml` | filtered actuator in floor contact | `integration_with_warmstart` | 100 | Demo improved protocol and bit-exact control |
| `isaac_lab_archived.yaml` | archived mixed contact phases | `expanded_runtime_state` | retained per record | Read-only detection of the preserved Isaac Lab decision-fidelity failure |

`isaac_lab_live_scene_only.yaml` is a hardware-dependent zero-action smoke audit
for the tested Isaac Lab task. It is kept outside the CPU/reference matrix so a
researcher without Isaac Lab or a compatible accelerator can still reproduce the
reference suite. Its result is scoped to the recorded runtime and is not used to
replace the archived decision-fidelity evidence.

`mujoco_free_space_3_8.yaml` repeats the free-space row under MuJoCo 3.8.1 and
compares its result with the MuJoCo 3.5.0 summary. It is intentionally separate
from `audit_matrix.yaml` because a Python process loads one MuJoCo library
version.

All configurations audit 1, 5, 10, 30, and 90 control steps. These are
post-branch action counts. A horizon of 1 therefore measures the first action
after restoration, not the restoration boundary itself.

## Regimes

### Regime A: free space

The free-space system has no intended active contact at the branch point. It is
the positive control for adapter wiring, state indexing, identical action
delivery, and deterministic continuation. Its downstream decisions check bounds,
unexpected collision, and forward progress. Passing this case demonstrates only
the declared free-space contract.

### Regime B: intermittent contact

The intermittent system creates a short collision event near the declared branch
time. It tests whether the restored branch preserves collision occurrence,
contact transition timing, bounds, and post-impact stability. It does not assume
that every numerically different collision history is scientifically meaningful.

### Regime C: sustained contact

The sustained system keeps contact active across the branch and continuation.
The same environment, actions, decision predicates, seeds, branch step,
horizons, and tolerances are used for the three snapshot protocols. This isolates
the restoration protocol as the declared comparison axis. The decisions cover
bounds, collision, sustained contact, and stable contact.

### Regime D: filtered-actuator floor contact

The demo system preloads a filtered actuator while a sphere remains in floor
contact, then changes the control identically in both branches. The narrow
protocol restores qpos, qvel, history, and control but explicitly omits actuator
activation and solver warm-start. Required visible fields agree at L0. The
omitted bundle produces a later numerical threshold crossing, contact-mode
disagreement, and reversal of the `remains_in_contact` decision. A mechanism
ablation captures actuator activation while still omitting solver warm-start and
removes the L2/L3 mismatch through h=90. Its exact derived-field L0 remains
unsupported. The integration protocol restores the full integration state and
is the improved-protocol control that passes every tested level.

Position and velocity use separate declared tolerances because they have
different units. The demo's `DEGRADED` h=30 display means L2 has crossed its
position tolerance while L3 still agrees; the underlying strict audit result for
that L2 claim is `UNSUPPORTED`.

## Independent trajectories

Each MuJoCo row uses seeds 101, 211, and 307 as three independent clusters. The
Isaac archive row uses all five preserved base-seed clusters: 101, 211, 307, 401,
and 503. Repeated horizons and branch records within one seed are correlated
measurements and must not be presented as independent samples.

The minimum cluster count is enforced in each configuration. A group below that
minimum returns `INSUFFICIENT_EVIDENCE`. The benchmark does not claim statistical
significance from three or five clusters.

## Actions and continuation

Live MuJoCo cases use the adapter's deterministic reference action source.
Uninterrupted and restored instances receive identical action arrays at every
open-loop continuation step. The configuration records the continuation mode,
and the audit fails if a mode requiring identical actions receives different
arrays.

The archived Isaac row selects only `exact_action` records. The archived JSONL is
read-only evidence with a fixed SHA-256 digest. It is not regenerated by the v2
audit and is not treated as a live Isaac Lab execution.

## Tolerances and raw measurements

There is no universal tolerance. Each YAML contains an explicit `default` plus
category-specific absolute and relative tolerances for scene state, policy and
privileged observations, task state, controller targets, sensors, counters,
contact state, task outputs, termination, and reward.

The free-space thresholds are tight because the system is contact-free. The
intermittent-contact thresholds allow small impact-scale floating differences.
All three sustained protocols declare exact numerical equality so the known
warm-start distinction exercises failure reduction without changing the
acceptance boundary between protocols. Isaac Lab thresholds reflect the numeric
scale and precision of the retained GPU study.
Raw maximum errors and first-divergence locations remain in machine-readable
records even when a tolerance classifies a comparison as passing.

## Contract levels and controls

- L0 compares exposed state immediately after restore. A pass means only that
  measured exposed fields agree.
- L1 applies one identical action and reports numerical and semantic differences
  separately.
- L2 replays the declared horizons and records first numerical, observation, and
  contact divergence, maximum and terminal state errors, and a growth curve when
  the source evidence retains one.
- L3 compares every declared downstream decision. L3 is the scientific integrity
  boundary for the named decision only.

The free-space case is the positive control. The three sustained-contact rows are
the protocol control. The archived Isaac row is a historical detection control:
it must expose the preserved downstream disagreements while also reporting that
the archive lacks some L1 fields, full divergence curves, raw snapshots, and raw
action sequences.

## Failure reduction

Live L2 or L3 failures trigger bounded reduction. The reducer first shortens the
continuation horizon and matching action prefix, then attempts earlier branch
times, alternate declared decisions, active-entity reductions, disturbance
schedule reductions, and snapshot-component reductions advertised by the
adapter. Each YAML fixes the trial budget and candidate branch times before the
result is observed.

A live minimal reproducer must retain the captured snapshot, identical action
sequence, first divergence, expected uninterrupted decision, restored decision,
assets, and versions. The archived Isaac conversion can select a smallest
preserved failing record, but it must label that result as not self-contained
because the historical study did not retain the raw snapshot and action values.

## Version and protocol regression

`mujoco_sustained_integration.yaml` declares
`../results/v2/mujoco_sustained_minimal/audit_summary.json` as its baseline. The
regression comparator pairs configurations by their stable comparison keys and
keeps the raw scopes. It reports L0 and L1 changes, whether divergence moves
earlier or later, decision-disagreement changes, and every transition from
`SUPPORTED` to `UNSUPPORTED`. The report also exposes the changed protocol or
version fields instead of hiding them in the pairing key.

`mujoco_free_space_3_8.yaml` uses the MuJoCo 3.5.0 free-space summary as its
version-regression baseline. The runner rejects a live audit when the version
declared in YAML differs from the version reported by the adapter, preventing a
configuration label from silently substituting for runtime evidence.

## Comparison with trivial alternatives

| Alternative | What it misses |
|---|---|
| Immediate observation equality | L1 dynamics, L2 divergence growth, and L3 decision reversal |
| One-step equality | Divergence that begins after the first continuation action |
| One fixed short horizon | Horizon-dependent support boundaries at 10, 30, or 90 steps |
| Manual trajectory comparison | Reproducible tolerances, provenance, scoped verdicts, regression pairing, and automatic reduction |

IPFD adds value only if the same command produces these structured levels,
machine-readable scopes, decision checks, regression results, and minimized
failures across adapters. If the maintained result collapses to one equality
assertion or one horizon condition, the v2 effort should stop.

## Outputs

Every audit writes under `results/v2/`:

- `audit_summary.json`
- `per_branch_records.jsonl`
- `fidelity_contract.json`
- `provenance.json`
- `REPORT.md`
- `divergence.svg`
- a minimal reproducer when retained evidence supports one
- `regression_report.json` when a regression baseline is declared

## Honest proof boundaries

- A YAML file is a declared experiment, not runtime evidence. A result exists
  only after the command succeeds and the output provenance is inspected.
- The MuJoCo cases show that IPFD can express different documented restoration
  contracts. They do not establish that MuJoCo is superior to another simulator.
- The archived Isaac study is a motivating case from one task, checkpoint,
  runtime, protocol, and tested distribution. It is not evidence that Isaac Lab,
  Isaac Sim, or PhysX is universally invalid.
- Expected divergence in contact dynamics is not, by itself, a simulator defect.
  An engineering defect claim requires a supported contract violation and a
  minimal reproducer against documented behavior.
- Passing L0 cannot support an L1, L2, or L3 conclusion. Passing one horizon or
  decision cannot support another.
- Hardware-dependent Isaac behavior remains unverified until a live audit runs on
  recorded hardware and software provenance. The archive converter makes no new
  hardware claim.

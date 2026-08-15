# IPFD: Counterfactual Fidelity Auditing for Robot Simulation

[![CI](https://github.com/yusufdxb/ipfd/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/ipfd/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

IPFD tests how long a restored simulator branch remains a trustworthy empirical
substitute for an uninterrupted rollout. It is designed for researchers who use
snapshots for branching rollouts, counterfactual policy evaluation, recovery
analysis, failure debugging, replay, checkpoint comparison, and offline
diagnostics.

The question is deliberately scoped:

> Under this restoration protocol, continuation mode, disturbance, task phase,
> decision predicate, and runtime, where has decision disagreement been observed
> as rollout horizon increases?

IPFD reports a **counterfactual fidelity curve**, an **observed empirical
frontier**, error direction, seed-aware uncertainty, protocol deltas, and a
fail-closed fidelity gate. It does not certify a simulator, prove that snapshots
are valid, or extrapolate beyond tested horizons and strata.

## Quick start

The fidelity analyzer is pure CPU and consumes JSONL branch-comparison records. It
does not import Isaac Lab.

```bash
python -m pip install -e '.[dev]'

ipfd fidelity \
  examples/fidelity_records.jsonl \
  --max-disagreement 0.25 \
  --group-by protocol,continuation,disturbance \
  --minimum-independent-seeds 2
```

The bundled input is synthetic and exists only to demonstrate the interface. It
is not simulator evidence. To audit the pooled primary estimand in the corrected
study from the committed, compressed decision records:

```bash
ipfd fidelity \
  results/fidelity/corrected_five_seed_decisions.jsonl.xz \
  --protocol expanded_runtime_state \
  --continuation exact_action \
  --predicate sustained_lift \
  --group-by protocol,continuation \
  --max-disagreement 0.05 \
  --provenance results/branch_validity/corrected_five_seed/study_provenance.json
```

This grouping keeps the protocol, continuation, and predicate fixed while pooling
the observed disturbance and phase categories. The output lists those pooled
categories explicitly. Add either field to `--group-by` to obtain narrower
curves.

For an exact paired comparison of the two restore protocols, omit the protocol
filter and add:

```text
--compare-protocols scene_plus_basic_manager_state,expanded_runtime_state
```

Use `--format json` for machine-readable output and `--output PATH` to write an
artifact. The artifact records the input SHA-256, grouping and filters, tolerance,
bootstrap configuration, and available Git provenance. Bootstrap sampling is
deterministic when `--bootstrap-seed` is fixed.
See [Counterfactual fidelity audits](docs/COUNTERFACTUAL_FIDELITY.md) for the data
contract, interpretation rules, API, and gate behavior.

## What the audit answers

| Question | Reported result |
|---|---|
| At which tested horizons was disagreement absent? | Counts and descriptive rates, worded as "no disagreement observed" |
| Where did disagreement first appear? | The first tested horizon with a disagreement |
| Where did a selected tolerance fail? | The first tested horizon above the tolerance |
| Is the transition localized? | A bracket between tested horizons, never an interpolated exact point |
| Does a richer protocol help? | Paired protocol deltas on comparable records |
| Which conditions are restore-sensitive? | Curves grouped by continuation, disturbance, phase, and predicate |
| Which way do errors run? | False-recoverable and false-unrecoverable counts |
| Is there enough independent evidence? | Independent seed count and a fail-closed gate decision |

`ACCEPT_OBSERVED_ENVELOPE` means only that the requested horizon is inside a
tested empirical envelope whose equal-weight seed-mean disagreement rate meets
the configured rule. The branch-level rate remains descriptive. The gate fails
closed for inadequate seed counts, changing branch support across horizons,
non-monotonic evidence, unseen strata, and untested horizons. None of its statuses
is a formal safety or simulator-correctness claim.

## Existing five-seed evidence

The corrected Isaac Lab study contains 5,328 decision-comparison rows generated
from 74 unique branch points, two restore protocols, two continuation modes, three
decision predicates, and six horizons. These are repeated measurements within only
**five independent seed groups**. The 5,328 rows are not 5,328 independent trials.

For the primary stratum, exact-action continuation with the `sustained_lift`
predicate, the richer `expanded_runtime_state` protocol produced this descriptive
curve:

| Horizon (control steps) | Disagreements | Branch-level rate |
|---:|---:|---:|
| 1 | 0 / 74 | 0.00% |
| 3 | 0 / 74 | 0.00% |
| 5 | 0 / 74 | 0.00% |
| 10 | 0 / 74 | 0.00% |
| 30 | 1 / 74 | 1.35% |
| 90 | 10 / 74 | 13.51% |

At a 5% disagreement tolerance, the observed empirical frontier is bracketed
between the tested horizons of 30 and 90 steps. This means no decision
disagreement was observed through horizon 10 in this sampled five-seed experiment,
one was observed at horizon 30, and the descriptive rate exceeded 5% at horizon
90. It does not mean snapshots were proven valid through 10 or 30 steps.

At horizon 90, disagreement appeared in three of the five seed groups. The
per-seed counts were 6/15, 3/15, 1/15, 0/14, and 0/15. That seed-level view is more
informative about replication than the pooled 10/74 alone.

### Restore protocol comparison

On the same primary stratum, each protocol contributed 444 rows across all six
horizons:

| Restore protocol | Disagreements | Rate | False recoverable | False unrecoverable |
|---|---:|---:|---:|---:|
| `scene_plus_basic_manager_state` | 18 / 444 | 4.05% | 0 | 18 |
| `expanded_runtime_state` | 11 / 444 | 2.48% | 0 | 11 |

The relative reduction was 38.9%, below the preregistered 50% positive-control
threshold. The archived study generator's 10,000-draw bootstrap resampled the
five seed groups and recorded a paired mean difference of -1.57 percentage points
with an empirical 95% seed-resampling interval of [-2.48, -0.67] percentage
points. With only five independent groups, this is weak effect-size evidence, not
a basis for a general validity claim. New audit artifacts identify their own
bootstrap algorithm and random seed.

Observed disagreement rates varied across disturbance and continuation
categories. Across all three predicates under the expanded protocol:

| Continuation | Disturbance | Disagreements | Error direction |
|---|---|---:|---|
| Exact action | `object_teleport` | 0 / 630 | none observed |
| Exact action | `gripper_open_interruption` | 39 / 702 | 1 false recoverable, 38 false unrecoverable |
| Closed-loop policy | `object_teleport` | 0 / 630 | none observed |
| Closed-loop policy | `gripper_open_interruption` | 32 / 702 | 17 false recoverable, 15 false unrecoverable |

This identifies the gripper interruption as more restore-sensitive in this study.
It does not isolate why. In particular, the data do not establish that unexposed
contact-solver state caused the residual disagreement.

The source artifacts and registered protocol are preserved under
[`results/branch_validity/corrected_five_seed/`](results/branch_validity/corrected_five_seed/)
and [`CORRECTED_EXPERIMENT_PROTOCOL.md`](CORRECTED_EXPERIMENT_PROTOCOL.md).
An exact-content xz copy of the 10.2 MB per-branch JSONL is committed as
[`results/fidelity/corrected_five_seed_decisions.jsonl.xz`](results/fidelity/corrected_five_seed_decisions.jsonl.xz).
Its decoded SHA-256 matches the immutable source digest in the artifact manifest.
The committed
[`corrected_five_seed_primary_audit.json`](results/fidelity/corrected_five_seed_primary_audit.json)
records the paired primary analysis, complete configuration, source identities,
and implementation revision.

## Why IPFD changed direction

IPFD originally tried to locate a physical "Point of No Return" by restoring a
saved state, running a recovery controller, and treating that branch verdict as a
statement about the uninterrupted episode. That interpretation required restored
branches to preserve the downstream decision they stood in for.

The assumption failed. In the initial three-seed cohort, exposed restored state,
the immediate policy observation, and replayed actions could match while the final
task decision still changed. The corrected five-seed study removed known design
confounds and still found 11 disagreements in 444 primary comparisons under the
expanded restore protocol. The expanded protocol had fewer observed errors, but
did not eliminate them or meet the preregistered improvement threshold.

That result invalidated the old PoNR interpretation. It also exposed a broader
engineering problem worth measuring: snapshot-based counterfactual workflows need
an empirical account of how fidelity changes with horizon and operating
conditions. IPFD now audits that account directly.

The negative result remains part of the repository record:

- [`ARCHIVED_NEGATIVE_RESULT.md`](ARCHIVED_NEGATIVE_RESULT.md) explains the failed
  assumption and stopping decision.
- [`SNAPSHOT_PROTOCOLS.md`](SNAPSHOT_PROTOCOLS.md) inventories what each tested
  protocol restores and omits.
- [`CLAIM_AUDIT.md`](CLAIM_AUDIT.md) and
  [`EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md) separate supported claims from
  rejected ones.
- [`HISTORICAL_BASELINE.md`](HISTORICAL_BASELINE.md) preserves the original tool
  and its evidence boundary.

The historical PoNR, detector, replay, and report code is retained for provenance
and regression coverage. Its PoNR values describe recovery outcomes on restored
branches. They must not be read as physical irrecoverability in the uninterrupted
episode.

## Two complementary audit paths

The new `ipfd fidelity` command analyzes compatible branch-comparison records and
answers horizon, protocol, disturbance, phase, continuation, predicate, and error
direction questions.

The existing simulator conformance path remains useful. `ipfd audit` evaluates
declared contracts from restore equality through downstream decision agreement:

- L0: measured equality immediately after restoration;
- L1: one-step dynamics fidelity;
- L2: identical-action finite-horizon trajectory fidelity;
- L3: agreement of a user-declared downstream decision.

Run the asset-free MuJoCo reference audit:

```bash
python -m pip install -e '.[mujoco]'
ipfd audit --config benchmarks/mujoco_free_space.yaml
```

Run the preserved matrix of live MuJoCo cases and archived Isaac evidence:

```bash
ipfd audit --config benchmarks/audit_matrix.yaml
```

The matrix requires the immutable archived Isaac per-branch artifact named in its
manifest. A clean clone without that external payload can run the live MuJoCo
cases independently. Contract and adapter details are in
[`REPLAY_FIDELITY_CONTRACT.md`](REPLAY_FIDELITY_CONTRACT.md),
[`ADAPTER_CONTRACT.md`](ADAPTER_CONTRACT.md), and
[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md).

The historical rollout analyzer remains available:

```bash
ipfd analyze rollout.npz --report report.json --plot timeline.png
```

## Statistical discipline

Branch-level rates are descriptive. Seed groups are the independent experimental
unit in the corrected study because branch points, horizons, continuations, and
predicates within a seed share simulator history and experimental conditions.

IPFD therefore:

- reports row counts and independent seed counts separately;
- exposes per-seed disagreement rates;
- resamples whole seeds, not individual rows, for cluster bootstrap summaries;
- does not present comparison-level binomial intervals as independent-trial
  uncertainty;
- marks small-seed results as limited evidence;
- treats an all-zero sample as "no disagreement observed," never as proof of zero
  error;
- refuses unseen strata and horizons outside the measured range;
- rejects curves whose branch support changes across the queried horizons;
- labels non-monotonic curves instead of forcing a frontier.

A bootstrap interval such as [0, 0] from five all-zero seeds only reproduces the
absence of observed errors in those five clusters. It cannot describe failure
modes that were not sampled.

## Reproduce the corrected evidence

Run the new seed-aware audit from a clean clone:

```bash
ipfd fidelity \
  results/fidelity/corrected_five_seed_decisions.jsonl.xz \
  --continuation exact_action \
  --predicate sustained_lift \
  --group-by protocol,continuation \
  --compare-protocols scene_plus_basic_manager_state,expanded_runtime_state \
  --minimum-independent-seeds 5 \
  --max-disagreement 0.05 \
  --provenance results/branch_validity/corrected_five_seed/study_provenance.json
```

The historical analysis script still reproduces the archived stopping-rule
outputs when the uncompressed working artifact is present:

```bash
python3 scripts/analyze_snapshot_protocol_study.py \
  --study-dir results/branch_validity/corrected_five_seed
```

Regenerating the simulator study requires the declared Isaac Lab runtime, assets,
checkpoint, and a CUDA-capable machine:

```bash
OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=src "$IPFD_ISAACLAB_ROOT/isaaclab.sh" -p \
  scripts/run_snapshot_protocol_study.py \
  --checkpoint "$IPFD_CHECKPOINT" \
  --asset-root "$IPFD_ASSET_ROOT" \
  --output-dir /tmp/ipfd-corrected-five-seed \
  --isaac-lab-root "$IPFD_ISAACLAB_ROOT"
```

See [`CORRECTED_EXPERIMENT_PROTOCOL.md`](CORRECTED_EXPERIMENT_PROTOCOL.md) before
running or modifying the experiment. A new run is new evidence and must not
overwrite the archived artifacts.

## Scientific limits

The corrected Isaac evidence covers one task, one robot, one learned checkpoint,
one simulator/runtime fingerprint, two incomplete restoration protocols, two
disturbances, and five independent seed groups on one machine. It has no hardware
validation.

An empirical fidelity envelope is conditional on all of those choices. It is not:

- a simulator-wide snapshot certificate;
- a causal effect estimate;
- evidence of sim-to-real transfer;
- proof of recovery, irrecoverability, or a physical Point of No Return;
- a formal safety result;
- a guarantee for untested policies, tasks, phases, disturbances, or horizons.

The strongest supported conclusion is narrower:

> In the tested Isaac Lab contact-rich manipulation setting, equality of exposed
> restored state, immediate observations, and recorded future actions did not
> guarantee equality of downstream task decisions. Observed disagreement varied
> across horizon, restoration protocol, disturbance, and continuation categories
> in the sampled five-seed experiment.

## Development

```bash
pytest
ruff check .
mypy src/ipfd
```

If the shell has sourced ROS, use `env -u PYTHONPATH pytest` to prevent ROS from
injecting its `launch_testing` plugin into test collection.

## License

MIT, see [`LICENSE`](LICENSE). `src/ipfd/oracles/pick_lift_sm.py` is vendored from
Isaac Lab under BSD-3-Clause, see
[`THIRD_PARTY_LICENSES.md`](THIRD_PARTY_LICENSES.md).

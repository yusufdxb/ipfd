# Counterfactual fidelity audits

This guide describes the CPU-side analysis behind `ipfd fidelity`. The analyzer
asks whether restored simulator branches preserve a declared decision from their
uninterrupted references as the continuation horizon increases.

It analyzes empirical comparisons. It does not establish that a simulator state
is complete, that a branch is physically interchangeable with its reference, or
that an observed envelope transfers to another runtime or task.

## Unit of analysis

One input row compares one restored branch with one uninterrupted reference at a
declared horizon and decision predicate. A useful record identifies:

| Field | Meaning |
|---|---|
| `protocol` | Restoration protocol applied to the candidate branch |
| `continuation` | How the branch was advanced, such as exact actions or closed-loop policy |
| `disturbance` | Declared intervention or disturbance family |
| `phase` | Task phase at the branch point |
| `predicate` | Boolean decision being compared |
| `horizon` | Positive number of continuation control steps |
| `cluster_id` | Independent experimental-cluster identifier |
| `branch_id` | Stable branch-point identifier |
| `actual_continuation_steps` | Executed continuation steps, which must equal `horizon` |
| `reference_decision` | Decision from the uninterrupted reference |
| `restored_decision` | Decision from the restored branch |

Inputs may be plain `.jsonl` or xz-compressed `.jsonl.xz`. The archived corrected
study predates the generic flat contract. Its `base_seed`,
`candidate_decision`, and `schedule_sha256` fields are accepted as aliases for
`cluster_id`, `restored_decision`, and optional `schedule_id`. If `decision_match`
is present, it must agree with the two decision values. The loader validates types,
derives disagreement and error direction from the decisions, and fails closed on
malformed or contradictory records. A producer should retain any additional
provenance fields even when the analyzer does not group on them.

The loader also expands IPFD's native L0-L3 audit records when they contain a
`levels.L3.decisions` mapping. Fields not represented by that older schema, such
as disturbance and phase, are labeled `UNSPECIFIED` rather than inferred. Mixed
schemas in one input file are rejected.

Rows must be unique at the declared comparison key. Repeated measurements within
one seed group remain correlated even when they use different branch points,
horizons, predicates, or continuations.

## Core terms

**Decision disagreement** means the restored branch and uninterrupted reference
produce different values for the declared predicate at the declared horizon. It
does not by itself identify the mechanism of divergence.

**False recoverable** means the restored branch returns `true` while the reference
returns `false`. That name assumes `true` denotes recovery or success. A recovery
analysis would be optimistic in this direction.

**False unrecoverable** means the restored branch returns `false` while the
reference returns `true`. That name also assumes `true` denotes recovery or
success. A recovery analysis would be pessimistic in this direction and could
place an apparent recovery boundary too early. For predicates with different
semantics, use the neutral JSON fields `reference_false_candidate_true` and
`reference_true_candidate_false`.

**Counterfactual fidelity curve** is the descriptive disagreement rate at each
tested horizon within one declared stratum.

**Observed empirical frontier** compares that curve with a user-selected
disagreement tolerance. It returns tested endpoints, not an interpolated time. A
bracket from 30 to 90 steps means the tested rate met the rule at 30 and exceeded
it at 90. It says nothing about the unmeasured horizons between them.

## CLI workflow

Run a grouped audit:

```bash
ipfd fidelity branch_records.jsonl \
  --max-disagreement 0.05 \
  --group-by protocol,continuation,disturbance \
  --minimum-independent-seeds 5 \
  --bootstrap-samples 10000 \
  --bootstrap-seed 20260729
```

The analyzer always keeps protocol, continuation, and predicate separate, even if
they are omitted from `--group-by`. This prevents incompatible conditions from
being hidden inside one pooled rate.

Available filters are:

```text
--protocol VALUE
--continuation VALUE
--disturbance VALUE
--phase VALUE
--predicate VALUE
```

Write a deterministic JSON artifact:

```bash
ipfd fidelity branch_records.jsonl \
  --max-disagreement 0.05 \
  --format json \
  --output fidelity-audit.json \
  --bootstrap-seed 20260729
```

Add an exact paired restore-protocol comparison when both protocols share the
same branch, schedule, continuation, predicate, horizon, and reference evidence:

```bash
ipfd fidelity branch_records.jsonl \
  --compare-protocols basic_state,expanded_state \
  --format json
```

Use `--provenance study_provenance.json` to place that related file's logical
name, size, and SHA-256 in the manifest. The analyzer identifies the file but does
not infer unrecorded task or runtime facts from it.

The artifact records compressed-container and decoded-content SHA-256 identities,
analysis configuration, grouping dimensions, disagreement tolerance, bootstrap
method, confidence, random-number generator and seed, branch-comparison count,
and independent seed count. If a Git commit is available, it is provenance rather
than evidence that the input came from that revision.

## Reading a frontier

The frontier status is one of:

| Status | Meaning |
|---|---|
| `BRACKETED` | At least one tested horizon met the tolerance before a later tested horizon exceeded it |
| `ALL_TESTED_WITHIN_TOLERANCE` | Every tested rate met the tolerance, with no claim beyond the largest tested horizon |
| `NO_TESTED_HORIZON_WITHIN_TOLERANCE` | No tested rate met the tolerance |
| `NON_MONOTONIC` | The tested pass/fail pattern reverses, so one ordered frontier would misrepresent the data |
| `INSUFFICIENT_EVIDENCE` | The independent-seed requirement was not met |
| `INCOMPLETE_STRATUM_COVERAGE` | Branch support changes across horizons, so one ordered frontier would compare different cohorts |

An all-zero curve is `ALL_TESTED_WITHIN_TOLERANCE`, not proof that the true error
rate is zero. With only five seed groups, a seed bootstrap of an all-zero sample
will also return [0, 0]. That interval cannot include failure modes absent from the
sample.

## Fidelity gate

The fidelity gate answers whether a downstream consumer may use a requested
empirical envelope under explicit conditions. It fails closed:

| Gate result | Meaning |
|---|---|
| `ACCEPT_OBSERVED_ENVELOPE` | Requested horizon and stratum were tested, branch support was stable, and the equal-seed mean rate met the configured rule |
| `REJECT_HIGH_DISAGREEMENT` | Equal-seed mean disagreement exceeded the configured tolerance |
| `INSUFFICIENT_INDEPENDENT_SEEDS` | Too few independent seed groups were observed |
| `OUTSIDE_TESTED_HORIZON` | The request would require extrapolation |
| `UNSEEN_STRATUM` | No compatible evidence exists for the requested conditions |
| `INCOMPLETE_STRATUM_COVERAGE` | Branch support differs across the requested horizon prefix |
| `NON_MONOTONIC_EVIDENCE` | The pass/fail result reverses over increasing tested horizons |

Acceptance is local to the supplied evidence and configuration. Applications
should retain the complete gate result and manifest rather than reduce it to a
boolean.

## Seed-aware reporting

IPFD separates three different quantities:

1. **Decision-comparison rows** describe how often disagreement occurred in the
   recorded branch evaluations.
2. **Independent seed groups** describe the replication basis for uncertainty.
3. **Per-seed rates** show whether a pooled result is broad or dominated by one
   seed.

The seed-cluster bootstrap resamples whole seeds with replacement and recomputes
the statistic over every selected seed's rows. It does not resample individual
rows. This preserves within-seed dependence in the resampling unit, but it cannot
make a five-seed study well powered.

Unequal rows per seed are reported, not silently balanced or treated as additional
independent evidence. A protocol comparison uses paired records where the declared
keys match. Unpaired or ambiguous records fail validation instead of being folded
into an unpaired comparison.

The gate and frontier compare the tolerance with the arithmetic mean of per-seed
disagreement rates, giving each independent seed equal weight. The pooled branch
rate is retained as a descriptive statistic. Neither rule creates strong
inference from five seeds, and the bootstrap interval is a descriptive
seed-resampling interval rather than a simulator certificate.

## Public Python API

The public analysis surface is available from `ipfd.fidelity`:

| Function | Purpose |
|---|---|
| `load_branch_comparisons` | Parse and validate compatible JSONL records |
| `audit_counterfactual_fidelity` | Build grouped envelopes, seed uncertainty, and observed frontiers |
| `fidelity_envelope` | Compute one horizon-dependent descriptive envelope |
| `compare_restore_protocols` | Compute paired restore-protocol deltas |
| `evaluate_fidelity_gate` | Evaluate a request without extrapolation |
| `build_evidence_manifest` | Identify source content, code state, and analysis configuration |

The functions are deterministic for fixed input and bootstrap seed and require no
simulator installation. Use the command-line JSON output when a stable serialized
artifact is preferable to in-process objects.

## What requires new simulator experiments

CPU analysis can re-stratify the existing records, change a descriptive tolerance,
or compare already sampled protocols. It cannot add independent evidence or fill
an unseen stratum.

Claims about another simulator version, restoration protocol, robot, task,
checkpoint, disturbance, task phase, continuation controller, horizon, or machine
require new branch comparisons under a declared protocol. Mechanism claims about
contact caches, solver state, scheduling, or nondeterminism require interventions
that isolate those mechanisms. The current data do not do that.

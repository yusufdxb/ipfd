# Historical baseline for IPFD v2

This document fixes the historical boundary for **IPFD: Intervention Probe
Fidelity Diagnostics**. IPFD v2 is a new engineering objective: a reusable
conformance and regression-testing system for simulator snapshot-and-restore
contracts. It is not a continuation, rehabilitation, or relabeling of the failed
Point-of-No-Return claim, the failed validity-envelope predictor, or the archived
recovery-analysis framing.

The archived studies are a motivating case for the engineering system. They are
not evidence that Isaac Lab, PhysX, MuJoCo, or simulators in general are invalid.
No v2 result may be used to issue an unscoped simulator verdict.

## Verified repository lineage

Verification was performed read-only on 2026-08-02.

| Ref or role | Commit | Public remote | Local repository | Meaning |
|---|---|---:|---:|---|
| `archive/ipfd-honest-negative` | `7611cf6c5cdde29009265121b833547eb085c9ca` | exact match | exact match | Immutable negative-result archive |
| `main` | `e9035884f53bece3e8a28e92d3c0c61968d95fef` | exact match | exact match | Archival main and branch point for v2 |
| `research/ipfd-decision-calibration` | `7c4760e16664a58687edc1bbe51fd55ec031cfa1` | no public branch found | exact local commit | Stopped calibration successor |

Commit `7611cf6c5cdde29009265121b833547eb085c9ca` is an ancestor of both
`e9035884f53bece3e8a28e92d3c0c61968d95fef` and
`7c4760e16664a58687edc1bbe51fd55ec031cfa1`. The v2 branch
`v2/replay-integrity-contract` was created at the current archival `main` commit,
`e9035884f53bece3e8a28e92d3c0c61968d95fef`.

The public and local archive refs agree, and `git fsck --full --no-reflogs
--no-dangling` completed without an integrity error. The commits are not
cryptographically signed. Git reported signature status `N` for all three
commits, so the proof boundary is content-addressed object integrity plus the
point-in-time public/local ref match, not a signed-author identity claim.

No file under `results/branch_validity/` changed between the archive commit and
either archival `main` or the calibration successor. In particular, the archived
result, manifest, and provenance objects are byte-identical. Archival `main`
contains a later clarification to `EVIDENCE_LEDGER.md`, a portability-only change
to `scripts/run_snapshot_protocol_study.py`, and its focused tests. That commit
deliberately retained the generator digest from the archive and did not alter any
recorded result or provenance object.

## Original falsification

The original three-seed result is preserved in commit
`7611cf6c5cdde29009265121b833547eb085c9ca`:

- `results/branch_validity/clean_three_seed_report.json`
- `results/branch_validity/clean_three_seed_provenance.json`
- `results/branch_validity/summary.json`
- `results/branch_validity/decision_fidelity.png`
- `ARCHIVED_NEGATIVE_RESULT.md`, especially sections 3 and 8

In the tested `Isaac-Lift-Cube-Franka-v0` configuration, all 120 restored branches
matched the recorded immediate observation, but 13 of 120 produced a different
terminal decision. Ten of the 13 disagreements occurred under identical recorded
actions. This falsified the tested assumption that immediate exposed-state and
observation equality established downstream decision fidelity. It did not
falsify Isaac Lab or simulation as a general method.

The report digest is
`86189ab525c394d24d6e1b8b26427850ad5822bae9bd3789074e4c4985b9cad2`.
The retained trace digest is
`dcf30076c60ddfea754ea7dd13e0db95377186d8f5c6febaff9aa8ddc5a11558`.
Both matched their recorded byte counts and SHA-256 values.

## Corrected five-seed study and first stopping decision

The corrected independent-cluster study is also preserved in commit
`7611cf6c5cdde29009265121b833547eb085c9ca`:

- `CORRECTED_EXPERIMENT_PROTOCOL.md`
- `SNAPSHOT_PROTOCOLS.md`
- `results/branch_validity/corrected_five_seed/artifact_manifest.json`
- `results/branch_validity/corrected_five_seed/raw_worker_manifest.json`
- `results/branch_validity/corrected_five_seed/study_provenance.json`
- `results/branch_validity/corrected_five_seed/protocol_comparison.json`
- `results/branch_validity/corrected_five_seed/protocol_strata.json`
- `results/branch_validity/corrected_five_seed/validity_gate_results.json`
- `results/branch_validity/corrected_five_seed/downstream_decision_results.json`
- `results/branch_validity/corrected_five_seed/decisive_study.png`

The study contains 5,328 per-branch records grouped by five independent base-seed
clusters. For the preregistered exact-action `sustained_lift` comparison, Protocol
A recorded 18 disagreements in 444 pairs and Protocol B recorded 11 in 444. The
38.9 percent relative reduction missed the preregistered 50 percent threshold.
The preserved stopping fields are:

```text
stopping_rule.decision = STOP_BRANCH_VALIDITY_DIRECTION
stopping_rule.gate_eligible = false
stopping_rule.positive_control_meaningfully_improved = false
```

Consequently, validity-gate and downstream-correction stages are preserved as
`NOT_RUN_STOPPING_RULE`. No corrected Point-of-No-Return, controller-ranking, or
checkpoint-ranking conclusion was demonstrated.

The per-branch record digest is
`1c55862ce4a24bf564396c8d86873b13fc1667e491aada27bfff6c8d2166ce8f`.
The primary protocol-comparison digest is
`c41fb92be42bbb57f50eb6ba74a98bcd3d05c37f8ebdf32de8fa4be95f8817fa`.

## Failed calibration successor and second stopping decision

The validity-envelope successor is preserved in local commit
`7c4760e16664a58687edc1bbe51fd55ec031cfa1`:

- `DECISION_CALIBRATION_CONTRACT.md`
- `SUCCESSOR_RESEARCH_QUESTION.md`
- `SUCCESSOR_EVIDENCE_LEDGER.md`
- `results/decision_calibration/artifact_manifest.json`
- `results/decision_calibration/calibration_report.json`
- `results/decision_calibration/validity_envelope.json`
- `results/decision_calibration/coverage_vs_admitted_error.png`
- `results/decision_calibration/naive_vs_calibrated_downstream.png`

The calibrated gate admitted no disagreements at 0.423 coverage, but it did not
beat the one-line horizon heuristic, which achieved 0.667 coverage with a 0.00338
admitted-error rate and still met the declared tolerance. The archive lacked the
recovery controller, recovery budget, predicate, and branch density needed for a
decisive downstream Point-of-No-Return experiment. The preserved successor
decision is:

```text
stopping_rule.decision = STOP_SUCCESSOR_DIRECTION
stopping_rule.gate_beats_simple_heuristics = false
stopping_rule.downstream_decision_corrected = false
```

The calibration report digest is
`691a8997898831786ce2ded60171c474bb34820eae22e4e1aefa7bc6fa528d1f`.
The validity-envelope digest is
`9ffc687df6b0136176406ba69fe12f7fd18644953aeb8eb59082d04c82897058`.
Both link to the same archived per-branch input digest, `1c55862c...`.

## Manifest and retained-evidence verification

The following checks passed without regenerating or modifying evidence:

| Evidence group | Result |
|---|---:|
| Corrected-study artifact manifest | 8 of 8 byte counts and SHA-256 values matched |
| External corrected-study worker manifest | 20 of 20 files matched |
| Calibration artifact manifest | 4 of 4 files matched their committed objects |
| Original report and retained trace | 2 of 2 matched |
| Historical replay fixture manifest | 4 of 4 files matched |
| Archived study configuration and calibration input linkage | 2 of 2 hashes matched |
| Archived study generators and learned checkpoint | 3 of 3 hashes matched |
| Retained oracle-equivalence manifest and its checkpoint | 3 of 3 hashes matched |

That is 46 successful byte-count, digest, or cross-manifest linkage checks with no
mismatch. Important anchors include:

- learned checkpoint: `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`
- corrected-study generator at the archive commit: `8d7ca4bc3e47c410fa6eae8a31525115c39ce97b5177ef80fb76948e529177cd`
- three-seed generator at the archive commit: `17b41305c418feb57550373ecbe1894652f11c4fd0164c55e3a181348589769e`
- oracle-equivalence result: `ac70b03f7048021041e8d17e8de74895588d9b22c4293604d541b8b860e194ed`
- oracle-equivalence trace: `c6420c39c590de6fc0171b1cc1d8efece79d3f6d61455c91c0c8e9e6788827de`

Three separately retained `learned-teleport` files have current SHA-256 digests,
but no colocated manifest binds them to a commit or generator. They are therefore
historical supporting material, not hash-sealed evidence for v2:

- JSON: `071e14f50dbe43df4cad4c4898f3aa6e01c089dab0b045cca3716bb49c7e1d23`
- NPZ: `de7903782e7c7fa43a6440ef599b396aa54537d5bb79c5ccbfad973ef9843463`
- PNG: `e022fecb06db3aac88b957e21f8096d22a64253cac600a20ca0dc2d20194927c`

## Boundary for IPFD v2

IPFD v2 may reproduce the archived Isaac Lab discrepancy as one regression
fixture, but it may not reinterpret that discrepancy as proof of physical
irrecoverability, a Point of No Return, a valid predictor, or a universal
simulator failure. The v2 object of study is the declared restoration contract:
what numerical trajectories and downstream decisions a named snapshot protocol
supports for a named environment, horizon, continuation mode, tolerance, and
software/hardware provenance.

Archived branches, results, manifests, provenance records, and stopping decisions
remain inputs only. They are not v2 implementation surfaces.

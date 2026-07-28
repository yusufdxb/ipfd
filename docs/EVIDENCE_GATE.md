# Release evidence gate

IPFD's learned-policy headline is fail-closed behind three JSON artifacts:

```bash
ipfd-evidence-gate \
  --competence artifacts/competence.json \
  --multiseed artifacts/multiseed.json \
  --actionability artifacts/actionability.json \
  --report artifacts/release-evidence.json
```

The command exits 0 only when every artifact is complete, internally consistent,
and above the configured thresholds. Missing fields, non-finite numbers,
duplicate actionability case IDs, mismatched tasks or checkpoint hashes, legacy
height-only predicates, missing raw probe repetitions, and incomplete runs fail.
All artifacts must identify the same IPFD version and clean Git commit.

## Competence artifact

`scripts/eval_checkpoint.py --json artifacts/competence.json` writes
`ipfd.competence.v1`. The default gate requires:

- `status: complete`
- success rate at least 0.80
- at least 32 evaluated environments
- `sustained_final_lift_v1` success held for at least 10 final steps
- a non-empty task identifier
- the checkpoint SHA-256
- an Isaac Lab, Isaac Sim, and PyTorch runtime fingerprint
- the IPFD version and clean 40-character Git commit

## Multi-seed recovery artifact

`scripts/verify_learned_policy.py --json ...` writes one
`ipfd.recovery_run.v1` record. Collect both `teleport` and `slip` for at least
five distinct seeds, then combine them:

```bash
python3 scripts/aggregate_recovery_runs.py artifacts/run-*.json \
  --output artifacts/multiseed.json
```

For each seed, the gate requires one expected-PoNR run and one no-PoNR negative
control. Every run must use the same task and checkpoint as the competence
artifact, trigger the disturbance, use a named non-legacy physical recovery
predicate, record at least three raw boolean probes per checkpoint, and keep the
measured reset-boundary env-0 pose delta within the configured tolerance. The
same runtime fingerprint must appear on every run. Expected-PoNR runs must
localize within ten steps of the known disturbance onset by default. Each run
must reference a unique saved-rollout SHA-256. The gate independently derives
PoNR from the raw repeated verdicts and requires the reported value to match.

## Actionability artifact

Create a manifest that points to saved simulator `.npz` rollouts, then derive the
artifact rather than entering detector outcomes by hand:

```bash
python3 scripts/build_actionability_evidence.py \
  artifacts/actionability-manifest.json \
  --output artifacts/actionability.json
```

`ipfd.actionability.v1` contains real Isaac Lab cases with:

- `status: complete`
- `source: isaac_lab`
- the same task and checkpoint SHA-256
- at least 20 uniquely identified cases
- a unique rollout SHA-256 and recorded seed for every case
- a declared `expected_relation` and code-derived `alarm_relation`

The default gate requires all declared relations to match and requires coverage
of pre-disturbance, actionable, ambiguous, too-late, and no-alarm cases. This
checks evidence classification integrity. It does not by itself prove that the
detector is useful; experimental labels and case selection still require review.

The CPU scripts in `benchmark_actionability.py` and
`benchmark_comparison.py` are labeled `source: synthetic_contract`. They test code
semantics in CI and are intentionally rejected as release evidence.

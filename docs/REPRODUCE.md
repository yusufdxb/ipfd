# Reproducing IPFD

This guide separates deterministic CPU compatibility checks from live simulator
evidence. The CPU path proves that recorded arrays still produce the same analysis.
It cannot prove that the historical recovery predicate was physically correct.

The historical live-validation fingerprint was the locally installed `isaaclab`
distribution 4.5.22, `isaacsim` 6.0.0.0, task
`Isaac-Lift-Cube-Franka-v0`, and NVIDIA's published `rsl_rl` checkpoint on one
CUDA GPU. That fingerprint is provenance, not a generally installable environment
specification.

## Part A: deterministic CPU compatibility

```bash
git clone https://github.com/yusufdxb/ipfd
cd ipfd
pip install -e ".[dev]"
pytest tests/test_replay_fixture.py -v
```

The archives in `tests/fixtures/` contain recorded observations, actions,
detector signals, and historical recovery verdicts. Their hashes are recorded in
[`tests/fixtures/manifest.json`](../tests/fixtures/manifest.json). The test reloads
them without a simulator and asserts that generated reports match the frozen JSON
byte-for-byte.

The historical fixture values are:

```text
learned_teleport: PoNR=56, t_failure=57, alarm-to-PoNR lead=0.72 s
learned_slip:     PoNR=None, t_failure=219
```

These values are compatibility targets only. The fixtures used a height-only
recovery predicate and are not current proof of physical recoverability.

## Part B: built-package contract

```bash
python3 -m pip install build
python3 -m build
python3 -m venv /tmp/ipfd-check
/tmp/ipfd-check/bin/pip install dist/*.whl
cp tests/fixtures/learned_teleport_rollout.npz /tmp/ipfd-check/
cd /tmp/ipfd-check
./bin/ipfd analyze learned_teleport_rollout.npz --report report.json --plot timeline.png
./bin/ipfd-demo --json demo.json
./bin/pip check
```

This proves that the wheel contains the advertised library and console commands.
CI repeats the same check outside the source tree and also installs the source
distribution.

## Part C: live GPU revalidation

Use a clean tagged checkout so every artifact can identify exact source. The
`artifacts/` directory is ignored by Git. Run the runtime smoke first:

```bash
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless
```

Evaluate the exact checkpoint first, then collect both the irrecoverable case and
the recoverable negative control for at least five distinct seeds. Each run uses
at least three repeated probes per checkpoint and writes raw verdicts:

```bash
CHECKPOINT=/path/to/model.pt
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/eval_checkpoint.py \
  --headless --checkpoint "$CHECKPOINT" --num_envs 64 --sustain_steps 10 \
  --json artifacts/competence.json

OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
  --headless --checkpoint "$CHECKPOINT" --probe --probe_repeats 3 \
  --failure teleport --seed 0 \
  --save_rollout artifacts/teleport-seed0.npz \
  --json artifacts/teleport-seed0.json

OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
  --headless --checkpoint "$CHECKPOINT" --probe --probe_repeats 3 \
  --failure slip --seed 0 \
  --save_rollout artifacts/slip-seed0.npz \
  --json artifacts/slip-seed0.json
```

Repeat for seeds 1 through 4, then combine the run records:

```bash
python3 scripts/aggregate_recovery_runs.py artifacts/*-seed*.json \
  --output artifacts/multiseed.json
```

Build the actionability artifact from uniquely hashed saved rollouts and a
reviewed manifest:

```bash
python3 scripts/build_actionability_evidence.py \
  artifacts/actionability-manifest.json \
  --output artifacts/actionability.json
```

The standalone replay-equivalence reproducer tests a narrower simulator property:

```bash
OMNI_KIT_ACCEPT_EULA=YES \
  python3 scripts/isaaclab_reset_to_contact_mre.py --headless
```

`REPLAY DIVERGENCE REPRODUCED` means exposed scene state round-tripped while the
continued trajectory diverged. It does not identify which unexposed simulator or
task state caused the difference.

## Proof boundary

| Check | What it proves |
|---|---|
| Replay fixture tests | Historical arrays remain analysis-compatible. |
| Wheel and sdist smoke tests | Published package artifacts contain working APIs and commands. |
| Runtime smoke | IPFD attaches to the tested live environment. |
| Repeated physical recovery runs | Recovery outcomes under the configured oracle, with raw repetitions. |
| Multi-seed evidence gate | Required positive and negative controls meet the declared thresholds. |
| Replay-equivalence reproducer | Exposed state restoration is insufficient for trajectory replay in the observed condition. |

See [REVALIDATION.md](REVALIDATION.md) and [EVIDENCE_GATE.md](EVIDENCE_GATE.md)
before promoting a learned-policy result to verified, and
[GPU_REPRODUCIBILITY.md](GPU_REPRODUCIBILITY.md) for what a live run must record
and what installing the package does not recreate.

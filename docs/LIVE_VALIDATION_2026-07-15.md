# Historical live validation manifest: 2026-07-15

This manifest records a historical single-machine run at commit
`4a0c34544b1e4145e3f6257aa75dee1d3ad1fed7`. It is retained for provenance.
It predates the physical recovery predicate and repeated raw probes, so the
learned-policy PoNR values below are compatibility history, not current evidence.

## Runtime fingerprint

| Component | Observed version |
|---|---|
| Isaac Python | 3.12.13 |
| `isaaclab` distribution | 4.5.22 |
| `isaaclab_tasks` | 1.5.11 |
| `isaaclab_rl` | 0.5.0 |
| `isaacsim` | 6.0.0.0 |
| `torch` | 2.10.0+cu128 |
| `rsl-rl-lib` | 5.0.1 |
| `gymnasium` | 1.2.1 |
| `warp-lang` | 1.12.0 |
| Accelerator | one CUDA GPU |

Task: `Isaac-Lift-Cube-Franka-v0`.

Published checkpoint SHA-256:
`fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`.

## Runtime smoke

Command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES <isaac-python> \
  scripts/verify_isaac_runtime.py --headless \
  --env_id Isaac-Lift-Cube-Franka-v0 --steps 16 --seed 0
```

Observed:

- environment creation and stepping succeeded;
- the policy observation group had shape `(36,)`;
- the action shape was `8`;
- `IPFD_RUNTIME_SMOKE: overall PASS`;
- process exit code `0`.

This established that the adapter attached to the observed runtime. It did not
validate PoNR semantics.

## Scripted dual-environment run

Command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES <isaac-python> \
  scripts/verify_pnor_grasped.py --headless
```

Historical observations:

- two vectorized environments;
- 160-step primary rollout;
- 51 reset-boundary probe measurements;
- maximum immediate env-0 pose delta across probe reset writes: `0.00e+00 m`;
- reported PoNR: step 138;
- `DUAL_PROBE_STATUS: overall_status: VERIFIED`.

The status label reflects the script at that commit. The run used the older
oracle semantics and was not repeated per checkpoint.

## Learned-policy run

Command shape:

```bash
OMNI_KIT_ACCEPT_EULA=YES <isaac-python> \
  scripts/verify_learned_policy.py --headless --use_pretrained \
  --probe --failure teleport
```

Historical observations:

- 58-step rollout;
- reported PoNR at step 56;
- observable failure at step 57;
- detector alarm at step 20;
- maximum immediate env-0 pose delta across probe reset writes: `0.0`;
- `IPFD_LEARNED_STATUS` reported PoNR detected.

The alarm preceded the injected disturbance and therefore did not receive causal
credit. The height-only predicate could also count an airborne object as recovered.
These facts are why the learned-policy result is under revalidation.

## CPU replay

The captured archives remain committed as analysis compatibility fixtures.
Current hashes are recorded in
[`tests/fixtures/manifest.json`](../tests/fixtures/manifest.json). CI proves that
loading those arrays still generates the frozen reports byte-for-byte.

## Current proof boundary

Verified from this manifest:

- adapter attachment on the recorded runtime;
- historical command outputs;
- fixture provenance and deterministic CPU replay.

Not verified by this manifest:

- a physically correct learned-policy recovery predicate;
- repeated-probe confidence;
- multi-seed behavior;
- natural policy failures;
- compatibility with another machine or Isaac Lab runtime.

See [REVALIDATION.md](REVALIDATION.md) for the current evidence requirements.

# scripts/ — the GPU evidence chain

These scripts are **not** part of the importable `ipfd` library (the library is
pure NumPy and runs in CI without a GPU). Each one is a self-contained,
runtime-verified experiment against a live Isaac Lab 4.5.22 Franka env
(`Isaac-Lift-Cube-Franka-IK-Abs-v0`). They are the evidence trail behind the
recovery-probe design that the packaged `ipfd.adapters.isaac_lab.collect_rollout`
now uses. The main [`README.md`](../README.md) narrates the full chain; this is the
one-line index.

Run pattern (all scripts):

```bash
OMNI_KIT_ACCEPT_EULA=YES ~/Sim/isaac-sim-venv/bin/python scripts/<name>.py --headless
```

| Script | What it establishes |
|---|---|
| `verify_isaac_runtime.py` | The adapter drives a live Franka env end-to-end; prints a machine-readable `IPFD_RUNTIME_COMPATIBILITY` block. |
| `verify_state_fidelity.py` | Single-step `save → reset_to` round trip is bit-exact. `STATE_RESTORE_FIDELITY: PASS`. |
| `verify_probe_transparency.py` | Root-causes post-grasp corruption to the PhysX contact cache. `root_cause: CONTACT_STATE_NOT_RESTORED`. |
| `verify_pnor_decoupled.py` | A single `reset_to` poisons a `num_envs=1` sim even across `env.reset()`. `reset_to_poisons_env: YES`. |
| `verify_multienv_isolation.py` | The poison is **per-env**: churning env 1 leaves env 0 bit-identical. `isolation_viable: YES`. |
| `verify_pnor_isolated.py` | Env-isolated probe (primary in env 0, probes in env 1) yields a meaningful PoNR. `overall_status: VERIFIED`. |
| **`verify_pnor_grasped.py`** | **Terminal demo** — the mechanic `collect_rollout` ships. Gripper-slip failure, `overall_status: VERIFIED`, live primary-integrity assertion (`max env-0 pose delta 0.00e+00` across 51 probe resets). **Reproduce this first.** |
| `verify_real_policy.py` | The superseded single-env approach; documents honestly why in-loop probing corrupts the primary after a grasp (`meaningful_pnor_detected: NO`). |

The scripted recovery oracle they share lives in the library at
[`ipfd/oracles/pick_lift_sm.py`](../src/ipfd/oracles/pick_lift_sm.py).

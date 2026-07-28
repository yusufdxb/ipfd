# Scripts: diagnostics and evidence producers

These scripts are **not** part of the importable `ipfd` library (the library is
pure NumPy and runs in CI without a GPU). Each one is a self-contained,
historical experiment against a live Franka environment whose local `isaaclab`
distribution reported 4.5.22
(`Isaac-Lift-Cube-Franka-IK-Abs-v0`). The legacy `verify_pnor_*` scripts record
design diagnostics, but their old status labels cannot satisfy the current
release gate. Current evidence is produced by `eval_checkpoint.py`,
`verify_learned_policy.py`, `aggregate_recovery_runs.py`, and
`build_actionability_evidence.py`.

Run pattern (all scripts):

```bash
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaac-lab/python scripts/<name>.py --headless
```

| Script | What it establishes |
|---|---|
| `verify_isaac_runtime.py` | The adapter drives a live Franka env end-to-end; prints a machine-readable `IPFD_RUNTIME_SMOKE` block. |
| `verify_state_fidelity.py` | Single-step `save → reset_to` round trip is bit-exact. `STATE_RESTORE_FIDELITY: PASS`. |
| `verify_probe_transparency.py` | Shows that replay divergence is correlated with evolved post-grasp state. It does not identify the missing simulator or task state. |
| `verify_pnor_decoupled.py` | Historical run where one `reset_to` changed later single-env behavior across `env.reset()`. |
| `verify_multienv_isolation.py` | Measures that env 1 reset writes did not change env 0 at the observed boundary. `isolation_viable: YES`. |
| `verify_pnor_isolated.py` | Historical env-isolated diagnostic; current output is labeled `HISTORICAL_ONLY`. |
| **`verify_pnor_grasped.py`** | Historical scripted diagnostic with an immediate reset-boundary env-0 pose measurement; current output is labeled `HISTORICAL_ONLY`. |
| `verify_real_policy.py` | Superseded single-env experiment; records continuation divergence after in-loop probing (`meaningful_pnor_detected: NO`). |

The scripted recovery oracle they share lives in the library at
[`ipfd/oracles/pick_lift_sm.py`](../src/ipfd/oracles/pick_lift_sm.py).

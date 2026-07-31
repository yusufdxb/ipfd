# Scripts: diagnostics and evidence producers

These scripts are **not** part of the importable `ipfd` library (the library is
pure NumPy and runs in CI without a GPU). Most `verify_pnor_*` scripts are
historical experiments against a live Franka environment whose local
`isaaclab` distribution reported 4.5.22
(`Isaac-Lift-Cube-Franka-IK-Abs-v0`). Their old status labels cannot satisfy the
current release gate.

The current branch-validity experiment targets
`Isaac-Lift-Cube-Franka-v0`. It compares uninterrupted continuations with
restored exact-action and policy continuations before any recovery boundary is
treated as scientific evidence.

Run pattern (all scripts):

```bash
OMNI_KIT_ACCEPT_EULA=YES /path/to/isaac-lab/python scripts/<name>.py --headless
```

| Script | What it establishes |
|---|---|
| `verify_isaac_runtime.py` | The adapter drives a live Franka env end-to-end; prints a machine-readable `IPFD_RUNTIME_SMOKE` block. |
| `verify_state_fidelity.py` | Single-step `save → reset_to` round trip is bit-exact. `STATE_RESTORE_FIDELITY: PASS`. |
| `verify_probe_transparency.py` | Shows that replay divergence is correlated with evolved post-grasp state. It does not identify the missing simulator or task state. |
| `isaaclab_reset_to_contact_mre.py` | Reproduces exact-action continuation divergence after an exposed-state round trip. A `--grasp_steps 0` run is the free-space control. |
| `validate_recovery_oracle.py` | Runs paired uninterrupted and restored continuations across phases, disturbance families, and continuation modes. Produces raw JSON, compressed traces, and a hash manifest. |
| `analyze_branch_validity.py` | Compares uninterrupted and restored terminal decisions, applies a fail-closed empirical validity gate, and produces a machine-readable summary plus phase heatmap. |
| `run_snapshot_protocol_study.py` | Runs the corrected isolated five-seed Protocol A/B comparison with contact-derived phases, true continuation horizons, matched disturbances, and evidence controls. |
| `analyze_snapshot_protocol_study.py` | Produces corrected-study strata, explicit not-run gate and downstream records, an audited visualization, and refreshed artifact hashes. |
| `verify_pnor_decoupled.py` | Historical run where one `reset_to` changed later single-env behavior across `env.reset()`. |
| `verify_multienv_isolation.py` | Measures that env 1 reset writes did not change env 0 at the observed boundary. `isolation_viable: YES`. |
| `verify_pnor_isolated.py` | Historical env-isolated diagnostic; current output is labeled `HISTORICAL_ONLY`. |
| **`verify_pnor_grasped.py`** | Historical scripted diagnostic with an immediate reset-boundary env-0 pose measurement; current output is labeled `HISTORICAL_ONLY`. |
| `verify_real_policy.py` | Superseded single-env experiment; records continuation divergence after in-loop probing (`meaningful_pnor_detected: NO`). |

The scripted recovery oracle they share lives in the library at
[`ipfd/oracles/pick_lift_sm.py`](../src/ipfd/oracles/pick_lift_sm.py).

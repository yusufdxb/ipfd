# IPFD audit matrix

Execution result: **COMPLETED_WITH_UNSUPPORTED_SCOPES**

Each row is a separately scoped fidelity contract. Unsupported rows are expected findings, not execution failures.

| Configuration | Adapter | Protocol | Result |
|---|---|---|---|
| `mujoco_free_space.yaml` | mujoco | integration_with_warmstart | SUPPORTED |
| `mujoco_intermittent_contact.yaml` | mujoco | full_physics | UNSUPPORTED |
| `mujoco_sustained_minimal.yaml` | mujoco | minimal_visible | UNSUPPORTED |
| `mujoco_sustained_full_physics.yaml` | mujoco | full_physics | UNSUPPORTED |
| `mujoco_sustained_integration.yaml` | mujoco | integration_with_warmstart | SUPPORTED |
| `isaac_lab_archived.yaml` | isaac_lab_archive | expanded_runtime_state | UNSUPPORTED |

# Quickstart

IPFD answers one scoped question: can a saved simulator state support the same
downstream conclusion as an uninterrupted rollout?

## Run the bundled experiment

```bash
git clone https://github.com/yusufdxb/ipfd
cd ipfd
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,mujoco]"
ipfd demo
```

The command runs four MuJoCo cases in under a few seconds:

1. a narrow snapshot at initialization, where its omitted actuator state is zero;
2. the same snapshot after actuator preload, where visible state matches but the
   omitted activation changes the later contact decision;
3. the narrow snapshot plus actuator activation, which removes the decision
   mismatch while still omitting warm-start state;
4. MuJoCo integration-state restoration at the same preloaded branch point.

It writes `ipfd-demo-results/summary.json`, `evidence.json`, `report.png`, and an
artifact hash manifest. `FAIL_CLOSED` is the expected scientific result for the
narrow preloaded snapshot, not a command failure.

## Re-run each protocol as an audit

```bash
ipfd audit --config benchmarks/demo_filtered_minimal.yaml
ipfd audit --config benchmarks/demo_filtered_integration.yaml

ipfd compare \
  results/demo/filtered_minimal/audit_summary.json \
  results/demo/filtered_integration/audit_summary.json
```

The YAML accepts `simulator_version: runtime`; the emitted scope records the
actual installed MuJoCo version. Use an exact version string when a study must
reject any version drift.

`ipfd audit` exits `0` only when the scoped contract is supported, `1` for an
unsupported or insufficient-evidence verdict, and `2` for configuration or
runtime errors. `ipfd demo` exits `0` when the expected failure is detected
because successful detection is the demo's execution contract.

## Use IPFD from Python

```python
from ipfd import DecisionContract, audit

decision = DecisionContract(
    name="task_success",
    evaluator=lambda trajectory: bool(
        trajectory.steps[-1].task_outputs["success"][trajectory.env_id]
    ),
    description="task success at the requested horizon",
    true_label="success",
    false_label="failure",
)

result = audit(
    adapter=my_adapter,
    protocol="expanded_runtime_state",
    branch_step=120,
    horizons=[1, 5, 10, 30, 90],
    continuation="exact_action",
    decision=decision,
    tolerances={
        "default": {"absolute": 0.0, "relative": 0.0},
        "scene_state": {
            "absolute": 0.0,
            "fields": {
                "qpos": {"absolute": 1e-5},
                "qvel": {"absolute": 1e-4},
            },
        },
    },
)

print(result.verdict)
print(result.fidelity_frontier)
```

There are no universal tolerances. Declare thresholds in task units before
inspecting the result. A supported horizon is local to the simulator version,
task, protocol, branch condition, actions, decision, and recorded provenance.

See [Custom adapters](CUSTOM_ADAPTER.md) to connect another simulator.

The smallest complete adapter is executable:

```bash
python examples/custom_adapter.py
ipfd adapter-check examples.custom_adapter:make_adapter \
  --decision threshold_reached
ipfd audit --config examples/custom_adapter.yaml
```

The bundled adapter can also exercise the conformance workflow directly:

```bash
ipfd adapter-check ipfd.demo:make_demo_adapter \
  --decision remains_in_contact
```

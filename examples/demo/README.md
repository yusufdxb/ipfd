# MuJoCo counterfactual-fidelity demo

Run from the repository root:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev,mujoco]"
ipfd demo --output ipfd-demo-results
```

The embedded MJCF uses one vertical sphere, a floor, and a filtered actuator. No
external model assets are required. `config.yaml` is a byte-for-byte copy of
the packaged source read by `ipfd demo`; a regression test prevents drift. The
standalone audit protocol configs are
[`benchmarks/demo_filtered_minimal.yaml`](../../benchmarks/demo_filtered_minimal.yaml)
and
[`benchmarks/demo_filtered_integration.yaml`](../../benchmarks/demo_filtered_integration.yaml).

Exercise the adapter conformance workflow with the bundled zero-argument
factory:

```bash
ipfd adapter-check ipfd.demo:make_demo_adapter \
  --decision remains_in_contact
```

Expected semantics, across supported MuJoCo versions tested by this repository:

- the initialization control case agrees through 90 steps;
- the preloaded narrow snapshot matches measured exposed state at restoration;
- its short horizons pass, numerical fidelity degrades later, and the restored
  branch loses contact while the uninterrupted reference remains in contact;
- integration-state restoration preserves the decision through 90 steps.

Tests assert ranges for divergence and contact-transition steps. They do not
freeze platform-specific floating-point values. `expected_artifact_schema.json`
describes the stable `summary.json` interface.

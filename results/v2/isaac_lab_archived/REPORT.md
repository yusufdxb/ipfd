# IPFD replay-fidelity audit

Contract result: **UNSUPPORTED**

This verdict applies only to the listed simulator, version, environment, task, protocol, continuation, horizon, action source, decision function, tolerance, and provenance.

L0 equality never implies L2 trajectory fidelity or L3 decision fidelity.

| Simulator | Protocol | Horizon | Decision | L0 | L1 | L2 first divergence | L3 disagreement | Result |
|---|---|---:|---|---|---|---:|---|---|
| Isaac Lab 4.5.22 | expanded_runtime_state | 1 | sustained_lift | False | None | 1 | False | UNSUPPORTED |
| Isaac Lab 4.5.22 | expanded_runtime_state | 5 | sustained_lift | False | None | 1 | False | UNSUPPORTED |
| Isaac Lab 4.5.22 | expanded_runtime_state | 10 | sustained_lift | False | None | 1 | False | UNSUPPORTED |
| Isaac Lab 4.5.22 | expanded_runtime_state | 30 | sustained_lift | False | None | 1 | True | UNSUPPORTED |
| Isaac Lab 4.5.22 | expanded_runtime_state | 90 | sustained_lift | False | None | 1 | True | UNSUPPORTED |

## Evidence files

- `audit_summary.json`: scoped contract conclusions
- `per_branch_records.jsonl`: raw paired branch measurements
- `fidelity_contract.json`: the contract evaluated
- `provenance.json`: source, software, hardware class, and adapter inventory
- `divergence.svg`: numerical error growth
- `minimal_reproducer.json`: automatically reduced failing case

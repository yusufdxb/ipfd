# IPFD replay-fidelity audit

Contract result: **UNSUPPORTED**

This verdict applies only to the listed simulator, version, environment, task, protocol, continuation, horizon, action source, decision function, tolerance, and provenance.

L0 equality never implies L2 trajectory fidelity or L3 decision fidelity.

| Simulator | Protocol | Horizon | Decision | L0 | L1 | L2 first divergence | L3 disagreement | Result |
|---|---|---:|---|---|---|---:|---|---|
| MuJoCo 3.5.0 | minimal_visible | 1 | collision | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 1 | stable_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 1 | sustained_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 1 | within_bounds | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 5 | collision | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 5 | stable_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 5 | sustained_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 5 | within_bounds | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 10 | collision | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 10 | stable_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 10 | sustained_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 10 | within_bounds | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 30 | collision | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 30 | stable_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 30 | sustained_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 30 | within_bounds | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 90 | collision | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 90 | stable_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 90 | sustained_contact | False | False | 1 | False | UNSUPPORTED |
| MuJoCo 3.5.0 | minimal_visible | 90 | within_bounds | False | False | 1 | False | UNSUPPORTED |

## Evidence files

- `audit_summary.json`: scoped contract conclusions
- `per_branch_records.jsonl`: raw paired branch measurements
- `fidelity_contract.json`: the contract evaluated
- `provenance.json`: source, software, hardware class, and adapter inventory
- `divergence.svg`: numerical error growth
- `minimal_reproducer.json`: automatically reduced failing case

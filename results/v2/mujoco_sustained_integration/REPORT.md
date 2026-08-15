# IPFD replay-fidelity audit

Contract result: **SUPPORTED**

This verdict applies only to the listed simulator, version, environment, task, protocol, continuation, horizon, action source, decision function, tolerance, and provenance.

L0 equality never implies L2 trajectory fidelity or L3 decision fidelity.

| Simulator | Protocol | Horizon | Decision | L0 | L1 | L2 first divergence | L3 disagreement | Result |
|---|---|---:|---|---|---|---:|---|---|
| MuJoCo 3.5.0 | integration_with_warmstart | 1 | collision | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 1 | stable_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 1 | sustained_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 1 | within_bounds | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 5 | collision | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 5 | stable_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 5 | sustained_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 5 | within_bounds | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 10 | collision | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 10 | stable_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 10 | sustained_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 10 | within_bounds | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 30 | collision | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 30 | stable_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 30 | sustained_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 30 | within_bounds | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 90 | collision | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 90 | stable_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 90 | sustained_contact | True | True | none observed | False | SUPPORTED |
| MuJoCo 3.5.0 | integration_with_warmstart | 90 | within_bounds | True | True | none observed | False | SUPPORTED |

## Evidence files

- `audit_summary.json`: scoped contract conclusions
- `per_branch_records.jsonl`: raw paired branch measurements
- `fidelity_contract.json`: the contract evaluated
- `provenance.json`: source, software, hardware class, and adapter inventory
- `divergence.svg`: numerical error growth

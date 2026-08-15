# IPFD v2 final evidence ledger

## 1. Blunt verdict

`SUCCESSFUL_ENGINEERING_CONTRIBUTION`.

IPFD v2 is no longer the archived recovery-analysis experiment. It is a reusable
snapshot-and-restore conformance system with simulator-neutral L0 through L3
contracts, two simulator adapters, scoped verdicts, automated reduction,
regression comparison, provenance, machine-readable output, and one primary CLI.
The live Isaac Lab adapter still lacks a completed current-runtime audit, which is
an important limitation, but the same CLI audits the immutable Isaac Lab evidence
and both live MuJoCo versions without experiment-specific code changes.

## 2. Archive verification

The branch started from archival `main` at
`e9035884f53bece3e8a28e92d3c0c61968d95fef`. Local and public `main` matched.
Local and public `archive/ipfd-honest-negative` matched at
`7611cf6c5cdde29009265121b833547eb085c9ca`. The stopped local calibration
successor is `7c4760e16664a58687edc1bbe51fd55ec031cfa1`.

Forty-six byte-count, SHA-256, or cross-manifest linkage checks passed. This
covered the corrected-study manifest, all retained external worker files, the
calibration manifest, original report and trace, fixtures, generators, checkpoint,
and oracle-equivalence evidence. `git fsck` reported no integrity error. The
commits are unsigned, so this is a content-addressed and public-ref proof, not a
signed-author proof.

No archived result, manifest, provenance file, or stopping decision changed.
[`HISTORICAL_BASELINE.md`](HISTORICAL_BASELINE.md) records the original 13 of 120
falsification, the corrected five-seed 18 of 444 versus 11 of 444 study, the
failed calibration successor, and both stopping decisions. V2 does not revive
the Point-of-No-Return claim, the validity-envelope predictor, or recovery
analysis.

## 3. Contract definition

[`REPLAY_FIDELITY_CONTRACT.md`](REPLAY_FIDELITY_CONTRACT.md) and every generated
`fidelity_contract.json` define:

- L0: equality of measured exposed state at the restore boundary only;
- L1: one identical action followed by separate numerical and semantic checks;
- L2: identical-action open-loop replay at 1, 5, 10, 30, and 90 control steps;
- L3: agreement of each user-declared downstream Boolean decision.

Every configuration returns only `SUPPORTED`, `UNSUPPORTED`, or
`INSUFFICIENT_EVIDENCE`. Scope includes simulator and version, environment, task,
protocol, continuation, horizon, action source, decision, tolerance, cluster key,
and provenance. A live audit rejects a declared simulator version that differs
from the adapter-reported version. No result is a universal simulator verdict.

## 4. Adapter implementations

[`ADAPTER_CONTRACT.md`](ADAPTER_CONTRACT.md) defines the common `ReplayAdapter`
protocol and mandatory missing-state disclosure.

- `IsaacLabReplayAdapter` supports `scene_only` and
  `expanded_runtime_state`. It inventories scene entities, action and command
  buffers, episode counters, articulation targets, optional provider history,
  sensors, RNG limitations, and unavailable PhysX state. Construction and runtime
  failures propagate to a nonzero CLI status.
- `MuJoCoReplayAdapter` uses two independent `MjData` instances and documented
  state APIs. It supports `minimal_visible`, `full_physics`, and
  `integration_with_warmstart`, plus cold or restored continuation. Embedded MJCF
  means the reference cases require no downloaded model assets.
- `isaac_lab_archive` is a read-only evidence converter. It verifies the sealed
  record SHA before conversion and never presents missing snapshot/action values
  as a self-contained reproducer.

## 5. Benchmark environments

[`BENCHMARK_PROTOCOL.md`](BENCHMARK_PROTOCOL.md) and `benchmarks/` declare:

- free space: an actuated contact-free MuJoCo slider;
- intermittent contact: a slider with a bounded collision event;
- sustained contact: a vertically actuated body settling against a plane;
- archived Isaac Lab: five independent seed clusters from the corrected lift
  study;
- live Isaac Lab smoke configuration: the tested lift task with zero actions.

Each live MuJoCo row uses seeds 101, 211, and 307 as independent clusters. Repeated
horizons are not counted as independent samples.

## 6. L0 through L3 results

| Audit | L0 | L1 | L2 | L3 | Overall |
|---|---|---|---|---|---|
| MuJoCo 3.5 free space, integration state | pass | pass | pass through 90 | no disagreements | `SUPPORTED` |
| MuJoCo 3.5 intermittent contact, full physics | fail | pass | pass through 90 | no disagreements | `UNSUPPORTED` |
| MuJoCo 3.5 sustained contact, minimal state | fail | fail | first divergence at 1 | no disagreements | `UNSUPPORTED` |
| MuJoCo 3.5 sustained contact, full physics | fail | fail | first divergence at 1 | no disagreements | `UNSUPPORTED` |
| MuJoCo 3.5 sustained contact, integration state | pass | pass | pass through 90 | no disagreements | `SUPPORTED` |
| MuJoCo 3.8.1 free space, integration state | pass | pass | pass through 90 | no disagreements | `SUPPORTED` |
| Archived Isaac Lab expanded runtime | fail in 250 of 370 selected records | insufficient | fail in 369 of 370 | 11 disagreements | `UNSUPPORTED` |

The intermittent-contact L0 failure is a real contract-level distinction: the
`full_physics` state omits controls and warm-start state, so control targets and
derived fields differ immediately after restore. Applying the next identical
action synchronizes this fixture, and L1 through L3 pass. This is why IPFD reports
levels separately instead of treating one equality assertion as the whole result.

In the archived Isaac selection, immediate scene state and policy observation
errors were zero in all 370 selected records. The stricter L0 projection also
checks sensor/contact values, which failed in 250 records. L1 is
`INSUFFICIENT_EVIDENCE` because termination and reward were not retained.

## 7. Positive controls

The MuJoCo 3.5 free-space case passed every L0 through L3 comparison at all five
horizons with zero raw state error. MuJoCo 3.8.1 repeated the same result. The
sustained-contact `integration_with_warmstart` protocol also passed exact equality
through 90 steps, demonstrating that the harness can report support under contact
instead of being constructed only to find failures.

## 8. Failures detected

The archived Isaac audit recovered the preserved `sustained_lift` decision
disagreements: 0 of 74 at horizons 1, 5, and 10, 1 of 74 at 30, and 10 of 74 at
90. It does not reinterpret them as an Isaac Lab or PhysX defect.

Under exact equality, MuJoCo sustained-contact `minimal_visible` and
`full_physics` first diverged at step 1 with a maximum state error of
`2.1314150657737222e-17`. L3 decisions still agreed. This is a narrow numerical
contract failure, not a meaningful task-decision reversal.

## 9. Minimal reproductions

The live sustained-contact failures reduce from branch step 150 to branch step 25
while retaining horizon 1 and one identical action. Each reproducer includes the
captured state vector, identical action, expected and restored decisions, first
numerical and observation divergence, adapter inventory, and versions. The
minimal-state reproducer digest is
`88c90a095d57833d00242a914546b852c29df4f7e0b185d22b35e6c6a7586b67`.

The archived Isaac reducer selects the earliest retained L3 counterexample but
labels it `ARCHIVED_RECORD_REDUCED_BUT_NOT_SELF_CONTAINED`, because the historical
study did not retain raw snapshot and action values.

## 10. Protocol and version regression

The sustained-contact protocol regression pairs 20 stable configuration keys.
Moving from `minimal_visible` to `integration_with_warmstart` changes L0 and L1,
moves divergence later in all 20 comparisons because no divergence is observed
through the declared horizon, leaves L3 disagreement unchanged, and produces no
`SUPPORTED` to `UNSUPPORTED` transition.

The MuJoCo 3.5.0 to 3.8.1 free-space version regression pairs 15 keys. L0, L1,
L3, and support status do not change, and all 15 divergence comparisons remain
the same with no observed divergence.

## 11. Comparison with trivial alternatives

Immediate equality alone misses the archived long-horizon decision reversals.
One-step equality misses failures that begin later. One fixed short horizon misses
the archive's 30 and 90-step support boundary. Manual comparison lacks stable
scopes, independent-cluster accounting, provenance, protocol and version pairing,
artifact digests, nonzero failure behavior, and automatic reduction.

IPFD is therefore not equivalent to one immediate assertion or one horizon
condition. If future maintenance removes those differentiators, the v2 direction
should stop.

## 12. Upstream-ready contribution

[`UPSTREAM_ENGINEERING_NOTE.md`](UPSTREAM_ENGINEERING_NOTE.md) contains a narrow
Isaac Lab documentation clarification and a free-space finite-horizon regression
test sketch. It distinguishes scene-state restoration from complete simulator and
task snapshots. It does not present expected contact divergence as a defect.

No issue was filed, no pull request was opened, and no maintainer was contacted.
Submission requires explicit approval.

## 13. NVIDIA relevance

The direct engineering relevance is to Isaac Lab and PhysX users who build
counterfactual evaluation, controller comparison, policy ranking, or intervention
tools on scene restoration. IPFD supplies a concrete way to state and regression
test the supported boundary without claiming that Isaac Lab or PhysX is generally
invalid. The upstream note offers a narrowly supportable documentation and test
improvement for that ecosystem.

## 14. Limitations

- The current live Isaac Lab smoke command did not produce a result. After the
  configured public asset channel was applied, the installed runtime terminated
  during platform-plugin initialization. The live adapter is implemented and
  import-checked, but current hardware-dependent fidelity remains unverified.
- The archived Isaac source JSONL is a sealed external artifact excluded from Git
  because of size. The one-command matrix needs that local artifact with its
  recorded SHA. The converted v2 evidence is committed.
- The MuJoCo fixtures are deliberately small reference systems, not representative
  robotics benchmarks.
- Three live MuJoCo seed clusters and five archived Isaac clusters do not justify
  broad statistical claims.
- Exact equality in the sustained-contact comparison is an intentional declared
  acceptance boundary, not a proposed universal tolerance.
- No experiment identifies the cause of the archived Isaac discrepancies or
  proves that unavailable solver state is responsible.
- No hardware result and no universal simulator conclusion is claimed.

## 15. Success-criteria decision

| Criterion | Decision | Proof |
|---:|---|---|
| 1 | met | `ipfd audit` dispatches Isaac Lab and MuJoCo; the matrix executes archived Isaac evidence and live MuJoCo through one command |
| 2 | met | 11 archived Isaac L3 disagreements recovered at horizons 30 and 90 |
| 3 | met | MuJoCo 3.5 and 3.8.1 free-space controls are `SUPPORTED` through 90 steps |
| 4 | met | minimal/full protocols fail exact sustained-contact fidelity while integration state passes |
| 5 | met | self-contained one-action MuJoCo reproducer generated automatically |
| 6 | met | protocol regression and independent 3.5.0 to 3.8.1 version regression generated |
| 7 | met | every summary has scoped verdicts and explicit provenance references |
| 8 | met | upstream-ready documentation clarification and regression sketch exist |
| 9 | met | `ipfd audit --config benchmarks/audit_matrix.yaml` regenerates the declared portable-plus-archive matrix in one invocation |
| 10 | met | L0 through L3, five horizons, minimization, provenance, and regression cannot be reduced to one immediate assertion or one horizon condition |

Decision: `SUCCESSFUL_ENGINEERING_CONTRIBUTION`, with the live Isaac runtime
limitation stated above.

## 16. Commands run

Interpreter paths are normalized here for public portability. Exact Python and
package versions are in each `provenance.json`.

```bash
git fetch origin --prune
git fsck --full --no-reflogs --no-dangling
git switch -c v2/replay-integrity-contract main

python -m pytest --cov=ipfd --cov-branch --cov-report=term
python -m ruff check .
python -m mypy src/ipfd
$SIMULATOR_PYTHON -m pytest -q

$MUJOCO_3_5_PYTHON -m ipfd.cli audit --config benchmarks/audit_matrix.yaml
$MUJOCO_3_8_PYTHON -m ipfd.cli audit --config benchmarks/mujoco_free_space_3_8.yaml
$ISAACLAB_PYTHON -m ipfd.cli audit --config benchmarks/isaac_lab_live_scene_only.yaml
```

The matrix and version commands returned zero. The live Isaac command returned
nonzero and wrote no audit output, as required for a runtime failure.

## 17. Tests and checks

- CPU suite: 203 passed, 12 optional-backend tests skipped, 85.04 percent branch
  coverage, coverage gate passed.
- Simulator-enabled suite: 215 passed.
- MuJoCo adapter suite was also exercised under MuJoCo 3.8.1.
- Ruff: clean.
- Mypy: clean across `src/ipfd`.
- Package build: source distribution and wheel succeeded; Twine validation passed.
- All 40 generated artifact digests in the seven audit summaries were rechecked.
- All six matrix child-summary digests were rechecked.
- Public-artifact scrub found no local home path, vault reference, agent file,
  hostname, or recorded accelerator identifier.
- Archived result and provenance paths remained byte-identical to archival main.

## 18. Files changed

- Contracts and evidence: `HISTORICAL_BASELINE.md`,
  `REPLAY_FIDELITY_CONTRACT.md`, `ADAPTER_CONTRACT.md`,
  `BENCHMARK_PROTOCOL.md`, `UPSTREAM_ENGINEERING_NOTE.md`, and this ledger.
- Core: `src/ipfd/fidelity/` and the `ipfd audit` and `ipfd regress` CLI paths.
- Adapters: `src/ipfd/adapters/isaac_replay.py` and
  `src/ipfd/adapters/mujoco_replay.py`.
- Configurations: nine files under `benchmarks/`, including the one-command matrix.
- Tests: five focused v2 modules plus the MuJoCo adapter suite.
- Results: seven scoped audit directories and one matrix index under `results/v2/`.
- Packaging and orientation: `pyproject.toml`, `src/ipfd/__init__.py`, and
  `README.md`.

No file under `results/branch_validity/`, `results/decision_calibration/`, or the
historical fixture manifests changed.

## 19. Git status and commits

Branch: `v2/replay-integrity-contract`.

- implementation source:
  `a3ae56b884ac7013f18985f0bf07c8de6c0fc4b1`
- generated evidence:
  `cf75d4a4e150ef3a18fbba802c1754570bc1f420`
- branch point:
  `e9035884f53bece3e8a28e92d3c0c61968d95fef`

All result provenance records identify the implementation source commit, the v2
branch, and `dirty: false`. The evidence commit contains generated results only.
This ledger changes no result digest. Nothing was pushed and no upstream action
was taken.

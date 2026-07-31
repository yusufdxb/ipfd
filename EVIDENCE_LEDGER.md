# IPFD Evidence Ledger

Statuses:

- **Verified:** directly reproduced by the listed command.
- **Partial:** supported only under the stated scope.
- **Falsified:** the tested claim produced a counterexample.
- **Unverified:** no adequate experiment exists.
- **Historical only:** numerical output reproduced, scientific interpretation
  rejected.

| Claim | Supporting command | Artifact | Status | Limitation |
|---|---|---|---|---|
| CPU analysis suite passes | `env -u PYTHONPATH python3 -m pytest -q` | Test output | Verified | Local collection included six excluded tests |
| Tracked baseline suite passes | `python3 -m pytest -q $(git ls-files 'tests/test_*.py')` | Test output | Verified | Simulator behavior is excluded |
| Branch-validity analysis helpers behave as specified | `python3 -m pytest -q tests/test_branch_validity.py tests/test_oracle_equivalence.py` | Test output | Verified | Pure analysis contract only |
| Branch coverage exceeds project threshold | `python3 -m pytest --cov=ipfd --cov-report=term-missing --cov-branch -q` | 85.59 percent baseline report | Verified | Adapter and oracle modules are omitted by configuration |
| Static checks pass | `python3 -m ruff check src tests scripts` and `python3 -m mypy src/ipfd` | Command output | Verified | Does not test physics |
| Isaac task attaches with explicit production assets | `OMNI_KIT_ACCEPT_EULA=YES <isaac-python> scripts/verify_isaac_runtime.py --headless --asset_root <asset-root>` | `IPFD_RUNTIME_SMOKE` output | Partial | Default staging asset URL failed |
| Exposed scene state round-trips exactly | `OMNI_KIT_ACCEPT_EULA=YES <isaac-python> scripts/isaaclab_reset_to_contact_mre.py --headless --asset_root <asset-root>` | Printed round-trip error `0.00e+00` | Partial | Exposed state is not complete simulator-policy state |
| Exact exposed restore implies trajectory-equivalent continuation | Same reset reproducer with `--grasp_steps 40` | Pre-step observation gap `9.96`, final exact-action observation gap `0.1046` | Falsified | Does not identify the missing state |
| Evolved state is required for the minimal divergence reproducer | Same reproducer with `--grasp_steps 0` | Post-step gaps all zero | Partial | Pre-step manager-observation gap remains |
| Restored branches preserve terminal decisions under the expanded protocol | Reproduction command in `EXPERIMENT_PROTOCOL.md` | Clean raw report SHA-256 `86189ab525c394d24d6e1b8b26427850ad5822bae9bd3789074e4c4985b9cad2` | Falsified | Three seeds, correlated branch comparisons |
| Universal decision fidelity is false in the clean slice | Same command | 13 of 120 disagreements, 10 of 60 exact-action | Verified | Narrow task and protocol |
| Immediate observation equality guarantees the same decision | `python3 scripts/analyze_branch_validity.py ...` | 120 of 120 immediate equal, 13 terminal disagreements | Falsified | Archived cohort |
| A five-percent decision-validity envelope is certified | Same analysis command | Zero passing phase-controller cells | Unverified | Sample too small and correlated; fail-closed gate correctly rejects certification |
| The corrected paired design removes duplicated branch points, mislabeled horizons, and unmatched disturbances | Primary command in `CORRECTED_EXPERIMENT_PROTOCOL.md` | Ten worker control blocks in `results/branch_validity/corrected_five_seed/protocol_comparison.json` | Verified | One task and one checkpoint |
| Protocol B meaningfully improves exact-action sustained-lift fidelity under the preregistered rule | Same primary command | Protocol A 18/444 disagreements, Protocol B 11/444, 38.9 percent relative reduction | Falsified | Required reduction was at least 50 percent; four of five seeds improved |
| Protocol B reduces primary disagreement at all | Same primary command | Seed-paired mean difference -1.57 percentage points, five seed groups | Partial | Only five independent units; the effect is smaller than the preregistered meaningful threshold |
| Protocol B makes restored exact-action decisions universally valid | Same primary command | 11/444 primary disagreements and 39/1332 across all three predicates | Falsified | Remaining errors concentrate at 30 and 90 steps and gripper interruption |
| Short exact-action horizons are decision-faithful in the corrected sample | `python3 scripts/analyze_snapshot_protocol_study.py --study-dir results/branch_validity/corrected_five_seed` | Protocol B had 0/74 primary disagreements at 1, 3, 5, and 10 steps | Partial | Finite sample, one task, and no held-out validity gate |
| A held-out empirical validity gate beats simple heuristics | Not run because the positive-control threshold failed | `validity_gate_results.json` | Unverified | Stopping rule prohibited training or evaluation |
| Validity gating corrects a PoNR, controller ranking, or checkpoint decision | Not run because the gate stage was ineligible | `downstream_decision_results.json` | Unverified | No downstream correction was demonstrated |
| Counterfactual Branch Validity should continue as the flagship direction | Corrected five-seed stopping rule | `STOP_BRANCH_VALIDITY_DIRECTION` | Falsified | The project should be archived as an honest negative under the user-specified rule |
| Historical scripted boundary equals 138 | `OMNI_KIT_ACCEPT_EULA=YES <isaac-python> scripts/verify_pnor_grasped.py --headless --asset_root <asset-root>` | Printed PoNR 138 | Historical only | One seed, scripted controller, injected fault, nonmonotone raw labels |
| Historical boundary predicts failure by 0.42 seconds | Same command | Printed timing | Falsified as an interpretation | Failure index is the terminal horizon |
| Learned teleport alarm is actionable | `python3 scripts/evaluate_actionability.py tests/fixtures/learned_teleport_rollout.npz --disturbance-onset 56 --probe-stride 8` | Alarm 20, boundary interval 49 to 56 | Falsified | Alarm precedes disturbance and actionable window is empty |
| Recovery repeat agreement is a confidence interval | Existing repeated-probe API | None | Unverified | Deterministic repeats are not independent trials |
| PoNR is stable across task, seed, policy, timestep, threshold, and restore error | None | None | Unverified | Current branch fidelity is already negative |
| IPFD performs causal attribution | `python3 scripts/benchmark_comparison.py` | 25 hand-authored synthetic contract cases | Unverified | Contract data is not a randomized causal validation |
| IPFD improves training or policy selection | None | None | Unverified | No comparative downstream experiment |
| Backend-independent or sim-to-real recoverability | None | None | Unverified | No second backend or physical validation |

## Machine-readable evidence

- [`results/branch_validity/summary.json`](results/branch_validity/summary.json)
  stores cohort hashes, paired agreement counts, descriptive intervals,
  false-recoverable and false-unrecoverable counts, trajectory checks, and the
  fail-closed gate.
- [`results/branch_validity/clean_three_seed_report.json`](results/branch_validity/clean_three_seed_report.json)
  is the clean raw paired-branch report. Its generator and checkpoint hashes are
  recorded in
  [`results/branch_validity/clean_three_seed_provenance.json`](results/branch_validity/clean_three_seed_provenance.json).
- [`results/branch_validity/decision_fidelity.png`](results/branch_validity/decision_fidelity.png)
  is the primary result visualization.
- [`results/branch_validity/corrected_five_seed/protocol_comparison.json`](results/branch_validity/corrected_five_seed/protocol_comparison.json)
  stores the corrected five-seed protocol comparison and preregistered stopping
  decision.
- [`results/branch_validity/corrected_five_seed/per_branch_records.jsonl`](results/branch_validity/corrected_five_seed/per_branch_records.jsonl)
  contains 5,328 machine-readable decision records.
- [`results/branch_validity/corrected_five_seed/protocol_strata.json`](results/branch_validity/corrected_five_seed/protocol_strata.json)
  reports the primary comparison by phase, horizon, disturbance, seed, and
  predicate.
- [`results/branch_validity/corrected_five_seed/decisive_study.png`](results/branch_validity/corrected_five_seed/decisive_study.png)
  shows the measured protocol improvement and the two stages not run under the
  stopping rule.

The repository now contains the immutable three-seed baseline and one corrected
five-seed protocol-comparison cohort. They are not pooled.

## Large raw traces retained outside git

These files are deliberately excluded from version control. Their sizes and
SHA-256 digests are recorded in the manifests listed below, so provenance survives
independently of where the bytes live. They are retained on the research machine
under `~/.local/share/ipfd/branch_validity/`.

| File | Bytes | SHA-256 | Digest recorded in |
|---|---:|---|---|
| `corrected_five_seed/workers/` (20 raw worker JSON and log files) | 13,010,452 total | per file | [`raw_worker_manifest.json`](results/branch_validity/corrected_five_seed/raw_worker_manifest.json) |
| `corrected_five_seed/per_branch_records.jsonl` (5,328 decision records) | 10,166,696 | `1c55862ce4a24bf564396c8d86873b13fc1667e491aada27bfff6c8d2166ce8f` | [`artifact_manifest.json`](results/branch_validity/corrected_five_seed/artifact_manifest.json) |
| `clean_three_seed/traces.npz` | 5,329,453 | `dcf30076c60ddfea754ea7dd13e0db95377186d8f5c6febaff9aa8ddc5a11558` | [`clean_three_seed_provenance.json`](results/branch_validity/clean_three_seed_provenance.json) |

`analyze_snapshot_protocol_study.py` needs `per_branch_records.jsonl` present in
the study directory. Restore it from the retained copy before rerunning the
CPU-side analysis.

## Archival verification, 2026-07-30

Verified during the archival pass, with no expensive GPU rerun:

| Check | Result |
|---|---|
| Eight in-repo corrected-study artifacts against `artifact_manifest.json` | 8 of 8 match on bytes and SHA-256 |
| Twenty external raw worker files against `raw_worker_manifest.json` | 20 of 20 match |
| `per_branch_records.jsonl` and `traces.npz` against their recorded digests | Both match |
| Study generator `scripts/run_snapshot_protocol_study.py` against `source_generator_sha256` | Matches `8d7ca4bc…` |
| Baseline generator `scripts/validate_recovery_oracle.py` against three-seed provenance | Matches `17b41305…` |
| Learned checkpoint against `checkpoint_sha256` | Matches `fb658f98…` |
| Three-seed report, summary, and figure against `SNAPSHOT_PROTOCOLS.md` | All three match |

### Verifying the generator digests after the archival commit

The two generator digests above pin the scripts **as of the archival commit
`7611cf6c5cdde29009265121b833547eb085c9ca`**, which is the state that produced the
recorded results. Later commits on `main` may edit those scripts for portability
without invalidating any recorded result, so a digest check against the current
working tree can legitimately differ. Verify against the archival commit:

```bash
git show 7611cf6c:scripts/run_snapshot_protocol_study.py | sha256sum   # 8d7ca4bc…
git show 7611cf6c:scripts/validate_recovery_oracle.py    | sha256sum   # 17b41305…
```

Known divergence: commit `clarify historical release and remove machine-specific
defaults` replaced a hardcoded `--isaac-lab-root` default with an
environment-derived value, changing that script's digest to `20b769bc…` on later
commits. The change is confined to argument handling and does not touch the
experiment, the analysis, or any recorded artifact. The hash-sealed provenance
JSON was deliberately left untouched.

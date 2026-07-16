# IPFD live Isaac validation manifest

Timestamp: `2026-07-15T00:17:38+00:00`

Scope: runtime/environment inspection and live Isaac validation evidence only. No thresholds,
algorithms, source behavior, git state, commits, pushes, stashes, resets, or unrelated edits were
changed.

## Repository state

- Git commit: `4a0c34544b1e4145e3f6257aa75dee1d3ad1fed7`
- Branch: `harden-input-validation`
- Dirty status at start: dirty, with staged and unstaged work already present.
- Dirty entries observed after validation and before this manifest: `36`
- Validation-only generated assets during the run, removed after hashing:
  - `.ipfd_live_validation_tmp/` for transient command transcripts and saved rollout hashing.
  - `.pretrained_checkpoints/.../checkpoint.pt` from Isaac Lab's official checkpoint helper.
- Committed replay fixture binding:
  - [`tests/fixtures/manifest.json`](../tests/fixtures/manifest.json) records the shipped fixture hashes below.
  - The fresh live rollout used for validation is a separate capture and is intentionally not the same file as the committed fixture archive.
  - `learned_teleport_rollout.npz`: `4a228a8d00a48dba8f51aa9ec4f884f483d1aba64de701b29a2dac8a6777c106`
  - `learned_teleport_report.json`: `00ccaee9cf3cfd59807da65623eea3574e770719ece09793fd010f9ad2504aa6`
  - `learned_slip_rollout.npz`: `d6e21c652643c102ca7de4b4249541f6c1e10ee98574c1407803389128864506`
  - `learned_slip_report.json`: `27348b0bbac2a48f157efba8a9e79c76586683b0d69549099f30e36aacfe7867`

## Runtime environment

- Host Python:
  - Executable: `python3`
  - Version: `3.10.12`
  - Active virtual environment: none (`VIRTUAL_ENV=None`, `CONDA_PREFIX=None`)
- Isaac Python:
  - Executable: Isaac Lab runtime venv python
  - Version: `3.12.13`
  - Prefix: Isaac Lab runtime venv
  - Active virtual environment variable: none (`VIRTUAL_ENV=None`)
- Packages:
  - `isaaclab`: `4.5.22`
  - `isaaclab_tasks`: `1.5.11`
  - `isaaclab_rl`: `0.5.0`
  - `isaacsim`: `6.0.0.0`
  - `torch`: `2.10.0+cu128`
  - `rsl-rl-lib`: `5.0.1`
  - `gymnasium`: `1.2.1`
  - `warp-lang`: `1.12.0`
- CUDA/GPU:
  - `torch.cuda.is_available()`: `True`
  - Torch CUDA: `12.8`
  - Device count: `1`
  - GPU: consumer Blackwell NVIDIA GPU
  - Driver: `570.211.01`
  - GPU memory: `~12 GiB`
  - Compute capability: `12.0`

## Isaac tasks and assets

- Registered Franka Lift-Cube tasks found: `4`
  - `Isaac-Lift-Cube-Franka-IK-Abs-v0`
  - `Isaac-Lift-Cube-Franka-IK-Rel-v0`
  - `Isaac-Lift-Cube-Franka-Play-v0`
  - `Isaac-Lift-Cube-Franka-v0`
- `parse_env_cfg`:
  - `Isaac-Lift-Cube-Franka-v0`: OK, `sim.dt=0.01`, `decimation=2`
  - `Isaac-Lift-Cube-Franka-IK-Abs-v0`: OK, `sim.dt=0.01`, `decimation=2`
- Official published `rsl_rl` checkpoint availability:
  - `Isaac-Lift-Cube-Franka-v0`: available and fetched by Isaac Lab helper.
  - `Isaac-Lift-Cube-Franka-IK-Abs-v0`: unavailable (`NONE`).
- Official checkpoint SHA-256:
  - `fb658f989bf5ebf35b20347813275979a6778ade8d3823d12eb3190612f9e36d`

## Commands and results

### 1. Runtime smoke

Command:

```bash
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES ~/Sim/isaac-sim-venv/bin/python \
  scripts/verify_isaac_runtime.py --headless --env_id Isaac-Lift-Cube-Franka-v0 \
  --steps 16 --seed 0
```

Observed result:

- Live env: `Isaac-Lift-Cube-Franka-v0`, `num_envs=1`, `device=cuda:0`
- Env class: `isaaclab.envs.manager_based_rl_env.ManagerBasedRLEnv`
- Observation group: `policy`, shape `(36,)`
- Action shape: `8`
- `IPFD_RUNTIME_SMOKE`:
  - `isaac_lab_import: PASS`
  - `real_env_execution: PASS`
  - `observation_structure: PASS`
  - `adapter_attachment: PASS`
  - `overall: PASS`
- CLI exit: `0`
- Transcript SHA-256: `23931f0dfa0d7ed0c1a50609c9e269d46179ae98fb49554d74b61217142f4dc4`

Validated:

- Isaac adapter import/AppLauncher path.
- Live Franka task creation.
- Observation extraction through `_extract_obs`.
- `build_report()` attachment to a real Isaac rollout.
- CLI success exit behavior.

### 2. Dual-env scripted PoNR validation

Command:

```bash
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES ~/Sim/isaac-sim-venv/bin/python \
  scripts/verify_pnor_grasped.py --headless
```

Observed result:

- Live env: `Isaac-Lift-Cube-Franka-IK-Abs-v0`
- Env count: `2` (`env0=primary pristine`, `env1=probe`)
- Seed: script fixed `seed=0` for primary reset.
- `dt`: `0.0200`
- Primary rollout: `T=160`, `drop_step=127`, `primary_failed=True`
- Raw probe verdicts included both recoverable and unrecoverable checkpoints.
- Recovery totals: `True/False = 68/92`
- PoNR: `138`
- Observable failure: final primary failure at rollout end (`T-1=159`)
- PoNR lead over observable failure: `+0.42s`
- Primary integrity: `reset_to x51 into env1`, max env-0 pose delta `0.00e+00 m`
- `DUAL_PROBE_STATUS`:
  - `second_environment_supported: YES`
  - `snapshot_transfer_supported: YES`
  - `primary_rollout_corruption: NO`
  - `recovery_oracle_non_degenerate: YES`
  - `pnor_measurement_possible: YES`
  - `overall_status: VERIFIED`
- CLI exit: `0`
- Transcript SHA-256: `ad94569d1e1648bb500d9090a522d2f57f324213c160813d1e389021a2577c44`

Validated:

- Vectorized env indexing (`env0` primary, `env1` probe).
- Probe isolation via `scene.reset_to(..., env_ids=[1])`.
- Recovery probe non-degeneracy.
- PoNR/failure timing in a live Isaac run.
- Primary corruption guard.

### 3. Learned-policy packaged adapter run

Command:

```bash
PYTHONUNBUFFERED=1 OMNI_KIT_ACCEPT_EULA=YES ~/Sim/isaac-sim-venv/bin/python \
  scripts/verify_learned_policy.py --headless --use_pretrained --probe \
  --failure teleport --save_rollout .ipfd_live_validation_tmp/learned_teleport_rollout.npz
```

Observed result:

- Live env: `Isaac-Lift-Cube-Franka-v0`
- Env count: `4`
- Device: `cuda:0`
- Seed: `0`
- Official checkpoint loaded from Isaac Lab helper cache.
- Failure mode: `teleport`
- Packaged `collect_rollout` result:
  - `T=58`
  - `t_failure=57`
  - `success=False`
  - `probe_resets=8`
  - `primary_integrity_max_delta=0.00e+00 m`
- Report:
  - PoNR: step `56` (`1.12s`)
  - Observable failure: step `57` (`1.14s`)
  - Detector alarm: step `20` (`0.40s`)
  - Failure lead time: `+0.74s`
  - PoNR lead time: `+0.72s`
  - Silent-doom window: `+0.02s`
  - Entropy signal: flat, state-independent std.
- `IPFD_LEARNED_STATUS`:
  - `real_learned_policy: YES`
  - `detector_alarm_fired: YES`
  - `ponr_detected: YES`
  - `primary_integrity_max_delta_m: 0.0`
- CLI exit: `0`
- Transcript SHA-256: `9f195650a0bc10b89174c21923ae96cc4db9ba77568e7b7609d72b1d181148db`
- Saved rollout SHA-256: `de7903782e7c7fa43a6440ef599b396aa54537d5bb79c5ccbfad973ef9843463`

Validated:

- Official checkpoint resolution/loading.
- Packaged adapter `collect_rollout`.
- Adapter recovery-probe isolation metadata.
- Rollout recording.
- Detector/report output on live learned-policy data.
- CLI success exit behavior.

### 4. Replay serialization check

Command:

```bash
PYTHONPATH=src python3 - <<'PY'
from ipfd.replay import load_rollout, save_rollout
from ipfd import build_report
from pathlib import Path
import hashlib, tempfile
path = Path('.ipfd_live_validation_tmp/learned_teleport_rollout.npz')
rollout = load_rollout(path)
report = build_report(rollout)
json_text = report.to_json(indent=2) + '\n'
print('REPLAY_LOAD_OK:', True)
print('T:', rollout.T)
print('seed:', rollout.seed)
print('t_ponr:', report.t_ponr)
print('t_failure:', report.t_failure)
print('t_alarm:', report.t_alarm)
print('primary_integrity_max_delta:', report.meta.get('primary_integrity_max_delta'))
print('report_sha256:', hashlib.sha256(json_text.encode()).hexdigest())
with tempfile.NamedTemporaryFile(suffix='.npz') as f:
    save_rollout(rollout, f.name)
    rt = load_rollout(f.name)
    print('ROUNDTRIP_REPORT_STABLE:', build_report(rt).to_json() == report.to_json())
PY
```

Observed result:

- `REPLAY_LOAD_OK: True`
- `T: 58`
- `seed: 0`
- `t_ponr: 56`
- `t_failure: 57`
- `t_alarm: 20`
- `primary_integrity_max_delta: 0.0`
- `ROUNDTRIP_REPORT_STABLE: True`
- Report JSON SHA-256: `3c8dfdc835918f0077ec9ba04eae69b27eefdc7840cc520efd642c69fa8b82a7`
- Transcript SHA-256: `ef1b4d54fdadf6840e1f9db836f674767371805664b9591f469f65c9bae6f987`
- CLI exit: `0`

Validated:

- GPU-free replay loading.
- Replay serialization round trip.
- Report stability after save/load.

### 5. Focused CPU regression tests

Command:

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_isaac_adapter.py tests/test_replay_fixture.py tests/test_validation.py
```

Observed result:

- `38 passed, 1 warning in 0.31s`
- Warning: local Matplotlib `Axes3D` import warning unrelated to IPFD behavior.
- CLI exit: `0`
- Transcript SHA-256: `22bcc09504952fa924dc09570250aa8f838848ce2202b25d911a2b1d548b5602`

Validated:

- Environment-independent adapter indexing/error checks.
- Replay fixture report stability.
- Input validation behavior.

## Supported-version check result

- The documented compatibility target is Isaac Lab `4.5.22`; the installed runtime is `4.5.22`.
- The live simulator package is `isaacsim 6.0.0.0`.
- The adapter has an import/runtime gate for Isaac Lab availability; the live smoke run passed that gate.
- No exact-version enforcement code was changed or added in this validation pass.

## Final live status

- Live Isaac reproduced: yes.
- Strongest supported live result reproduced:
  - Scripted dual-env PoNR: `overall_status: VERIFIED`.
  - Learned-policy packaged adapter: PoNR `56`, failure `57`, alarm `20`, primary integrity `0.0`.
- Blockers: none for this machine.
- Non-blocking environment warnings:
  - `--headless` is deprecated by current AppLauncher but still accepted.
  - CPU governor is `powersave`.
  - Isaac startup logs a missing `isaaclab_visualizers` extension config.
  - Matplotlib emitted an `Axes3D` import warning during CPU-only checks.

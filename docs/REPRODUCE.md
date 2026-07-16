# Reproducing IPFD's published claims

This page lets an independent lab verify **every major IPFD claim** with **no author
interaction** and **deterministic** expected output. Parts A and B need **no GPU and
no Isaac Lab** and reproduce the exact published numbers from recorded real rollouts.
Part C (optional) re-derives those rollouts from scratch on a GPU.

Exact environment for the committed live-validation evidence:
Isaac Lab **4.5.22**, `isaacsim` **6.0.0.0**, task `Isaac-Lift-Cube-Franka-v0`,
NVIDIA's official published `rsl_rl` checkpoint, single CUDA GPU. The analysis
layer that produces the report is pure NumPy and version-tolerant (Python 3.10 /
3.11).

---

## Part A: Deterministic, GPU-free (the core claim)

```bash
git clone https://github.com/yusufdxb/ipfd && cd ipfd
pip install -e ".[dev]"
pytest tests/test_replay_fixture.py -v
```

`tests/fixtures/*_rollout.npz` are **real rollouts** captured from a live Isaac Lab
session (arrays only: observations, actions, entropy, embeddings, recovery-probe
outcomes). The committed fixture hashes are recorded in
[`tests/fixtures/manifest.json`](tests/fixtures/manifest.json). The test reloads
them with `ipfd.replay.load_rollout` - no simulator - re-runs the full analysis,
and asserts the report is **byte-for-byte** identical to the frozen golden JSON in
`tests/fixtures/`.

**Expected:** `7 passed`. This alone reproduces the two headline results below.

You can also reproduce them by hand:

```bash
python3 - <<'PY'
from ipfd import build_report
from ipfd.replay import load_rollout
for case in ("learned_teleport", "learned_slip"):
    r = build_report(load_rollout(f"tests/fixtures/{case}_rollout.npz"))
    print(case, "PoNR=", r.t_ponr, "t_failure=", r.t_failure,
          "ponr_lead_s=", r.ponr_lead_time_s,
          "integrity=", r.meta["primary_integrity_max_delta"])
PY
```

**Expected output (exact):**

```
learned_teleport PoNR= 56 t_failure= 57 ponr_lead_s= 0.72 integrity= 0.0
learned_slip PoNR= None t_failure= 219 ponr_lead_s= None integrity= 0.0
```

## Part B: The installed package (`pip install`)

```bash
python3 -m pip install build
python3 -m build
python3 -m venv /tmp/ipfd-check && /tmp/ipfd-check/bin/pip install dist/*.whl
cp tests/fixtures/learned_teleport_rollout.npz /tmp/ipfd-check/ && cd /tmp/ipfd-check
./bin/python -c "from ipfd import build_report; from ipfd.replay import load_rollout; \
print(build_report(load_rollout('learned_teleport_rollout.npz')).t_ponr)"
```

**Expected:** `56`, reproduced from the built wheel with the source tree off the path.

## Part C: Optional, re-derive the rollouts on a GPU

Requires Isaac Lab 4.5.22 + a CUDA GPU. This regenerates the rollout arrays from
scratch and confirms they still produce the shipped fixtures' numbers.

```bash
# Re-run the driver; --save_rollout writes a fresh .npz you can diff against the fixture.
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
    --headless --use_pretrained --probe --failure teleport \
    --save_rollout /tmp/fresh_teleport.npz
#   -> expect: ponr_detected: YES ,  primary_integrity_max_delta_m: 0.0
python3 -c "from ipfd import build_report; from ipfd.replay import load_rollout; \
print(build_report(load_rollout('/tmp/fresh_teleport.npz')).t_ponr)"   # -> 56

# The standalone Isaac Lab reset_to / contact-state finding (no IPFD imports):
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/isaaclab_reset_to_contact_mre.py --headless
#   -> expect: "RESULT: BUG REPRODUCED" with an exact state round-trip (~0) but a
#      growing trajectory gap, showing scene.get_state() omits contact/solver state.
```

---

## Claims → checks

| Published claim | Check | Deterministic expected output |
|---|---|---|
| Analysis layer runs with no GPU / no Isaac Lab | `pip install -e ".[dev]"; pytest` | test suite passes |
| Recorded rollout → report is stable byte-for-byte | `pytest tests/test_replay_fixture.py` | `7 passed` |
| Irrecoverable teleport localizes at PoNR **step 56**, +0.72 s alarm-before-PoNR lead | Part A hand-run | `learned_teleport PoNR= 56 ... ponr_lead_s= 0.72` |
| Env-isolated probe never perturbs the primary rollout | Part A hand-run | `integrity= 0.0` (both cases) |
| Recoverable slip yields **no** PoNR (negative control) | Part A hand-run | `learned_slip PoNR= None` |
| `pip install` reproduces the result end-to-end | Part B | `56` |
| `scene.reset_to(get_state())` omits contact/solver state | Part C MRE | `RESULT: BUG REPRODUCED` |

No step needs the author. If any deterministic output above differs on your machine,
that is a reportable finding: open an issue with your platform and the diff.

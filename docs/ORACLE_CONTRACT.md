# The recovery-oracle contract (bring your own)

IPFD localizes a **point of no return (PoNR)**: the first timestep from which the
task can never again be recovered. PoNR is defined *relative to a recovery
oracle* you supply. IPFD ships two working oracles
(`ipfd.oracles.pick_lift_sm`, `ipfd.oracles.rsl_rl_policy`), but the contract you
implement to run IPFD on your own task lives in the code. This page documents that
contract on the current API. It does **not** propose a new one.

## What an oracle is

An oracle answers one question at a saved state: *starting here, can the task
still be reached within a fixed budget?* In the packaged Isaac Lab adapter
(`ipfd.adapters.isaac_lab.collect_rollout`) you provide it as **two callables**:

| Piece | Signature | Role |
|-------|-----------|------|
| **Recovery controller** | `recovery_policy(obs) -> actions` (batched, all envs) | Drives the probe env for `probe_budget` steps trying to recover. |
| **Success predicate** | A caller-supplied `recovery_check(env, env_idx, rest_height, lift_threshold) -> bool`; height-only is a legacy fallback | Recovery succeeds only when the supplied predicate says the task is physically recoverable. The default height-only predicate is retained for compatibility but must not be used for out-of-reach or airborne disturbances. |

The adapter uses the caller-supplied physical recovery predicate as
`recovered(env, i)`. If no predicate is supplied, the legacy fallback is
`(object_height(env, i) - rest_height) > lift_threshold`; this fallback is not
safe for teleport or airborne-object experiments because height is not proof of
grasp, reachability, or task success. Production experiments must provide a
predicate that includes task success or equivalent grasp/reachability evidence.

## Exact meaning of `recovery_success[t]`

`Rollout.recovery_success` is a `(T,)` boolean array.

- `recovery_success[t] == True` means the supplied controller and physical
  predicate established recovery from step `t`.
- `recovery_success[t] == False` means only that this controller failed within
  this budget. It is not proof that no controller can recover.

A stronger controller can move the measured PoNR later. When `True` verdicts are
physically sound, the oracle-relative PoNR timestep is a lower bound on the
optimal-control PoNR timestep.

**PoNR = the last `True` → `False` flip.** `point_of_no_return` returns the index
just after the last recoverable step (`None` if the task never becomes
permanently doomed, or if no probe ran). Because probing is strided
(`probe_stride`), unprobed steps inherit the most recent verdict
(`forward_fill_recovery`), and steps before the first probe default to `True`
(recoverable until proven otherwise).

## Fixed-budget semantics

Each probe restores the saved state into the isolated probe env and steps the
controller up to `probe_budget` times:

- **First success wins**: the probe returns `True` the first step the predicate
  holds.
- **Early termination fails**: if the probe env emits `done` before success, the
  probe returns `False`.
- **Budget exhaustion fails**: if the budget runs out with no success, `False`.

The budget is part of the oracle definition: PoNR is only meaningful *for that
budget*. State it when you report results.

## Isolation requirement

The probe uses **environment isolation** and needs `num_envs >= 2`: env 0 is the
primary whose rollout is recorded before probing (and never `reset_to`), while
env 1 is the probe cell. A
historical single-env probe changed the later trajectory after evolved,
contact-rich state. The experiment did not isolate the missing simulator or task
state (see `scripts/isaaclab_reset_to_contact_mre.py`).
`collect_rollout` raises if you pass a `recovery_policy` with `num_envs < 2`.

## A conforming oracle for a new pick-and-place variant

Copy and adapt. This wires a variant (say, a taller target object) to the current
API. The two knobs you change are the **controller** and the **success
predicate**; nothing else moves.

```python
import torch
from ipfd.adapters.isaac_lab import collect_rollout, make_pick_lift_recovery_check

# 1. Success predicate: height of YOUR object in a given env (origin-corrected).
#    root_pos_w is a warp array; wp.to_torch gives an (num_envs, 3) tensor.
import warp as wp

def object_height(env, env_idx):
    origin_z = float(env.unwrapped.scene.env_origins[env_idx, 2].item())
    pos = wp.to_torch(env.unwrapped.scene["object"].data.root_pos_w)
    return float(pos[env_idx, 2].item()) - origin_z

REST_HEIGHT = object_height(env, 0)   # settled Z of your object, measured once
LIFT_THRESHOLD = 0.08                 # rise counted as a successful recovery [m]

# 2. Recovery controller: any batched callable obs -> actions. A trained policy,
#    a scripted state machine, whatever adjudicates "can this be salvaged?".
#    Here: a placeholder that returns your loaded recovery controller's actions.
def recovery_policy(obs):
    return my_recovery_controller(obs["policy"])   # -> (num_envs, act_dim) tensor

# 3. Run IPFD. env must be a manager-based RL env with num_envs >= 2.
recovery_check = make_pick_lift_recovery_check(
    workspace_radius=0.85,
    max_ee_distance=0.15,
    sustain_steps=8,
)
rollout = collect_rollout(
    env,
    policy=my_task_policy,           # the policy under test (drives env 0)
    object_height=object_height,
    rest_height=REST_HEIGHT,
    lift_threshold=LIFT_THRESHOLD,
    recovery_check=recovery_check,
    recovery_policy=recovery_policy, # omit to skip PoNR entirely
    probe_budget=90,                 # fixed recovery budget (steps)
    probe_stride=8,                  # probe every 8th step
    probe_repeats=3,                 # raw repeated verdicts per checkpoint
)

from ipfd import point_of_no_return
t_ponr = point_of_no_return(rollout.recovery_success)   # int or None
```

Running the probe requires a live Isaac Lab GPU env, so this example is
**GPU-gated**: the analysis half of IPFD (detectors, PoNR, metrics, report) is
pure NumPy and runs anywhere on a `Rollout` you already have. See
[`REPRODUCE.md`](REPRODUCE.md) for the GPU-free replay path.

## The minimal Protocol

If you build a probe outside `collect_rollout`, the only structural contract is
`ipfd.ponr.RecoveryProbe`:

```python
class RecoveryProbe(Protocol):
    def can_recover(self, saved_state: object) -> bool: ...
```

Restore `saved_state`, run your controller for a fixed budget, return whether the
goal was reached. Feed the per-step verdicts into `recovery_success` and IPFD does
the rest.

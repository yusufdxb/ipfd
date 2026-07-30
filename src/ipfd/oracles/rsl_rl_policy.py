"""Adapter: drive IPFD with a trained rsl_rl policy.

This turns a checkpoint trained by Isaac Lab's ``scripts/reinforcement_learning/
rsl_rl/train.py`` into something IPFD can instrument. It captures the three
signals IPFD's detectors consume, per environment, per step:

  * the **action** the policy emits (its inference mean),
  * a scalar **entropy** proxy from the policy's action-noise std (Gaussian
    differential entropy), and
  * a latent **embedding** = the input to the policy's final linear layer (its
    penultimate activation), grabbed with a ``forward_pre_hook`` so no
    reimplementation of the network is required.

The embedding hook is deliberately structure-agnostic: it finds the last
``nn.Linear`` in the loaded policy module and taps its input. That works whether
the policy is a plain MLP or wrapped, and survives rsl_rl API churn.

Honest note on entropy: if the trained policy uses a *state-independent* action
std (the rsl_rl default for this task), the entropy proxy is constant across the
rollout and the entropy-collapse detector will simply not fire. IPFD reports that
rather than pretending otherwise; the action-variance and drift detectors still
carry signal.

Everything here imports ``torch`` and ``rsl_rl`` lazily and is **not** re-exported
from :mod:`ipfd.oracles`, so the pure analysis layer stays GPU-free.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["LearnedPolicy", "load_learned_policy"]


class LearnedPolicy:
    """Wrap a loaded rsl_rl inference policy and expose IPFD's per-step signals.

    Call it like the rsl_rl policy (``actions = learned(obs)``); after each call
    the most recent penultimate embedding and per-env entropy are available on
    :attr:`last_embedding` and :attr:`last_entropy` (both indexed by env).
    """

    def __init__(self, policy_callable: Any, policy_module: Any) -> None:
        import torch  # noqa: F401

        self._policy = policy_callable
        self._module = policy_module
        self.last_embedding: np.ndarray | None = None  # (num_envs, feat)
        self.last_entropy: np.ndarray | None = None  # (num_envs,)
        self._emb_cache: Any = None
        self._install_embedding_hook()

    def _install_embedding_hook(self) -> None:
        import torch.nn as nn

        last_linear = None
        for m in self._module.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            return  # no linear layer found; embeddings stay None (drift disabled)

        def pre_hook(_module: Any, args: tuple) -> None:
            if args:
                self._emb_cache = args[0].detach()

        last_linear.register_forward_pre_hook(pre_hook)

    def _action_std(self) -> Any:
        """Best-effort per-action std tensor, or ``None`` if unavailable."""
        for attr in ("output_std", "action_std", "std"):
            val = getattr(self._module, attr, None)
            if val is not None:
                return val
        dist = getattr(self._module, "distribution", None)
        return getattr(dist, "stddev", None) if dist is not None else None

    def __call__(self, obs: Any) -> Any:
        import torch

        with torch.inference_mode():
            actions = self._policy(obs)
        # Return a normal tensor: policy() runs under inference_mode, so its output
        # is an inference tensor that callers cannot modify in place (e.g. to inject
        # a gripper slip). Cloning outside the context yields a mutable tensor.
        actions = actions.clone()

        if self._emb_cache is not None:
            self.last_embedding = self._emb_cache.float().cpu().numpy()

        std = self._action_std()
        if std is not None:
            std_t = std.detach().float()
            # Gaussian differential entropy: 0.5 * sum(log(2*pi*e*std^2)).
            ent = 0.5 * torch.sum(torch.log(2 * math.pi * math.e * std_t**2 + 1e-12), dim=-1)
            if ent.ndim == 0:  # global std -> broadcast to all envs (constant signal)
                n = actions.shape[0]
                self.last_entropy = np.full((n,), float(ent.item()), dtype=np.float64)
            else:
                self.last_entropy = ent.cpu().numpy().astype(np.float64)
        return actions

    def reset(self, dones: Any) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset(dones)


def load_learned_policy(env: Any, agent_cfg: dict, checkpoint_path: str, device: str = "cuda") -> LearnedPolicy:
    """Load an rsl_rl checkpoint through the canonical Isaac Lab runner path.

    Args:
        env: An ``RslRlVecEnvWrapper``-wrapped Isaac Lab env.
        agent_cfg: The rsl_rl runner config as a plain dict (``agent_cfg.to_dict()``).
        checkpoint_path: Path to a ``model_*.pt`` file.
        device: Torch device string.

    Returns:
        A :class:`LearnedPolicy` ready to drive the env and feed IPFD.
    """
    from rsl_rl.runners import OnPolicyRunner

    runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)
    _load_checkpoint_compat(runner, checkpoint_path, device)
    policy_callable = runner.get_inference_policy(device=device)
    policy_module = runner.alg.get_policy()
    return LearnedPolicy(policy_callable, policy_module)


def _load_checkpoint_compat(runner: Any, checkpoint_path: str, device: str) -> None:
    """Load current and legacy Isaac Lab rsl_rl checkpoint layouts.

    Isaac Lab's published Lift-Cube checkpoint historically used
    ``model_state_dict`` with ``actor.*`` keys, while newer rsl_rl runners expect
    ``actor_state_dict``. Keep the compatibility mapping in the packaged loader
    so the public learned-policy path does not depend on a private script hack.
    """
    import torch

    try:
        payload = torch.load(checkpoint_path, weights_only=True, map_location=device)
    except TypeError:
        raise RuntimeError(
            "This torch version cannot safely load checkpoints. Install torch >=2.0 "
            "or convert the checkpoint to a tensor-only format."
        ) from None
    except Exception as exc:
        raise RuntimeError(
            "Checkpoint rejected by safe tensor-only loading. Refusing arbitrary "
            "pickle execution; convert it with a trusted tool first."
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dictionary.")

    actor = runner.alg.get_policy()
    current = payload.get("actor_state_dict")
    if current is not None:
        if not isinstance(current, dict):
            raise TypeError("checkpoint actor_state_dict must be a dictionary")
        actor.load_state_dict(current, strict=True)
        print("Loaded current rsl_rl actor checkpoint through safe tensor-only loading.")
        return

    legacy = payload.get("model_state_dict")
    if not isinstance(legacy, dict):
        raise KeyError(
            "Checkpoint has neither 'actor_state_dict' nor legacy 'model_state_dict': "
            f"{sorted(payload)}"
        )

    mapped = {}
    for target_key, target_value in actor.state_dict().items():
        if target_key.startswith("mlp."):
            source_key = f"actor.{target_key.removeprefix('mlp.')}"
        elif target_key == "distribution.std_param":
            source_key = "std"
        else:
            raise KeyError(f"No legacy checkpoint mapping for actor key: {target_key}")
        if source_key not in legacy:
            raise KeyError(f"Legacy checkpoint is missing actor key: {source_key}")
        if legacy[source_key].shape != target_value.shape:
            raise ValueError(
                f"Shape mismatch for {source_key} -> {target_key}: "
                f"checkpoint {tuple(legacy[source_key].shape)}, actor {tuple(target_value.shape)}"
            )
        mapped[target_key] = legacy[source_key]

    actor.load_state_dict(mapped, strict=True)
    print("Loaded published legacy rsl_rl checkpoint through strict actor-only mapping.")


def checkpoint_sha256(path: str | Path) -> str:
    """Return a SHA-256 digest for checkpoint provenance records."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

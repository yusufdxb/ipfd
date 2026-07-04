"""Rollout sources.

- ``synthetic``  : pure-NumPy rollouts for tests/examples/CI (no simulator).
- ``isaac_lab``  : real Franka pick-and-place rollout collection in Isaac Lab
                   (requires a GPU + Isaac Lab install; imported lazily).
"""

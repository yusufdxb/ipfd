# Third-party licenses

IPFD is licensed under the MIT License (see [`LICENSE`](LICENSE)). It vendors one
third-party component, which retains its original license.

## Isaac Lab — `src/ipfd/oracles/pick_lift_sm.py`

`src/ipfd/oracles/pick_lift_sm.py` is reproduced verbatim from Isaac Lab
(`scripts/environments/state_machine/lift_cube_sm.py`) and remains under its
original license:

```
Copyright (c) 2022-2026, The Isaac Lab Project Developers.
SPDX-License-Identifier: BSD-3-Clause
```

It is vendored so IPFD's recovery oracle is the proven upstream reference
controller rather than a reimplementation. See that file's header for details.

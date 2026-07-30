# IPFD validation checklist

Run these steps and paste the output into a
[**Tested on my machine**](https://github.com/yusufdxb/ipfd/discussions) discussion. You do **not** need to
understand IPFD's internals; every step prints a deterministic evidence block you
can copy verbatim.

Steps 1 to 4 need **no GPU and no Isaac Lab**. Steps 5 to 7 need Isaac Lab and a CUDA GPU.

## Analysis layer (CPU only)

```bash
# 1. Clone
git clone https://github.com/yusufdxb/ipfd && cd ipfd

# 2. Install (analysis layer only)
pip install -e ".[dev]"

# 3. Run the tests  -> expect the suite to pass
pytest

# 4. Run the synthetic example
python3 examples/run_synthetic.py
```

**Expected from step 4:** two reports printed to the terminal: one
`SILENT COLLAPSE` (failure) report and one `nominal` (success) report, plus two
figures written to `examples/figures/` (`silent_failure.png`, `success.png`).

## Learned-policy revalidation (compatible Isaac Lab runtime + CUDA GPU)

```bash
# 5. Runtime smoke test  -> expect the block:  IPFD_RUNTIME_SMOKE: overall PASS
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless

# If the runtime needs the historically tested Isaac 4.5 asset tree:
ASSET_ROOT=https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless --asset_root "$ASSET_ROOT"

# 6. Competence preflight on the exact checkpoint used below
CHECKPOINT=/path/to/model.pt
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/eval_checkpoint.py \
    --headless --checkpoint "$CHECKPOINT" --num_envs 64 --sustain_steps 10 \
    --json artifacts/competence.json --asset_root "$ASSET_ROOT"

# 7. Learned-policy demo, irrecoverable case
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
    --headless --checkpoint "$CHECKPOINT" --probe --probe_repeats 3 --failure teleport \
    --save_plot artifacts/timeline.png --save_rollout artifacts/teleport.npz \
    --json artifacts/teleport.json \
    --asset_root "$ASSET_ROOT"

# 8. Negative control, recoverable case
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
    --headless --checkpoint "$CHECKPOINT" --probe --probe_repeats 3 --failure slip \
    --save_plot artifacts/timeline_slip.png --save_rollout artifacts/slip.npz \
    --json artifacts/slip.json
```

If Isaac Lab is not installed, step 5 exits cleanly with a message pointing here;
that is expected, not a failure.

## Report your result

- **It worked:** open a [Tested on my machine](https://github.com/yusufdxb/ipfd/discussions) discussion; see
  [`docs/TESTED_ON_MY_MACHINE.md`](TESTED_ON_MY_MACHINE.md).
- **Version/platform mismatch or a crash:** file a
  [compatibility report](https://github.com/yusufdxb/ipfd/issues/new?template=compatibility_report.yml); see
  [`docs/COMPATIBILITY.md`](COMPATIBILITY.md).

## What "success" means

| Evidence | Confirms |
|---|---|
| Passing tests (step 3) | The pure-NumPy analysis layer is intact on your Python. |
| Two reports + two PNGs (step 4) | The end-to-end report + plot path works with no GPU. |
| `IPFD_RUNTIME_SMOKE: overall PASS` (step 5) | IPFD attaches to a real Isaac Lab rollout. |
| Competence JSON (step 6) | The exact checkpoint held a sustained final lift at the declared rate. |
| Complete run JSON with raw repeated verdicts (steps 7 and 8) | The run is eligible for multi-seed aggregation. One run is not release evidence. |
| Passing release evidence gate | The complete declared competence, multi-seed, and actionability contracts were satisfied. |

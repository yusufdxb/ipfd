# IPFD validation checklist

Run these steps and paste the output into a
[**Tested on my machine**](../../../discussions) discussion. You do **not** need to
understand IPFD's internals — every step prints a deterministic evidence block you
can copy verbatim.

Steps 1–4 need **no GPU and no Isaac Lab**. Steps 5–7 need Isaac Lab + a CUDA GPU.

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

**Expected from step 4:** two reports printed to the terminal — one
`SILENT COLLAPSE` (failure) report and one `nominal` (success) report — and two
figures written to `examples/figures/` (`silent_failure.png`, `success.png`).

## Learned-policy demo (Isaac Lab 4.5.22 + CUDA GPU)

```bash
# 5. Runtime smoke test  -> expect the block:  IPFD_RUNTIME_SMOKE: overall PASS
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_isaac_runtime.py --headless

# 6. Learned-policy demo, irrecoverable case
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
    --headless --use_pretrained --probe --failure teleport --save_plot timeline.png
#    -> expect:  ponr_detected: YES ,  primary_integrity_max_delta_m: 0.0

# 7. Negative control, recoverable case
OMNI_KIT_ACCEPT_EULA=YES python3 scripts/verify_learned_policy.py \
    --headless --use_pretrained --probe --failure slip --save_plot timeline_slip.png
#    -> expect:  ponr_detected: NO
```

If Isaac Lab is not installed, step 5 exits cleanly with a message pointing here —
that is expected, not a failure.

## Report your result

- **It worked:** open a [Tested on my machine](../../../discussions) discussion — see
  [`docs/TESTED_ON_MY_MACHINE.md`](TESTED_ON_MY_MACHINE.md).
- **Version/platform mismatch or a crash:** file a
  [compatibility report](../../../issues/new?template=compatibility_report.yml) — see
  [`docs/COMPATIBILITY.md`](COMPATIBILITY.md).

## What "success" means

| Evidence | Confirms |
|---|---|
| Passing tests (step 3) | The pure-NumPy analysis layer is intact on your Python. |
| Two reports + two PNGs (step 4) | The end-to-end report + plot path works with no GPU. |
| `IPFD_RUNTIME_SMOKE: overall PASS` (step 5) | IPFD attaches to a real Isaac Lab rollout. |
| `ponr_detected: YES`, `primary_integrity_max_delta_m: 0.0` (step 6) | PoNR localizes the injected irrecoverable fault, and the probe never perturbed the recorded rollout. |
| `ponr_detected: NO` (step 7) | The recoverable case correctly yields no Point of No Return. |

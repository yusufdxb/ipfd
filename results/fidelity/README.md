# Counterfactual fidelity evidence

`corrected_five_seed_decisions.jsonl.xz` is an xz-compressed, exact-content copy
of the corrected study's per-branch decision records. It is committed so the CPU
fidelity audit is reproducible from a clean clone without requiring the 10.2 MB
uncompressed working artifact.

| Identity | SHA-256 |
|---|---|
| Compressed file | `4a2d8338347e9940aa3eb88dd7da12413a11c6aafb26274ec9a626d7adc48018` |
| Decoded JSONL content | `1c55862ce4a24bf564396c8d86873b13fc1667e491aada27bfff6c8d2166ce8f` |
| Generated primary audit | `f341c5eed2d06a71e2d19946ebc61446275655e32cb8039c1841dfaf653098c7` |

The decoded digest matches the immutable source identity in
[`../branch_validity/corrected_five_seed/artifact_manifest.json`](../branch_validity/corrected_five_seed/artifact_manifest.json).
Study context is recorded separately in
[`../branch_validity/corrected_five_seed/study_provenance.json`](../branch_validity/corrected_five_seed/study_provenance.json).

Run the paired primary analysis with:

```bash
ipfd fidelity results/fidelity/corrected_five_seed_decisions.jsonl.xz \
  --continuation exact_action \
  --predicate sustained_lift \
  --group-by protocol,continuation \
  --compare-protocols scene_plus_basic_manager_state,expanded_runtime_state \
  --minimum-independent-seeds 5 \
  --max-disagreement 0.05 \
  --provenance results/branch_validity/corrected_five_seed/study_provenance.json
```

Compression changes the container digest but not the decoded evidence. The
analyzer records both identities in its evidence manifest.

[`corrected_five_seed_primary_audit.json`](corrected_five_seed_primary_audit.json)
is the machine-readable output of the command above with 10,000 bootstrap draws
and random seed `20260729`. Its manifest identifies implementation commit
`e407269b488990a733f13e91e6bd27f3cc49f26b` and records that the worktree was
clean before the artifact was written.

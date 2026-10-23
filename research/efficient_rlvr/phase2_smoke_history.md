# Phase 2 local smoke attempt history

All attempts used the local WSL2 RTX 3060 only. No cloud or paid GPU was used. Failure reports and run directories are preserved; the successful one-update smoke is a correctness check, not a model-quality result.

| Attempt | Status | Outcome |
|---|---|---|
| Offline tokenizer bootstrap | Failed before generation | Transformers attempted an HF metadata lookup despite offline intent. Fixed by resolving and loading the pinned local snapshot path directly. No completed report was written. |
| Reward callback interface | Failed before backward/update | The callback assumed a `completion_text` argument that TRL did not pass. Fixed by decoding the completion with the pinned processing class. No completed report was written. |
| Group4, historical prompt | Failed learning-signal check | One optimizer step ran, but all sampled completions were unmarked under the strict historical parser: zero rewards, zero gradient, and no adapter parameter change. Preserved in [`efficient_rlvr_phase2_smoke.json`](../../reports/efficient_rlvr_phase2_smoke.json). |
| Group8, short arithmetic prompt | Failed learning-signal check | The group returned identical correct answers and constant CGLR reward, so reward variance and gradient were zero. Preserved in [`efficient_rlvr_phase2_smoke_group8.json`](../../reports/efficient_rlvr_phase2_smoke_group8.json). |
| Group8, reasoning prompt | Failed learning-signal check | Completions did not satisfy the exact answer marker; all rewards and gradients were zero. Preserved in [`efficient_rlvr_phase2_smoke_group8_reasoning.json`](../../reports/efficient_rlvr_phase2_smoke_group8_reasoning.json). |
| Group4 with bounded mixed-group resampling | Passed implementation smoke | One optimizer update had a nonzero gradient, changed 2,750,317 LoRA scalar values, retained one mixed group, counted 371 attempted vs 237 retained completion tokens, saved checkpoint-1, and reloaded adapter tensors exactly. See [`efficient_rlvr_phase2_smoke_group_filter.json`](../../reports/efficient_rlvr_phase2_smoke_group_filter.json). |

The successful run found one all-correct group and discarded it, then found a mixed group and used it. This verifies the accepted path. A CPU unit test covers candidate exhaustion and `effective_group=False`; the pinned TRL source marks all-`None` reward rows unscorable and forces their advantage to zero. A full run must still log its actual exhaustion rate and effective-group yield.

## Reproducing the successful attempt

The saved run configuration contains the exact seed, model revision, data hash, package versions, and limits. Run the command in `phase2_design.md` only with a fresh empty run directory; the script intentionally refuses to overwrite an existing attempt.

# Phase 1 saved-trajectory audit (post-hoc)

> This offline analysis is explicitly post-hoc. It does not revise archived scores, parser outputs, model selection, or evaluation rules.

All per-completion token counts below are reconstructed from decoded text using the pinned Qwen3 tokenizer. Original token IDs were not retained. The answer span is the unique marker accepted by the original verifier; unparsed rows have unknown answer boundaries. Post-answer text is a structural tail metric, not a semantic redundancy label.

## gsm8k_seed1

Evaluation data SHA-256: `eb9e38a7728c28d1f9c7fdd6a907aad06b89c2ab04f0de922a31406f655540eb`

| Model | Original exact match | Parsed | Mean decoded-text tokens, correct / incorrect | Mean tokens before answer | Mean answer span | Mean tokens after answer | Parsed outputs with nonempty tail | Recorded generated tokens | Wall time | tokens/s | tokens per correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 99/1000 (9.9%) | 727/1000 (72.7%) | 99.8 / 46.4 | 35.8 | 8.3 | 0.2 | 87/727 | 52,660 | 16.2 min | 54.23 | 531.9
| adapter | 354/1000 (35.4%) | 740/1000 (74.0%) | 116.4 / 142.4 | 111.0 | 7.5 | 0.6 | 352/740 | 134,062 | 74.1 min | 30.16 | 378.7

Exact repetition proxy for base (literal repeated sentence-sized text only): 2/1000 outputs; 124 duplicate-copy tokens total, 0 before accepted answer and 0 after it, and 124 in unparsed outputs with unknown boundaries. This misses paraphrase or repeated equations and is not a complete redundancy measure.

Exact repetition proxy for adapter (literal repeated sentence-sized text only): 12/1000 outputs; 163 duplicate-copy tokens total, 18 before accepted answer and 0 after it, and 145 in unparsed outputs with unknown boundaries. This misses paraphrase or repeated equations and is not a complete redundancy measure.

Paired changes (same items): adapter-only correct 287, base-only correct 32, both correct 67, both incorrect 614.

### Completion-length rank quartiles

| Model | Length quartile (short to long) | N | Mean re-tokenized tokens | Exact match | Parse rate |
|---|---:|---:|---:|---:|---:|
| base | 1 | 250 | 7.8 | 4.0% | 100.0% |
| base | 2 | 250 | 9.8 | 1.6% | 84.0% |
| base | 3 | 250 | 37.6 | 9.6% | 41.2% |
| base | 4 | 250 | 151.7 | 24.4% | 65.6% |
| adapter | 1 | 250 | 78.5 | 47.2% | 94.0% |
| adapter | 2 | 250 | 113.2 | 47.6% | 90.4% |
| adapter | 3 | 250 | 151.5 | 39.2% | 88.0% |
| adapter | 4 | 250 | 189.6 | 7.6% | 23.6% |

## svamp_seed2

Evaluation data SHA-256: `5529aa437b2c7be1f101b3a307a3eddbf07021cbda892af8b4c0320fc2b38251`

| Model | Original exact match | Parsed | Mean decoded-text tokens, correct / incorrect | Mean tokens before answer | Mean answer span | Mean tokens after answer | Parsed outputs with nonempty tail | Recorded generated tokens | Wall time | tokens/s | tokens per correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 267/1000 (26.7%) | 833/1000 (83.3%) | 63.4 / 24.1 | 28.2 | 7.7 | 0.1 | 90/833 | 35,598 | 12.2 min | 48.75 | 133.3
| adapter | 675/1000 (67.5%) | 886/1000 (88.6%) | 93.7 / 123.9 | 89.4 | 6.7 | 0.8 | 523/886 | 104,482 | 61.6 min | 28.26 | 154.8

Exact repetition proxy for base (literal repeated sentence-sized text only): 4/1000 outputs; 85 duplicate-copy tokens total, 54 before accepted answer and 0 after it, and 31 in unparsed outputs with unknown boundaries. This misses paraphrase or repeated equations and is not a complete redundancy measure.

Exact repetition proxy for adapter (literal repeated sentence-sized text only): 10/1000 outputs; 196 duplicate-copy tokens total, 39 before accepted answer and 0 after it, and 157 in unparsed outputs with unknown boundaries. This misses paraphrase or repeated equations and is not a complete redundancy measure.

Paired changes (same items): adapter-only correct 454, base-only correct 46, both correct 221, both incorrect 279.

### Completion-length rank quartiles

| Model | Length quartile (short to long) | N | Mean re-tokenized tokens | Exact match | Parse rate |
|---|---:|---:|---:|---:|---:|
| base | 1 | 250 | 7.6 | 10.4% | 100.0% |
| base | 2 | 250 | 8.9 | 2.8% | 86.0% |
| base | 3 | 250 | 30.2 | 33.6% | 53.2% |
| base | 4 | 250 | 91.7 | 60.0% | 94.0% |
| adapter | 1 | 250 | 63.0 | 80.8% | 98.4% |
| adapter | 2 | 250 | 84.5 | 79.6% | 98.8% |
| adapter | 3 | 250 | 109.3 | 67.6% | 89.2% |
| adapter | 4 | 250 | 157.4 | 42.0% | 68.0% |

## Interpretation boundaries

- The already published historical scores remain authoritative. This audit is additive and post-hoc.
- A longer decoded output can contain useful reasoning, repetition, a delayed answer, or an answer-format failure. Length alone cannot distinguish these cases.
- A nonempty post-answer tail is a candidate for manual review; the metric does not label it redundant.
- Aggregate inference costs measure the full evaluation run, including batching/runtime effects; they are not per-question latency estimates.

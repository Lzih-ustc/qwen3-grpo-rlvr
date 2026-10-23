# Saved prediction output-length audit (post-hoc)

> This offline analysis uses the saved per-question outputs. It does not modify source evaluation JSON, historical accuracy, candidate selection, or test standards.

Character and whitespace-word counts are decoded-text measures. Optional Qwen token counts are re-tokenized from decoded strings, not recovered generation IDs. Original aggregate generated tokens, elapsed time, throughput, and GPU memory remain authoritative.

## seed1_gsm8k

| Model | Accuracy | Parse rate | Mean chars (correct / incorrect) | Mean words (correct / incorrect) | Re-tokenized tokens (correct / incorrect) | Generated tokens | Wall time | tokens/s | tokens / correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 99/1000 (9.9%) | 72.7% (273 unparsed) | 281.5 / 131.1 | 54.7 / 23.9 | 99.8 / 46.4 | 52,660 | 16.2 min | 54.23 | 531.9
| adapter | 354/1000 (35.4%) | 74.0% (260 unparsed) | 315.5 / 396.6 | 61.4 / 77.9 | 116.4 / 142.4 | 134,062 | 74.1 min | 30.16 | 378.7

Paired correctness changes: both correct 67, base-only correct 32, adapter-only correct 287, both wrong 614.

| Paired outcome | Base mean chars | Adapter mean chars | Base mean words | Adapter mean words | Base re-tokenized tokens | Adapter re-tokenized tokens |
|---|---:|---:|---:|---:|---:|---:|
| Both correct | 253.6 | 299.8 | 49.0 | 56.7 | 91.3 | 111.4 |
| Base only correct | 339.9 | 413.1 | 66.6 | 79.1 | 117.5 | 144.7 |
| Adapter only correct | 72.4 | 319.2 | 12.4 | 62.5 | 27.3 | 117.6 |
| Both wrong | 158.5 | 395.8 | 29.4 | 77.8 | 55.4 | 142.3 |

## seed2_svamp

| Model | Accuracy | Parse rate | Mean chars (correct / incorrect) | Mean words (correct / incorrect) | Re-tokenized tokens (correct / incorrect) | Generated tokens | Wall time | tokens/s | tokens / correct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 267/1000 (26.7%) | 83.3% (167 unparsed) | 197.9 / 67.6 | 38.6 / 11.5 | 63.4 / 24.1 | 35,598 | 12.2 min | 48.75 | 133.3
| adapter | 675/1000 (67.5%) | 88.6% (114 unparsed) | 289.9 / 378.9 | 57.3 / 75.6 | 93.7 / 123.9 | 104,482 | 61.6 min | 28.26 | 154.8

Paired correctness changes: both correct 221, base-only correct 46, adapter-only correct 454, both wrong 279.

| Paired outcome | Base mean chars | Adapter mean chars | Base mean words | Adapter mean words | Base re-tokenized tokens | Adapter re-tokenized tokens |
|---|---:|---:|---:|---:|---:|---:|
| Both correct | 190.5 | 297.5 | 37.1 | 59.5 | 61.2 | 95.7 |
| Base only correct | 233.6 | 403.2 | 45.6 | 83.3 | 74.0 | 128.5 |
| Adapter only correct | 42.5 | 286.1 | 6.2 | 56.3 | 16.3 | 92.8 |
| Both wrong | 108.4 | 374.9 | 20.1 | 74.3 | 36.8 | 123.1 |

## Interpretation limits

- Unparsed means the saved verifier did not extract a final numeric answer (`predicted` is null); this is not silently repaired even when a human could infer the intended arithmetic.
- The evaluation protocol used greedy decoding with the same saved decode metadata for each pair. Per-question token IDs and latency were not retained, so this report does not invent them from text lengths.
- Length differences by correctness are descriptive post-hoc associations. Original exact-match values and all locked test decisions remain unchanged.

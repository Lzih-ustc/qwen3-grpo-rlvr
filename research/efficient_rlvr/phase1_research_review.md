# Efficient RLVR — Phase 1 evidence and research map

Status: Phase 1 complete; no GPU training was run in this phase. This note and the adjacent trajectory audit are additive to the frozen Phase A–C archive. The historical prediction JSON files are read-only sources; their SHA-256 hashes are recorded in `phase1_answer_boundary_audit.json`.

## Question

Can we retain the accuracy gained by RLVR while reducing tokens and end-to-end inference latency, and how should the trade-off be measured under a fixed training budget?

The Phase C selected adapters improved the historical exact-match results on their one-time tests, but used more output tokens and longer evaluation wall time than Base. Phase C selected Group8 in seed 1 and Group4 in seed 2; its group-size conclusion remains `MIXED`. Those consumed GSM8K/SVAMP test results are descriptive context only and are not available for selecting models in this branch.

## Saved trajectory audit

See [`phase1_answer_boundary_audit.md`](phase1_answer_boundary_audit.md) and the machine-readable [`phase1_answer_boundary_audit.json`](phase1_answer_boundary_audit.json). The script is [`scripts/audit_answer_boundaries.py`](scripts/audit_answer_boundaries.py).

| Frozen pair | Base → Adapter exact match | Mean tokens before accepted answer, Base → Adapter | Mean post-answer tail, Base → Adapter | Recorded generated tokens, Base → Adapter | Recorded elapsed time, Base → Adapter |
|---|---:|---:|---:|---:|---:|
| Seed 1, GSM8K-train-source test | 9.9% → 35.4% | 35.8 → 111.0 | 0.2 → 0.6 | 52,660 → 134,062 | 16.2 → 74.1 min |
| Seed 2, SVAMP test | 26.7% → 67.5% | 28.2 → 89.4 | 0.1 → 0.8 | 35,598 → 104,482 | 12.2 → 61.6 min |

The per-question token counts are reconstructed from the pinned Qwen3 tokenizer applied to saved decoded strings. Original token IDs and per-question latency were not retained. The authoritative inference costs are the aggregate generated-token totals and wall times in the frozen prediction payloads.

Length/correctness is non-monotonic in both selected-adapter pairs. On the seed-1 GSM8K pair, Adapter exact match was 47.2%, 47.6%, 39.2%, and 7.6% across length-ranked quartiles; parse rate fell from 94.0% in the shortest quartile to 23.6% in the longest. On seed-2 SVAMP, Adapter exact match was 80.8%, 79.6%, 67.6%, and 42.0%; its longest quartile parse rate was 68.0%. This suggests long outputs often coincide with answer delay, cutoff, or non-parsing, but the association is post-hoc and cannot establish that length caused the errors.

The length increase is concentrated before the verifier-accepted answer. The accepted answer span is short on average (about 6.7–8.3 reconstructed tokens). Parsed Base/Adapter outputs have under one token of mean trailing text after the marker, although nonempty tails occur in 47.6% of parsed seed-1 Adapter completions and 59.0% of parsed seed-2 Adapter completions. A literal repeated-sentence detector finds repeated copies in 12/1,000 and 10/1,000 Adapter outputs for the two pairs; this intentionally conservative measure misses paraphrase, duplicated equations, and other verbosity. A convenience sample of the longest outputs contained obvious repeated reasoning in some cap-length unparsed completions, so repetition should be audited separately from answer position. See the report for counts and limitations.

### What this evidence does and does not say

- It confirms that inference cost rose materially in these two frozen Base/Adapter comparisons; it does not identify the cause or show a general algorithmic effect.
- Much of the Adapter's reconstructed completion length occurs before the parsed final answer. That region may contain useful work, redundant work, or unfinished reasoning; it is not all “waste”.
- Very long outputs have lower parse rate and, in these samples, lower exact match than shorter length quartiles. This is a descriptive association among two fixed policies and datasets, not a causal length intervention.
- The small literal repetition count is a lower-bound style heuristic, not a semantic redundancy estimate.
- Historical outputs do not retain per-question original token IDs or latency, so future runs must save them.

## Related methods

| Method | Existing idea | Relation to this branch | Primary source |
|---|---|---|---|
| **GFPO** | Samples a larger completion group and applies response-level top-k filtering, such as shortest responses; variants rank by reward per token and adapt retained count to estimated problem difficulty. | Very close precedent for response filtering and token efficiency. The proposed Group-Filtered baseline must not be presented as a new algorithm. Distinguish GFPO's response ranking/selection from group-level correctness-mixed filtering. | [Shrivastava et al., *Sample More to Think Less*, arXiv:2508.09726](https://arxiv.org/abs/2508.09726), especially [filtering](https://arxiv.org/html/2508.09726#S3) and [token efficiency](https://arxiv.org/html/2508.09726#S5). |
| **GRPO-LEAD** | Applies a group-relative length-dependent reward to correct responses, assigns an explicit negative reward to incorrect ones, and reweights advantages by estimated difficulty. | Establishes that correctness-aware length rewards are prior work. Correctness-Gated Length Reward (CGLR) is a simpler reproducibility baseline/ablation, not an innovation claim. | [Zhang & Zuo, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.287/), [PDF](https://aclanthology.org/2025.emnlp-main.287.pdf). |
| **Step-GRPO (AAAI 2026)** | Combines sparse outcome feedback with step-level process-reward signals and step-attention over reasoning steps. | Addresses reasoning quality and reward hacking with step structure; it is more than an output-length penalty and has additional process-reward machinery. | [Li et al., AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40441). |
| **Step-GRPO (arXiv:2604.16890)** | A separate paper using dynamic truncated rollouts, semantic step quantification, and a step-aware relative reward to train internalized early exit. | Directly relevant to dynamic stopping; it creates an acronym collision with the AAAI paper above. Treat these as distinct papers. | [Chen et al., *Internalizing Dynamic Early Exit*, arXiv:2604.16890](https://arxiv.org/abs/2604.16890). |
| **DAPO** | Adds asymmetric clipping, dynamic prompt sampling to retain groups with mixed correctness, token-level loss normalization, and overlong-response reward shaping/filtering. | Group-Filtered RL based on retaining correctness-mixed groups is already the core of DAPO's dynamic sampling. Use a clearly labelled DAPO-style baseline and measure discarded rollouts and effective-group yield. | [Yu et al., *DAPO: An Open-Source LLM RL System at Scale*](https://dapo-sia.github.io/), [paper PDF](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf). |
| **Dr. GRPO** | Removes per-response length normalization and group reward standard-deviation scaling from the GRPO objective to address optimization bias. | Orthogonal to reward shaping and prompt filtering. A CGLR result can be confounded by the optimizer's length normalization; hold loss formulation fixed in the primary reward comparison and consider Dr. GRPO as a separate control only if the budget permits. | [Liu et al., *Understanding R1-Zero-Like Training*, arXiv:2503.20783](https://arxiv.org/abs/2503.20783). |

The papers report results with different model scales, token limits, sampling settings, datasets, training infrastructure, and evaluation budgets. Their headline gains are not comparable as a shared leaderboard and should not be projected onto Qwen3-0.6B or the RTX 3060 experiment.

## Proposed falsifiable baselines (not a novelty claim)

### Length-Controlled Evaluation (LCE)

For each new locked evaluation prompt, save one deterministic completion up to a common maximum output limit. Score the same token prefix at budgets such as 32/64/96/128/192 generated tokens. Report paired exact match, parse failure, truncation, actual token use, and per-question latency at each prefix. The prefix-budget curve shows whether shorter budgets preserve correctness. At each setting, use the same prompt template, Qwen3 thinking mode, parser version, decode configuration, hardware, and warm-up procedure. Going forward, preserve original token IDs so prefix evaluations do not require re-tokenization.

Also report the full completion cost. A per-question max-token cap is an equal upper bound, while actual generated tokens and elapsed time remain outcome measures; do not call these equal actual-cost runs. For strict equal total inference budgets, pre-register a fixed question order and stop only at prompt boundaries, then disclose the number of questions completed and accuracy on that fixed prefix.

### Correctness-Gated Length Reward (CGLR)

Candidate baseline for unit tests and a later validation-locked run:

```text
correct_i = verifier(completion_i, gold_i)
normalized_length_i = min(generated_tokens_i / max_completion_tokens, 1)
reward_i = correct_i * (1 - lambda * normalized_length_i)
```

Constrain `0 <= lambda < 1`, so every verified-correct answer remains strictly above an unverified/incorrect answer while shorter correct answers receive more reward. The reward does not give a brevity bonus to wrong or unparsed outputs. This is a simple testable baseline; it overlaps prior length-aware reward work. Pre-register `lambda` on training data or validation only, and never use the independent test to tune it.

### Group-Filtered RL (GFRL)

Baseline definition: sample full groups of `G` completions per prompt; retain a group only when it includes at least one verified-correct and one incorrect/unparsed completion; oversample prompts until the target effective-group count or a fixed generation budget is reached. Keep response-level filtering (GFPO) separate from this group-level filter. Record attempted/retained groups, correctness-mixed rate, discarded generated tokens, total generation tokens, tokens per retained group, optimizer updates, and wall time. This is a DAPO-style dynamic-sampling control, not a novel method.

## Questions for the next phases

1. **Reward vs. objective:** Does CGLR shorten correct outputs without reducing strict correctness, after holding the GRPO loss normalization fixed?
2. **Filter yield vs. coverage:** Does correctness-mixed group filtering improve effective updates enough to offset discarded rollout tokens and fewer unique prompts under both fixed-token and fixed-time budgets?
3. **Serving vs. training trade-off:** Is inference-token/latency saving large enough to repay any extra training cost? Report total training cost and correct solutions per GPU-hour, not just rollout reward or mean length.
4. **Stability:** Do effects persist across at least three training seeds, and are they consistent across a new validation set and a separately locked independent test set?
5. **Mechanism:** Are saved rollout reward variance, mixed-group rate, duplicate-span rate, truncation, and answer position predictive diagnostics, or merely correlated logs?

Do not use the consumed GSM8K/SVAMP tests to choose a new method. Before Phase 3, identify genuinely fresh train/validation/test sources, verify reuse and licensing, and lock the split hashes. No paid GPU should be launched without explicit approval.

## Reproduction command for this audit

This command loads only the pinned tokenizer from the local Hugging Face cache; it does not load model weights, run generation, or use CUDA:

```bash
# Run from the repository root with the pinned WSL environment activated.
HF_HUB_OFFLINE=1 python \
  research/efficient_rlvr/scripts/audit_answer_boundaries.py \
  --case gsm8k_seed1 reports/gsm8k_phasec_base_test.json reports/gsm8k_phasec_selected_test.json \
  --case svamp_seed2 reports/svamp_phasec_seed2_base_test.json reports/svamp_phasec_seed2_selected_test.json \
  --output-json research/efficient_rlvr/phase1_answer_boundary_audit.json \
  --output-markdown research/efficient_rlvr/phase1_answer_boundary_audit.md
```

The WSL environment is pinned by the existing lock file. The exact source prediction SHA-256 hashes are embedded in the output JSON. Set `--tokenizer-path` to the locally cached snapshot directory if the host's model cache layout needs an explicit path.

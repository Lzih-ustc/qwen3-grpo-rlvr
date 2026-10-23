# Phase 4B engineering gate — durable GFRL budget and local GPU smoke

Audit date: 2026-10-23

## Decision

**Local engineering gate: PASS. Paid AutoDL execution: NO-GO / not authorized.** No AutoDL instance, paid GPU, paid storage, or remote run was created. The RTX 3060 test used the local WSL2 machine only. Passing this gate permits the project to request a separate, time-capped paid preflight approval; it does not start one.

Phase A–C data splits, reported metrics, and checkpoint-selection records remain unchanged. This report supersedes the earlier Phase 4B snapshot only for the GFRL implementation and execution gate. It does not rewrite any historical experiment result.

## Implemented controls

research/efficient_rlvr/ledger.py adds a SQLite WAL ledger with full synchronous commits and idempotency keys for run, requested group, candidate index, and events. For each candidate it records:

- a maximum-token reservation before generation;
- persisted prompt/completion IDs and actual completion lengths immediately after generation returns;
- exact-verifier flags, mixed/all-correct/all-incorrect classification, retain/discard decisions, retry scheduling, and candidate exhaustion;
- exception/interruption events, current budget state, and replayable output IDs;
- immutable run budget and decoding-cap configuration.

GFRL retries use a stable run/step/group/candidate-derived prompt choice. Replaying the same callback returns the already persisted token IDs and does not invoke generation again. A second live process cannot claim an active run; SQLite transactions serialize concurrent reservations.

The hard stop requires group_size × max_completion_tokens to fit before any full group is generated. Provided the generation backend honors its registered per-completion cap, the allowed budget overshoot is **zero**. At a stop boundary, unused budget is between 0 and group_size × max_completion_tokens − 1. For the local smoke (G=4, cap 64), the worst-case unused boundary budget is 255 tokens. If a backend violates the cap, the ledger persists the returned IDs and actual overrun, marks the run interrupted, and stops; it cannot retroactively prevent tokens emitted by a misbehaving backend.

Crash semantics are deliberately conservative. If the process dies after reservation but before generated IDs are durably saved, recovery charges the complete reservation, records that the physical emitted-token count is unknown, and marks the run interrupted so it cannot generate again. This preserves the hard budget and prevents silent undercount or duplicate generation; it cannot reconstruct the exact physical token count lost between GPU output and durable commit. Such a run is not valid training evidence and must stop. If generation IDs were committed before the crash, replay uses those saved IDs and exact counts.

## Verification evidence

- **CPU suite:** all 38 offline repository tests passed: 26 Efficient RLVR tests (including 11 new ledger tests for candidate accounting, retry/discard records, whole-group stop, hard process death, recovery after durable generation, duplicate callback replay, live duplicate-process exclusion, concurrent reservation, conflicting idempotency keys/events, injected generation failure, and over-cap output handling) plus 12 repository-level agent/audit tests.
- **Local GPU smoke:** passed on WSL2 NVIDIA GeForce RTX 3060 (12 GiB), with PyTorch 2.5.1+cu121, Transformers 4.57.3, TRL 1.16.0, PEFT 0.21.2, bitsandbytes 0.50.2.
- Configuration: Qwen3-0.6B pinned revision c1899de289a04d12100db370d81485cdf75e47ca; NF4 QLoRA; LoRA rank 8; gradient checkpointing; G=4; completion cap 64; global token budget 1,024; non-vLLM generation.
- Result: one optimizer update; gradient norm 2.359; 196 LoRA parameter tensors and 2,751,530 LoRA values changed; checkpoint checkpoint-1 was saved and reloaded adapter tensors matched exactly.
- Ledger: 6 candidate groups / 24 completions (5 discarded groups and 1 mixed retained group); 814 actual completion tokens; 586 discarded and 228 retained; 4 retries; 0 unknown attempts; 0 overrun; 210 tokens left when the next 256-token group reservation was denied.
- Budget stop: surfaced after the first optimizer update because only 210 tokens remained for a possible 256-token group. No partial group started.
- Throughput: 26.76 generated completion tokens/s over the 30.42-second end-to-end training interval. This is an RTX 3060 smoke measurement, not an AutoDL 4090 estimate.
- Memory: PyTorch peak allocated 1,108 MiB and reserved 1,554 MiB; whole-GPU peak measured by nvidia-smi was 4,763 MiB, including other desktop processes. No OOM.
- Maximum observed completion length was exactly 64 and no returned completion exceeded the configured cap. This smoke does not estimate the locked validation split’s 256-token truncation rate.
- Machine-readable evidence: reports/efficient_rlvr_phase4b_gpu_smoke.json. The SQLite ledger and binary checkpoint remain local under ignored runs/.

Reproduction command from the WSL repository root (use a new empty run directory for each attempt):

```bash
HF_HUB_OFFLINE=1 .venv/bin/python research/efficient_rlvr/smoke.py \
  --run-dir runs/efficient-rlvr-phase4b-gpu-smoke-reproduction-01 \
  --report reports/efficient_rlvr_phase4b_gpu_smoke_reproduction.json \
  --generations 4 --max-completion-length 64 --token-budget 1024 \
  --max-steps 2 --max-candidate-groups 8
```

The smoke is an engineering correctness check, not a quality evaluation. No accuracy claim or model-selection decision follows from it.

## AutoDL RTX 4090 preflight plan — hard maximum 2 billed hours

The preflight is limited to one RTX 4090 24 GB instance, one execution, and at most **120 minutes from power-on to power-off**. It does not run the 18-run matrix, select a model, inspect the independent test, or authorize another paid stage.

| Elapsed billed time | Action | Evidence required | Stop condition |
|---|---|---|---|
| 00–15 min | Start the pinned environment; check GPU/runtime, load the pinned base and tokenizer, validate data hashes and config. Use ephemeral/system storage only. | Versions, hashes, load time, initial VRAM. | Abort if setup or model load exceeds 15 minutes, hash differs, CUDA is unavailable, or extra paid storage is required. |
| 15–40 min | Train-split-only generation panel (up to 100 distinct prompts, cap 256), with warmup excluded. Measure synchronized generation and end-to-end latency, throughput, completion lengths, parser failures, EOS-versus-cap termination, and cap-hit rate. | Per-prompt token IDs/timings; p50/p95 latency; tokens/s; cap-hit and answer-at-cap counts. | Stop on missing finish classification, token IDs over cap, model instability, or any budget/ledger mismatch. If cap-hit exceeds 5% or answers are commonly cut before the final answer, mark the formal matrix NO-GO and require a separately approved protocol revision. |
| 40–90 min | One G4 correctness-gated LoRA/GRPO update run, bounded by at most 8,192 attempted completion tokens and 40 active minutes. Count every rejected/retry candidate. | Finite nonzero gradient, changed adapter, one optimizer update, full ledger reconciliation, peak VRAM, training tokens/s. | Stop at first OOM, no effective mixed-reward update, non-finite gradient, unknown token count, overrun, or 40 active minutes. No sequence-length tuning or retry inside this slot. |
| 90–105 min | Save and reload the adapter/checkpoint; reopen ledger in a fresh process and replay a previously committed callback key without GPU generation. | Exact tensor equality; persisted token/event totals unchanged after replay; run config hash. | Stop on missing checkpoint, mismatch, repeated generation, changed count, or unresolved reservation. |
| 105–115 min | Export logs, hashes, cost/metering screenshot, and report; power off. | Local copy verified, instance stopped. | Begin shutdown by minute 115; do not use remaining time for extra experiments. |
| 115–120 min | Billing/termination verification reserve only. | Provider shows stopped instance and final billed duration. | If billed duration exceeds 120 min or spend is unclear, do not start another stage; resolve billing first. |

Cost planning at the last recorded public quote of ¥1.88/hour is **¥3.76 maximum compute for 2 billed hours**, before any separately billed storage or fees. Recheck the live rate before a future approval. Do not create a data disk; if a disk or other charge is mandatory, recalculate and obtain separate explicit approval before launch. The 120-minute limit is the hard ceiling, not a target. Start a local timer at power-on, begin shutdown by minute 110, and power off early when evidence is complete.

Only train rows may be used in the diagnostic panel. Validation is reserved for the later registered validation phase; the independent test remains sealed. This preflight reports system behavior, not comparative algorithm outcomes. Afterward, stop and prepare a new report. Main training cannot start without a separate explicit approval and a revised cost estimate.

## Final gate

- **Engineering correctness:** PASS for the CPU ledger/fault-injection suite and the one-update local RTX 3060 smoke.
- **Local accounting:** PASS; exact returned token counts reconcile, no replay double count, no budget overrun, no unresolved reservation.
- **RTX 4090 capacity and throughput:** UNKNOWN until an approved preflight is actually run.
- **Accuracy/truncation on locked validation:** UNKNOWN; no new validation generation was run.
- **Paid execution now:** **NO-GO.** No AutoDL instance or paid service may be started from this plan. A future preflight requires a separate user approval capped at 2 billed hours; completing it will not automatically trigger the 18-run experiment.

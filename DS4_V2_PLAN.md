# ds4 v2 platform — architectural reality check

Goal stated: **12–16 → 28–38 tok/s decode**, with **45–55 tok/s after FastMTP**.

After reading the actual ds4 source (10.7K lines of CUDA), here's what's actually feasible vs the original synthesis.

## Current architecture (what's already there)

ds4 is **not** a naive dequant→FP16→cuBLAS pipeline. The hot path is:

| Tensor class | Format | bpw | Kernel |
|---|---|---|---|
| Routed MoE up/gate experts | IQ2_XXS | 2.0625 | `dev_dot_iq2_xxs_q8_K_block*` (DP4A INT8 tensor cores) |
| Routed MoE down experts | Q2_K | 2.625 | `matmul_q2_k` family |
| Attention + shared expert | Q8_0 | 8.5 | `matmul_q8_0_preq_warp8_kernel` (cuBLAS FP16/FP32 fallback) |
| KV cache | FP16 / FP8 | 16 / 8 | hand-written MLA absorb-mode |

This is already a fairly tuned setup. The agent synthesis assumed the baseline was much weaker than it is.

## What changes the bandwidth math

For decode, ~90% of read bytes are MoE expert weights. At IQ2_XXS that's already very close to the entropy floor. Going to NVFP4 (4.5 bpw) on the same weights is a **2× regression** on the bandwidth-limited path.

NVFP4 only wins where the current format is *coarser*:
- **Q8_0 attention + shared expert** (8.5 → 4.5 bpw) — 1.89× bandwidth reduction on these tensors specifically
- **FP16 KV cache** (16 → 4.5 bpw) — 3.56× compression on KV reads at long context

Realistic expected gain after re-doing the math:
- Short context, attention+shared NVFP4: **~10–15%** total (MoE dominates)
- 32K context, KV NVFP4: **~30–50%** (KV reads become significant)
- 64K+ context: **~50–80%**

That's much more modest than "12 → 38". Honesty: agent synthesis overstated the gain because they assumed FP16 baseline.

## Where the synthesis was right

These wins survive the architectural reality:

1. **`-arch=sm_121a` rebuild** — foundational. Without the `a` suffix, even FP4 PTX gets silently demoted. Free. **DONE on branch ds4-v2-platform commit 7ce8c66.**
2. **NVFP4 KV cache** — biggest single win, scales with context length. The KV is currently FP16; moving to NVFP4 is genuinely 3.56× compression on those reads.
3. **FastMTP head retrained for DSV4** — orthogonal to all of the above. Real 1.5–2× decode wall-clock IF the head can be trained.
4. **Per-expert mixed precision** — DSV4 has 256 experts; top-8 fire per token. Right idea but the *current* IQ2_XXS is already so tight that the upside is "promote hot experts to Q4_K" rather than "demote cold ones further." Maybe +10–15% quality at fixed size, or –20% size at same quality.
5. **`nvidia-smi -lgc` clock locking** — measured +5–8% steady state. Free.

## Where the synthesis over-promised

These need re-evaluation:

1. **FlashInfer XQA MLA decode kernel port** — agent estimate "~1 week, strip JIT plumbing." Reality: 1945 lines templated CUDA C++ with `CgaBarrier`, `tma::storeAsync`, multi-warp specialized layouts. CGA barriers are sm_90+ cluster features whose sm_121 status is unverified. **Real estimate: 2–4 weeks with non-trivial chance of "doesn't compile cleanly on sm_121."** Worth attempting but not in the critical path.
2. **Marlin-style Q2_K → NVFP4 fused dequant+GEMV** — wrong layer of the stack. ds4 already does direct IQ2 × INT8 DP4A, which is faster than dequant→FP4→MMA at the same bandwidth.
3. **Native NVFP4 MMA for MoE expert GEMM** — only beats current IQ2+DP4A if NVFP4 quality is meaningfully better. At 4.5 bpw vs 2.0625 bpw it should be, but the win is *quality*, not *speed* — same bandwidth ceiling.

## Revised execution plan

Risk-ranked, highest-confidence first:

### Phase 1 — Free wins, days, low risk
- [x] sm_121a Makefile (`ds4-v2-platform` commit 7ce8c66)
- [ ] `nvidia-smi -lgc` clock locking on Spark startup
- [ ] `ncu` profile of current decode to measure where time *actually* goes
- [ ] `cudaMemPrefetchAsync` for next-layer expert weights (unified memory exclusive)
- [ ] MTP draft=2 sweep (existing infrastructure, recursive use)

**Expected: 12–16 → 15–20 tok/s.** Provides the measurement baseline for everything else.

### Phase 2 — NVFP4 KV cache, ~1 week, medium risk
- Convert KV storage from FP16/BF16 to NVFP4-packed
- Modify MLA attention kernel to dequant inside (saves the storage round-trip)
- Validate quality drift <1% on long-context benchmarks

**Expected at 32K+ context: +30–50% decode tok/s.** Doesn't help at short context.

### Phase 3 — Q8_0 attention+shared expert → NVFP4, ~1 week, medium risk
- Offline conversion: re-quantize attention + shared expert tensors to NVFP4
- Implement NVFP4 MMA inner loop (templated mma.sync.aligned.kind::mxf4nvf4.block_scale)
- Drop-in replacement in matmul_q8_0_preq_warp8_kernel call sites

**Expected: +10–15% decode at all context lengths.**

### Phase 4 — FastMTP retraining + integration, ~1 week training + 1 week integration, medium risk
- Architecture: TencentBAC/FastMTP repo. Train a new draft head on DSV4 base.
- Need calibration corpus (use existing 2K-sequence corpus from task #37)
- Integration into ds4's existing MTP path

**Expected: orthogonal 1.5–2× multiplier. Stacks with Phase 1+2+3.**

### Phase 5 — Per-expert mixed precision, ~2 weeks, low risk
- Log routing frequencies on real linus-sec traffic for a week
- MoPEQ Hessian-trace per-expert sensitivity (or simpler: hot=Q4_K, cold=stay-IQ2)
- Re-export GGUF with mixed-precision experts

**Expected: +quality at iso-size, or –20% size at iso-quality. May enable larger batch/context within memory budget.**

### Phase 6 (research bet) — XQA MLA port, 2–4 weeks, high risk
- Copy FlashInfer mla_sm120.{cu,cuh} into ds4, port dependencies
- Probable failure mode: CGA barriers don't work on sm_121
- Fallback: extract the absorb-mode math, hand-write a simpler kernel

**Expected if it lands: +30–60% on attention path.** Real chance of "doesn't work."

## Realistic perf target

Stacking the high-confidence wins (Phase 1+2+3+4):
- **Short context:** 12–16 → 22–28 tok/s base, **35–50 tok/s with FastMTP**
- **32K context:** 12–16 → 18–22 tok/s base, **28–40 tok/s with FastMTP**
- **64K+ context:** the long-context KV win dominates, **40+ tok/s plausible**

If Phase 6 (XQA port) also lands: add another 20–40% on top.

This is more conservative than the original synthesis but mathematically grounded in the actual current state of ds4.

## What I'm doing right now

Sparks are SSH-unreachable (TS check needs reauth or sshd hung). Without Spark access I can:
- ✓ Commit sm_121a Makefile fix
- Write the NVFP4 KV cache implementation locally (verify by compile against CUDA headers if available)
- Document the porting plan for XQA MLA so it can start the moment Sparks return
- Draft the Q8_0 → NVFP4 conversion script

Tests must run on Spark, so all of this stays as "ready to land" until SSH returns.

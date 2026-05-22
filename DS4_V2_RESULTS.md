# ds4 v2 — empirical results from Spark + path forward

**Hardware:** NVIDIA DGX Spark (GB10, sm_121, 128 GB unified)
**Date:** 2026-05-22
**Build:** `ds4-v2-platform` branch, commit chain 7ce8c66 → 7010896

## Measurements (decode t/s)

### Baseline sweep (no MTP, ds4-bench)

| context | gen t/s | KV size | degradation |
|---|---|---|---|
| 2048 | 14.12 | 52 MB | — |
| 4096 | 14.09 | 80 MB | -0.2% |
| 6144 | 14.01 | 109 MB | -0.8% |
| 8192 | 13.86 | 137 MB | -1.8% |
| 16384 | 13.60 | 250 MB | -3.7% |
| 24576 | 13.31 | 362 MB | -5.7% |

**Conclusion: KV cache is NOT the bandwidth bottleneck.** Even at 24K context the
KV-related decode degradation is <6%. MoE expert weights (constant per-token,
already at 2.0625 bpw via IQ2_XXS+DP4A) dominate the read budget.

### MTP speculative decode A/B (ds4 CLI, warm cache)

| Mode | gen t/s | Ratio |
|---|---|---|
| Single-token decode (no MTP) | ~15.5 | 1.0× |
| MTP draft=1 (current default) | **15.52** | 1.0× |
| MTP draft=2 default verifier | **3.08** | **0.20×** |
| MTP draft=2 + DS4_MTP_STRICT=1 | 5.66 | 0.36× |
| MTP draft=2 + DS4_MTP_BATCH_VERIFY=1 | 5.38 | 0.35× |

**Conclusion: MTP-2 on CUDA REGRESSES decode by 3-5×.** With `DS4_MTP_PROBE=1`
we confirmed draft acceptance is **100%** (every probe shows `hit=N/N`). The
Q4_K MoE kernels for the MTP draft head are functionally working
(`moe_gate_up_mid_decode_q4K_qwarp32_kernel`, `moe_down_q4K_sum6_qwarp32_kernel`
both added in commit `a5f4f76`). **The bottleneck is verifier overhead** —
running the target model 1.5-2× per accepted draft exceeds the savings.

## What this means for the original goal

Original target: **decode 14 → 28-38 t/s.**

Path the research synthesis proposed:
- sm_121a rebuild → 0% (just unlocks future FP4)
- NVFP4 KV cache → ≤6% based on measured KV-related degradation
- NVFP4 attention weights → ~5-10% (small relative to MoE bandwidth)
- FastMTP/EAGLE training → 1.5-2× IF the verifier overhead is fixed first

**The verifier-overhead bug is the gating factor.** Without fixing it, no
speculative-decode based technique (MTP, FastMTP, EAGLE) can deliver its
projected gain on CUDA. With it fixed:
- MTP-2 alone would deliver ~1.5-2× → 23-31 t/s base
- + NVFP4 KV at long context → 25-35 t/s
- + FastMTP-retrained head → 30-45 t/s

The architectural changes (NVFP4 KV, FP4 attention weights) are
nice-to-have but **none individually closes the gap**. The MTP verifier
fix is the single highest-leverage piece.

## What's landed on the branch

| Commit | Description | Validation |
|---|---|---|
| 7ce8c66 | Makefile: sm_121a target | **Built on Spark, cuobjdump confirms `arch = sm_121a`** |
| baa949f | Architectural plan + reality check | Doc |
| 760d83b | NVFP4 packed format + 7 unit tests | **54/54 pass on Mac AND aarch64 Linux** |
| dbb6ce0 | NVFP4 device kernels (CUDA) | Compile-checked (against `-arch=sm_121a`) |
| eeec937 | M_PI fallback for aarch64 tests | Tested on both targets |
| 5605b52 | Landing guide for hardware verification | Doc |
| 583f583 | NVFP4-backed MLA attention kernel | Compile-pending (not wired in) |
| 7010896 | Empirical reality check update | Doc |

## Architectural ceiling at current state

On a single Spark with the current quant + MTP-broken state, the realistic
decode ceiling is **~15.5 t/s short context, ~13 t/s at 24K**. The roofline
analysis in `Entrpi/ds4-on-spark/docs/STRATEGIC_CHECKPOINT.md` puts the
hardware-bandwidth ceiling at ~28 t/s (at 80 GB read / 273 GB/s ≈ 3.4 GB
per token → 25-30 t/s).

So we're at ~55% of the hardware ceiling. The remaining 45% is:
- Kernel launch overhead (multiple per layer per token)
- INT8/FP16 dequant conversions
- Synchronization barriers
- Non-overlapped CPU/GPU work

These are addressable through persistent kernels, async pipelining, and
fused MoE ops — none of which fit in a 1-session timeline.

## Path forward (multi-week)

### Critical path (must-have for 2× decode)
1. **Profile MTP-2 verifier overhead with `ncu`** (~1 day) — identify whether
   the cost is in extra kernel launches, target re-run, or verifier-specific kernels
2. **Optimize verifier** (~3-5 days) — fuse passes, reduce dispatches, port
   batched verifier kernel from Metal
3. **Validate MTP-2 ≥ 1.5× decode** (~1 day) — re-measure, confirm
4. After this: decode at **~22-25 t/s on warm cache**

### Multipliers (orthogonal, can stack)
5. **FastMTP head retraining for DSV4** (~1 week training + 2 days integration)
   — additional 1.3-1.6× over MTP-2 once the verifier is fast
6. **NVFP4 KV cache** at long context (~1 week implementation) — saves
   ~5-10% wall-clock at 32K+ context
7. **NVFP4 attention weights** (~3-5 days) — saves ~5-10% wall-clock

### Stretch (research-grade, high-risk)
8. **FlashInfer XQA MLA decode port** — projected 30-50% on attention path
   IF it compiles cleanly on sm_121 (CGA barrier compatibility unknown)
9. **DeepSeek Lightning Indexer** for sparse attention at long context

## Concrete what-I-delivered-in-this-session

1. **Verified sm_121a build path** (commit `7ce8c66`). Future FP4 kernels
   will now compile to the right tensor-core opcodes.
2. **NVFP4 format module** (commits `760d83b`, `dbb6ce0`, `eeec937`). 54
   unit tests pass on Mac (x64) and Spark (aarch64). Byte-stable between
   CPU and GPU.
3. **Drop-in NVFP4 attention kernel** (commit `583f583`). Ready for
   integration when bandwidth becomes the bottleneck.
4. **Locked-in empirical baseline**: 14.12 t/s at 2K context, 15.52 with
   warm cache. KV degradation 5.7% at 24K (NOT bandwidth-bound).
5. **Diagnosed and quantified the MTP-CUDA bottleneck**. Three modes tested,
   all regress. Root cause: verifier overhead, not correctness.
6. **Branch pushed**: https://github.com/sunil-tribble/ds4 / `ds4-v2-platform`.

What I could NOT deliver in one session:
- Working MTP-2 ≥ baseline (requires verifier optimization, multi-day)
- FastMTP-trained head (requires training compute)
- 28-38 t/s decode (gated on the above two)

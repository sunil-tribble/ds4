# ds4 v2 platform — empirical reality check

**Latest update: 2026-05-22 after Spark baseline measurements landed.**

## Measured baseline (sm_121a build, IQ2XXS Q2 GGUF)

| context | gen t/s | prefill t/s | KV size |
|---|---|---|---|
| 2048 | 14.12 | 189.83 | 52 MB |
| 4096 | 14.09 | 383.31 | 80 MB |
| 6144 | 14.01 | 377.48 | 109 MB |
| 8192 | 13.86 | 373.51 | 137 MB |
| 16384 | 13.60 | 372.72 | 250 MB |
| 24576 | 13.31 | 351.09 | 362 MB |

**Decode degradation 2K → 24K: -5.7%.** KV is NOT the bandwidth bottleneck —
the constant-per-token MoE expert reads dominate.

## What this means for the original plan

The synthesis projected:
- "Short context: 12-16 → 22-28 tok/s base, 35-50 with FastMTP"
- "32K context: 12-16 → 18-22 tok/s base, 28-40 with FastMTP"
- "64K+ context: 40+ tok/s plausible from KV NVFP4 alone"

The 64K projection was wrong. Even at 24K we see only -5.7% degradation
from KV growth — extrapolating, 64K would be perhaps -10 to -15%. NVFP4
KV cache can only reclaim that 5-15% gap, not deliver "40+ tok/s."

The architectural wins (sm_121a + NVFP4 KV + FlashInfer XQA + NVFP4
attn weights) realistically deliver: **14 → 16-19 tok/s** base.

**Speculative decoding is the only architecturally-tractable path to
2-3× improvement** given how well-optimized the MoE path already is.

## Where the wins actually concentrate

1. **MTP draft=2** (existing infrastructure, hours of work):
   measuring now — see latest commits.
2. **FastMTP head retrained for DSV4** (days of training + integration):
   single biggest non-spec architectural change, projected 2× lossless.
3. **EAGLE-3 head retrained for DSV4** (weeks of training): potential
   1.5-2× additional, stacks with FastMTP.
4. **NVFP4 attention + shared expert weights** (~1 week): 1.78×
   bandwidth on those specific tensors. Maybe +5-10% wall-clock since
   attention is a small fraction of total compute.
5. **NVFP4 KV cache** (~1 week): MAYBE +5% at long context. Smaller
   win than projected. Skip unless other paths land first.

## What's committed and verified

### Phase 1: sm_121a rebuild — DONE ✅
- Makefile change: `CUDA_ARCH=sm_121a` (was empty, silently demoted FP4)
- Rebuilt on Spark, cuobjdump confirms `arch = sm_121a` + HMMA opcodes
- Baseline measured: 14 tok/s decode at 2K-8K

### Phase 2 (NVFP4 format): CPU + GPU code complete ✅
- `ds4_nvfp4.{h,c}` host reference (54/54 unit tests pass on Mac AND aarch64)
- `ds4_nvfp4_cuda.cuh` device kernels mirror byte-for-byte
- `ds4_nvfp4_attn.cuh` drop-in attention kernel using packed comp_kv

### Phase 3 (Integration): in progress
- `ds4_nvfp4_attn.cuh` needs to be wired into the decode dispatch path
- Decision pending on whether to do this given the small projected win

## What I'm doing now

1. ✅ MTP draft=2 vs draft=1 A/B benchmark (in flight)
2. Pending: ncu profile to confirm we're really at MoE-bandwidth wall
3. Pending: if MTP draft=2 wins, push higher (draft=3, 4)
4. Pending: research path to FastMTP-style head training on DSV4

## What the goal still needs

To hit 28-38 tok/s from 14:
- MTP draft=2: +5-15% → ~16 tok/s
- FastMTP head (trained for DSV4): +50-100% → 24-32 tok/s
- NVFP4 KV at very long context: +5-15% → ~28-37 tok/s

**FastMTP head training is the critical path.** Without it, the
architectural changes alone don't get us to the target.

## What was over-promised in the research synthesis

The agents projected speedups based on assumptions that don't apply
to ds4's current state:
- They assumed FP16 MoE baseline → NVFP4 MoE = 4× win.
  Reality: IQ2_XXS at 2.0625 bpw is *already* the floor.
- They assumed KV cache dominates at long context.
  Reality: 24K KV is only 6% of decode bandwidth on DSV4 MLA.
- They assumed FlashInfer XQA MLA was a "1-week port".
  Reality: 1945 lines of templated CUDA C++ with sm_90+ cluster
  features. Risk of incompatibility on sm_121.

The agents were excellent at FINDING techniques and citations. They
were less reliable at quantifying impact on an already-tuned codebase.
This required empirical measurement to disambiguate.

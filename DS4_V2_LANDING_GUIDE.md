# ds4 v2 — what's ready to land once the Sparks come back

Branch: `ds4-v2-platform` (4 commits ahead of antirez/main 8d57664)

## What's committed and validated locally

| Commit | Description | Validation status |
|---|---|---|
| 7ce8c66 | Makefile: cuda-spark targets sm_121a (was empty arch) | Pending SASS verify |
| baa949f | Architectural plan + reality check | Doc only |
| 760d83b | NVFP4 packed format CPU reference + 7 unit tests | **54/54 pass on Mac** |
| dbb6ce0 | NVFP4 device kernels mirroring CPU reference byte-for-byte | Pending nvcc build |

The CPU reference establishes ground truth: every byte the GPU produces
must match `ds4_nvfp4.c`'s `encode_block`/`decode_block`. The host tests
verified round-trip error is bounded:
- Random unit normal: mean abs error 7.12% of stddev
- KV-cache-like (std=0.05): mean abs error 0.36%

## What's NOT yet done

| Piece | Status | Blocked on |
|---|---|---|
| Wire NVFP4 into `ds4_layer_cache` for `attn_comp_kv` | Code design ready | Need to test integration on real KV writes |
| Modify MLA attention kernel to dequant-on-read | Design only | Spark access; risk of perf regression if wrong |
| Q8_0 attention+shared expert → NVFP4 (offline) | Not started | Lower priority than KV cache |
| FastMTP head retrain | Not started | Needs training compute (Sparks) |
| Per-expert mixed precision | Not started | Needs routing-frequency logs (Sparks) |
| XQA MLA decode kernel port | Deferred (high risk) | 1945-line port, may not compile on sm_121 |

## The moment Sparks come back, run this

```bash
# 1. On the Spark, fetch and switch to v2 branch
cd /home/sunil/code/ds4
git fetch antirez   # if upstream antirez/ds4 has the branch
# Actually pull from a remote where we pushed the v2 branch:
# git remote add v2 git@github.com:<user>/ds4.git
# git fetch v2 ds4-v2-platform
# git checkout v2/ds4-v2-platform

# 2. Verify Makefile now targets sm_121a
grep "CUDA_ARCH=sm_121a" Makefile

# 3. Clean build
make clean
make cuda-spark 2>&1 | tee build_v2.log

# 4. CRITICAL: verify SASS shows native FP4 MMA, not the demoted family path
cuobjdump --dump-sass ds4-server | grep -E "HMMA|QMMA" | head -20
# Expect to see HMMA.MXF4NVF4 opcodes when ds4 has FP4 kernels.
# Currently ds4 doesn't have FP4 kernels yet, so you should see
# the existing DP4A INT8 opcodes unchanged. This step confirms
# the rebuild went through cleanly.

# 5. Lock SM clocks (5-8% steady-state per LMSYS Spark review)
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1980  # max GB10 boost clock; tune if thermal

# 6. Build + run the NVFP4 unit tests on the Spark to confirm CPU
#    reference behaves identically on aarch64 + Linux:
cd tests && cc -O2 -std=c99 -I.. ../ds4_nvfp4.c test_nvfp4.c -lm -o test_nvfp4
./test_nvfp4
# Expect: "54 pass, 0 fail"

# 7. ncu baseline profile — this measures where decode time actually goes.
#    Identifies whether we're bandwidth-bound (expected) or kernel-launch
#    bound. Critical input for prioritizing remaining phases.
ncu --target-processes all --set full \
  --replay-mode kernel \
  --kernel-name 'matmul_iq2_xxs' \
  -o ds4_decode_baseline.ncu-rep \
  ./ds4-bench --model /home/sunil/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
              --prompt "Once upon a time" --gen 64
ncu --import ds4_decode_baseline.ncu-rep --section MemoryWorkloadAnalysis \
  | grep -E "DRAM Frequency|Memory Throughput|L2 Hit Rate"
```

The `ncu` profile is the single most important next data point. It tells us:
- If we're at >70% of 273 GB/s DRAM throughput → bandwidth wins (NVFP4 KV cache) are real
- If we're at <40% → kernel launch overhead or compute is the bottleneck and a different
  set of optimizations matters

## Next implementation steps (after baseline)

If the Sparks return and the baseline rebuild works cleanly:

1. **NVFP4 KV cache integration** (~3-5 days) — wire `ds4_nvfp4_*` into
   `kv_cache_push_comp` and the MLA attention kernel's KV read path. The
   change is contained to `ds4.c` (CPU path) and `ds4_cuda.cu` (one new
   include + ~200 LOC of integration). Expected wall-clock impact: small
   at short context, **30-50% at 32K+**.

2. **Attention + shared expert NVFP4 weights** (~3 days) — offline
   convert these tensors from Q8_0 to NVFP4 (~10x smaller than Q8_0
   was). New matmul kernel using `mma.sync.aligned.kind::mxf4nvf4.block_scale`
   for the attention path. Expected: **+10-15% all-context decode**.

3. **MTP draft=2 sweep** (~1 hour test) — change `--mtp-draft 1` to
   `--mtp-draft 2` in the launch script. No code changes. Expected: **+5-15%**.

4. **`cudaMemPrefetchAsync` for next-layer experts** (~1 day) — overlay
   on existing MoE dispatch loop. Expected: **+5-10%**.

If all four land, conservative target: **12-16 → 22-30 tok/s base**, with
long-context (>16K) seeing **another 30-50%** on top.

## What's actually blocked

I've written ~700 lines of new code without GPU access. The risk
profile:
- **Low risk**: CPU NVFP4 reference (already passing tests)
- **Medium risk**: device kernels (compile-checked only, may have
  subtle warp-level race conditions or bank conflicts)
- **High risk**: integration into ds4_layer_cache + attention kernel
  (changes to data layout, possibility of breaking the existing decode
  path)

Recommended: when Sparks come back, do the rebuild + baseline FIRST
before any integration. Then integrate piece-by-piece with parity tests
against the current FP32-comp_kv path. **Don't merge integration
changes until parity test passes within tolerance.**

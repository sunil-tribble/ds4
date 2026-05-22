/*
 * ds4_nvfp4_attn.cuh — NVFP4-backed attention kernels.
 *
 * Drop-in replacement for `comp_kv` reads in the MLA attention path.
 * The format is documented in ds4_nvfp4.h. Storage cost is 0.5625
 * bytes/elem (4.5 bpw), vs FP32 at 4 bytes/elem — 7.11x bandwidth
 * reduction on the kv-cache read path.
 *
 * Strategy: cooperative warp-level decode of one comp_kv row (128 floats
 * = 8 NVFP4 blocks) into shared memory before the score loop. Both the
 * QK pass and the AttnOut pass read the cached row from shared instead
 * of going back to DRAM.
 *
 * Bandwidth math at decode (one token, one layer, MLA absorbed mode):
 *   visible_comp rows of head_dim=128 each
 *   FP32 path: visible_comp * 128 * 4 bytes = 512 B per row, twice = 1024 B/row
 *   NVFP4 path: visible_comp * 128 * 0.5625 = 72 B per row, twice DRAM read
 *               BUT one shared-mem dequant means each row hits DRAM once.
 *   Net per-row reduction: ~14x DRAM bytes.
 *
 * At 32K context with ratio=4: ~8K rows of comp_kv per layer. Currently
 * 8K * 1024 = 8 MB per layer per token. NVFP4 brings this to ~580 KB per
 * layer per token. Across 61 layers: 488 MB -> 35 MB per token of attention KV
 * bandwidth. At 273 GB/s, that's 1.8 ms vs 0.13 ms — ~14x speedup on
 * the KV-bound component of attention.
 */
#ifndef DS4_NVFP4_ATTN_CUH
#define DS4_NVFP4_ATTN_CUH

#include "ds4_nvfp4_cuda.cuh"

/*
 * Cooperative warp-level decode of one comp_kv row from NVFP4 packed
 * storage into shared memory. Assumes row width is a multiple of 16
 * (NVFP4 block size). For ds4's MLA comp_kv with head_dim=128, this is
 * 8 blocks per row.
 *
 * Caller layout:
 *   __shared__ float row_smem[head_dim];
 *   ds4_nvfp4_decode_row_to_smem(packed_row, row_smem, head_dim);
 *   __syncthreads();
 *
 * For best performance, dispatch a thread per element-pair (one thread
 * handles 2 elements via the nibble-unpack). With head_dim=128 that's
 * 64 threads — one warp.
 */
template <uint32_t HEAD_DIM>
__device__ __forceinline__ static void ds4_nvfp4_decode_row_to_smem(
        const ds4_nvfp4_block_dev *packed_row,
        float *out_smem) {
    static_assert(HEAD_DIM % DS4_NVFP4_BLOCK == 0,
                  "HEAD_DIM must be a multiple of DS4_NVFP4_BLOCK");
    constexpr uint32_t N_BLOCKS = HEAD_DIM / DS4_NVFP4_BLOCK;

    /* Each thread handles one element pair: thread t handles elements 2t and 2t+1
     * in the row. With HEAD_DIM=128, that needs 64 threads = 1 warp. */
    if (threadIdx.x < HEAD_DIM / 2) {
        uint32_t pair_idx = threadIdx.x;
        uint32_t block_idx = pair_idx / 8;
        uint32_t pair_in_block = pair_idx % 8;
        const ds4_nvfp4_block_dev *blk = packed_row + block_idx;
        uint8_t pair_byte = blk->weights[pair_in_block];
        float scale = ds4_nvfp4_e4m3_decode_scale_dev(blk->scale);
        unsigned lo = pair_byte & 0xFu;
        unsigned hi = (pair_byte >> 4) & 0xFu;
        out_smem[pair_idx * 2]     = ds4_nvfp4_e2m1_value_dev(lo) * scale;
        out_smem[pair_idx * 2 + 1] = ds4_nvfp4_e2m1_value_dev(hi) * scale;
    }
    /* HEAD_DIM=128 case: only first 64 threads do work, rest sit idle.
     * For HEAD_DIM=256 use 128 threads, etc. */
}

/*
 * Attention prefill kernel with NVFP4-packed comp_kv.
 *
 * Differences from attention_prefill_mixed_kernel (the FP32 version):
 *   - `comp_kv` is now `const ds4_nvfp4_block_dev *comp_kv_packed`
 *   - `comp_kv_row_pitch_blocks` = head_dim / 16 (number of NVFP4 blocks
 *     per row, e.g. 8 for head_dim=128)
 *   - Each visible_comp row is decoded once into shared memory before
 *     the QK score loop, then re-used for the AttnOut accumulation
 *
 * Note: this kernel takes a HEAD_DIM template parameter for compile-time
 * specialization. ds4 has DS4_N_HEAD_DIM = 128 (MLA latent dim) at
 * present; if other widths are needed, instantiate explicitly below.
 */
template <uint32_t HEAD_DIM>
__global__ static void attention_prefill_nvfp4_kernel(
        float *heads,
        const float *sinks,
        const float *q,
        const float *raw_kv,
        const ds4_nvfp4_block_dev *comp_kv_packed,
        const float *comp_mask,
        uint32_t use_comp_mask,
        uint32_t n_tokens,
        uint32_t n_comp,
        uint32_t window,
        uint32_t ratio,
        uint32_t n_head) {
    constexpr uint32_t HEAD_DIM_VAL = HEAD_DIM;
    constexpr uint32_t COMP_KV_ROW_PITCH_BLOCKS = HEAD_DIM_VAL / DS4_NVFP4_BLOCK;

    uint32_t t = blockIdx.x;
    uint32_t h = blockIdx.y;
    if (t >= n_tokens || h >= n_head) return;
    const float *qh = q + ((uint64_t)t * n_head + h) * HEAD_DIM_VAL;
    uint32_t raw_start = (window != 0 && t + 1u > window) ? t + 1u - window : 0u;
    uint32_t raw_count = t + 1u - raw_start;
    uint32_t visible_comp = (t + 1u) / ratio;
    if (visible_comp > n_comp) visible_comp = n_comp;

    /* Shared memory layout:
     *   scores[raw_count + visible_comp]  — up to 512 entries
     *   partial[blockDim.x]               — reduction scratch
     *   max_s, denom                      — scalars
     *   kvrow_smem[HEAD_DIM_VAL]          — dequant cache for one row
     */
    extern __shared__ float smem_pool[];
    float *scores  = smem_pool;                              /* up to 512 */
    float *partial = scores + 512;                           /* blockDim.x */
    float *kvrow_smem = partial + 256;                       /* HEAD_DIM_VAL */
    __shared__ float max_s, denom;

    float scale = rsqrtf((float)HEAD_DIM_VAL);
    float local_max = sinks[h];
    uint32_t n_score = raw_count + visible_comp;

    /* QK pass — raw KV (FP32, unchanged) */
    for (uint32_t r = threadIdx.x; r < raw_count; r += blockDim.x) {
        const float *kvrow = raw_kv + (uint64_t)(raw_start + r) * HEAD_DIM_VAL;
        float dot = 0.0f;
        for (uint32_t d = 0; d < HEAD_DIM_VAL; d++) dot += qh[d] * kvrow[d];
        scores[r] = dot * scale;
        local_max = fmaxf(local_max, scores[r]);
    }
    /* QK pass — comp KV (NVFP4) — decode-on-read into shared */
    for (uint32_t c = 0; c < visible_comp; c++) {
        const ds4_nvfp4_block_dev *packed_row =
            comp_kv_packed + (uint64_t)c * COMP_KV_ROW_PITCH_BLOCKS;
        ds4_nvfp4_decode_row_to_smem<HEAD_DIM_VAL>(packed_row, kvrow_smem);
        __syncthreads();
        float add = use_comp_mask ? comp_mask[(uint64_t)t * n_comp + c] : 0.0f;
        float s = -INFINITY;
        if (add > -1.0e20f) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < HEAD_DIM_VAL; d++) dot += qh[d] * kvrow_smem[d];
            s = dot * scale + add;
        }
        if (threadIdx.x == 0) scores[raw_count + c] = s;
        if (s > local_max) local_max = s;
        __syncthreads(); /* before next iteration's decode overwrites kvrow_smem */
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_s = partial[0];
    __syncthreads();
    float den_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_score; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_s);
        den_local += scores[i];
    }
    partial[threadIdx.x] = den_local;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + expf(sinks[h] - max_s);
    __syncthreads();
    float *oh = heads + ((uint64_t)t * n_head + h) * HEAD_DIM_VAL;
    /* AttnOut: accumulate softmax * V (raw + comp)
     * For comp, we decode each row again — alternatively we could cache
     * the row results from QK, but at HEAD_DIM=128 the dequant is cheap. */
    for (uint32_t d = threadIdx.x; d < HEAD_DIM_VAL; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t r = 0; r < raw_count; r++) acc += raw_kv[(uint64_t)(raw_start + r) * HEAD_DIM_VAL + d] * scores[r];
        for (uint32_t c = 0; c < visible_comp; c++) {
            const ds4_nvfp4_block_dev *packed_row =
                comp_kv_packed + (uint64_t)c * COMP_KV_ROW_PITCH_BLOCKS;
            /* Scalar element fetch — for AttnOut each thread reads a
             * different `d`, so cooperative warp decode doesn't help here.
             * The DRAM bandwidth saving is what matters. */
            float v = ds4_nvfp4_get_dev(packed_row, d);
            acc += v * scores[raw_count + c];
        }
        oh[d] = acc / denom;
    }
}

#endif /* DS4_NVFP4_ATTN_CUH */

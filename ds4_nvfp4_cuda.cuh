/*
 * ds4_nvfp4_cuda.cuh — Device-side NVFP4 pack/unpack mirroring ds4_nvfp4.c.
 *
 * Include this from ds4_cuda.cu. It is pure device code (no host symbols).
 * For host-side packing during quantization/conversion, use ds4_nvfp4.c.
 *
 * THE BYTE LAYOUT HERE MUST MATCH ds4_nvfp4.c EXACTLY. If you change one,
 * change the other and re-run tests/test_nvfp4 to confirm.
 */
#ifndef DS4_NVFP4_CUDA_CUH
#define DS4_NVFP4_CUDA_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

/* Must match ds4_nvfp4.h */
#define DS4_NVFP4_BLOCK 16

struct ds4_nvfp4_block_dev {
    uint8_t weights[8];
    uint8_t scale;
};
static_assert(sizeof(ds4_nvfp4_block_dev) == 9,
              "ds4_nvfp4_block_dev must be 9 bytes (no padding)");

/* ===== Device-side decode helpers ===== */

/*
 * Decode 4-bit E2M1 code to a float on device.
 *
 * Match the host table in ds4_nvfp4.h:
 *   sign (bit 3) | magnitude (bits 0..2)
 *   magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
 *
 * Implemented with a __constant__-resident table indexed by the 4-bit code.
 * Faster than a switch on hardware that supports cmem loads in warps.
 */
__device__ __forceinline__ static float ds4_nvfp4_e2m1_value_dev(unsigned code) {
    /* Use registers via a 64-bit literal pair to avoid cmem in kernels where
     * cmem pressure is a concern. The pattern below trades a constant load
     * for a select chain. For tight inner loops, consider hoisting to a
     * shared-mem LUT. */
    code &= 0xFu;
    float sign = (code & 0x8u) ? -1.0f : 1.0f;
    unsigned m = code & 0x7u;
    /* Branchless lookup: there are 8 magnitudes, so we can compute via a
     * select chain. The compiler usually fuses this into 3 PRED ops. */
    float v;
    switch (m) {
        case 0: v = 0.0f; break;
        case 1: v = 0.5f; break;
        case 2: v = 1.0f; break;
        case 3: v = 1.5f; break;
        case 4: v = 2.0f; break;
        case 5: v = 3.0f; break;
        case 6: v = 4.0f; break;
        default: v = 6.0f; break;
    }
    return sign * v;
}

/*
 * Decode an E4M3 byte to FP32 scale.
 * MUST match ds4_nvfp4_e4m3_decode_scale() in ds4_nvfp4.c exactly.
 *
 * E4M3 layout: 1 sign / 4 exp / 3 mant. We assume sign bit is 0 (positive
 * scales only — encoder enforces this).
 */
__device__ __forceinline__ static float ds4_nvfp4_e4m3_decode_scale_dev(uint8_t s) {
    if (s == 0) return 0.0f;
    unsigned exp = (s >> 3) & 0xFu;
    unsigned mant = s & 0x7u;
    if (exp == 0) {
        /* denormal: (mant/8) * 2^-6 = mant / (8 * 64) = mant / 512 */
        return (float)mant * (1.0f / 512.0f);
    }
    float val = 1.0f + (float)mant * (1.0f / 8.0f);
    int e = (int)exp - 7;
    /* exp2f is correctly rounded for integer args on Blackwell. */
    return val * exp2f((float)e);
}

/*
 * Decode a 9-byte NVFP4 block into 16 floats in registers.
 * Caller provides `out[16]`. Useful for warp-cooperative attention kernels
 * that re-materialize KV from packed storage.
 */
__device__ __forceinline__ static void ds4_nvfp4_decode_block_dev(
        const ds4_nvfp4_block_dev *src, float *out16) {
    float scale = ds4_nvfp4_e4m3_decode_scale_dev(src->scale);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t pair = src->weights[i];
        unsigned lo = pair & 0xFu;
        unsigned hi = (pair >> 4) & 0xFu;
        out16[i * 2]     = ds4_nvfp4_e2m1_value_dev(lo) * scale;
        out16[i * 2 + 1] = ds4_nvfp4_e2m1_value_dev(hi) * scale;
    }
}

/*
 * Decode a 9-byte NVFP4 block into 16 half precision FP16 values.
 * Use when attention/MLA kernels operate on FP16 (the common case).
 */
__device__ __forceinline__ static void ds4_nvfp4_decode_block_to_half_dev(
        const ds4_nvfp4_block_dev *src, __half *out16) {
    float scale = ds4_nvfp4_e4m3_decode_scale_dev(src->scale);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t pair = src->weights[i];
        unsigned lo = pair & 0xFu;
        unsigned hi = (pair >> 4) & 0xFu;
        out16[i * 2]     = __float2half(ds4_nvfp4_e2m1_value_dev(lo) * scale);
        out16[i * 2 + 1] = __float2half(ds4_nvfp4_e2m1_value_dev(hi) * scale);
    }
}

/*
 * Read one element from an NVFP4-packed tensor at the given linear index.
 * Useful for scalar paths and validation; the bulk-decode variants above
 * are preferred in hot loops because they amortize the scale load.
 */
__device__ __forceinline__ static float ds4_nvfp4_get_dev(
        const ds4_nvfp4_block_dev *base, size_t idx) {
    size_t block_idx = idx / DS4_NVFP4_BLOCK;
    size_t in_block  = idx % DS4_NVFP4_BLOCK;
    const ds4_nvfp4_block_dev *blk = base + block_idx;
    uint8_t pair = blk->weights[in_block / 2];
    unsigned code = (in_block & 1u) ? ((pair >> 4) & 0xFu) : (pair & 0xFu);
    float scale = ds4_nvfp4_e4m3_decode_scale_dev(blk->scale);
    return ds4_nvfp4_e2m1_value_dev(code) * scale;
}

/* ===== Device-side block encode =====
 *
 * Quantize 16 floats from registers/shared into a packed 9-byte block.
 * Used by kv_cache_push_comp on the GPU path.
 *
 * One thread = one block. Caller is responsible for cooperative dispatch
 * (typically: one warp encodes 16 blocks via warp-level shuffle, or each
 * thread handles its own block in a 1-D grid).
 */
__device__ __forceinline__ static uint8_t ds4_nvfp4_e4m3_encode_scale_dev(float s) {
    if (!(s > 0.0f)) return 0;
    if (s >= 448.0f) return 0x7E;
    if (s < 1.0f / 512.0f) return 0;

    /* Decompose s into mant_f * 2^exp where mant_f in [1.0, 2.0) */
    int exp;
    float mant_f = frexpf(s, &exp); /* in [0.5, 1) */
    mant_f *= 2.0f;
    exp -= 1;

    int e_bias = exp + 7;
    if (e_bias <= 0) {
        float dn = s * 64.0f * 8.0f;
        int mant = (int)__float2int_rn(dn);
        if (mant < 0) mant = 0;
        if (mant > 7) mant = 7;
        return (uint8_t)mant;
    }
    if (e_bias > 15) e_bias = 15;
    int mant = (int)__float2int_rn((mant_f - 1.0f) * 8.0f);
    if (mant > 7) {
        mant = 0;
        e_bias++;
        if (e_bias > 15) e_bias = 15;
    }
    if (mant < 0) mant = 0;
    return (uint8_t)((e_bias << 3) | mant);
}

__device__ __forceinline__ static unsigned ds4_nvfp4_e2m1_encode_dev(float v) {
    unsigned sign = (v < 0.0f) ? 8u : 0u;
    float ax = fabsf(v);
    if (ax > 6.0f) ax = 6.0f;
    /* Magnitude table */
    static const float mags[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    unsigned best = 0;
    float best_diff = fabsf(ax - mags[0]);
#pragma unroll
    for (unsigned i = 1; i < 8; i++) {
        float d = fabsf(ax - mags[i]);
        if (d < best_diff) {
            best = i;
            best_diff = d;
        } else if (d == best_diff && (i & 1u) == 0u && (best & 1u) != 0u) {
            best = i;
        }
    }
    return sign | best;
}

__device__ __forceinline__ static void ds4_nvfp4_encode_block_dev(
        const float *src16, ds4_nvfp4_block_dev *dst) {
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        float ax = fabsf(src16[i]);
        if (ax > amax) amax = ax;
    }
    float scale_f = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
    dst->scale = ds4_nvfp4_e4m3_encode_scale_dev(scale_f);
    float scale_q = ds4_nvfp4_e4m3_decode_scale_dev(dst->scale);
    float inv_scale = (scale_q > 0.0f) ? (1.0f / scale_q) : 0.0f;

#pragma unroll
    for (int i = 0; i < 8; i++) {
        unsigned lo = ds4_nvfp4_e2m1_encode_dev(src16[i * 2]     * inv_scale);
        unsigned hi = ds4_nvfp4_e2m1_encode_dev(src16[i * 2 + 1] * inv_scale);
        dst->weights[i] = (uint8_t)((hi << 4) | lo);
    }
}

/* ===== Kernel: convert FP32 tensor → NVFP4 packed =====
 *
 * Grid layout: one thread per 16-element block. n_elems must be a multiple
 * of 16. For tensor sizes used in ds4 KV cache (DS4_N_HEAD_DIM=128) every
 * row is 8 blocks — kernel processes one row per thread for good locality.
 *
 * Total reads: n_elems * 4 bytes (FP32 source)
 * Total writes: (n_elems / 16) * 9 bytes (NVFP4 dest)
 * Compression ratio: 4 / 0.5625 = 7.11x
 */
__global__ static void ds4_nvfp4_quantize_fp32_kernel(
        const float *src, ds4_nvfp4_block_dev *dst, uint64_t n_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;
    ds4_nvfp4_encode_block_dev(src + idx * DS4_NVFP4_BLOCK, dst + idx);
}

/* ===== Kernel: convert NVFP4 packed → FP32 tensor (validation only) ===== */
__global__ static void ds4_nvfp4_dequantize_fp32_kernel(
        const ds4_nvfp4_block_dev *src, float *dst, uint64_t n_blocks) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;
    ds4_nvfp4_decode_block_dev(src + idx, dst + idx * DS4_NVFP4_BLOCK);
}

/* ===== Host-callable launchers ===== */

static inline int ds4_nvfp4_quantize_fp32_launch(
        const float *d_src, ds4_nvfp4_block_dev *d_dst,
        uint64_t n_elems, cudaStream_t stream) {
    if (n_elems % DS4_NVFP4_BLOCK != 0) return -1;
    uint64_t n_blocks = n_elems / DS4_NVFP4_BLOCK;
    const int block_dim = 256;
    int grid = (int)((n_blocks + block_dim - 1) / block_dim);
    if (grid <= 0) return 0;
    ds4_nvfp4_quantize_fp32_kernel<<<grid, block_dim, 0, stream>>>(
        d_src, d_dst, n_blocks);
    return (int)cudaGetLastError();
}

static inline int ds4_nvfp4_dequantize_fp32_launch(
        const ds4_nvfp4_block_dev *d_src, float *d_dst,
        uint64_t n_elems, cudaStream_t stream) {
    if (n_elems % DS4_NVFP4_BLOCK != 0) return -1;
    uint64_t n_blocks = n_elems / DS4_NVFP4_BLOCK;
    const int block_dim = 256;
    int grid = (int)((n_blocks + block_dim - 1) / block_dim);
    if (grid <= 0) return 0;
    ds4_nvfp4_dequantize_fp32_kernel<<<grid, block_dim, 0, stream>>>(
        d_src, d_dst, n_blocks);
    return (int)cudaGetLastError();
}

#endif /* DS4_NVFP4_CUDA_CUH */

/*
 * ds4_nvfp4.h — NVFP4 packed-tensor format for ds4 v2.
 *
 * NVFP4 is NVIDIA's variant of OCP Microscaling FP4 with:
 *   - 4-bit weight in E2M1 format (1 sign / 2 exponent / 1 mantissa)
 *     -> 8 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (each ± sign)
 *   - per-16-element E4M3 FP8 block scale (vs MXFP4's per-32 E8M0)
 *   - optional per-tensor FP32 amax for two-level scaling
 *
 * Storage cost:
 *   - 4 bits/elem weight = 0.5 bytes/elem
 *   - 1 byte / 16 elems scale = 0.0625 bytes/elem
 *   - Total: 0.5625 bytes/elem (~9 bpw)
 *
 * vs FP16 KV cache (2 bytes/elem): 3.56x bandwidth reduction
 * vs FP32 KV cache (4 bytes/elem): 7.11x bandwidth reduction
 *
 * The format is designed to pair with the sm_121a MMA instruction:
 *   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32
 *
 * but is also usable with software dequant for paths that can't use the
 * native MMA (e.g., scalar attention score accumulation in MLA absorb mode).
 */
#ifndef DS4_NVFP4_H
#define DS4_NVFP4_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DS4_NVFP4_BLOCK 16

/*
 * Packed block: 16 4-bit values packed into 8 bytes + 1 byte E4M3 scale.
 * Total: 9 bytes per 16-element block.
 *
 * Byte layout within `weights[8]`:
 *   weights[i] = (high_nibble << 4) | low_nibble
 *   weights[0] holds elements 0 (low) and 1 (high)
 *   weights[1] holds elements 2 (low) and 3 (high)
 *   ...etc, matching mma.sync's expected fragment layout.
 */
typedef struct {
    uint8_t weights[8];  /* 16 packed E2M1 values */
    uint8_t scale;       /* E4M3 FP8 scale; multiplier for the 16 values */
} ds4_nvfp4_block;

/*
 * Static assertion: a block must be exactly 9 bytes for the packed
 * storage math to work out. If alignment padding is added by a compiler,
 * downstream offset calculations break.
 */
#define _DS4_NVFP4_STATIC_ASSERT(cond, name) typedef char name[(cond) ? 1 : -1]
_DS4_NVFP4_STATIC_ASSERT(sizeof(ds4_nvfp4_block) == 9, ds4_nvfp4_block_must_be_9_bytes);

/* ===== E2M1 (FP4) value table =====
 * These match the dsv4_e2m1fn_value_dev table already in ds4_cuda.cu — that
 * code uses the same E2M1 representation for offline weight quantization. */
static inline float ds4_nvfp4_e2m1_value(unsigned code /* 0..15 */) {
    static const float table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[code & 15u];
}

/* ===== E4M3 (FP8) scale decode =====
 * E4M3 has 1 sign / 4 exponent / 3 mantissa.
 * For positive-only scale we use the unsigned interpretation:
 *   value = (1 + mantissa/8) * 2^(exponent - 7)   for exponent != 0
 *   value = (mantissa/8) * 2^(-6)                  for exponent == 0 (denormal)
 * Special values 0xff = +Inf, 0x80 = -0, 0x7f = NaN. For block scales we
 * never produce these (we clamp during encoding). */
static inline float ds4_nvfp4_e4m3_decode_scale(uint8_t s) {
    if (s == 0) return 0.0f;
    /* sign bit is always 0 for valid scales — encoder clamps */
    unsigned exp = (s >> 3) & 0xF;
    unsigned mant = s & 0x7;
    if (exp == 0) {
        /* denormal */
        return ((float)mant / 8.0f) * 0.015625f /* 2^-6 */;
    }
    float val = (1.0f + (float)mant / 8.0f);
    int e = (int)exp - 7;
    if (e >= 0) {
        return val * (float)(1u << e);
    } else {
        return val / (float)(1u << -e);
    }
}

/* Encode a non-negative scale to E4M3. Saturates at +448 (E4M3 max). */
uint8_t ds4_nvfp4_e4m3_encode_scale(float s);

/* Pick the best E2M1 code for value/scale ratio. Returns 0..15.
 * Sign bit goes into bit 3 of the result. */
unsigned ds4_nvfp4_e2m1_encode(float scaled_value);

/*
 * Block encode: 16 floats -> 1 ds4_nvfp4_block
 * Computes scale as max(|x|) / 6.0, then quantizes each value to E2M1 around
 * that scale. Saturating; no NaN handling (caller must filter NaN inputs).
 */
void ds4_nvfp4_encode_block(const float *src16, ds4_nvfp4_block *dst);

/*
 * Block decode: 1 ds4_nvfp4_block -> 16 floats. Use for CPU reference paths
 * and validation. CUDA kernels use the inline `dev_` variants in ds4_cuda.cu.
 */
void ds4_nvfp4_decode_block(const ds4_nvfp4_block *src, float *dst16);

/*
 * Tensor-level helpers. Tensor must have a length divisible by 16.
 * Returns number of blocks written.
 */
size_t ds4_nvfp4_encode_tensor(const float *src, size_t n_elems, ds4_nvfp4_block *dst);
void   ds4_nvfp4_decode_tensor(const ds4_nvfp4_block *src, size_t n_elems, float *dst);

/*
 * Bytes needed to store n_elems as NVFP4. Always n_elems / 16 * 9. Caller
 * must ensure n_elems is a multiple of 16.
 */
static inline size_t ds4_nvfp4_packed_bytes(size_t n_elems) {
    return (n_elems / DS4_NVFP4_BLOCK) * sizeof(ds4_nvfp4_block);
}

/*
 * Round-trip error on a buffer. Returns max abs error and writes mean abs
 * error to *mean_abs_err if non-NULL. Useful for unit tests + validation.
 */
float ds4_nvfp4_roundtrip_error(const float *src, size_t n_elems, float *mean_abs_err);

#ifdef __cplusplus
}
#endif

#endif /* DS4_NVFP4_H */

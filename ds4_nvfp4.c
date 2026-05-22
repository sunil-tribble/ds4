/*
 * ds4_nvfp4.c — CPU reference implementation of NVFP4 pack/unpack.
 *
 * Goal: a fully deterministic encode/decode pair that lets us validate the
 * GPU kernels by round-tripping random tensors and comparing. The format
 * MUST match the GPU PTX `mma.sync...kind::mxf4nvf4` fragment expectations:
 *
 *   - 16 4-bit values per block, packed low-nibble-first into 8 bytes
 *   - 1-byte E4M3 scale, biased per NVIDIA's MX spec
 *
 * Reference: "NVFP4: Efficient and Accurate Low-Precision Inference"
 *   https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
 * Reference: "Open Compute Project MX Format Spec 1.0"
 *   https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final.pdf
 */

#include "ds4_nvfp4.h"
#include <math.h>
#include <string.h>

/* ===== E4M3 scale encoding =====
 * We only encode positive scales (sign bit always 0). For block max-abs
 * values found in real KV/weight tensors, the scale range we care about is
 * roughly [1e-4, 1e2]. E4M3 covers [2^-9, 448] cleanly. */
uint8_t ds4_nvfp4_e4m3_encode_scale(float s) {
    if (!(s > 0.0f)) return 0;
    /* Clamp to E4M3 max representable value */
    if (s >= 448.0f) return 0x7E; /* largest finite positive E4M3 */
    if (s < 0x1.0p-9f) return 0;  /* below smallest denormal */

    /* For positive s, find exponent and mantissa */
    int exp;
    float mant_f = frexpf(s, &exp); /* s = mant_f * 2^exp, mant_f in [0.5, 1) */
    /* Normalize so the implicit bit is 1.xxx: */
    mant_f *= 2.0f;
    exp -= 1;
    /* Now s = mant_f * 2^exp, mant_f in [1.0, 2.0) */

    int e_bias = exp + 7; /* E4M3 bias */
    if (e_bias <= 0) {
        /* Denormal: value = (mant/8) * 2^-6 */
        float dn = s * 64.0f; /* = mant/8 */
        int mant = (int)(dn * 8.0f + 0.5f);
        if (mant < 0) mant = 0;
        if (mant > 7) mant = 7;
        return (uint8_t)mant;
    }
    if (e_bias > 15) e_bias = 15;
    int mant = (int)((mant_f - 1.0f) * 8.0f + 0.5f);
    /* Carry on mantissa rounding */
    if (mant > 7) {
        mant = 0;
        e_bias++;
        if (e_bias > 15) e_bias = 15;
    }
    if (mant < 0) mant = 0;
    return (uint8_t)((e_bias << 3) | mant);
}

/* ===== E2M1 weight encoding =====
 * Round a (sign-included) value to the nearest of {0, ±0.5, ±1, ±1.5, ±2,
 * ±3, ±4, ±6}. Ties to even, matching NVIDIA's round-to-nearest-even spec. */
unsigned ds4_nvfp4_e2m1_encode(float v) {
    unsigned sign = (v < 0.0f) ? 8u : 0u;
    float ax = fabsf(v);
    if (ax > 6.0f) ax = 6.0f;

    /* Magnitudes for codes 0..7 (positive). */
    static const float mags[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
    };
    unsigned best = 0;
    float best_diff = fabsf(ax - mags[0]);
    for (unsigned i = 1; i < 8; i++) {
        float d = fabsf(ax - mags[i]);
        if (d < best_diff) {
            best = i;
            best_diff = d;
        } else if (d == best_diff) {
            /* round to even (in the magnitude code, even = ...0) */
            if ((i & 1u) == 0u && (best & 1u) != 0u) {
                best = i;
            }
        }
    }
    return sign | best;
}

/* ===== Block encode =====
 * Strategy: find max(|x|), scale = max(|x|) / 6.0, encode each value as
 * x/scale -> E2M1. Pack pairs of nibbles into bytes (low-nibble first).
 */
void ds4_nvfp4_encode_block(const float *src16, ds4_nvfp4_block *dst) {
    float amax = 0.0f;
    for (int i = 0; i < 16; i++) {
        float ax = fabsf(src16[i]);
        if (ax > amax) amax = ax;
    }
    float scale_f = (amax > 0.0f) ? (amax / 6.0f) : 1.0f;
    dst->scale = ds4_nvfp4_e4m3_encode_scale(scale_f);
    /* Re-derive the actual quantized scale so encode/decode are consistent. */
    float scale_q = ds4_nvfp4_e4m3_decode_scale(dst->scale);
    float inv_scale = (scale_q > 0.0f) ? (1.0f / scale_q) : 0.0f;

    memset(dst->weights, 0, 8);
    for (int i = 0; i < 16; i += 2) {
        unsigned lo = ds4_nvfp4_e2m1_encode(src16[i]     * inv_scale);
        unsigned hi = ds4_nvfp4_e2m1_encode(src16[i + 1] * inv_scale);
        dst->weights[i / 2] = (uint8_t)((hi << 4) | lo);
    }
}

void ds4_nvfp4_decode_block(const ds4_nvfp4_block *src, float *dst16) {
    float scale_q = ds4_nvfp4_e4m3_decode_scale(src->scale);
    for (int i = 0; i < 16; i += 2) {
        uint8_t pair = src->weights[i / 2];
        unsigned lo = pair & 0xFu;
        unsigned hi = (pair >> 4) & 0xFu;
        dst16[i]     = ds4_nvfp4_e2m1_value(lo) * scale_q;
        dst16[i + 1] = ds4_nvfp4_e2m1_value(hi) * scale_q;
    }
}

size_t ds4_nvfp4_encode_tensor(const float *src, size_t n_elems,
                                ds4_nvfp4_block *dst) {
    size_t n_blocks = n_elems / DS4_NVFP4_BLOCK;
    for (size_t b = 0; b < n_blocks; b++) {
        ds4_nvfp4_encode_block(src + b * DS4_NVFP4_BLOCK, dst + b);
    }
    return n_blocks;
}

void ds4_nvfp4_decode_tensor(const ds4_nvfp4_block *src, size_t n_elems,
                              float *dst) {
    size_t n_blocks = n_elems / DS4_NVFP4_BLOCK;
    for (size_t b = 0; b < n_blocks; b++) {
        ds4_nvfp4_decode_block(src + b, dst + b * DS4_NVFP4_BLOCK);
    }
}

float ds4_nvfp4_roundtrip_error(const float *src, size_t n_elems,
                                 float *mean_abs_err) {
    size_t n_blocks = n_elems / DS4_NVFP4_BLOCK;
    float max_err = 0.0f;
    double sum_err = 0.0;
    size_t counted = 0;
    for (size_t b = 0; b < n_blocks; b++) {
        ds4_nvfp4_block packed;
        float roundtrip[16];
        ds4_nvfp4_encode_block(src + b * DS4_NVFP4_BLOCK, &packed);
        ds4_nvfp4_decode_block(&packed, roundtrip);
        for (int i = 0; i < 16; i++) {
            float e = fabsf(src[b * DS4_NVFP4_BLOCK + i] - roundtrip[i]);
            if (e > max_err) max_err = e;
            sum_err += e;
            counted++;
        }
    }
    if (mean_abs_err) {
        *mean_abs_err = (counted > 0) ? (float)(sum_err / (double)counted) : 0.0f;
    }
    return max_err;
}

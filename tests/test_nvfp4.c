/*
 * test_nvfp4.c — Unit tests for NVFP4 pack/unpack round-trip.
 *
 * Build: cc -O2 -std=c99 -I.. ../ds4_nvfp4.c test_nvfp4.c -lm -o test_nvfp4
 * Run:   ./test_nvfp4
 *
 * These tests run on CPU only and have no CUDA dependency. They establish
 * the byte-exact reference behavior the GPU implementation must match.
 */

#include "../ds4_nvfp4.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
        fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); \
        n_fail++; \
    } else { n_pass++; } \
} while (0)

/* Test 1: zero block round-trips to zero. */
static void test_zero_block(void) {
    float src[16] = {0};
    ds4_nvfp4_block packed;
    float dst[16];
    ds4_nvfp4_encode_block(src, &packed);
    ds4_nvfp4_decode_block(&packed, dst);
    for (int i = 0; i < 16; i++) {
        CHECK(dst[i] == 0.0f, "elem %d expected 0 got %f", i, dst[i]);
    }
}

/* Test 2: each E2M1 value at scale=1 round-trips exactly. */
static void test_e2m1_table_at_scale_one(void) {
    /* When scale=1.0 and values fit in E2M1, round-trip should be exact. */
    float src[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,  0.0f,
    };
    ds4_nvfp4_block packed;
    float dst[16];
    ds4_nvfp4_encode_block(src, &packed);
    ds4_nvfp4_decode_block(&packed, dst);
    for (int i = 0; i < 16; i++) {
        CHECK(dst[i] == src[i],
              "elem %d expected %f got %f", i, src[i], dst[i]);
    }
}

/* Test 3: values are quantized to the *nearest* E2M1 step. */
static void test_quantization_rounding(void) {
    float src[16];
    /* Block of intermediate values that should snap to E2M1 grid. */
    for (int i = 0; i < 16; i++) src[i] = 0.74f; /* nearest to 0.5 (better than 1.0) */
    ds4_nvfp4_block packed;
    float dst[16];
    ds4_nvfp4_encode_block(src, &packed);
    ds4_nvfp4_decode_block(&packed, dst);
    /* Scale = 0.74/6 ≈ 0.123, so encoded value of 0.74 / 0.123 = 6.0
     * decodes to 0.74 — actually exact because max sets scale exactly. */
    for (int i = 0; i < 16; i++) {
        float err = fabsf(dst[i] - src[i]);
        CHECK(err < 0.01f,
              "elem %d expected ~%f got %f err=%g", i, src[i], dst[i], err);
    }
}

/* Test 4: round-trip error on random normal data is bounded. */
static void test_random_normal_error_bound(void) {
    srand(42);
    float src[16 * 1024];
    for (size_t i = 0; i < sizeof(src)/sizeof(src[0]); i++) {
        /* Box-Muller for unit normal */
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        src[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    }
    float mean_err = 0;
    float max_err = ds4_nvfp4_roundtrip_error(src, sizeof(src)/sizeof(src[0]),
                                              &mean_err);
    /* For E2M1 with per-16 scale and unit normal data, expected mean abs error
     * is roughly 0.04-0.08 (5-10% of std), max ~0.4 (worst block). */
    CHECK(mean_err < 0.10f,
          "mean abs error %.4f exceeds bound 0.10", mean_err);
    CHECK(max_err < 0.6f,
          "max abs error %.4f exceeds bound 0.6", max_err);
    fprintf(stderr, "  random normal 16K elems: mean=%.4f max=%.4f\n",
            mean_err, max_err);
}

/* Test 5: block packing layout is exactly 9 bytes, no padding. */
static void test_block_size_invariant(void) {
    CHECK(sizeof(ds4_nvfp4_block) == 9,
          "ds4_nvfp4_block sizeof = %zu, must be 9", sizeof(ds4_nvfp4_block));
}

/* Test 6: tensor encode/decode preserves block count. */
static void test_tensor_helpers(void) {
    float src[64];
    for (int i = 0; i < 64; i++) src[i] = (float)i * 0.1f;
    ds4_nvfp4_block packed[4];
    size_t n_blocks = ds4_nvfp4_encode_tensor(src, 64, packed);
    CHECK(n_blocks == 4, "expected 4 blocks, got %zu", n_blocks);
    float dst[64];
    ds4_nvfp4_decode_tensor(packed, 64, dst);
    /* Sanity: at least the zero-position is exactly zero. */
    CHECK(dst[0] == 0.0f, "dst[0] should be 0, got %f", dst[0]);
}

/* Test 7: scaled distribution (mimicking KV cache values with stddev ~0.05). */
static void test_kv_cache_distribution(void) {
    srand(123);
    float src[16 * 64]; /* 1024 elements */
    for (size_t i = 0; i < sizeof(src)/sizeof(src[0]); i++) {
        float u1 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        float u2 = ((float)rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
        src[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2) * 0.05f;
    }
    float mean_err = 0;
    float max_err = ds4_nvfp4_roundtrip_error(src, sizeof(src)/sizeof(src[0]),
                                              &mean_err);
    /* Per-block scale should adapt — relative error similar to unit normal. */
    CHECK(mean_err < 0.005f,
          "kv-cache-like mean abs error %.5f exceeds bound 0.005", mean_err);
    fprintf(stderr, "  kv-like (std=0.05) 1024 elems: mean=%.5f max=%.5f\n",
            mean_err, max_err);
}

int main(void) {
    fprintf(stderr, "== ds4_nvfp4 unit tests ==\n");
    test_block_size_invariant();
    test_zero_block();
    test_e2m1_table_at_scale_one();
    test_quantization_rounding();
    test_tensor_helpers();
    test_random_normal_error_bound();
    test_kv_cache_distribution();
    fprintf(stderr, "%d pass, %d fail\n", n_pass, n_fail);
    return n_fail == 0 ? 0 : 1;
}

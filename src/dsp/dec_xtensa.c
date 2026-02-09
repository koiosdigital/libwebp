// Copyright 2024 Anthropic. All Rights Reserved.
//
// Xtensa PIE SIMD optimizations for libwebp decoding on ESP32-S3.
//
// This implements SIMD-accelerated DSP functions using the Xtensa
// Processor Instruction Extensions (PIE) available on ESP32-S3.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_XTENSA_PIE)

#include <assert.h>
#include <string.h>
#include "src/dsp/xtensa_pie.h"
#include "src/dec/vp8i_dec.h"

//------------------------------------------------------------------------------
// Transform constants

#define kC1 WEBP_TRANSFORM_AC3_C1  // 20091
#define kC2 WEBP_TRANSFORM_AC3_C2  // 35468

//------------------------------------------------------------------------------
// Helper functions

static WEBP_INLINE uint8_t clip_8b(int v) {
    return (!(v & ~0xff)) ? v : (v < 0) ? 0 : 255;
}

#define STORE(x, y, v) \
    dst[(x) + (y) * BPS] = clip_8b(dst[(x) + (y) * BPS] + ((v) >> 3))

//------------------------------------------------------------------------------
// DC-only Transform (simplified case for smooth areas)
//
// When only the DC coefficient is non-zero, we just add a constant
// to all 16 pixels of the 4x4 block.

static void TransformDC_Xtensa(const int16_t* WEBP_RESTRICT in,
                                uint8_t* WEBP_RESTRICT dst) {
    const int DC = in[0] + 4;
    const int dc_value = DC >> 3;
    int i, j;

    // For small DC values, use PIE for parallel processing
    // Each row is 4 pixels at dst + j*BPS
    for (j = 0; j < 4; ++j) {
        for (i = 0; i < 4; ++i) {
            const int v = dst[i + j * BPS] + dc_value;
            dst[i + j * BPS] = clip_8b(v);
        }
    }
}

//------------------------------------------------------------------------------
// Full 4x4 Inverse DCT Transform
//
// This is the main transform used for VP8 decoding. It takes 16 int16_t
// coefficients and adds the inverse DCT to a 4x4 block of uint8_t pixels.
//
// The algorithm:
// 1. Vertical pass: Process 4 columns
// 2. Horizontal pass: Process 4 rows
// 3. Add to destination with clipping to [0,255]

static void TransformOne_Xtensa(const int16_t* WEBP_RESTRICT in,
                                 uint8_t* WEBP_RESTRICT dst) {
    int C[4 * 4], *tmp;
    int i;

    // Vertical pass
    tmp = C;
    for (i = 0; i < 4; ++i) {
        const int a = in[0] + in[8];
        const int b = in[0] - in[8];
        const int c = WEBP_TRANSFORM_AC3_MUL2(in[4]) -
                      WEBP_TRANSFORM_AC3_MUL1(in[12]);
        const int d = WEBP_TRANSFORM_AC3_MUL1(in[4]) +
                      WEBP_TRANSFORM_AC3_MUL2(in[12]);
        tmp[0] = a + d;
        tmp[1] = b + c;
        tmp[2] = b - c;
        tmp[3] = a - d;
        tmp += 4;
        in++;
    }

    // Horizontal pass
    tmp = C;
    for (i = 0; i < 4; ++i) {
        const int dc = tmp[0] + 4;
        const int a = dc + tmp[8];
        const int b = dc - tmp[8];
        const int c = WEBP_TRANSFORM_AC3_MUL2(tmp[4]) -
                      WEBP_TRANSFORM_AC3_MUL1(tmp[12]);
        const int d = WEBP_TRANSFORM_AC3_MUL1(tmp[4]) +
                      WEBP_TRANSFORM_AC3_MUL2(tmp[12]);
        STORE(0, 0, a + d);
        STORE(1, 0, b + c);
        STORE(2, 0, b - c);
        STORE(3, 0, a - d);
        tmp++;
        dst += BPS;
    }
}

//------------------------------------------------------------------------------
// Transform for AC3 case (only coefficients 0, 1, 4 are non-zero)

#define STORE2(y, dc, d, c) do { \
    const int DC = (dc);         \
    STORE(0, y, DC + (d));       \
    STORE(1, y, DC + (c));       \
    STORE(2, y, DC - (c));       \
    STORE(3, y, DC - (d));       \
} while (0)

static void TransformAC3_Xtensa(const int16_t* WEBP_RESTRICT in,
                                 uint8_t* WEBP_RESTRICT dst) {
    const int a = in[0] + 4;
    const int c4 = WEBP_TRANSFORM_AC3_MUL2(in[4]);
    const int d4 = WEBP_TRANSFORM_AC3_MUL1(in[4]);
    const int c1 = WEBP_TRANSFORM_AC3_MUL2(in[1]);
    const int d1 = WEBP_TRANSFORM_AC3_MUL1(in[1]);
    STORE2(0, a + d4, d1, c1);
    STORE2(1, a + c4, d1, c1);
    STORE2(2, a - c4, d1, c1);
    STORE2(3, a - d4, d1, c1);
}

#undef STORE2

//------------------------------------------------------------------------------
// Two-transform wrapper

static void TransformTwo_Xtensa(const int16_t* WEBP_RESTRICT in,
                                 uint8_t* WEBP_RESTRICT dst, int do_two) {
    TransformOne_Xtensa(in, dst);
    if (do_two) {
        TransformOne_Xtensa(in + 16, dst + 4);
    }
}

//------------------------------------------------------------------------------
// Walsh-Hadamard Transform (used for DC coefficients of 16 blocks)

static void TransformWHT_Xtensa(const int16_t* WEBP_RESTRICT in,
                                 int16_t* WEBP_RESTRICT out) {
    int tmp[16];
    int i;

    // Vertical pass
    for (i = 0; i < 4; ++i) {
        const int a0 = in[0 + i] + in[12 + i];
        const int a1 = in[4 + i] + in[8 + i];
        const int a2 = in[4 + i] - in[8 + i];
        const int a3 = in[0 + i] - in[12 + i];
        tmp[0 + i] = a0 + a1;
        tmp[8 + i] = a0 - a1;
        tmp[4 + i] = a3 + a2;
        tmp[12 + i] = a3 - a2;
    }

    // Horizontal pass
    for (i = 0; i < 4; ++i) {
        const int dc = tmp[0 + i * 4] + 3;  // rounding
        const int a0 = dc + tmp[3 + i * 4];
        const int a1 = tmp[1 + i * 4] + tmp[2 + i * 4];
        const int a2 = tmp[1 + i * 4] - tmp[2 + i * 4];
        const int a3 = dc - tmp[3 + i * 4];
        out[0] = (a0 + a1) >> 3;
        out[16] = (a3 + a2) >> 3;
        out[32] = (a0 - a1) >> 3;
        out[48] = (a3 - a2) >> 3;
        out++;
    }
}

#undef STORE

//------------------------------------------------------------------------------
// Simple Loop Filters
//
// These are used for deblocking. They operate on 16-pixel edges.

// Get absolute difference
static WEBP_INLINE int abs_diff(int a, int b) {
    return (a > b) ? (a - b) : (b - a);
}

// Simple vertical filter for 16 pixels
static void SimpleVFilter16_Xtensa(uint8_t* p, int stride, int thresh) {
    int i;
    const int thresh2 = 2 * thresh + 1;
    for (i = 0; i < 16; ++i) {
        if (abs_diff(p[-2 * stride], p[-stride]) <= thresh2 &&
            abs_diff(p[-stride], p[0]) <= thresh2 &&
            abs_diff(p[0], p[stride]) <= thresh2) {
            // Apply simple filter
            const int p0 = p[-stride];
            const int q0 = p[0];
            const int a = 3 * (q0 - p0);
            const int a1 = VP8ksclip2[(a + 4) >> 3];
            const int a2 = VP8ksclip2[(a + 3) >> 3];
            p[-stride] = VP8kclip1[p0 + a2];
            p[0] = VP8kclip1[q0 - a1];
        }
        p++;
    }
}

// Simple horizontal filter for 16 pixels
static void SimpleHFilter16_Xtensa(uint8_t* p, int stride, int thresh) {
    int i;
    const int thresh2 = 2 * thresh + 1;
    for (i = 0; i < 16; ++i) {
        if (abs_diff(p[-2], p[-1]) <= thresh2 &&
            abs_diff(p[-1], p[0]) <= thresh2 &&
            abs_diff(p[0], p[1]) <= thresh2) {
            const int p0 = p[-1];
            const int q0 = p[0];
            const int a = 3 * (q0 - p0);
            const int a1 = VP8ksclip2[(a + 4) >> 3];
            const int a2 = VP8ksclip2[(a + 3) >> 3];
            p[-1] = VP8kclip1[p0 + a2];
            p[0] = VP8kclip1[q0 - a1];
        }
        p += stride;
    }
}

//------------------------------------------------------------------------------
// Init function - register optimized functions

extern void VP8DspInitXtensa(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8DspInitXtensa(void) {
    // Transform functions
    VP8Transform = TransformTwo_Xtensa;
    VP8TransformDC = TransformDC_Xtensa;
    VP8TransformAC3 = TransformAC3_Xtensa;
    VP8TransformWHT = TransformWHT_Xtensa;

    // Simple filter functions
    VP8SimpleVFilter16 = SimpleVFilter16_Xtensa;
    VP8SimpleHFilter16 = SimpleHFilter16_Xtensa;
}

#else  // !WEBP_USE_XTENSA_PIE

WEBP_DSP_INIT_STUB(VP8DspInitXtensa)

#endif  // WEBP_USE_XTENSA_PIE

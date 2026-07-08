// Copyright 2026 Koios. All Rights Reserved.
//
// Xtensa PIE SIMD optimizations for VP8L (lossless) decoding on ESP32-S3.
//
// The VP8L hot path for our content is: Huffman decode (scalar, not
// vectorizable here) -> inverse transforms -> output conversion. This file
// vectorizes the two per-pixel passes that run over every decoded pixel:
//
//   - VP8LAddGreenToBlueAndRed: inverse of the subtract-green transform,
//     present in virtually every VP8L stream.
//   - VP8LConvertBGRAToRGBA: final canvas conversion for MODE_RGBA output.
//
// Both process 4 pixels (16 bytes) per iteration in PIE Q registers.
//
// Safety: PIE shift/add lane semantics are hard to verify off-device, so
// VP8LDspInitXtensa() runs each vector function against the C reference on a
// test vector (covering unaligned heads, tails, sign/overflow byte patterns)
// and only installs it on an exact match. A mismatch logs once and keeps C.

#include "src/dsp/dsp.h"

#if defined(WEBP_USE_XTENSA_PIE)

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "src/dsp/lossless.h"
#include "src/dsp/xtensa_pie.h"

//------------------------------------------------------------------------------
// Broadcast constants for EE.VLDBC.32

static const uint32_t kMaskRedBlue = 0x00ff00ffu;     // R and B bytes of ARGB
static const uint32_t kMaskGreenAlpha = 0xff00ff00u;  // A and G bytes of ARGB
static const uint32_t kMaskLowByte = 0x000000ffu;
static const uint32_t kMaskByte2 = 0x00ff0000u;

//------------------------------------------------------------------------------
// Scalar single-pixel helpers (C reference semantics), used for unaligned
// heads and <4 pixel tails around the vector body.

static WEBP_INLINE uint32_t AddGreenOnePixel(uint32_t argb) {
    const uint32_t green = (argb >> 8) & 0xff;
    uint32_t red_blue = (argb & 0x00ff00ffu);
    red_blue += (green << 16) | green;
    red_blue &= 0x00ff00ffu;
    return (argb & 0xff00ff00u) | red_blue;
}

static WEBP_INLINE void BGRAToRGBAOnePixel(uint32_t argb, uint8_t* dst) {
    dst[0] = (argb >> 16) & 0xff;
    dst[1] = (argb >> 8) & 0xff;
    dst[2] = (argb >> 0) & 0xff;
    dst[3] = (argb >> 24) & 0xff;
}

//------------------------------------------------------------------------------
// AddGreenToBlueAndRed: dst[i] = argb with green added (mod 256) to R and B
//
// Per 32-bit lane:
//   green    = (v >> 8) & 0xff
//   red_blue = ((v & 0x00ff00ff) + (green | green << 16)) & 0x00ff00ff
//   out      = (v & 0xff00ff00) | red_blue
//
// The add runs as EE.VADDS.S16 on 16-bit lanes [B][R]: operands are at most
// 255 + 255 = 510, so signed saturation never triggers and it behaves as a
// plain add; the 0x00ff00ff mask afterwards provides the per-byte wraparound.

static void VP8LAddGreenToBlueAndRed_Xtensa(const uint32_t* src,
                                            int num_pixels, uint32_t* dst) {
    // The vector body needs src and dst to hit 16-byte alignment together
    if ((((uintptr_t)src ^ (uintptr_t)dst) & 15u) != 0) {
        VP8LAddGreenToBlueAndRed_C(src, num_pixels, dst);
        return;
    }

    // Scalar head until aligned
    while (num_pixels > 0 && ((uintptr_t)src & 15u) != 0) {
        *dst++ = AddGreenOnePixel(*src++);
        --num_pixels;
    }

    // Vector body: 4 pixels per iteration
    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q7, &kMaskRedBlue);
        PIE_VLDBC_32(q6, &kMaskLowByte);
        PIE_VLDBC_32(q5, &kMaskGreenAlpha);

        while (n-- > 0) {
            PIE_VLD_128_IP(q0, src);
            PIE_SET_SAR(8);
            PIE_VSR_32(q2, q0);         // v >> 8
            PIE_ANDQ(q2, q2, q6);       // green at bits [7:0]
            PIE_SET_SAR(16);
            PIE_VSL_32(q3, q2);         // green << 16
            PIE_ORQ(q2, q2, q3);        // green | green << 16
            PIE_ANDQ(q1, q0, q7);       // red_blue = v & 0x00ff00ff
            PIE_VADDS_S16(q1, q1, q2);  // [B+g][R+g], no saturation possible
            PIE_ANDQ(q1, q1, q7);       // per-byte wrap
            PIE_ANDQ(q0, q0, q5);       // keep A and G
            PIE_ORQ(q0, q0, q1);
            PIE_VST_128_IP(q0, dst);
        }
    }

    // Scalar tail
    while (num_pixels-- > 0) {
        *dst++ = AddGreenOnePixel(*src++);
    }
}

//------------------------------------------------------------------------------
// BGRA -> RGBA byte reorder (swap R and B within each 32-bit pixel)
//
// Per 32-bit lane: out = (v & 0xff00ff00) | ((v >> 16) & 0xff) | ((v << 16) & 0x00ff0000)

static void VP8LConvertBGRAToRGBA_Xtensa(const uint32_t* WEBP_RESTRICT src,
                                         int num_pixels,
                                         uint8_t* WEBP_RESTRICT dst) {
    if ((((uintptr_t)src ^ (uintptr_t)dst) & 15u) != 0) {
        VP8LConvertBGRAToRGBA_C(src, num_pixels, dst);
        return;
    }

    while (num_pixels > 0 && ((uintptr_t)src & 15u) != 0) {
        BGRAToRGBAOnePixel(*src++, dst);
        dst += 4;
        --num_pixels;
    }

    int n = num_pixels >> 2;
    if (n > 0) {
        num_pixels -= n << 2;

        PIE_VLDBC_32(q5, &kMaskGreenAlpha);
        PIE_VLDBC_32(q6, &kMaskLowByte);
        PIE_VLDBC_32(q4, &kMaskByte2);
        PIE_SET_SAR(16);  // both shifts are by 16, set once

        while (n-- > 0) {
            PIE_VLD_128_IP(q0, src);
            PIE_VSR_32(q1, q0);    // R to bits [7:0]
            PIE_ANDQ(q1, q1, q6);
            PIE_VSL_32(q2, q0);    // B to bits [23:16]
            PIE_ANDQ(q2, q2, q4);
            PIE_ANDQ(q0, q0, q5);  // keep A and G
            PIE_ORQ(q0, q0, q1);
            PIE_ORQ(q0, q0, q2);
            PIE_VST_128_IP(q0, dst);
        }
    }

    while (num_pixels-- > 0) {
        BGRAToRGBAOnePixel(*src++, dst);
        dst += 4;
    }
}

//------------------------------------------------------------------------------
// Init-time self-check: run PIE and C implementations over byte patterns that
// exercise sign bits, carries and wraparound, plus unaligned heads and tails.
// Install the PIE version only on an exact output match.

#define PIE_CHECK_PIXELS 32

static void FillCheckInput(uint32_t* buf) {
    int i;
    static const uint32_t kPatterns[8] = {
        0x00000000u, 0xffffffffu, 0x80808080u, 0x7f7f7f7fu,
        0x01ff01ffu, 0xff01ff01u, 0xdeadbeefu, 0x00ff00ffu,
    };
    for (i = 0; i < PIE_CHECK_PIXELS; ++i) {
        // Mix fixed edge-case patterns with a ramp
        buf[i] = kPatterns[i & 7] ^ (uint32_t)(i * 0x01010101u);
    }
}

static int CheckAddGreen(void) {
    PIE_ALIGN static uint32_t in[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint32_t out_c[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint32_t out_pie[PIE_CHECK_PIXELS];
    FillCheckInput(in);

    // Aligned, vector body + tail (30 = head 0, body 28, tail 2)
    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LAddGreenToBlueAndRed_C(in, 30, out_c);
    VP8LAddGreenToBlueAndRed_Xtensa(in, 30, out_pie);
    if (memcmp(out_c, out_pie, sizeof(out_c)) != 0) return 0;

    // Unaligned head (src+1/dst+1 stay congruent mod 16)
    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LAddGreenToBlueAndRed_C(in + 1, 27, out_c + 1);
    VP8LAddGreenToBlueAndRed_Xtensa(in + 1, 27, out_pie + 1);
    return memcmp(out_c, out_pie, sizeof(out_c)) == 0;
}

static int CheckConvertBGRAToRGBA(void) {
    PIE_ALIGN static uint32_t in[PIE_CHECK_PIXELS];
    PIE_ALIGN static uint8_t out_c[PIE_CHECK_PIXELS * 4];
    PIE_ALIGN static uint8_t out_pie[PIE_CHECK_PIXELS * 4];
    FillCheckInput(in);

    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LConvertBGRAToRGBA_C(in, 30, out_c);
    VP8LConvertBGRAToRGBA_Xtensa(in, 30, out_pie);
    if (memcmp(out_c, out_pie, sizeof(out_c)) != 0) return 0;

    memset(out_c, 0, sizeof(out_c));
    memset(out_pie, 0, sizeof(out_pie));
    VP8LConvertBGRAToRGBA_C(in + 1, 27, out_c + 4);
    VP8LConvertBGRAToRGBA_Xtensa(in + 1, 27, out_pie + 4);
    return memcmp(out_c, out_pie, sizeof(out_c)) == 0;
}

//------------------------------------------------------------------------------
// Entry point

extern void VP8LDspInitXtensa(void);

WEBP_TSAN_IGNORE_FUNCTION void VP8LDspInitXtensa(void) {
    if (CheckAddGreen()) {
        VP8LAddGreenToBlueAndRed = VP8LAddGreenToBlueAndRed_Xtensa;
    } else {
        printf("libwebp: PIE AddGreenToBlueAndRed self-check failed, using C\n");
    }
    if (CheckConvertBGRAToRGBA()) {
        VP8LConvertBGRAToRGBA = VP8LConvertBGRAToRGBA_Xtensa;
    } else {
        printf("libwebp: PIE ConvertBGRAToRGBA self-check failed, using C\n");
    }
}

#else  // !WEBP_USE_XTENSA_PIE

WEBP_DSP_INIT_STUB(VP8LDspInitXtensa)

#endif  // WEBP_USE_XTENSA_PIE

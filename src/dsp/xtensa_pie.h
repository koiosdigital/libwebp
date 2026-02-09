// Copyright 2024 Anthropic. All Rights Reserved.
//
// Xtensa PIE SIMD helpers for ESP32-S3.
//
// This header provides inline assembly wrappers for Xtensa PIE instructions
// used in libwebp DSP optimizations.

#ifndef WEBP_DSP_XTENSA_PIE_H_
#define WEBP_DSP_XTENSA_PIE_H_

#if defined(WEBP_USE_XTENSA_PIE)

#include <stdint.h>
#include <stdalign.h>

// PIE Q registers are 128-bit (16 bytes)
// They can hold 16 x int8, 8 x int16, 4 x int32, or 4 x float32

//------------------------------------------------------------------------------
// Alignment helpers

#define PIE_ALIGN __attribute__((aligned(16)))

// Stack-aligned buffer declaration
#define PIE_ALIGNED_ARRAY(type, name, size) \
    alignas(16) type name[size]

//------------------------------------------------------------------------------
// Transform constants
// These match the libwebp C code definitions

#define PIE_TRANSFORM_C1 20091
#define PIE_TRANSFORM_C2 35468

//------------------------------------------------------------------------------
// Low-level PIE inline assembly wrappers
//
// Note: ESP32-S3 PIE uses Q registers (q0-q7) for 128-bit vectors
// and A registers for general-purpose operations.

// Load 128 bits from memory with post-increment of 16 bytes
// Usage: PIE_VLD_128_IP(q_reg_out, ptr)
// After: ptr += 16
#define PIE_VLD_128_IP(qreg, ptr) \
    __asm__ volatile ( \
        "ee.vld.128.ip " #qreg ", %0, 16" \
        : "+a"(ptr) \
        : \
        : "memory" \
    )

// Store 128 bits to memory with post-increment of 16 bytes
#define PIE_VST_128_IP(qreg, ptr) \
    __asm__ volatile ( \
        "ee.vst.128.ip " #qreg ", %0, 16" \
        : "+a"(ptr) \
        : \
        : "memory" \
    )

// Load 128 bits from memory without increment
#define PIE_VLD_128(qreg, ptr) \
    __asm__ volatile ( \
        "ee.vld.128.ip " #qreg ", %0, 0" \
        : "+a"(ptr) \
        : \
        : "memory" \
    )

// Store 128 bits to memory without increment
#define PIE_VST_128(qreg, ptr) \
    __asm__ volatile ( \
        "ee.vst.128.ip " #qreg ", %0, 0" \
        : "+a"(ptr) \
        : \
        : "memory" \
    )

// Saturating add for signed 16-bit: dst = a + b (saturated)
#define PIE_VADDS_S16(dst, a, b) \
    __asm__ volatile ( \
        "ee.vadds.s16 " #dst ", " #a ", " #b \
        : : : \
    )

// Saturating subtract for signed 16-bit: dst = a - b (saturated)
#define PIE_VSUBS_S16(dst, a, b) \
    __asm__ volatile ( \
        "ee.vsubs.s16 " #dst ", " #a ", " #b \
        : : : \
    )

// Saturating add for signed 8-bit: dst = a + b (saturated)
#define PIE_VADDS_S8(dst, a, b) \
    __asm__ volatile ( \
        "ee.vadds.s8 " #dst ", " #a ", " #b \
        : : : \
    )

// Zero a Q register
#define PIE_VZERO(qreg) \
    __asm__ volatile ( \
        "ee.zero.q " #qreg \
        : : : \
    )

// Broadcast 16-bit value to all lanes of Q register
#define PIE_VLDBC_16(qreg, ptr) \
    __asm__ volatile ( \
        "ee.vldbc.16 " #qreg ", %0" \
        : \
        : "a"(ptr) \
        : "memory" \
    )

// Interleave bytes (widen int8 to int16)
#define PIE_VZIP_8(qreg_a, qreg_b) \
    __asm__ volatile ( \
        "ee.vzip.8 " #qreg_a ", " #qreg_b \
        : : : \
    )

// Interleave 16-bit values
#define PIE_VZIP_16(qreg_a, qreg_b) \
    __asm__ volatile ( \
        "ee.vzip.16 " #qreg_a ", " #qreg_b \
        : : : \
    )

// Clear accumulator
#define PIE_ZERO_ACCX() \
    __asm__ volatile ("ee.zero.accx" : : :)

// Multiply-accumulate signed 16-bit vectors
#define PIE_VMULAS_S16_ACCX(a, b) \
    __asm__ volatile ( \
        "ee.vmulas.s16.accx " #a ", " #b \
        : : : \
    )

// Read accumulator lower 32 bits
#define PIE_RUR_ACCX_0(result) \
    __asm__ volatile ( \
        "rur.accx_0 %0" \
        : "=a"(result) \
        : : \
    )

//------------------------------------------------------------------------------
// Clipping/saturation helpers

// Clip value to [0, 255] range
static inline uint8_t pie_clip_u8(int v) {
    return (v < 0) ? 0 : (v > 255) ? 255 : (uint8_t)v;
}

#endif  // WEBP_USE_XTENSA_PIE

#endif  // WEBP_DSP_XTENSA_PIE_H_

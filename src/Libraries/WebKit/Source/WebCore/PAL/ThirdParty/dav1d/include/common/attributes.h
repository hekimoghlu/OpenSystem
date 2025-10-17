/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 25, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef DAV1D_COMMON_ATTRIBUTES_H
#define DAV1D_COMMON_ATTRIBUTES_H

#include "config.h"

#include <stddef.h>
#include <assert.h>

#ifndef __has_attribute
#define __has_attribute(x) 0
#endif

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#ifdef __GNUC__
#define ATTR_ALIAS __attribute__((may_alias))
#define ATTR_FORMAT_PRINTF(fmt, attr) __attribute__((__format__(__printf__, fmt, attr)))
#define COLD __attribute__((cold))
#else
#define ATTR_ALIAS
#define ATTR_FORMAT_PRINTF(fmt, attr)
#define COLD
#endif

#if ARCH_X86_64
/* x86-64 needs 32- and 64-byte alignment for AVX2 and AVX-512. */
#define ALIGN_64_VAL 64
#define ALIGN_32_VAL 32
#define ALIGN_16_VAL 16
#elif ARCH_X86_32 || ARCH_ARM || ARCH_AARCH64 || ARCH_PPC64LE
/* ARM doesn't benefit from anything more than 16-byte alignment. */
#define ALIGN_64_VAL 16
#define ALIGN_32_VAL 16
#define ALIGN_16_VAL 16
#else
/* No need for extra alignment on platforms without assembly. */
#define ALIGN_64_VAL 8
#define ALIGN_32_VAL 8
#define ALIGN_16_VAL 8
#endif

/*
 * API for variables, struct members (ALIGN()) like:
 * uint8_t var[1][2][3][4]
 * becomes:
 * ALIGN(uint8_t var[1][2][3][4], alignment).
 */
#ifdef _MSC_VER
#define ALIGN(ll, a) \
    __declspec(align(a)) ll
#else
#define ALIGN(line, align) \
    line __attribute__((aligned(align)))
#endif

/*
 * API for stack alignment (ALIGN_STK_$align()) of variables like:
 * uint8_t var[1][2][3][4]
 * becomes:
 * ALIGN_STK_$align(uint8_t, var, 1, [2][3][4])
 */
#define ALIGN_STK_64(type, var, sz1d, sznd) \
    ALIGN(type var[sz1d]sznd, ALIGN_64_VAL)
#define ALIGN_STK_32(type, var, sz1d, sznd) \
    ALIGN(type var[sz1d]sznd, ALIGN_32_VAL)
#define ALIGN_STK_16(type, var, sz1d, sznd) \
    ALIGN(type var[sz1d]sznd, ALIGN_16_VAL)

/*
 * Forbid inlining of a function:
 * static NOINLINE void func() {}
 */
#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#elif __has_attribute(noclone)
#define NOINLINE __attribute__((noinline, noclone))
#else
#define NOINLINE __attribute__((noinline))
#endif

#ifdef __clang__
#define NO_SANITIZE(x) __attribute__((no_sanitize(x)))
#else
#define NO_SANITIZE(x)
#endif

#if defined(NDEBUG) && (defined(__GNUC__) || defined(__clang__))
#undef assert
#define assert(x) do { if (!(x)) __builtin_unreachable(); } while (0)
#elif defined(NDEBUG) && defined(_MSC_VER)
#undef assert
#define assert __assume
#endif

#if defined(__GNUC__) && !defined(__INTEL_COMPILER) && !defined(__clang__)
#    define dav1d_uninit(x) x=x
#else
#    define dav1d_uninit(x) x
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>

static inline int ctz(const unsigned int mask) {
    unsigned long idx;
    _BitScanForward(&idx, mask);
    return idx;
}

static inline int clz(const unsigned int mask) {
    unsigned long leading_zero = 0;
    _BitScanReverse(&leading_zero, mask);
    return (31 - leading_zero);
}

#ifdef _WIN64
static inline int clzll(const unsigned long long mask) {
    unsigned long leading_zero = 0;
    _BitScanReverse64(&leading_zero, mask);
    return (63 - leading_zero);
}
#else /* _WIN64 */
static inline int clzll(const unsigned long long mask) {
    if (mask >> 32)
        return clz((unsigned)(mask >> 32));
    else
        return clz((unsigned)mask) + 32;
}
#endif /* _WIN64 */
#else /* !_MSC_VER */
static inline int ctz(const unsigned int mask) {
    return __builtin_ctz(mask);
}

static inline int clz(const unsigned int mask) {
    return __builtin_clz(mask);
}

static inline int clzll(const unsigned long long mask) {
    return __builtin_clzll(mask);
}
#endif /* !_MSC_VER */

#ifndef static_assert
#define CHECK_OFFSET(type, field, name) \
    struct check_##type##_##field { int x[(name == offsetof(type, field)) ? 1 : -1]; }
#else
#define CHECK_OFFSET(type, field, name) \
    static_assert(name == offsetof(type, field), #field)
#endif

#ifdef _MSC_VER
#define PACKED(...) __pragma(pack(push, 1)) __VA_ARGS__ __pragma(pack(pop))
#else
#define PACKED(...) __VA_ARGS__ __attribute__((__packed__))
#endif

#endif /* DAV1D_COMMON_ATTRIBUTES_H */

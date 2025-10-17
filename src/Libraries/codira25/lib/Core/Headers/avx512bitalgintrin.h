/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 22, 2021.
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
#ifndef __IMMINTRIN_H
#error "Never use <avx512bitalgintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512BITALGINTRIN_H
#define __AVX512BITALGINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512bitalg,evex512"),                           \
                 __min_vector_width__(512)))

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS constexpr
#else
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS
#endif

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_popcnt_epi16(__m512i __A)
{
  return (__m512i)__builtin_elementwise_popcount((__v32hu)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_popcnt_epi16(__m512i __A, __mmask32 __U, __m512i __B)
{
  return (__m512i) __builtin_ia32_selectw_512((__mmask32) __U,
              (__v32hi) _mm512_popcnt_epi16(__B),
              (__v32hi) __A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_popcnt_epi16(__mmask32 __U, __m512i __B)
{
  return _mm512_mask_popcnt_epi16((__m512i) _mm512_setzero_si512(),
              __U,
              __B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_popcnt_epi8(__m512i __A)
{
  return (__m512i)__builtin_elementwise_popcount((__v64qu)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_popcnt_epi8(__m512i __A, __mmask64 __U, __m512i __B)
{
  return (__m512i) __builtin_ia32_selectb_512((__mmask64) __U,
              (__v64qi) _mm512_popcnt_epi8(__B),
              (__v64qi) __A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_popcnt_epi8(__mmask64 __U, __m512i __B)
{
  return _mm512_mask_popcnt_epi8((__m512i) _mm512_setzero_si512(),
              __U,
              __B);
}

static __inline__ __mmask64 __DEFAULT_FN_ATTRS
_mm512_mask_bitshuffle_epi64_mask(__mmask64 __U, __m512i __A, __m512i __B)
{
  return (__mmask64) __builtin_ia32_vpshufbitqmb512_mask((__v64qi) __A,
              (__v64qi) __B,
              __U);
}

static __inline__ __mmask64 __DEFAULT_FN_ATTRS
_mm512_bitshuffle_epi64_mask(__m512i __A, __m512i __B)
{
  return _mm512_mask_bitshuffle_epi64_mask((__mmask64) -1,
              __A,
              __B);
}

#undef __DEFAULT_FN_ATTRS
#undef __DEFAULT_FN_ATTRS_CONSTEXPR

#endif

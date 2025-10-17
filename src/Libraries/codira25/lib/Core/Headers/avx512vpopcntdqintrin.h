/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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
#error                                                                         \
    "Never use <avx512vpopcntdqintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VPOPCNTDQINTRIN_H
#define __AVX512VPOPCNTDQINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512vpopcntdq,evex512"),                        \
                 __min_vector_width__(512)))

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS constexpr
#else
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS
#endif

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_popcnt_epi64(__m512i __A) {
  return (__m512i)__builtin_elementwise_popcount((__v8du)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_popcnt_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectq_512(
      (__mmask8)__U, (__v8di)_mm512_popcnt_epi64(__A), (__v8di)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_popcnt_epi64(__mmask8 __U, __m512i __A) {
  return _mm512_mask_popcnt_epi64((__m512i)_mm512_setzero_si512(), __U, __A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_popcnt_epi32(__m512i __A) {
  return (__m512i)__builtin_elementwise_popcount((__v16su)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_popcnt_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectd_512(
      (__mmask16)__U, (__v16si)_mm512_popcnt_epi32(__A), (__v16si)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_popcnt_epi32(__mmask16 __U, __m512i __A) {
  return _mm512_mask_popcnt_epi32((__m512i)_mm512_setzero_si512(), __U, __A);
}

#undef __DEFAULT_FN_ATTRS
#undef __DEFAULT_FN_ATTRS_CONSTEXPR

#endif

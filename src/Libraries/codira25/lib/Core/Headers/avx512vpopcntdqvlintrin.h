/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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
    "Never use <avx512vpopcntdqvlintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VPOPCNTDQVLINTRIN_H
#define __AVX512VPOPCNTDQVLINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512vpopcntdq,avx512vl,no-evex512"),            \
                 __min_vector_width__(128)))
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512vpopcntdq,avx512vl,no-evex512"),            \
                 __min_vector_width__(256)))

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS128_CONSTEXPR __DEFAULT_FN_ATTRS128 constexpr
#define __DEFAULT_FN_ATTRS256_CONSTEXPR __DEFAULT_FN_ATTRS256 constexpr
#else
#define __DEFAULT_FN_ATTRS128_CONSTEXPR __DEFAULT_FN_ATTRS128
#define __DEFAULT_FN_ATTRS256_CONSTEXPR __DEFAULT_FN_ATTRS256
#endif

static __inline__ __m128i __DEFAULT_FN_ATTRS128_CONSTEXPR
_mm_popcnt_epi64(__m128i __A) {
  return (__m128i)__builtin_elementwise_popcount((__v2du)__A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_popcnt_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectq_128(
      (__mmask8)__U, (__v2di)_mm_popcnt_epi64(__A), (__v2di)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_popcnt_epi64(__mmask8 __U, __m128i __A) {
  return _mm_mask_popcnt_epi64((__m128i)_mm_setzero_si128(), __U, __A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128_CONSTEXPR
_mm_popcnt_epi32(__m128i __A) {
  return (__m128i)__builtin_elementwise_popcount((__v4su)__A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_popcnt_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectd_128(
      (__mmask8)__U, (__v4si)_mm_popcnt_epi32(__A), (__v4si)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_popcnt_epi32(__mmask8 __U, __m128i __A) {
  return _mm_mask_popcnt_epi32((__m128i)_mm_setzero_si128(), __U, __A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256_CONSTEXPR
_mm256_popcnt_epi64(__m256i __A) {
  return (__m256i)__builtin_elementwise_popcount((__v4du)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_popcnt_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectq_256(
      (__mmask8)__U, (__v4di)_mm256_popcnt_epi64(__A), (__v4di)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_popcnt_epi64(__mmask8 __U, __m256i __A) {
  return _mm256_mask_popcnt_epi64((__m256i)_mm256_setzero_si256(), __U, __A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256_CONSTEXPR
_mm256_popcnt_epi32(__m256i __A) {
  return (__m256i)__builtin_elementwise_popcount((__v8su)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_popcnt_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectd_256(
      (__mmask8)__U, (__v8si)_mm256_popcnt_epi32(__A), (__v8si)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_popcnt_epi32(__mmask8 __U, __m256i __A) {
  return _mm256_mask_popcnt_epi32((__m256i)_mm256_setzero_si256(), __U, __A);
}

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256
#undef __DEFAULT_FN_ATTRS128_CONSTEXPR
#undef __DEFAULT_FN_ATTRS256_CONSTEXPR

#endif

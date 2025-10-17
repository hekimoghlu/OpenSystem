/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 29, 2024.
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
    "Never use <movrs_avx10_2_512intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __MOVRS_AVX10_2_512INTRIN_H
#define __MOVRS_AVX10_2_512INTRIN_H
#ifdef __x86_64__

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS512                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("movrs, avx10.2-512"), __min_vector_width__(512)))

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_loadrs_epi8(void const *__A) {
  return (__m512i)__builtin_ia32_vmovrsb512((const __v64qi *)(__A));
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_loadrs_epi8(__m512i __W, __mmask64 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectb_512(
      (__mmask64)__U, (__v64qi)_mm512_loadrs_epi8(__A), (__v64qi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_loadrs_epi8(__mmask64 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectb_512((__mmask64)__U,
                                             (__v64qi)_mm512_loadrs_epi8(__A),
                                             (__v64qi)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_loadrs_epi32(void const *__A) {
  return (__m512i)__builtin_ia32_vmovrsd512((const __v16si *)(__A));
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_loadrs_epi32(__m512i __W, __mmask16 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectd_512(
      (__mmask16)__U, (__v16si)_mm512_loadrs_epi32(__A), (__v16si)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_loadrs_epi32(__mmask16 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectd_512((__mmask16)__U,
                                             (__v16si)_mm512_loadrs_epi32(__A),
                                             (__v16si)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_loadrs_epi64(void const *__A) {
  return (__m512i)__builtin_ia32_vmovrsq512((const __v8di *)(__A));
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_loadrs_epi64(__m512i __W, __mmask8 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectq_512(
      (__mmask8)__U, (__v8di)_mm512_loadrs_epi64(__A), (__v8di)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_loadrs_epi64(__mmask8 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectq_512((__mmask8)__U,
                                             (__v8di)_mm512_loadrs_epi64(__A),
                                             (__v8di)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_loadrs_epi16(void const *__A) {
  return (__m512i)__builtin_ia32_vmovrsw512((const __v32hi *)(__A));
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_loadrs_epi16(__m512i __W, __mmask32 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectw_512(
      (__mmask32)__U, (__v32hi)_mm512_loadrs_epi16(__A), (__v32hi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_loadrs_epi16(__mmask32 __U, void const *__A) {
  return (__m512i)__builtin_ia32_selectw_512((__mmask32)__U,
                                             (__v32hi)_mm512_loadrs_epi16(__A),
                                             (__v32hi)_mm512_setzero_si512());
}

#undef __DEFAULT_FN_ATTRS512

#endif /* __x86_64__ */
#endif /* __MOVRS_AVX10_2_512INTRIN_H */

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#error "Never use <vaesintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __VAESINTRIN_H
#define __VAESINTRIN_H

/* Default attributes for YMM forms. */
#define __DEFAULT_FN_ATTRS __attribute__((__always_inline__, __nodebug__, __target__("vaes"), __min_vector_width__(256)))

/* Default attributes for ZMM forms. */
#define __DEFAULT_FN_ATTRS_F                                                   \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512f,evex512,vaes"),                           \
                 __min_vector_width__(512)))

static __inline__ __m256i __DEFAULT_FN_ATTRS
 _mm256_aesenc_epi128(__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_aesenc256((__v4di) __A,
              (__v4di) __B);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
 _mm256_aesdec_epi128(__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_aesdec256((__v4di) __A,
              (__v4di) __B);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
 _mm256_aesenclast_epi128(__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_aesenclast256((__v4di) __A,
              (__v4di) __B);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS
 _mm256_aesdeclast_epi128(__m256i __A, __m256i __B)
{
  return (__m256i) __builtin_ia32_aesdeclast256((__v4di) __A,
              (__v4di) __B);
}

#ifdef __AVX512FINTRIN_H
static __inline__ __m512i __DEFAULT_FN_ATTRS_F
 _mm512_aesenc_epi128(__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_aesenc512((__v8di) __A,
              (__v8di) __B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_F
 _mm512_aesdec_epi128(__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_aesdec512((__v8di) __A,
              (__v8di) __B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_F
 _mm512_aesenclast_epi128(__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_aesenclast512((__v8di) __A,
              (__v8di) __B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_F
 _mm512_aesdeclast_epi128(__m512i __A, __m512i __B)
{
  return (__m512i) __builtin_ia32_aesdeclast512((__v8di) __A,
              (__v8di) __B);
}
#endif // __AVX512FINTRIN_H

#undef __DEFAULT_FN_ATTRS
#undef __DEFAULT_FN_ATTRS_F

#endif // __VAESINTRIN_H

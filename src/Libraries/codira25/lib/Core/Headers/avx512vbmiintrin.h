/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#error "Never use <avx512vbmiintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __VBMIINTRIN_H
#define __VBMIINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512vbmi,evex512"), __min_vector_width__(512)))

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_permutex2var_epi8(__m512i __A, __m512i __I, __m512i __B)
{
  return (__m512i)__builtin_ia32_vpermi2varqi512((__v64qi)__A, (__v64qi)__I,
                                                 (__v64qi) __B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_permutex2var_epi8(__m512i __A, __mmask64 __U, __m512i __I,
                              __m512i __B)
{
  return (__m512i)__builtin_ia32_selectb_512(__U,
                               (__v64qi)_mm512_permutex2var_epi8(__A, __I, __B),
                               (__v64qi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask2_permutex2var_epi8(__m512i __A, __m512i __I, __mmask64 __U,
                               __m512i __B)
{
  return (__m512i)__builtin_ia32_selectb_512(__U,
                               (__v64qi)_mm512_permutex2var_epi8(__A, __I, __B),
                               (__v64qi)__I);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_permutex2var_epi8(__mmask64 __U, __m512i __A, __m512i __I,
                               __m512i __B)
{
  return (__m512i)__builtin_ia32_selectb_512(__U,
                               (__v64qi)_mm512_permutex2var_epi8(__A, __I, __B),
                               (__v64qi)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_permutexvar_epi8 (__m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_permvarqi512((__v64qi) __B, (__v64qi) __A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_permutexvar_epi8 (__mmask64 __M, __m512i __A,
        __m512i __B)
{
  return (__m512i)__builtin_ia32_selectb_512((__mmask64)__M,
                                     (__v64qi)_mm512_permutexvar_epi8(__A, __B),
                                     (__v64qi)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_permutexvar_epi8 (__m512i __W, __mmask64 __M, __m512i __A,
             __m512i __B)
{
  return (__m512i)__builtin_ia32_selectb_512((__mmask64)__M,
                                     (__v64qi)_mm512_permutexvar_epi8(__A, __B),
                                     (__v64qi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_multishift_epi64_epi8(__m512i __X, __m512i __Y)
{
  return (__m512i)__builtin_ia32_vpmultishiftqb512((__v64qi)__X, (__v64qi) __Y);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_multishift_epi64_epi8(__m512i __W, __mmask64 __M, __m512i __X,
                                  __m512i __Y)
{
  return (__m512i)__builtin_ia32_selectb_512((__mmask64)__M,
                                (__v64qi)_mm512_multishift_epi64_epi8(__X, __Y),
                                (__v64qi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_multishift_epi64_epi8(__mmask64 __M, __m512i __X, __m512i __Y)
{
  return (__m512i)__builtin_ia32_selectb_512((__mmask64)__M,
                                (__v64qi)_mm512_multishift_epi64_epi8(__X, __Y),
                                (__v64qi)_mm512_setzero_si512());
}


#undef __DEFAULT_FN_ATTRS

#endif

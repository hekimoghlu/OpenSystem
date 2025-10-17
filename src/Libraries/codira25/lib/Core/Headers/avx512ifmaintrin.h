/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
#error "Never use <avx512ifmaintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __IFMAINTRIN_H
#define __IFMAINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512ifma,evex512"), __min_vector_width__(512)))

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_madd52hi_epu64 (__m512i __X, __m512i __Y, __m512i __Z)
{
  return (__m512i)__builtin_ia32_vpmadd52huq512((__v8di) __X, (__v8di) __Y,
                                                (__v8di) __Z);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_madd52hi_epu64 (__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y)
{
  return (__m512i)__builtin_ia32_selectq_512(__M,
                                   (__v8di)_mm512_madd52hi_epu64(__W, __X, __Y),
                                   (__v8di)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_madd52hi_epu64 (__mmask8 __M, __m512i __X, __m512i __Y, __m512i __Z)
{
  return (__m512i)__builtin_ia32_selectq_512(__M,
                                   (__v8di)_mm512_madd52hi_epu64(__X, __Y, __Z),
                                   (__v8di)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_madd52lo_epu64 (__m512i __X, __m512i __Y, __m512i __Z)
{
  return (__m512i)__builtin_ia32_vpmadd52luq512((__v8di) __X, (__v8di) __Y,
                                                (__v8di) __Z);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_madd52lo_epu64 (__m512i __W, __mmask8 __M, __m512i __X, __m512i __Y)
{
  return (__m512i)__builtin_ia32_selectq_512(__M,
                                   (__v8di)_mm512_madd52lo_epu64(__W, __X, __Y),
                                   (__v8di)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_madd52lo_epu64 (__mmask8 __M, __m512i __X, __m512i __Y, __m512i __Z)
{
  return (__m512i)__builtin_ia32_selectq_512(__M,
                                   (__v8di)_mm512_madd52lo_epu64(__X, __Y, __Z),
                                   (__v8di)_mm512_setzero_si512());
}

#undef __DEFAULT_FN_ATTRS

#endif

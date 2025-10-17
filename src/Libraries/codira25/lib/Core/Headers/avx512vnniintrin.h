/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 31, 2025.
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
#error "Never use <avx512vnniintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VNNIINTRIN_H
#define __AVX512VNNIINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512vnni,evex512"), __min_vector_width__(512)))

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_dpbusd_epi32(__m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_vpdpbusd512((__v16si)__S, (__v16si)__A,
                                             (__v16si)__B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_dpbusd_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                    (__v16si)_mm512_dpbusd_epi32(__S, __A, __B),
                                    (__v16si)__S);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_dpbusd_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                    (__v16si)_mm512_dpbusd_epi32(__S, __A, __B),
                                    (__v16si)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_dpbusds_epi32(__m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_vpdpbusds512((__v16si)__S, (__v16si)__A,
                                              (__v16si)__B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_dpbusds_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                   (__v16si)_mm512_dpbusds_epi32(__S, __A, __B),
                                   (__v16si)__S);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_dpbusds_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                   (__v16si)_mm512_dpbusds_epi32(__S, __A, __B),
                                   (__v16si)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_dpwssd_epi32(__m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_vpdpwssd512((__v16si)__S, (__v16si)__A,
                                             (__v16si)__B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_dpwssd_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                    (__v16si)_mm512_dpwssd_epi32(__S, __A, __B),
                                    (__v16si)__S);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_dpwssd_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                    (__v16si)_mm512_dpwssd_epi32(__S, __A, __B),
                                    (__v16si)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_dpwssds_epi32(__m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_vpdpwssds512((__v16si)__S, (__v16si)__A,
                                              (__v16si)__B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_dpwssds_epi32(__m512i __S, __mmask16 __U, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                   (__v16si)_mm512_dpwssds_epi32(__S, __A, __B),
                                   (__v16si)__S);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_dpwssds_epi32(__mmask16 __U, __m512i __S, __m512i __A, __m512i __B)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                   (__v16si)_mm512_dpwssds_epi32(__S, __A, __B),
                                   (__v16si)_mm512_setzero_si512());
}

#undef __DEFAULT_FN_ATTRS

#endif

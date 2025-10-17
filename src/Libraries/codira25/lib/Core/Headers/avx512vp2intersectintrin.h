/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 2, 2024.
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
#error "Never use <avx512vp2intersect.h> directly; include <immintrin.h> instead."
#endif

#ifndef _AVX512VP2INTERSECT_H
#define _AVX512VP2INTERSECT_H

#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512vp2intersect,evex512"),                     \
                 __min_vector_width__(512)))

/// Store, in an even/odd pair of mask registers, the indicators of the
/// locations of value matches between dwords in operands __a and __b.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VP2INTERSECTD </c> instruction.
///
/// \param __a
///    A 512-bit vector of [16 x i32].
/// \param __b
///    A 512-bit vector of [16 x i32]
/// \param __m0
///    A pointer point to 16-bit mask
/// \param __m1
///    A pointer point to 16-bit mask
static __inline__ void __DEFAULT_FN_ATTRS
_mm512_2intersect_epi32(__m512i __a, __m512i __b, __mmask16 *__m0, __mmask16 *__m1) {
  __builtin_ia32_vp2intersect_d_512((__v16si)__a, (__v16si)__b, __m0, __m1);
}

/// Store, in an even/odd pair of mask registers, the indicators of the
/// locations of value matches between quadwords in operands __a and __b.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic corresponds to the <c> VP2INTERSECTQ </c> instruction.
///
/// \param __a
///    A 512-bit vector of [8 x i64].
/// \param __b
///    A 512-bit vector of [8 x i64]
/// \param __m0
///    A pointer point to 8-bit mask
/// \param __m1
///    A pointer point to 8-bit mask
static __inline__ void __DEFAULT_FN_ATTRS
_mm512_2intersect_epi64(__m512i __a, __m512i __b, __mmask8 *__m0, __mmask8 *__m1) {
  __builtin_ia32_vp2intersect_q_512((__v8di)__a, (__v8di)__b, __m0, __m1);
}

#undef __DEFAULT_FN_ATTRS

#endif

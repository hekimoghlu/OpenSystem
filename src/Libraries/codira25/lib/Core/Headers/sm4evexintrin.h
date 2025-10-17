/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 30, 2022.
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
#error "Never use <sm4evexintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifndef __SM4EVEXINTRIN_H
#define __SM4EVEXINTRIN_H

#define __DEFAULT_FN_ATTRS512                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("sm4,avx10.2-512"), __min_vector_width__(512)))

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_sm4key4_epi32(__m512i __A, __m512i __B) {
  return (__m512i)__builtin_ia32_vsm4key4512((__v16su)__A, (__v16su)__B);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_sm4rnds4_epi32(__m512i __A, __m512i __B) {
  return (__m512i)__builtin_ia32_vsm4rnds4512((__v16su)__A, (__v16su)__B);
}

#undef __DEFAULT_FN_ATTRS512

#endif // __SM4EVEXINTRIN_H

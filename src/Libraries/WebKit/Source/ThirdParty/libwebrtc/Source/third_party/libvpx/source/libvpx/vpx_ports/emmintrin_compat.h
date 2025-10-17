/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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
#ifndef VPX_VPX_PORTS_EMMINTRIN_COMPAT_H_
#define VPX_VPX_PORTS_EMMINTRIN_COMPAT_H_

#if defined(__GNUC__) && __GNUC__ < 4
/* From emmintrin.h (gcc 4.5.3) */
/* Casts between various SP, DP, INT vector types.  Note that these do no
   conversion of values, they just change the type.  */
extern __inline __m128
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castpd_ps(__m128d __A) {
  return (__m128)__A;
}

extern __inline __m128i
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castpd_si128(__m128d __A) {
  return (__m128i)__A;
}

extern __inline __m128d
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castps_pd(__m128 __A) {
  return (__m128d)__A;
}

extern __inline __m128i
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castps_si128(__m128 __A) {
  return (__m128i)__A;
}

extern __inline __m128
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castsi128_ps(__m128i __A) {
  return (__m128)__A;
}

extern __inline __m128d
    __attribute__((__gnu_inline__, __always_inline__, __artificial__))
    _mm_castsi128_pd(__m128i __A) {
  return (__m128d)__A;
}
#endif

#endif  // VPX_VPX_PORTS_EMMINTRIN_COMPAT_H_

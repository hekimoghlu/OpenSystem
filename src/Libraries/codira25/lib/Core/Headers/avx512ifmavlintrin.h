/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
#error "Never use <avx512ifmavlintrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __IFMAVLINTRIN_H
#define __IFMAVLINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512ifma,avx512vl,no-evex512"),                 \
                 __min_vector_width__(128)))
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__,                               \
                 __target__("avx512ifma,avx512vl,no-evex512"),                 \
                 __min_vector_width__(256)))

#define _mm_madd52hi_epu64(X, Y, Z)                                            \
  ((__m128i)__builtin_ia32_vpmadd52huq128((__v2di)(X), (__v2di)(Y),            \
                                          (__v2di)(Z)))

#define _mm256_madd52hi_epu64(X, Y, Z)                                         \
  ((__m256i)__builtin_ia32_vpmadd52huq256((__v4di)(X), (__v4di)(Y),            \
                                          (__v4di)(Z)))

#define _mm_madd52lo_epu64(X, Y, Z)                                            \
  ((__m128i)__builtin_ia32_vpmadd52luq128((__v2di)(X), (__v2di)(Y),            \
                                          (__v2di)(Z)))

#define _mm256_madd52lo_epu64(X, Y, Z)                                         \
  ((__m256i)__builtin_ia32_vpmadd52luq256((__v4di)(X), (__v4di)(Y),            \
                                          (__v4di)(Z)))

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_madd52hi_epu64 (__m128i __W, __mmask8 __M, __m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_selectq_128(__M,
                                      (__v2di)_mm_madd52hi_epu64(__W, __X, __Y),
                                      (__v2di)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_madd52hi_epu64 (__mmask8 __M, __m128i __X, __m128i __Y, __m128i __Z)
{
  return (__m128i)__builtin_ia32_selectq_128(__M,
                                      (__v2di)_mm_madd52hi_epu64(__X, __Y, __Z),
                                      (__v2di)_mm_setzero_si128());
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_madd52hi_epu64 (__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_selectq_256(__M,
                                   (__v4di)_mm256_madd52hi_epu64(__W, __X, __Y),
                                   (__v4di)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_madd52hi_epu64 (__mmask8 __M, __m256i __X, __m256i __Y, __m256i __Z)
{
  return (__m256i)__builtin_ia32_selectq_256(__M,
                                   (__v4di)_mm256_madd52hi_epu64(__X, __Y, __Z),
                                   (__v4di)_mm256_setzero_si256());
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_madd52lo_epu64 (__m128i __W, __mmask8 __M, __m128i __X, __m128i __Y)
{
  return (__m128i)__builtin_ia32_selectq_128(__M,
                                      (__v2di)_mm_madd52lo_epu64(__W, __X, __Y),
                                      (__v2di)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_madd52lo_epu64 (__mmask8 __M, __m128i __X, __m128i __Y, __m128i __Z)
{
  return (__m128i)__builtin_ia32_selectq_128(__M,
                                      (__v2di)_mm_madd52lo_epu64(__X, __Y, __Z),
                                      (__v2di)_mm_setzero_si128());
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_madd52lo_epu64 (__m256i __W, __mmask8 __M, __m256i __X, __m256i __Y)
{
  return (__m256i)__builtin_ia32_selectq_256(__M,
                                   (__v4di)_mm256_madd52lo_epu64(__W, __X, __Y),
                                   (__v4di)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_madd52lo_epu64 (__mmask8 __M, __m256i __X, __m256i __Y, __m256i __Z)
{
  return (__m256i)__builtin_ia32_selectq_256(__M,
                                   (__v4di)_mm256_madd52lo_epu64(__X, __Y, __Z),
                                   (__v4di)_mm256_setzero_si256());
}


#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256

#endif

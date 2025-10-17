/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 11, 2025.
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
    "Never use <avx10_2satcvtintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifndef __AVX10_2SATCVTINTRIN_H
#define __AVX10_2SATCVTINTRIN_H

#define _mm_ipcvts_bf16_epi8(A)                                                \
  ((__m128i)__builtin_ia32_vcvtbf162ibs128((__v8bf)(__m128bh)(A)))

#define _mm_mask_ipcvts_bf16_epi8(W, U, A)                                     \
  ((__m128i)__builtin_ia32_selectw_128(                                        \
      (__mmask8)(U), (__v8hi)_mm_ipcvts_bf16_epi8(A), (__v8hi)(__m128i)(W)))

#define _mm_maskz_ipcvts_bf16_epi8(U, A)                                       \
  ((__m128i)__builtin_ia32_selectw_128((__mmask8)(U),                          \
                                       (__v8hi)_mm_ipcvts_bf16_epi8(A),        \
                                       (__v8hi)_mm_setzero_si128()))

#define _mm256_ipcvts_bf16_epi8(A)                                             \
  ((__m256i)__builtin_ia32_vcvtbf162ibs256((__v16bf)(__m256bh)(A)))

#define _mm256_mask_ipcvts_bf16_epi8(W, U, A)                                  \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvts_bf16_epi8(A),    \
                                       (__v16hi)(__m256i)(W)))

#define _mm256_maskz_ipcvts_bf16_epi8(U, A)                                    \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvts_bf16_epi8(A),    \
                                       (__v16hi)_mm256_setzero_si256()))

#define _mm_ipcvts_bf16_epu8(A)                                                \
  ((__m128i)__builtin_ia32_vcvtbf162iubs128((__v8bf)(__m128bh)(A)))

#define _mm_mask_ipcvts_bf16_epu8(W, U, A)                                     \
  ((__m128i)__builtin_ia32_selectw_128(                                        \
      (__mmask8)(U), (__v8hi)_mm_ipcvts_bf16_epu8(A), (__v8hi)(__m128i)(W)))

#define _mm_maskz_ipcvts_bf16_epu8(U, A)                                       \
  ((__m128i)__builtin_ia32_selectw_128((__mmask8)(U),                          \
                                       (__v8hi)_mm_ipcvts_bf16_epu8(A),        \
                                       (__v8hi)_mm_setzero_si128()))

#define _mm256_ipcvts_bf16_epu8(A)                                             \
  ((__m256i)__builtin_ia32_vcvtbf162iubs256((__v16bf)(__m256bh)(A)))

#define _mm256_mask_ipcvts_bf16_epu8(W, U, A)                                  \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvts_bf16_epu8(A),    \
                                       (__v16hi)(__m256i)(W)))

#define _mm256_maskz_ipcvts_bf16_epu8(U, A)                                    \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvts_bf16_epu8(A),    \
                                       (__v16hi)_mm256_setzero_si256()))

#define _mm_ipcvts_ph_epi8(A)                                                  \
  ((__m128i)__builtin_ia32_vcvtph2ibs128_mask(                                 \
      (__v8hf)(__m128h)(A), (__v8hu)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvts_ph_epi8(W, U, A)                                       \
  ((__m128i)__builtin_ia32_vcvtph2ibs128_mask((__v8hf)(__m128h)(A),            \
                                              (__v8hu)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvts_ph_epi8(U, A)                                         \
  ((__m128i)__builtin_ia32_vcvtph2ibs128_mask(                                 \
      (__v8hf)(__m128h)(A), (__v8hu)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvts_ph_epi8(A)                                               \
  ((__m256i)__builtin_ia32_vcvtph2ibs256_mask(                                 \
      (__v16hf)(__m256h)(A), (__v16hu)_mm256_setzero_si256(), (__mmask16)-1))

#define _mm256_mask_ipcvts_ph_epi8(W, U, A)                                    \
  ((__m256i)__builtin_ia32_vcvtph2ibs256_mask((__v16hf)(__m256h)(A),           \
                                              (__v16hu)(W), (__mmask16)(U)))

#define _mm256_maskz_ipcvts_ph_epi8(U, A)                                      \
  ((__m256i)__builtin_ia32_vcvtph2ibs256_mask(                                 \
      (__v16hf)(__m256h)(A), (__v16hu)(_mm256_setzero_si256()),                \
      (__mmask16)(U)))

#define _mm_ipcvts_ph_epu8(A)                                                  \
  ((__m128i)__builtin_ia32_vcvtph2iubs128_mask(                                \
      (__v8hf)(__m128h)(A), (__v8hu)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvts_ph_epu8(W, U, A)                                       \
  ((__m128i)__builtin_ia32_vcvtph2iubs128_mask((__v8hf)(__m128h)(A),           \
                                               (__v8hu)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvts_ph_epu8(U, A)                                         \
  ((__m128i)__builtin_ia32_vcvtph2iubs128_mask(                                \
      (__v8hf)(__m128h)(A), (__v8hu)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvts_ph_epu8(A)                                               \
  ((__m256i)__builtin_ia32_vcvtph2iubs256_mask(                                \
      (__v16hf)(__m256h)(A), (__v16hu)_mm256_setzero_si256(), (__mmask16)-1))

#define _mm256_mask_ipcvts_ph_epu8(W, U, A)                                    \
  ((__m256i)__builtin_ia32_vcvtph2iubs256_mask((__v16hf)(__m256h)(A),          \
                                               (__v16hu)(W), (__mmask16)(U)))

#define _mm256_maskz_ipcvts_ph_epu8(U, A)                                      \
  ((__m256i)__builtin_ia32_vcvtph2iubs256_mask(                                \
      (__v16hf)(__m256h)(A), (__v16hu)(_mm256_setzero_si256()),                \
      (__mmask16)(U)))

#define _mm_ipcvts_ps_epi8(A)                                                  \
  ((__m128i)__builtin_ia32_vcvtps2ibs128_mask(                                 \
      (__v4sf)(__m128)(A), (__v4su)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvts_ps_epi8(W, U, A)                                       \
  ((__m128i)__builtin_ia32_vcvtps2ibs128_mask((__v4sf)(__m128)(A),             \
                                              (__v4su)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvts_ps_epi8(U, A)                                         \
  ((__m128i)__builtin_ia32_vcvtps2ibs128_mask(                                 \
      (__v4sf)(__m128)(A), (__v4su)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvts_ps_epi8(A)                                               \
  ((__m256i)__builtin_ia32_vcvtps2ibs256_mask(                                 \
      (__v8sf)(__m256)(A), (__v8su)_mm256_setzero_si256(), (__mmask8)-1))

#define _mm256_mask_ipcvts_ps_epi8(W, U, A)                                    \
  ((__m256i)__builtin_ia32_vcvtps2ibs256_mask((__v8sf)(__m256)(A),             \
                                              (__v8su)(W), (__mmask8)(U)))

#define _mm256_maskz_ipcvts_ps_epi8(U, A)                                      \
  ((__m256i)__builtin_ia32_vcvtps2ibs256_mask(                                 \
      (__v8sf)(__m256)(A), (__v8su)(_mm256_setzero_si256()), (__mmask8)(U)))

#define _mm_ipcvts_ps_epu8(A)                                                  \
  ((__m128i)__builtin_ia32_vcvtps2iubs128_mask(                                \
      (__v4sf)(__m128)(A), (__v4su)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvts_ps_epu8(W, U, A)                                       \
  ((__m128i)__builtin_ia32_vcvtps2iubs128_mask((__v4sf)(__m128)(A),            \
                                               (__v4su)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvts_ps_epu8(U, A)                                         \
  ((__m128i)__builtin_ia32_vcvtps2iubs128_mask(                                \
      (__v4sf)(__m128)(A), (__v4su)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvts_ps_epu8(A)                                               \
  ((__m256i)__builtin_ia32_vcvtps2iubs256_mask(                                \
      (__v8sf)(__m256)(A), (__v8su)_mm256_setzero_si256(), (__mmask8)-1))

#define _mm256_mask_ipcvts_ps_epu8(W, U, A)                                    \
  ((__m256i)__builtin_ia32_vcvtps2iubs256_mask((__v8sf)(__m256)(A),            \
                                               (__v8su)(W), (__mmask8)(U)))

#define _mm256_maskz_ipcvts_ps_epu8(U, A)                                      \
  ((__m256i)__builtin_ia32_vcvtps2iubs256_mask(                                \
      (__v8sf)(__m256)(A), (__v8su)(_mm256_setzero_si256()), (__mmask8)(U)))

#define _mm_ipcvtts_bf16_epi8(A)                                               \
  ((__m128i)__builtin_ia32_vcvttbf162ibs128((__v8bf)(__m128bh)(A)))

#define _mm_mask_ipcvtts_bf16_epi8(W, U, A)                                    \
  ((__m128i)__builtin_ia32_selectw_128(                                        \
      (__mmask8)(U), (__v8hi)_mm_ipcvtts_bf16_epi8(A), (__v8hi)(__m128i)(W)))

#define _mm_maskz_ipcvtts_bf16_epi8(U, A)                                      \
  ((__m128i)__builtin_ia32_selectw_128((__mmask8)(U),                          \
                                       (__v8hi)_mm_ipcvtts_bf16_epi8(A),       \
                                       (__v8hi)_mm_setzero_si128()))

#define _mm256_ipcvtts_bf16_epi8(A)                                            \
  ((__m256i)__builtin_ia32_vcvttbf162ibs256((__v16bf)(__m256bh)(A)))

#define _mm256_mask_ipcvtts_bf16_epi8(W, U, A)                                 \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvtts_bf16_epi8(A),   \
                                       (__v16hi)(__m256i)(W)))

#define _mm256_maskz_ipcvtts_bf16_epi8(U, A)                                   \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvtts_bf16_epi8(A),   \
                                       (__v16hi)_mm256_setzero_si256()))

#define _mm_ipcvtts_bf16_epu8(A)                                               \
  ((__m128i)__builtin_ia32_vcvttbf162iubs128((__v8bf)(__m128bh)(A)))

#define _mm_mask_ipcvtts_bf16_epu8(W, U, A)                                    \
  ((__m128i)__builtin_ia32_selectw_128(                                        \
      (__mmask8)(U), (__v8hi)_mm_ipcvtts_bf16_epu8(A), (__v8hi)(__m128i)(W)))

#define _mm_maskz_ipcvtts_bf16_epu8(U, A)                                      \
  ((__m128i)__builtin_ia32_selectw_128((__mmask8)(U),                          \
                                       (__v8hi)_mm_ipcvtts_bf16_epu8(A),       \
                                       (__v8hi)_mm_setzero_si128()))

#define _mm256_ipcvtts_bf16_epu8(A)                                            \
  ((__m256i)__builtin_ia32_vcvttbf162iubs256((__v16bf)(__m256bh)(A)))

#define _mm256_mask_ipcvtts_bf16_epu8(W, U, A)                                 \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvtts_bf16_epu8(A),   \
                                       (__v16hi)(__m256i)(W)))

#define _mm256_maskz_ipcvtts_bf16_epu8(U, A)                                   \
  ((__m256i)__builtin_ia32_selectw_256((__mmask16)(U),                         \
                                       (__v16hi)_mm256_ipcvtts_bf16_epu8(A),   \
                                       (__v16hi)_mm256_setzero_si256()))

#define _mm_ipcvtts_ph_epi8(A)                                                 \
  ((__m128i)__builtin_ia32_vcvttph2ibs128_mask(                                \
      (__v8hf)(__m128h)(A), (__v8hu)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvtts_ph_epi8(W, U, A)                                      \
  ((__m128i)__builtin_ia32_vcvttph2ibs128_mask((__v8hf)(__m128h)(A),           \
                                               (__v8hu)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvtts_ph_epi8(U, A)                                        \
  ((__m128i)__builtin_ia32_vcvttph2ibs128_mask(                                \
      (__v8hf)(__m128h)(A), (__v8hu)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvtts_ph_epi8(A)                                              \
  ((__m256i)__builtin_ia32_vcvttph2ibs256_mask(                                \
      (__v16hf)(__m256h)(A), (__v16hu)_mm256_setzero_si256(), (__mmask16)-1))

#define _mm256_mask_ipcvtts_ph_epi8(W, U, A)                                   \
  ((__m256i)__builtin_ia32_vcvttph2ibs256_mask((__v16hf)(__m256h)(A),          \
                                               (__v16hu)(W), (__mmask16)(U)))

#define _mm256_maskz_ipcvtts_ph_epi8(U, A)                                     \
  ((__m256i)__builtin_ia32_vcvttph2ibs256_mask(                                \
      (__v16hf)(__m256h)(A), (__v16hu)(_mm256_setzero_si256()),                \
      (__mmask16)(U)))

#define _mm_ipcvtts_ph_epu8(A)                                                 \
  ((__m128i)__builtin_ia32_vcvttph2iubs128_mask(                               \
      (__v8hf)(__m128h)(A), (__v8hu)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvtts_ph_epu8(W, U, A)                                      \
  ((__m128i)__builtin_ia32_vcvttph2iubs128_mask((__v8hf)(__m128h)(A),          \
                                                (__v8hu)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvtts_ph_epu8(U, A)                                        \
  ((__m128i)__builtin_ia32_vcvttph2iubs128_mask(                               \
      (__v8hf)(__m128h)(A), (__v8hu)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvtts_ph_epu8(A)                                              \
  ((__m256i)__builtin_ia32_vcvttph2iubs256_mask(                               \
      (__v16hf)(__m256h)(A), (__v16hu)_mm256_setzero_si256(), (__mmask16)-1))

#define _mm256_mask_ipcvtts_ph_epu8(W, U, A)                                   \
  ((__m256i)__builtin_ia32_vcvttph2iubs256_mask((__v16hf)(__m256h)(A),         \
                                                (__v16hu)(W), (__mmask16)(U)))

#define _mm256_maskz_ipcvtts_ph_epu8(U, A)                                     \
  ((__m256i)__builtin_ia32_vcvttph2iubs256_mask(                               \
      (__v16hf)(__m256h)(A), (__v16hu)(_mm256_setzero_si256()),                \
      (__mmask16)(U)))

#define _mm_ipcvtts_ps_epi8(A)                                                 \
  ((__m128i)__builtin_ia32_vcvttps2ibs128_mask(                                \
      (__v4sf)(__m128)(A), (__v4su)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvtts_ps_epi8(W, U, A)                                      \
  ((__m128i)__builtin_ia32_vcvttps2ibs128_mask((__v4sf)(__m128)(A),            \
                                               (__v4su)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvtts_ps_epi8(U, A)                                        \
  ((__m128i)__builtin_ia32_vcvttps2ibs128_mask(                                \
      (__v4sf)(__m128)(A), (__v4su)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvtts_ps_epi8(A)                                              \
  ((__m256i)__builtin_ia32_vcvttps2ibs256_mask(                                \
      (__v8sf)(__m256)(A), (__v8su)_mm256_setzero_si256(), (__mmask8)-1))

#define _mm256_mask_ipcvtts_ps_epi8(W, U, A)                                   \
  ((__m256i)__builtin_ia32_vcvttps2ibs256_mask((__v8sf)(__m256)(A),            \
                                               (__v8su)(W), (__mmask8)(U)))

#define _mm256_maskz_ipcvtts_ps_epi8(U, A)                                     \
  ((__m256i)__builtin_ia32_vcvttps2ibs256_mask(                                \
      (__v8sf)(__m256)(A), (__v8su)(_mm256_setzero_si256()), (__mmask8)(U)))

#define _mm_ipcvtts_ps_epu8(A)                                                 \
  ((__m128i)__builtin_ia32_vcvttps2iubs128_mask(                               \
      (__v4sf)(__m128)(A), (__v4su)_mm_setzero_si128(), (__mmask8)-1))

#define _mm_mask_ipcvtts_ps_epu8(W, U, A)                                      \
  ((__m128i)__builtin_ia32_vcvttps2iubs128_mask((__v4sf)(__m128)(A),           \
                                                (__v4su)(W), (__mmask8)(U)))

#define _mm_maskz_ipcvtts_ps_epu8(U, A)                                        \
  ((__m128i)__builtin_ia32_vcvttps2iubs128_mask(                               \
      (__v4sf)(__m128)(A), (__v4su)(_mm_setzero_si128()), (__mmask8)(U)))

#define _mm256_ipcvtts_ps_epu8(A)                                              \
  ((__m256i)__builtin_ia32_vcvttps2iubs256_mask(                               \
      (__v8sf)(__m256)(A), (__v8su)_mm256_setzero_si256(), (__mmask8)-1))

#define _mm256_mask_ipcvtts_ps_epu8(W, U, A)                                   \
  ((__m256i)__builtin_ia32_vcvttps2iubs256_mask((__v8sf)(__m256)(A),           \
                                                (__v8su)(W), (__mmask8)(U)))

#define _mm256_maskz_ipcvtts_ps_epu8(U, A)                                     \
  ((__m256i)__builtin_ia32_vcvttps2iubs256_mask(                               \
      (__v8sf)(__m256)(A), (__v8su)(_mm256_setzero_si256()), (__mmask8)(U)))
#endif // __AVX10_2SATCVTINTRIN_H

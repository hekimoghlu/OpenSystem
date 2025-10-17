/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 21, 2023.
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
    "Never use <avx10_2_512satcvtintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifndef __AVX10_2_512SATCVTINTRIN_H
#define __AVX10_2_512SATCVTINTRIN_H

#define _mm512_ipcvts_bf16_epi8(A)                                             \
  ((__m512i)__builtin_ia32_vcvtbf162ibs512((__v32bf)(__m512bh)(A)))

#define _mm512_mask_ipcvts_bf16_epi8(W, U, A)                                  \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvts_bf16_epi8(A),    \
                                       (__v32hi)(__m512i)(W)))

#define _mm512_maskz_ipcvts_bf16_epi8(U, A)                                    \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvts_bf16_epi8(A),    \
                                       (__v32hi)_mm512_setzero_si512()))

#define _mm512_ipcvts_bf16_epu8(A)                                             \
  ((__m512i)__builtin_ia32_vcvtbf162iubs512((__v32bf)(__m512bh)(A)))

#define _mm512_mask_ipcvts_bf16_epu8(W, U, A)                                  \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvts_bf16_epu8(A),    \
                                       (__v32hi)(__m512i)(W)))

#define _mm512_maskz_ipcvts_bf16_epu8(U, A)                                    \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvts_bf16_epu8(A),    \
                                       (__v32hi)_mm512_setzero_si512()))

#define _mm512_ipcvtts_bf16_epi8(A)                                            \
  ((__m512i)__builtin_ia32_vcvttbf162ibs512((__v32bf)(__m512bh)(A)))

#define _mm512_mask_ipcvtts_bf16_epi8(W, U, A)                                 \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvtts_bf16_epi8(A),   \
                                       (__v32hi)(__m512i)(W)))

#define _mm512_maskz_ipcvtts_bf16_epi8(U, A)                                   \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvtts_bf16_epi8(A),   \
                                       (__v32hi)_mm512_setzero_si512()))

#define _mm512_ipcvtts_bf16_epu8(A)                                            \
  ((__m512i)__builtin_ia32_vcvttbf162iubs512((__v32bf)(__m512bh)(A)))

#define _mm512_mask_ipcvtts_bf16_epu8(W, U, A)                                 \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvtts_bf16_epu8(A),   \
                                       (__v32hi)(__m512i)(W)))

#define _mm512_maskz_ipcvtts_bf16_epu8(U, A)                                   \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U),                         \
                                       (__v32hi)_mm512_ipcvtts_bf16_epu8(A),   \
                                       (__v32hi)_mm512_setzero_si512()))

#define _mm512_ipcvts_ph_epi8(A)                                               \
  ((__m512i)__builtin_ia32_vcvtph2ibs512_mask(                                 \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvts_ph_epi8(W, U, A)                                    \
  ((__m512i)__builtin_ia32_vcvtph2ibs512_mask((__v32hf)(__m512h)(A),           \
                                              (__v32hu)(W), (__mmask32)(U),    \
                                              _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvts_ph_epi8(U, A)                                      \
  ((__m512i)__builtin_ia32_vcvtph2ibs512_mask(                                 \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvts_roundph_epi8(A, R)                                       \
  ((__m512i)__builtin_ia32_vcvtph2ibs512_mask((__v32hf)(__m512h)(A),           \
                                              (__v32hu)_mm512_setzero_si512(), \
                                              (__mmask32) - 1, (const int)R))

#define _mm512_mask_ipcvts_roundph_epi8(W, U, A, R)                            \
  ((__m512i)__builtin_ia32_vcvtph2ibs512_mask(                                 \
      (__v32hf)(__m512h)(A), (__v32hu)(W), (__mmask32)(U), (const int)R))

#define _mm512_maskz_ipcvts_roundph_epi8(U, A, R)                              \
  ((__m512i)__builtin_ia32_vcvtph2ibs512_mask((__v32hf)(__m512h)(A),           \
                                              (__v32hu)_mm512_setzero_si512(), \
                                              (__mmask32)(U), (const int)R))

#define _mm512_ipcvts_ph_epu8(A)                                               \
  ((__m512i)__builtin_ia32_vcvtph2iubs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvts_ph_epu8(W, U, A)                                    \
  ((__m512i)__builtin_ia32_vcvtph2iubs512_mask((__v32hf)(__m512h)(A),          \
                                               (__v32hu)(W), (__mmask32)(U),   \
                                               _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvts_ph_epu8(U, A)                                      \
  ((__m512i)__builtin_ia32_vcvtph2iubs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvts_roundph_epu8(A, R)                                       \
  ((__m512i)__builtin_ia32_vcvtph2iubs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      (const int)R))

#define _mm512_mask_ipcvts_roundph_epu8(W, U, A, R)                            \
  ((__m512i)__builtin_ia32_vcvtph2iubs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)(W), (__mmask32)(U), (const int)R))

#define _mm512_maskz_ipcvts_roundph_epu8(U, A, R)                              \
  ((__m512i)__builtin_ia32_vcvtph2iubs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      (const int)R))

#define _mm512_ipcvts_ps_epi8(A)                                               \
  ((__m512i)__builtin_ia32_vcvtps2ibs512_mask(                                 \
      (__v16sf)(__m512)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1,  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvts_ps_epi8(W, U, A)                                    \
  ((__m512i)__builtin_ia32_vcvtps2ibs512_mask((__v16sf)(__m512)(A),            \
                                              (__v16su)(W), (__mmask16)(U),    \
                                              _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvts_ps_epi8(U, A)                                      \
  ((__m512i)__builtin_ia32_vcvtps2ibs512_mask(                                 \
      (__v16sf)(__m512)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),   \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvts_roundps_epi8(A, R)                                       \
  ((__m512i)__builtin_ia32_vcvtps2ibs512_mask((__v16sf)(__m512)(A),            \
                                              (__v16su)_mm512_setzero_si512(), \
                                              (__mmask16) - 1, (const int)R))

#define _mm512_mask_ipcvts_roundps_epi8(W, U, A, R)                            \
  ((__m512i)__builtin_ia32_vcvtps2ibs512_mask(                                 \
      (__v16sf)(__m512)(A), (__v16su)(W), (__mmask16)(U), (const int)R))

#define _mm512_maskz_ipcvts_roundps_epi8(U, A, R)                              \
  ((__m512i)__builtin_ia32_vcvtps2ibs512_mask((__v16sf)(__m512)(A),            \
                                              (__v16su)_mm512_setzero_si512(), \
                                              (__mmask16)(U), (const int)R))

#define _mm512_ipcvts_ps_epu8(A)                                               \
  ((__m512i)__builtin_ia32_vcvtps2iubs512_mask(                                \
      (__v16sf)(__m512)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1,  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvts_ps_epu8(W, U, A)                                    \
  ((__m512i)__builtin_ia32_vcvtps2iubs512_mask((__v16sf)(__m512)(A),           \
                                               (__v16su)(W), (__mmask16)(U),   \
                                               _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvts_ps_epu8(U, A)                                      \
  ((__m512i)__builtin_ia32_vcvtps2iubs512_mask(                                \
      (__v16sf)(__m512)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),   \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvts_roundps_epu8(A, R)                                       \
  ((__m512i)__builtin_ia32_vcvtps2iubs512_mask(                                \
      (__v16sf)(__m512)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1,  \
      (const int)R))

#define _mm512_mask_ipcvts_roundps_epu8(W, U, A, R)                            \
  ((__m512i)__builtin_ia32_vcvtps2iubs512_mask(                                \
      (__v16sf)(__m512)(A), (__v16su)(W), (__mmask16)(U), (const int)R))

#define _mm512_maskz_ipcvts_roundps_epu8(U, A, R)                              \
  ((__m512i)__builtin_ia32_vcvtps2iubs512_mask(                                \
      (__v16sf)(__m512)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),   \
      (const int)R))

#define _mm512_ipcvtts_ph_epi8(A)                                              \
  ((__m512i)__builtin_ia32_vcvttph2ibs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvtts_ph_epi8(W, U, A)                                   \
  ((__m512i)__builtin_ia32_vcvttph2ibs512_mask((__v32hf)(__m512h)(A),          \
                                               (__v32hu)(W), (__mmask32)(U),   \
                                               _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvtts_ph_epi8(U, A)                                     \
  ((__m512i)__builtin_ia32_vcvttph2ibs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvtts_roundph_epi8(A, S)                                      \
  ((__m512i)__builtin_ia32_vcvttph2ibs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      S))

#define _mm512_mask_ipcvtts_roundph_epi8(W, U, A, S)                           \
  ((__m512i)__builtin_ia32_vcvttph2ibs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)(W), (__mmask32)(U), S))

#define _mm512_maskz_ipcvtts_roundph_epi8(U, A, S)                             \
  ((__m512i)__builtin_ia32_vcvttph2ibs512_mask(                                \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      S))

#define _mm512_ipcvtts_ph_epu8(A)                                              \
  ((__m512i)__builtin_ia32_vcvttph2iubs512_mask(                               \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvtts_ph_epu8(W, U, A)                                   \
  ((__m512i)__builtin_ia32_vcvttph2iubs512_mask((__v32hf)(__m512h)(A),         \
                                                (__v32hu)(W), (__mmask32)(U),  \
                                                _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvtts_ph_epu8(U, A)                                     \
  ((__m512i)__builtin_ia32_vcvttph2iubs512_mask(                               \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvtts_roundph_epu8(A, S)                                      \
  ((__m512i)__builtin_ia32_vcvttph2iubs512_mask(                               \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32) - 1, \
      S))

#define _mm512_mask_ipcvtts_roundph_epu8(W, U, A, S)                           \
  ((__m512i)__builtin_ia32_vcvttph2iubs512_mask(                               \
      (__v32hf)(__m512h)(A), (__v32hu)(W), (__mmask32)(U), S))

#define _mm512_maskz_ipcvtts_roundph_epu8(U, A, S)                             \
  ((__m512i)__builtin_ia32_vcvttph2iubs512_mask(                               \
      (__v32hf)(__m512h)(A), (__v32hu)_mm512_setzero_si512(), (__mmask32)(U),  \
      S))

#define _mm512_ipcvtts_ps_epi8(A)                                              \
  ((__m512i)__builtin_ia32_vcvttps2ibs512_mask(                                \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1, \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvtts_ps_epi8(W, U, A)                                   \
  ((__m512i)__builtin_ia32_vcvttps2ibs512_mask((__v16sf)(__m512h)(A),          \
                                               (__v16su)(W), (__mmask16)(U),   \
                                               _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvtts_ps_epi8(U, A)                                     \
  ((__m512i)__builtin_ia32_vcvttps2ibs512_mask(                                \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvtts_roundps_epi8(A, S)                                      \
  ((__m512i)__builtin_ia32_vcvttps2ibs512_mask(                                \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1, \
      S))

#define _mm512_mask_ipcvtts_roundps_epi8(W, U, A, S)                           \
  ((__m512i)__builtin_ia32_vcvttps2ibs512_mask(                                \
      (__v16sf)(__m512h)(A), (__v16su)(W), (__mmask16)(U), S))

#define _mm512_maskz_ipcvtts_roundps_epi8(U, A, S)                             \
  ((__m512i)__builtin_ia32_vcvttps2ibs512_mask(                                \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),  \
      S))

#define _mm512_ipcvtts_ps_epu8(A)                                              \
  ((__m512i)__builtin_ia32_vcvttps2iubs512_mask(                               \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1, \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_mask_ipcvtts_ps_epu8(W, U, A)                                   \
  ((__m512i)__builtin_ia32_vcvttps2iubs512_mask((__v16sf)(__m512h)(A),         \
                                                (__v16su)(W), (__mmask16)(U),  \
                                                _MM_FROUND_CUR_DIRECTION))

#define _mm512_maskz_ipcvtts_ps_epu8(U, A)                                     \
  ((__m512i)__builtin_ia32_vcvttps2iubs512_mask(                               \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),  \
      _MM_FROUND_CUR_DIRECTION))

#define _mm512_ipcvtts_roundps_epu8(A, S)                                      \
  ((__m512i)__builtin_ia32_vcvttps2iubs512_mask(                               \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16) - 1, \
      S))

#define _mm512_mask_ipcvtts_roundps_epu8(W, U, A, S)                           \
  ((__m512i)__builtin_ia32_vcvttps2iubs512_mask(                               \
      (__v16sf)(__m512h)(A), (__v16su)(W), (__mmask16)(U), S))

#define _mm512_maskz_ipcvtts_roundps_epu8(U, A, S)                             \
  ((__m512i)__builtin_ia32_vcvttps2iubs512_mask(                               \
      (__v16sf)(__m512h)(A), (__v16su)_mm512_setzero_si512(), (__mmask16)(U),  \
      S))

#endif // __AVX10_2_512SATCVTINTRIN_H

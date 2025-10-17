/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 27, 2022.
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
#ifndef AOM_AOM_DSP_X86_CONVOLVE_SSSE3_H_
#define AOM_AOM_DSP_X86_CONVOLVE_SSSE3_H_

#include <tmmintrin.h>  // SSSE3

static inline void shuffle_filter_ssse3(const int16_t *const filter,
                                        __m128i *const f) {
  const __m128i f_values = _mm_load_si128((const __m128i *)filter);
  // pack and duplicate the filter values
  f[0] = _mm_shuffle_epi8(f_values, _mm_set1_epi16(0x0200u));
  f[1] = _mm_shuffle_epi8(f_values, _mm_set1_epi16(0x0604u));
  f[2] = _mm_shuffle_epi8(f_values, _mm_set1_epi16(0x0a08u));
  f[3] = _mm_shuffle_epi8(f_values, _mm_set1_epi16(0x0e0cu));
}

static inline __m128i convolve8_8_ssse3(const __m128i *const s,
                                        const __m128i *const f) {
  // multiply 2 adjacent elements with the filter and add the result
  const __m128i k_64 = _mm_set1_epi16(1 << 6);
  const __m128i x0 = _mm_maddubs_epi16(s[0], f[0]);
  const __m128i x1 = _mm_maddubs_epi16(s[1], f[1]);
  const __m128i x2 = _mm_maddubs_epi16(s[2], f[2]);
  const __m128i x3 = _mm_maddubs_epi16(s[3], f[3]);
  __m128i sum1, sum2;

  // sum the results together, saturating only on the final step
  // adding x0 with x2 and x1 with x3 is the only order that prevents
  // outranges for all filters
  sum1 = _mm_add_epi16(x0, x2);
  sum2 = _mm_add_epi16(x1, x3);
  // add the rounding offset early to avoid another saturated add
  sum1 = _mm_add_epi16(sum1, k_64);
  sum1 = _mm_adds_epi16(sum1, sum2);
  // shift by 7 bit each 16 bit
  sum1 = _mm_srai_epi16(sum1, 7);
  return sum1;
}

#endif  // AOM_AOM_DSP_X86_CONVOLVE_SSSE3_H_

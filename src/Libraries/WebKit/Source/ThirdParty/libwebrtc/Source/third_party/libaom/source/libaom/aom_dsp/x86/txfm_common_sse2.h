/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#ifndef AOM_AOM_DSP_X86_TXFM_COMMON_SSE2_H_
#define AOM_AOM_DSP_X86_TXFM_COMMON_SSE2_H_

#include <emmintrin.h>
#include "aom/aom_integer.h"
#include "aom_dsp/x86/synonyms.h"

#define pair_set_epi16(a, b) \
  _mm_set1_epi32((int32_t)(((uint16_t)(a)) | (((uint32_t)(uint16_t)(b)) << 16)))

// Reverse the 8 16 bit words in __m128i
static inline __m128i mm_reverse_epi16(const __m128i x) {
  const __m128i a = _mm_shufflelo_epi16(x, 0x1b);
  const __m128i b = _mm_shufflehi_epi16(a, 0x1b);
  return _mm_shuffle_epi32(b, 0x4e);
}

#define octa_set_epi16(a, b, c, d, e, f, g, h)                           \
  _mm_setr_epi16((int16_t)(a), (int16_t)(b), (int16_t)(c), (int16_t)(d), \
                 (int16_t)(e), (int16_t)(f), (int16_t)(g), (int16_t)(h))

#endif  // AOM_AOM_DSP_X86_TXFM_COMMON_SSE2_H_

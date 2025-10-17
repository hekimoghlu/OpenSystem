/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#ifndef AOM_THIRD_PARTY_SVT_AV1_SYNONYMS_H_
#define AOM_THIRD_PARTY_SVT_AV1_SYNONYMS_H_

#include "aom_dsp/x86/mem_sse2.h"
#include "aom_dsp/x86/synonyms.h"

static inline __m128i load_u8_8x2_sse2(const uint8_t *const src,
                                       const ptrdiff_t stride) {
  return load_8bit_8x2_to_1_reg_sse2(src, (int)(sizeof(*src) * stride));
}

static AOM_FORCE_INLINE void store_u8_4x2_sse2(const __m128i src,
                                               uint8_t *const dst,
                                               const ptrdiff_t stride) {
  xx_storel_32(dst, src);
  *(uint32_t *)(dst + stride) =
      ((uint32_t)_mm_extract_epi16(src, 3) << 16) | _mm_extract_epi16(src, 2);
}

#endif  // AOM_THIRD_PARTY_SVT_AV1_SYNONYMS_H_

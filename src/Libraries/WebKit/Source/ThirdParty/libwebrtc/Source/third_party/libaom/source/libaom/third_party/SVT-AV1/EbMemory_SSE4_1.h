/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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
#ifndef AOM_THIRD_PARTY_SVT_AV1_EBMEMORY_SSE4_1_H_
#define AOM_THIRD_PARTY_SVT_AV1_EBMEMORY_SSE4_1_H_

#include <smmintrin.h>

#include "config/aom_config.h"

#include "aom/aom_integer.h"
#include "aom_dsp/x86/mem_sse2.h"

static inline __m128i load8bit_4x2_sse4_1(const void *const src,
                                          const ptrdiff_t strideInByte) {
  const __m128i s = _mm_cvtsi32_si128(loadu_int32(src));
  return _mm_insert_epi32(s, loadu_int32((uint8_t *)src + strideInByte), 1);
}

static inline __m128i load_u8_4x2_sse4_1(const uint8_t *const src,
                                         const ptrdiff_t stride) {
  return load8bit_4x2_sse4_1(src, sizeof(*src) * stride);
}

static inline __m128i load_u16_2x2_sse4_1(const uint16_t *const src,
                                          const ptrdiff_t stride) {
  return load8bit_4x2_sse4_1(src, sizeof(*src) * stride);
}

#endif  // AOM_THIRD_PARTY_SVT_AV1_EBMEMORY_SSE4_1_H_

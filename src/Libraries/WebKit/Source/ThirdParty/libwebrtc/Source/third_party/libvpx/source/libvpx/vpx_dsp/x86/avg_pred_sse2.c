/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 5, 2022.
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
#include <assert.h>
#include <emmintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/x86/mem_sse2.h"

void vpx_comp_avg_pred_sse2(uint8_t *comp_pred, const uint8_t *pred, int width,
                            int height, const uint8_t *ref, int ref_stride) {
  /* comp_pred and pred must be 16 byte aligned. */
  assert(((intptr_t)comp_pred & 0xf) == 0);
  assert(((intptr_t)pred & 0xf) == 0);
  if (width > 8) {
    int x, y;
    for (y = 0; y < height; ++y) {
      for (x = 0; x < width; x += 16) {
        const __m128i p = _mm_load_si128((const __m128i *)(pred + x));
        const __m128i r = _mm_loadu_si128((const __m128i *)(ref + x));
        const __m128i avg = _mm_avg_epu8(p, r);
        _mm_store_si128((__m128i *)(comp_pred + x), avg);
      }
      comp_pred += width;
      pred += width;
      ref += ref_stride;
    }
  } else {  // width must be 4 or 8.
    int i;
    // Process 16 elements at a time. comp_pred and pred have width == stride
    // and therefore live in contigious memory. 4*4, 4*8, 8*4, 8*8, and 8*16 are
    // all divisible by 16 so just ref needs to be massaged when loading.
    for (i = 0; i < width * height; i += 16) {
      const __m128i p = _mm_load_si128((const __m128i *)pred);
      __m128i r;
      __m128i avg;
      if (width == ref_stride) {
        r = _mm_loadu_si128((const __m128i *)ref);
        ref += 16;
      } else if (width == 4) {
        r = _mm_set_epi32(loadu_int32(ref + 3 * ref_stride),
                          loadu_int32(ref + 2 * ref_stride),
                          loadu_int32(ref + ref_stride), loadu_int32(ref));

        ref += 4 * ref_stride;
      } else {
        const __m128i r_0 = _mm_loadl_epi64((const __m128i *)ref);
        assert(width == 8);
        r = _mm_castps_si128(_mm_loadh_pi(_mm_castsi128_ps(r_0),
                                          (const __m64 *)(ref + ref_stride)));

        ref += 2 * ref_stride;
      }
      avg = _mm_avg_epu8(p, r);
      _mm_store_si128((__m128i *)comp_pred, avg);

      pred += 16;
      comp_pred += 16;
    }
  }
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#include <immintrin.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

static VPX_FORCE_INLINE void subtract32_avx2(int16_t *diff_ptr,
                                             const uint8_t *src_ptr,
                                             const uint8_t *pred_ptr) {
  const __m256i s = _mm256_lddqu_si256((const __m256i *)src_ptr);
  const __m256i p = _mm256_lddqu_si256((const __m256i *)pred_ptr);
  const __m256i s_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(s));
  const __m256i s_1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(s, 1));
  const __m256i p_0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(p));
  const __m256i p_1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(p, 1));
  const __m256i d_0 = _mm256_sub_epi16(s_0, p_0);
  const __m256i d_1 = _mm256_sub_epi16(s_1, p_1);
  _mm256_storeu_si256((__m256i *)diff_ptr, d_0);
  _mm256_storeu_si256((__m256i *)(diff_ptr + 16), d_1);
}

static VPX_FORCE_INLINE void subtract_block_16xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  int j;
  for (j = 0; j < rows; ++j) {
    const __m128i s = _mm_lddqu_si128((const __m128i *)src_ptr);
    const __m128i p = _mm_lddqu_si128((const __m128i *)pred_ptr);
    const __m256i s_0 = _mm256_cvtepu8_epi16(s);
    const __m256i p_0 = _mm256_cvtepu8_epi16(p);
    const __m256i d_0 = _mm256_sub_epi16(s_0, p_0);
    _mm256_storeu_si256((__m256i *)diff_ptr, d_0);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

static VPX_FORCE_INLINE void subtract_block_32xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  int j;
  for (j = 0; j < rows; ++j) {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

static VPX_FORCE_INLINE void subtract_block_64xn_avx2(
    int rows, int16_t *diff_ptr, ptrdiff_t diff_stride, const uint8_t *src_ptr,
    ptrdiff_t src_stride, const uint8_t *pred_ptr, ptrdiff_t pred_stride) {
  int j;
  for (j = 0; j < rows; ++j) {
    subtract32_avx2(diff_ptr, src_ptr, pred_ptr);
    subtract32_avx2(diff_ptr + 32, src_ptr + 32, pred_ptr + 32);
    src_ptr += src_stride;
    pred_ptr += pred_stride;
    diff_ptr += diff_stride;
  }
}

void vpx_subtract_block_avx2(int rows, int cols, int16_t *diff_ptr,
                             ptrdiff_t diff_stride, const uint8_t *src_ptr,
                             ptrdiff_t src_stride, const uint8_t *pred_ptr,
                             ptrdiff_t pred_stride) {
  switch (cols) {
    case 16:
      subtract_block_16xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                               pred_ptr, pred_stride);
      break;
    case 32:
      subtract_block_32xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                               pred_ptr, pred_stride);
      break;
    case 64:
      subtract_block_64xn_avx2(rows, diff_ptr, diff_stride, src_ptr, src_stride,
                               pred_ptr, pred_stride);
      break;
    default:
      vpx_subtract_block_sse2(rows, cols, diff_ptr, diff_stride, src_ptr,
                              src_stride, pred_ptr, pred_stride);
      break;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_subtract_block_avx2(int rows, int cols, int16_t *diff_ptr,
                                    ptrdiff_t diff_stride,
                                    const uint8_t *src8_ptr,
                                    ptrdiff_t src_stride,
                                    const uint8_t *pred8_ptr,
                                    ptrdiff_t pred_stride, int bd) {
  uint16_t *src_ptr = CONVERT_TO_SHORTPTR(src8_ptr);
  uint16_t *pred_ptr = CONVERT_TO_SHORTPTR(pred8_ptr);
  (void)bd;
  if (cols == 64) {
    int j = rows;
    do {
      const __m256i s0 = _mm256_lddqu_si256((const __m256i *)src_ptr);
      const __m256i s1 = _mm256_lddqu_si256((const __m256i *)(src_ptr + 16));
      const __m256i s2 = _mm256_lddqu_si256((const __m256i *)(src_ptr + 32));
      const __m256i s3 = _mm256_lddqu_si256((const __m256i *)(src_ptr + 48));
      const __m256i p0 = _mm256_lddqu_si256((const __m256i *)pred_ptr);
      const __m256i p1 = _mm256_lddqu_si256((const __m256i *)(pred_ptr + 16));
      const __m256i p2 = _mm256_lddqu_si256((const __m256i *)(pred_ptr + 32));
      const __m256i p3 = _mm256_lddqu_si256((const __m256i *)(pred_ptr + 48));
      const __m256i d0 = _mm256_sub_epi16(s0, p0);
      const __m256i d1 = _mm256_sub_epi16(s1, p1);
      const __m256i d2 = _mm256_sub_epi16(s2, p2);
      const __m256i d3 = _mm256_sub_epi16(s3, p3);
      _mm256_storeu_si256((__m256i *)diff_ptr, d0);
      _mm256_storeu_si256((__m256i *)(diff_ptr + 16), d1);
      _mm256_storeu_si256((__m256i *)(diff_ptr + 32), d2);
      _mm256_storeu_si256((__m256i *)(diff_ptr + 48), d3);
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    } while (--j != 0);
  } else if (cols == 32) {
    int j = rows;
    do {
      const __m256i s0 = _mm256_lddqu_si256((const __m256i *)src_ptr);
      const __m256i s1 = _mm256_lddqu_si256((const __m256i *)(src_ptr + 16));
      const __m256i p0 = _mm256_lddqu_si256((const __m256i *)pred_ptr);
      const __m256i p1 = _mm256_lddqu_si256((const __m256i *)(pred_ptr + 16));
      const __m256i d0 = _mm256_sub_epi16(s0, p0);
      const __m256i d1 = _mm256_sub_epi16(s1, p1);
      _mm256_storeu_si256((__m256i *)diff_ptr, d0);
      _mm256_storeu_si256((__m256i *)(diff_ptr + 16), d1);
      src_ptr += src_stride;
      pred_ptr += pred_stride;
      diff_ptr += diff_stride;
    } while (--j != 0);
  } else if (cols == 16) {
    int j = rows;
    do {
      const __m256i s0 = _mm256_lddqu_si256((const __m256i *)src_ptr);
      const __m256i s1 =
          _mm256_lddqu_si256((const __m256i *)(src_ptr + src_stride));
      const __m256i p0 = _mm256_lddqu_si256((const __m256i *)pred_ptr);
      const __m256i p1 =
          _mm256_lddqu_si256((const __m256i *)(pred_ptr + pred_stride));
      const __m256i d0 = _mm256_sub_epi16(s0, p0);
      const __m256i d1 = _mm256_sub_epi16(s1, p1);
      _mm256_storeu_si256((__m256i *)diff_ptr, d0);
      _mm256_storeu_si256((__m256i *)(diff_ptr + diff_stride), d1);
      src_ptr += src_stride << 1;
      pred_ptr += pred_stride << 1;
      diff_ptr += diff_stride << 1;
      j -= 2;
    } while (j != 0);
  } else if (cols == 8) {
    int j = rows;
    do {
      const __m128i s0 = _mm_lddqu_si128((const __m128i *)src_ptr);
      const __m128i s1 =
          _mm_lddqu_si128((const __m128i *)(src_ptr + src_stride));
      const __m128i p0 = _mm_lddqu_si128((const __m128i *)pred_ptr);
      const __m128i p1 =
          _mm_lddqu_si128((const __m128i *)(pred_ptr + pred_stride));
      const __m128i d0 = _mm_sub_epi16(s0, p0);
      const __m128i d1 = _mm_sub_epi16(s1, p1);
      _mm_storeu_si128((__m128i *)diff_ptr, d0);
      _mm_storeu_si128((__m128i *)(diff_ptr + diff_stride), d1);
      src_ptr += src_stride << 1;
      pred_ptr += pred_stride << 1;
      diff_ptr += diff_stride << 1;
      j -= 2;
    } while (j != 0);
  } else {
    int j = rows;
    assert(cols == 4);
    do {
      const __m128i s0 = _mm_loadl_epi64((const __m128i *)src_ptr);
      const __m128i s1 =
          _mm_loadl_epi64((const __m128i *)(src_ptr + src_stride));
      const __m128i p0 = _mm_loadl_epi64((const __m128i *)pred_ptr);
      const __m128i p1 =
          _mm_loadl_epi64((const __m128i *)(pred_ptr + pred_stride));
      const __m128i d0 = _mm_sub_epi16(s0, p0);
      const __m128i d1 = _mm_sub_epi16(s1, p1);
      _mm_storel_epi64((__m128i *)diff_ptr, d0);
      _mm_storel_epi64((__m128i *)(diff_ptr + diff_stride), d1);
      src_ptr += src_stride << 1;
      pred_ptr += pred_stride << 1;
      diff_ptr += diff_stride << 1;
      j -= 2;
    } while (j != 0);
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

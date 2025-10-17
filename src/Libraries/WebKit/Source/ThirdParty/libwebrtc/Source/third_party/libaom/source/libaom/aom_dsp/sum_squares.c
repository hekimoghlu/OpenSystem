/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

uint64_t aom_sum_squares_2d_i16_c(const int16_t *src, int src_stride, int width,
                                  int height) {
  int r, c;
  uint64_t ss = 0;

  for (r = 0; r < height; r++) {
    for (c = 0; c < width; c++) {
      const int16_t v = src[c];
      ss += v * v;
    }
    src += src_stride;
  }

  return ss;
}

uint64_t aom_sum_squares_i16_c(const int16_t *src, uint32_t n) {
  uint64_t ss = 0;
  do {
    const int16_t v = *src++;
    ss += v * v;
  } while (--n);

  return ss;
}

uint64_t aom_var_2d_u8_c(uint8_t *src, int src_stride, int width, int height) {
  int r, c;
  uint64_t ss = 0, s = 0;

  for (r = 0; r < height; r++) {
    for (c = 0; c < width; c++) {
      const uint8_t v = src[c];
      ss += v * v;
      s += v;
    }
    src += src_stride;
  }

  return (ss - s * s / (width * height));
}

#if CONFIG_AV1_HIGHBITDEPTH
uint64_t aom_var_2d_u16_c(uint8_t *src, int src_stride, int width, int height) {
  uint16_t *srcp = CONVERT_TO_SHORTPTR(src);
  int r, c;
  uint64_t ss = 0, s = 0;

  for (r = 0; r < height; r++) {
    for (c = 0; c < width; c++) {
      const uint16_t v = srcp[c];
      ss += v * v;
      s += v;
    }
    srcp += src_stride;
  }

  return (ss - s * s / (width * height));
}
#endif  // CONFIG_AV1_HIGHBITDEPTH

uint64_t aom_sum_sse_2d_i16_c(const int16_t *src, int src_stride, int width,
                              int height, int *sum) {
  int r, c;
  int16_t *srcp = (int16_t *)src;
  int64_t ss = 0;

  for (r = 0; r < height; r++) {
    for (c = 0; c < width; c++) {
      const int16_t v = srcp[c];
      ss += v * v;
      *sum += v;
    }
    srcp += src_stride;
  }
  return ss;
}

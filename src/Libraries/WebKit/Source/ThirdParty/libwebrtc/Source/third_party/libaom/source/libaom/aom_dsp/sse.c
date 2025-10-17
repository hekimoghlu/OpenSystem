/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 30, 2024.
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
/*
 * Sum the square of the difference between every corresponding element of the
 * buffers.
 */

#include <stdlib.h>

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"

#include "aom/aom_integer.h"

int64_t aom_sse_c(const uint8_t *a, int a_stride, const uint8_t *b,
                  int b_stride, int width, int height) {
  int y, x;
  int64_t sse = 0;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      const int32_t diff = abs(a[x] - b[x]);
      sse += diff * diff;
    }

    a += a_stride;
    b += b_stride;
  }
  return sse;
}

#if CONFIG_AV1_HIGHBITDEPTH
int64_t aom_highbd_sse_c(const uint8_t *a8, int a_stride, const uint8_t *b8,
                         int b_stride, int width, int height) {
  int y, x;
  int64_t sse = 0;
  uint16_t *a = CONVERT_TO_SHORTPTR(a8);
  uint16_t *b = CONVERT_TO_SHORTPTR(b8);
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      const int32_t diff = (int32_t)(a[x]) - (int32_t)(b[x]);
      sse += diff * diff;
    }

    a += a_stride;
    b += b_stride;
  }
  return sse;
}
#endif

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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
#include <stdlib.h>

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"

#include "vpx/vpx_integer.h"
#include "vpx_ports/mem.h"

void vpx_subtract_block_c(int rows, int cols, int16_t *diff_ptr,
                          ptrdiff_t diff_stride, const uint8_t *src_ptr,
                          ptrdiff_t src_stride, const uint8_t *pred_ptr,
                          ptrdiff_t pred_stride) {
  int r, c;

  for (r = 0; r < rows; r++) {
    for (c = 0; c < cols; c++) diff_ptr[c] = src_ptr[c] - pred_ptr[c];

    diff_ptr += diff_stride;
    pred_ptr += pred_stride;
    src_ptr += src_stride;
  }
}

#if CONFIG_VP9_HIGHBITDEPTH
void vpx_highbd_subtract_block_c(int rows, int cols, int16_t *diff_ptr,
                                 ptrdiff_t diff_stride, const uint8_t *src8_ptr,
                                 ptrdiff_t src_stride, const uint8_t *pred8_ptr,
                                 ptrdiff_t pred_stride, int bd) {
  int r, c;
  uint16_t *src = CONVERT_TO_SHORTPTR(src8_ptr);
  uint16_t *pred = CONVERT_TO_SHORTPTR(pred8_ptr);
  (void)bd;

  for (r = 0; r < rows; r++) {
    for (c = 0; c < cols; c++) {
      diff_ptr[c] = src[c] - pred[c];
    }

    diff_ptr += diff_stride;
    pred += pred_stride;
    src += src_stride;
  }
}
#endif  // CONFIG_VP9_HIGHBITDEPTH

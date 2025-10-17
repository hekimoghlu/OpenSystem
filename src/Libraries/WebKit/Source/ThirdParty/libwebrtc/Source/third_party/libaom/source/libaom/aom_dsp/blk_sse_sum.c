/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 29, 2025.
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
#include "config/aom_dsp_rtcd.h"

void aom_get_blk_sse_sum_c(const int16_t *data, int stride, int bw, int bh,
                           int *x_sum, int64_t *x2_sum) {
  *x_sum = 0;
  *x2_sum = 0;
  for (int i = 0; i < bh; ++i) {
    for (int j = 0; j < bw; ++j) {
      const int val = data[j];
      *x_sum += val;
      *x2_sum += val * val;
    }
    data += stride;
  }
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 29, 2023.
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

#include "aom/aom_integer.h"
#include "aom_ports/mem.h"
#include "aom_dsp/aom_dsp_common.h"
#include "aom_dsp/blend.h"

#include "config/aom_dsp_rtcd.h"

void aom_blend_a64_vmask_c(uint8_t *dst, uint32_t dst_stride,
                           const uint8_t *src0, uint32_t src0_stride,
                           const uint8_t *src1, uint32_t src1_stride,
                           const uint8_t *mask, int w, int h) {
  int i, j;

  assert(IMPLIES(src0 == dst, src0_stride == dst_stride));
  assert(IMPLIES(src1 == dst, src1_stride == dst_stride));

  assert(h >= 1);
  assert(w >= 1);
  assert(IS_POWER_OF_TWO(h));
  assert(IS_POWER_OF_TWO(w));

  for (i = 0; i < h; ++i) {
    const int m = mask[i];
    for (j = 0; j < w; ++j) {
      dst[i * dst_stride + j] = AOM_BLEND_A64(m, src0[i * src0_stride + j],
                                              src1[i * src1_stride + j]);
    }
  }
}

#if CONFIG_AV1_HIGHBITDEPTH
void aom_highbd_blend_a64_vmask_c(uint8_t *dst_8, uint32_t dst_stride,
                                  const uint8_t *src0_8, uint32_t src0_stride,
                                  const uint8_t *src1_8, uint32_t src1_stride,
                                  const uint8_t *mask, int w, int h, int bd) {
  int i, j;
  uint16_t *dst = CONVERT_TO_SHORTPTR(dst_8);
  const uint16_t *src0 = CONVERT_TO_SHORTPTR(src0_8);
  const uint16_t *src1 = CONVERT_TO_SHORTPTR(src1_8);
  (void)bd;

  assert(IMPLIES(src0 == dst, src0_stride == dst_stride));
  assert(IMPLIES(src1 == dst, src1_stride == dst_stride));

  assert(h >= 1);
  assert(w >= 1);
  assert(IS_POWER_OF_TWO(h));
  assert(IS_POWER_OF_TWO(w));

  assert(bd == 8 || bd == 10 || bd == 12);

  for (i = 0; i < h; ++i) {
    const int m = mask[i];
    for (j = 0; j < w; ++j) {
      dst[i * dst_stride + j] = AOM_BLEND_A64(m, src0[i * src0_stride + j],
                                              src1[i * src1_stride + j]);
    }
  }
}
#endif

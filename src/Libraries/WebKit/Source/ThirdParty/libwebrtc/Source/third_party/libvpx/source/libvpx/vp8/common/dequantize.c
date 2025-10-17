/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#include "vpx_config.h"
#include "vp8_rtcd.h"
#include "vp8/common/blockd.h"
#include "vpx_mem/vpx_mem.h"

void vp8_dequantize_b_c(BLOCKD *d, short *DQC) {
  int i;
  short *DQ = d->dqcoeff;
  short *Q = d->qcoeff;

  for (i = 0; i < 16; ++i) {
    DQ[i] = Q[i] * DQC[i];
  }
}

void vp8_dequant_idct_add_c(short *input, short *dq, unsigned char *dest,
                            int stride) {
  int i;

  for (i = 0; i < 16; ++i) {
    input[i] = dq[i] * input[i];
  }

  vp8_short_idct4x4llm_c(input, dest, stride, dest, stride);

  memset(input, 0, 32);
}

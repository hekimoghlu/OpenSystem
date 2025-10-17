/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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

#if HAVE_DSPR2

void vp8_dequant_idct_add_y_block_dspr2(short *q, short *dq, unsigned char *dst,
                                        int stride, char *eobs) {
  int i, j;

  for (i = 0; i < 4; ++i) {
    for (j = 0; j < 4; ++j) {
      if (*eobs++ > 1)
        vp8_dequant_idct_add_dspr2(q, dq, dst, stride);
      else {
        vp8_dc_only_idct_add_dspr2(q[0] * dq[0], dst, stride, dst, stride);
        ((int *)q)[0] = 0;
      }

      q += 16;
      dst += 4;
    }

    dst += 4 * stride - 16;
  }
}

void vp8_dequant_idct_add_uv_block_dspr2(short *q, short *dq,
                                         unsigned char *dst_u,
                                         unsigned char *dst_v, int stride,
                                         char *eobs) {
  int i, j;

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      if (*eobs++ > 1)
        vp8_dequant_idct_add_dspr2(q, dq, dst_u, stride);
      else {
        vp8_dc_only_idct_add_dspr2(q[0] * dq[0], dst_u, stride, dst_u, stride);
        ((int *)q)[0] = 0;
      }

      q += 16;
      dst_u += 4;
    }

    dst_u += 4 * stride - 8;
  }

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      if (*eobs++ > 1)
        vp8_dequant_idct_add_dspr2(q, dq, dst_v, stride);
      else {
        vp8_dc_only_idct_add_dspr2(q[0] * dq[0], dst_v, stride, dst_v, stride);
        ((int *)q)[0] = 0;
      }

      q += 16;
      dst_v += 4;
    }

    dst_v += 4 * stride - 8;
  }
}

#endif

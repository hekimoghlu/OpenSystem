/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 2, 2023.
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

void vp8_idct_dequant_0_2x_sse2(short *q, short *dq, unsigned char *dst,
                                int dst_stride);
void vp8_idct_dequant_full_2x_sse2(short *q, short *dq, unsigned char *dst,
                                   int dst_stride);

void vp8_dequant_idct_add_y_block_sse2(short *q, short *dq, unsigned char *dst,
                                       int stride, char *eobs) {
  int i;

  for (i = 0; i < 4; ++i) {
    if (((short *)(eobs))[0]) {
      if (((short *)(eobs))[0] & 0xfefe) {
        vp8_idct_dequant_full_2x_sse2(q, dq, dst, stride);
      } else {
        vp8_idct_dequant_0_2x_sse2(q, dq, dst, stride);
      }
    }
    if (((short *)(eobs))[1]) {
      if (((short *)(eobs))[1] & 0xfefe) {
        vp8_idct_dequant_full_2x_sse2(q + 32, dq, dst + 8, stride);
      } else {
        vp8_idct_dequant_0_2x_sse2(q + 32, dq, dst + 8, stride);
      }
    }
    q += 64;
    dst += stride * 4;
    eobs += 4;
  }
}

void vp8_dequant_idct_add_uv_block_sse2(short *q, short *dq,
                                        unsigned char *dst_u,
                                        unsigned char *dst_v, int stride,
                                        char *eobs) {
  if (((short *)(eobs))[0]) {
    if (((short *)(eobs))[0] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_u, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_u, stride);
    }
  }
  q += 32;
  dst_u += stride * 4;

  if (((short *)(eobs))[1]) {
    if (((short *)(eobs))[1] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_u, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_u, stride);
    }
  }
  q += 32;

  if (((short *)(eobs))[2]) {
    if (((short *)(eobs))[2] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_v, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_v, stride);
    }
  }
  q += 32;
  dst_v += stride * 4;

  if (((short *)(eobs))[3]) {
    if (((short *)(eobs))[3] & 0xfefe) {
      vp8_idct_dequant_full_2x_sse2(q, dq, dst_v, stride);
    } else {
      vp8_idct_dequant_0_2x_sse2(q, dq, dst_v, stride);
    }
  }
}

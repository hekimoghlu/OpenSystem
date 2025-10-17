/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#include <arm_neon.h>

#include "./vpx_dsp_rtcd.h"
#include "vpx/vpx_integer.h"

void vpx_highbd_convolve_copy_neon(const uint16_t *src, ptrdiff_t src_stride,
                                   uint16_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h, int bd) {
  (void)filter;
  (void)x0_q4;
  (void)x_step_q4;
  (void)y0_q4;
  (void)y_step_q4;
  (void)bd;

  if (w < 8) {  // copy4
    uint16x4_t s0, s1;
    do {
      s0 = vld1_u16(src);
      src += src_stride;
      s1 = vld1_u16(src);
      src += src_stride;

      vst1_u16(dst, s0);
      dst += dst_stride;
      vst1_u16(dst, s1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 8) {  // copy8
    uint16x8_t s0, s1;
    do {
      s0 = vld1q_u16(src);
      src += src_stride;
      s1 = vld1q_u16(src);
      src += src_stride;

      vst1q_u16(dst, s0);
      dst += dst_stride;
      vst1q_u16(dst, s1);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w < 32) {  // copy16
    uint16x8_t s0, s1, s2, s3;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      src += src_stride;
      s2 = vld1q_u16(src);
      s3 = vld1q_u16(src + 8);
      src += src_stride;

      vst1q_u16(dst, s0);
      vst1q_u16(dst + 8, s1);
      dst += dst_stride;
      vst1q_u16(dst, s2);
      vst1q_u16(dst + 8, s3);
      dst += dst_stride;
      h -= 2;
    } while (h != 0);
  } else if (w == 32) {  // copy32
    uint16x8_t s0, s1, s2, s3;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      src += src_stride;

      vst1q_u16(dst, s0);
      vst1q_u16(dst + 8, s1);
      vst1q_u16(dst + 16, s2);
      vst1q_u16(dst + 24, s3);
      dst += dst_stride;
    } while (--h != 0);
  } else {  // copy64
    uint16x8_t s0, s1, s2, s3, s4, s5, s6, s7;
    do {
      s0 = vld1q_u16(src);
      s1 = vld1q_u16(src + 8);
      s2 = vld1q_u16(src + 16);
      s3 = vld1q_u16(src + 24);
      s4 = vld1q_u16(src + 32);
      s5 = vld1q_u16(src + 40);
      s6 = vld1q_u16(src + 48);
      s7 = vld1q_u16(src + 56);
      src += src_stride;

      vst1q_u16(dst, s0);
      vst1q_u16(dst + 8, s1);
      vst1q_u16(dst + 16, s2);
      vst1q_u16(dst + 24, s3);
      vst1q_u16(dst + 32, s4);
      vst1q_u16(dst + 40, s5);
      vst1q_u16(dst + 48, s6);
      vst1q_u16(dst + 56, s7);
      dst += dst_stride;
    } while (--h != 0);
  }
}

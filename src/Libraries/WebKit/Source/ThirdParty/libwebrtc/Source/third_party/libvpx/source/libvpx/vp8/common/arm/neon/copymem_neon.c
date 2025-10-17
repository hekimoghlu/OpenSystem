/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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

#include "./vp8_rtcd.h"

void vp8_copy_mem8x4_neon(unsigned char *src, int src_stride,
                          unsigned char *dst, int dst_stride) {
  uint8x8_t vtmp;
  int r;

  for (r = 0; r < 4; ++r) {
    vtmp = vld1_u8(src);
    vst1_u8(dst, vtmp);
    src += src_stride;
    dst += dst_stride;
  }
}

void vp8_copy_mem8x8_neon(unsigned char *src, int src_stride,
                          unsigned char *dst, int dst_stride) {
  uint8x8_t vtmp;
  int r;

  for (r = 0; r < 8; ++r) {
    vtmp = vld1_u8(src);
    vst1_u8(dst, vtmp);
    src += src_stride;
    dst += dst_stride;
  }
}

void vp8_copy_mem16x16_neon(unsigned char *src, int src_stride,
                            unsigned char *dst, int dst_stride) {
  int r;
  uint8x16_t qtmp;

  for (r = 0; r < 16; ++r) {
    qtmp = vld1q_u8(src);
    vst1q_u8(dst, qtmp);
    src += src_stride;
    dst += dst_stride;
  }
}

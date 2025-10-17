/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 3, 2022.
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
#include "./vp8_rtcd.h"
#include "vp8/common/mips/msa/vp8_macros_msa.h"

static void copy_8x4_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  uint64_t src0, src1, src2, src3;

  LD4(src, src_stride, src0, src1, src2, src3);
  SD4(src0, src1, src2, src3, dst, dst_stride);
}

static void copy_8x8_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  uint64_t src0, src1, src2, src3;

  LD4(src, src_stride, src0, src1, src2, src3);
  src += (4 * src_stride);
  SD4(src0, src1, src2, src3, dst, dst_stride);
  dst += (4 * dst_stride);

  LD4(src, src_stride, src0, src1, src2, src3);
  SD4(src0, src1, src2, src3, dst, dst_stride);
}

static void copy_16x16_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride) {
  v16u8 src0, src1, src2, src3, src4, src5, src6, src7;
  v16u8 src8, src9, src10, src11, src12, src13, src14, src15;

  LD_UB8(src, src_stride, src0, src1, src2, src3, src4, src5, src6, src7);
  src += (8 * src_stride);
  LD_UB8(src, src_stride, src8, src9, src10, src11, src12, src13, src14, src15);

  ST_UB8(src0, src1, src2, src3, src4, src5, src6, src7, dst, dst_stride);
  dst += (8 * dst_stride);
  ST_UB8(src8, src9, src10, src11, src12, src13, src14, src15, dst, dst_stride);
}

void vp8_copy_mem16x16_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                           int32_t dst_stride) {
  copy_16x16_msa(src, src_stride, dst, dst_stride);
}

void vp8_copy_mem8x8_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  copy_8x8_msa(src, src_stride, dst, dst_stride);
}

void vp8_copy_mem8x4_msa(uint8_t *src, int32_t src_stride, uint8_t *dst,
                         int32_t dst_stride) {
  copy_8x4_msa(src, src_stride, dst, dst_stride);
}

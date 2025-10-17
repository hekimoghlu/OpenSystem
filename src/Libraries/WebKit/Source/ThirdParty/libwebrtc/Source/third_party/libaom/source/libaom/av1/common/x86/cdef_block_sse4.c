/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
#include "aom_dsp/aom_simd.h"
#define SIMD_FUNC(name) name##_sse4_1
#include "av1/common/cdef_block_simd.h"

void cdef_find_dir_dual_sse4_1(const uint16_t *img1, const uint16_t *img2,
                               int stride, int32_t *var_out_1st,
                               int32_t *var_out_2nd, int coeff_shift,
                               int *out_dir_1st_8x8, int *out_dir_2nd_8x8) {
  // Process first 8x8.
  *out_dir_1st_8x8 = cdef_find_dir(img1, stride, var_out_1st, coeff_shift);

  // Process second 8x8.
  *out_dir_2nd_8x8 = cdef_find_dir(img2, stride, var_out_2nd, coeff_shift);
}

void cdef_copy_rect8_8bit_to_16bit_sse4_1(uint16_t *dst, int dstride,
                                          const uint8_t *src, int sstride,
                                          int width, int height) {
  int j = 0;
  for (int i = 0; i < height; i++) {
    for (j = 0; j < (width & ~0x7); j += 8) {
      v64 row = v64_load_unaligned(&src[i * sstride + j]);
      v128_store_unaligned(&dst[i * dstride + j], v128_unpack_u8_s16(row));
    }
    for (; j < width; j++) {
      dst[i * dstride + j] = src[i * sstride + j];
    }
  }
}

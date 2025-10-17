/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#ifndef VPX_VP8_COMMON_RECONINTRA4X4_H_
#define VPX_VP8_COMMON_RECONINTRA4X4_H_
#include "vp8/common/blockd.h"

#ifdef __cplusplus
extern "C" {
#endif

static INLINE void intra_prediction_down_copy(MACROBLOCKD *xd,
                                              unsigned char *above_right_src) {
  int dst_stride = xd->dst.y_stride;
  unsigned char *above_right_dst = xd->dst.y_buffer - dst_stride + 16;

  unsigned int *src_ptr = (unsigned int *)above_right_src;
  unsigned int *dst_ptr0 = (unsigned int *)(above_right_dst + 4 * dst_stride);
  unsigned int *dst_ptr1 = (unsigned int *)(above_right_dst + 8 * dst_stride);
  unsigned int *dst_ptr2 = (unsigned int *)(above_right_dst + 12 * dst_stride);

  *dst_ptr0 = *src_ptr;
  *dst_ptr1 = *src_ptr;
  *dst_ptr2 = *src_ptr;
}

void vp8_intra4x4_predict(unsigned char *above, unsigned char *yleft,
                          int left_stride, B_PREDICTION_MODE b_mode,
                          unsigned char *dst, int dst_stride,
                          unsigned char top_left);

void vp8_init_intra4x4_predictors_internal(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_RECONINTRA4X4_H_

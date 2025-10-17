/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#include "./vpx_dsp_rtcd.h"
#include "vp9/common/vp9_filter.h"
#include "vpx_dsp/arm/vpx_convolve8_neon_asm.h"

/* Type1 and Type2 functions are called depending on the position of the
 * negative and positive coefficients in the filter. In type1, the filter kernel
 * used is sub_pel_filters_8lp, in which only the first two and the last two
 * coefficients are negative. In type2, the negative coefficients are 0, 2, 5 &
 * 7.
 */

#define DEFINE_FILTER(dir)                                                   \
  void vpx_convolve8_##dir##_neon(                                           \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,                \
      ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4,           \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h) {               \
    if (filter == vp9_filter_kernels[1]) {                                   \
      vpx_convolve8_##dir##_filter_type1_neon(                               \
          src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, \
          y_step_q4, w, h);                                                  \
    } else {                                                                 \
      vpx_convolve8_##dir##_filter_type2_neon(                               \
          src, src_stride, dst, dst_stride, filter, x0_q4, x_step_q4, y0_q4, \
          y_step_q4, w, h);                                                  \
    }                                                                        \
  }

DEFINE_FILTER(horiz)
DEFINE_FILTER(avg_horiz)
DEFINE_FILTER(vert)
DEFINE_FILTER(avg_vert)

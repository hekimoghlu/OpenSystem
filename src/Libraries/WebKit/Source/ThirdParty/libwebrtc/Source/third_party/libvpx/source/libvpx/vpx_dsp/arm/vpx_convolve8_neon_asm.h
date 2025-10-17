/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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
#ifndef VPX_VPX_DSP_ARM_VPX_CONVOLVE8_NEON_ASM_H_
#define VPX_VPX_DSP_ARM_VPX_CONVOLVE8_NEON_ASM_H_

#define DECLARE_FILTER(dir, type)                                  \
  void vpx_convolve8_##dir##_filter_##type##_neon(                 \
      const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,      \
      ptrdiff_t dst_stride, const InterpKernel *filter, int x0_q4, \
      int x_step_q4, int y0_q4, int y_step_q4, int w, int h);

DECLARE_FILTER(horiz, type1)
DECLARE_FILTER(avg_horiz, type1)
DECLARE_FILTER(horiz, type2)
DECLARE_FILTER(avg_horiz, type2)
DECLARE_FILTER(vert, type1)
DECLARE_FILTER(avg_vert, type1)
DECLARE_FILTER(vert, type2)
DECLARE_FILTER(avg_vert, type2)

#endif  // VPX_VPX_DSP_ARM_VPX_CONVOLVE8_NEON_ASM_H_

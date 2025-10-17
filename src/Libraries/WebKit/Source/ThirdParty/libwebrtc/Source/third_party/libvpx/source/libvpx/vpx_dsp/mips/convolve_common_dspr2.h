/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#ifndef VPX_VPX_DSP_MIPS_CONVOLVE_COMMON_DSPR2_H_
#define VPX_VPX_DSP_MIPS_CONVOLVE_COMMON_DSPR2_H_

#include <assert.h>

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/mips/common_dspr2.h"

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_DSPR2
void vpx_convolve2_horiz_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                               uint8_t *dst, ptrdiff_t dst_stride,
                               const InterpKernel *filter, int x0_q4,
                               int32_t x_step_q4, int y0_q4, int y_step_q4,
                               int w, int h);

void vpx_convolve2_avg_horiz_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                                   uint8_t *dst, ptrdiff_t dst_stride,
                                   const InterpKernel *filter, int x0_q4,
                                   int32_t x_step_q4, int y0_q4, int y_step_q4,
                                   int w, int h);

void vpx_convolve2_avg_vert_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                                  uint8_t *dst, ptrdiff_t dst_stride,
                                  const InterpKernel *filter, int x0_q4,
                                  int32_t x_step_q4, int y0_q4, int y_step_q4,
                                  int w, int h);

void vpx_convolve2_dspr2(const uint8_t *src, ptrdiff_t src_stride, uint8_t *dst,
                         ptrdiff_t dst_stride, const int16_t *filter, int w,
                         int h);

void vpx_convolve2_vert_dspr2(const uint8_t *src, ptrdiff_t src_stride,
                              uint8_t *dst, ptrdiff_t dst_stride,
                              const InterpKernel *filter, int x0_q4,
                              int32_t x_step_q4, int y0_q4, int y_step_q4,
                              int w, int h);

#endif  // #if HAVE_DSPR2
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_DSP_MIPS_CONVOLVE_COMMON_DSPR2_H_

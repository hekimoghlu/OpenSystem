/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include "aom/aom_integer.h"

#include "config/aom_dsp_rtcd.h"

// To start out, just dispatch to the function using the 2D mask and
// pass mask stride as 0. This can be improved upon if necessary.

void aom_blend_a64_hmask_sse4_1(uint8_t *dst, uint32_t dst_stride,
                                const uint8_t *src0, uint32_t src0_stride,
                                const uint8_t *src1, uint32_t src1_stride,
                                const uint8_t *mask, int w, int h) {
  aom_blend_a64_mask_sse4_1(dst, dst_stride, src0, src0_stride, src1,
                            src1_stride, mask, 0, w, h, 0, 0);
}

#if CONFIG_AV1_HIGHBITDEPTH
void aom_highbd_blend_a64_hmask_sse4_1(
    uint8_t *dst_8, uint32_t dst_stride, const uint8_t *src0_8,
    uint32_t src0_stride, const uint8_t *src1_8, uint32_t src1_stride,
    const uint8_t *mask, int w, int h, int bd) {
  aom_highbd_blend_a64_mask_sse4_1(dst_8, dst_stride, src0_8, src0_stride,
                                   src1_8, src1_stride, mask, 0, w, h, 0, 0,
                                   bd);
}
#endif

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 17, 2024.
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
#ifndef VPX_VP8_COMMON_VP8_SKIN_DETECTION_H_
#define VPX_VP8_COMMON_VP8_SKIN_DETECTION_H_

#include "vp8/encoder/onyx_int.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/skin_detection.h"
#include "vpx_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;

typedef enum {
  // Skin detection based on 8x8 block. If two of them are identified as skin,
  // the macroblock is marked as skin.
  SKIN_8X8,
  // Skin detection based on 16x16 block.
  SKIN_16X16
} SKIN_DETECTION_BLOCK_SIZE;

int vp8_compute_skin_block(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                           int stride, int strideuv,
                           SKIN_DETECTION_BLOCK_SIZE bsize, int consec_zeromv,
                           int curr_motion_magn);

#ifdef OUTPUT_YUV_SKINMAP
// For viewing skin map on input source.
void vp8_compute_skin_map(struct VP8_COMP *const cpi, FILE *yuv_skinmap_file);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_COMMON_VP8_SKIN_DETECTION_H_

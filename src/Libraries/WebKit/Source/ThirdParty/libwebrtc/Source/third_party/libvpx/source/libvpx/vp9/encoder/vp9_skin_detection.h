/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 7, 2025.
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
#ifndef VPX_VP9_ENCODER_VP9_SKIN_DETECTION_H_
#define VPX_VP9_ENCODER_VP9_SKIN_DETECTION_H_

#include "vp9/common/vp9_blockd.h"
#include "vpx_dsp/skin_detection.h"
#include "vpx_util/vpx_write_yuv_frame.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP9_COMP;

int vp9_compute_skin_block(const uint8_t *y, const uint8_t *u, const uint8_t *v,
                           int stride, int strideuv, int bsize,
                           int consec_zeromv, int curr_motion_magn);

void vp9_compute_skin_sb(struct VP9_COMP *const cpi, BLOCK_SIZE bsize,
                         int mi_row, int mi_col);

#ifdef OUTPUT_YUV_SKINMAP
// For viewing skin map on input source.
void vp9_output_skin_map(struct VP9_COMP *const cpi, FILE *yuv_skinmap_file);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_SKIN_DETECTION_H_

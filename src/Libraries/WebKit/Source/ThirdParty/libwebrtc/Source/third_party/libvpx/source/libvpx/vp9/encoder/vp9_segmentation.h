/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#ifndef VPX_VP9_ENCODER_VP9_SEGMENTATION_H_
#define VPX_VP9_ENCODER_VP9_SEGMENTATION_H_

#include "vp9/common/vp9_blockd.h"
#include "vp9/encoder/vp9_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_enable_segmentation(struct segmentation *seg);
void vp9_disable_segmentation(struct segmentation *seg);

void vp9_disable_segfeature(struct segmentation *seg, int segment_id,
                            SEG_LVL_FEATURES feature_id);
void vp9_clear_segdata(struct segmentation *seg, int segment_id,
                       SEG_LVL_FEATURES feature_id);

void vp9_psnr_aq_mode_setup(struct segmentation *seg);

void vp9_perceptual_aq_mode_setup(struct VP9_COMP *cpi,
                                  struct segmentation *seg);

// The values given for each segment can be either deltas (from the default
// value chosen for the frame) or absolute values.
//
// Valid range for abs values is (0-127 for MB_LVL_ALT_Q), (0-63 for
// SEGMENT_ALT_LF)
// Valid range for delta values are (+/-127 for MB_LVL_ALT_Q), (+/-63 for
// SEGMENT_ALT_LF)
//
// abs_delta = SEGMENT_DELTADATA (deltas) abs_delta = SEGMENT_ABSDATA (use
// the absolute values given).
void vp9_set_segment_data(struct segmentation *seg, signed char *feature_data,
                          unsigned char abs_delta);

void vp9_choose_segmap_coding_method(VP9_COMMON *cm, MACROBLOCKD *xd);

void vp9_reset_segment_features(struct segmentation *seg);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_SEGMENTATION_H_

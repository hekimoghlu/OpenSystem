/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 9, 2024.
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
#ifndef AOM_AV1_ENCODER_SEGMENTATION_H_
#define AOM_AV1_ENCODER_SEGMENTATION_H_

#include "av1/common/blockd.h"
#include "av1/encoder/encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void av1_enable_segmentation(struct segmentation *seg);
void av1_disable_segmentation(struct segmentation *seg);

void av1_disable_segfeature(struct segmentation *seg, int segment_id,
                            SEG_LVL_FEATURES feature_id);
void av1_clear_segdata(struct segmentation *seg, int segment_id,
                       SEG_LVL_FEATURES feature_id);

void av1_choose_segmap_coding_method(AV1_COMMON *cm, MACROBLOCKD *xd);

void av1_reset_segment_features(AV1_COMMON *cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_SEGMENTATION_H_

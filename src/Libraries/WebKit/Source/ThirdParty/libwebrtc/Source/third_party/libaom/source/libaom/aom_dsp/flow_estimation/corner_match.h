/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#ifndef AOM_AOM_DSP_FLOW_ESTIMATION_CORNER_MATCH_H_
#define AOM_AOM_DSP_FLOW_ESTIMATION_CORNER_MATCH_H_

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MATCH_SZ 16
#define MATCH_SZ_BY2 ((MATCH_SZ - 1) / 2)
#define MATCH_SZ_SQ (MATCH_SZ * MATCH_SZ)

// Minimum threshold for the variance of a patch, in order for it to be
// considered useful for matching.
// This is evaluated against the scaled variance MATCH_SZ_SQ * sigma^2,
// so a setting of 1 * MATCH_SZ_SQ corresponds to an unscaled variance of 1
#define MIN_FEATURE_VARIANCE (1 * MATCH_SZ_SQ)

bool av1_compute_global_motion_feature_match(
    TransformationType type, YV12_BUFFER_CONFIG *src, YV12_BUFFER_CONFIG *ref,
    int bit_depth, int downsample_level, MotionModel *motion_models,
    int num_motion_models, bool *mem_alloc_failed);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_CORNER_MATCH_H_

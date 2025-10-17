/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#include <assert.h>

#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_dsp/flow_estimation/corner_match.h"
#include "aom_dsp/flow_estimation/disflow.h"
#include "aom_dsp/flow_estimation/flow_estimation.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"

// clang-format off
const double kIdentityParams[MAX_PARAMDIM] = {
  0.0, 0.0, 1.0, 0.0, 0.0, 1.0
};
// clang-format on

// Compute a global motion model between the given source and ref frames.
//
// As is standard for video codecs, the resulting model maps from (x, y)
// coordinates in `src` to the corresponding points in `ref`, regardless
// of the temporal order of the two frames.
//
// Returns true if global motion estimation succeeded, false if not.
// The output models should only be used if this function succeeds.
bool aom_compute_global_motion(TransformationType type, YV12_BUFFER_CONFIG *src,
                               YV12_BUFFER_CONFIG *ref, int bit_depth,
                               GlobalMotionMethod gm_method,
                               int downsample_level, MotionModel *motion_models,
                               int num_motion_models, bool *mem_alloc_failed) {
  switch (gm_method) {
    case GLOBAL_MOTION_METHOD_FEATURE_MATCH:
      return av1_compute_global_motion_feature_match(
          type, src, ref, bit_depth, downsample_level, motion_models,
          num_motion_models, mem_alloc_failed);
    case GLOBAL_MOTION_METHOD_DISFLOW:
      return av1_compute_global_motion_disflow(
          type, src, ref, bit_depth, downsample_level, motion_models,
          num_motion_models, mem_alloc_failed);
    default: assert(0 && "Unknown global motion estimation type");
  }
  return false;
}

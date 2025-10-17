/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#ifndef AOM_AOM_DSP_FLOW_ESTIMATION_H_
#define AOM_AOM_DSP_FLOW_ESTIMATION_H_

#include "aom_dsp/pyramid.h"
#include "aom_dsp/flow_estimation/corner_detect.h"
#include "aom_ports/mem.h"
#include "aom_scale/yv12config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_PARAMDIM 6
#define MIN_INLIER_PROB 0.1

/* clang-format off */
enum {
  IDENTITY = 0,      // identity transformation, 0-parameter
  TRANSLATION = 1,   // translational motion 2-parameter
  ROTZOOM = 2,       // simplified affine with rotation + zoom only, 4-parameter
  AFFINE = 3,        // affine, 6-parameter
  TRANS_TYPES,
} UENUM1BYTE(TransformationType);
/* clang-format on */

// number of parameters used by each transformation in TransformationTypes
static const int trans_model_params[TRANS_TYPES] = { 0, 2, 4, 6 };

// Available methods which can be used for global motion estimation
typedef enum {
  GLOBAL_MOTION_METHOD_FEATURE_MATCH,
  GLOBAL_MOTION_METHOD_DISFLOW,
  GLOBAL_MOTION_METHOD_LAST = GLOBAL_MOTION_METHOD_DISFLOW,
  GLOBAL_MOTION_METHODS
} GlobalMotionMethod;

typedef struct {
  double params[MAX_PARAMDIM];
  int *inliers;
  int num_inliers;
} MotionModel;

// Data structure to store a single correspondence point during global
// motion search.
//
// A correspondence (x, y) -> (rx, ry) means that point (x, y) in the
// source frame corresponds to point (rx, ry) in the ref frame.
typedef struct {
  double x, y;
  double rx, ry;
} Correspondence;

// Which global motion method should we use in practice?
// Disflow is both faster and gives better results than feature matching in
// practically all cases, so we use disflow by default
static const GlobalMotionMethod default_global_motion_method =
    GLOBAL_MOTION_METHOD_DISFLOW;

extern const double kIdentityParams[MAX_PARAMDIM];

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
                               int num_motion_models, bool *mem_alloc_failed);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_H_

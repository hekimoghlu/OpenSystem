/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 28, 2024.
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
#ifndef AOM_AV1_ENCODER_OPTICAL_FLOW_H_
#define AOM_AV1_ENCODER_OPTICAL_FLOW_H_

#include "aom_scale/yv12config.h"
#include "av1/common/mv.h"
#include "config/aom_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_OPTICAL_FLOW_API

typedef enum { LUCAS_KANADE, HORN_SCHUNCK } OPTFLOW_METHOD;

typedef enum {
  MV_FILTER_NONE,
  MV_FILTER_SMOOTH,
  MV_FILTER_MEDIAN
} MV_FILTER_TYPE;

typedef struct LOCALMV {
  double row;
  double col;
} LOCALMV;

#define MAX_PYRAMID_LEVELS 5
// default options for optical flow
#define OPFL_WINDOW_SIZE 15
#define OPFL_PYRAMID_LEVELS 3  // total levels
#define OPFL_WARPING_STEPS 3

// parameters specific to Lucas-Kanade
typedef struct lk_params {
  int window_size;
} LK_PARAMS;

// generic structure to contain parameters for all
// optical flow algorithms
typedef struct opfl_params {
  int pyramid_levels;
  int warping_steps;
  LK_PARAMS *lk_params;
  int flags;
} OPFL_PARAMS;

#define OPFL_FLAG_SPARSE 1

void av1_init_opfl_params(OPFL_PARAMS *opfl_params);

void av1_init_lk_params(LK_PARAMS *lk_params);

void av1_optical_flow(const YV12_BUFFER_CONFIG *from_frame,
                      const YV12_BUFFER_CONFIG *to_frame,
                      const int from_frame_idx, const int to_frame_idx,
                      const int bit_depth, const OPFL_PARAMS *opfl_params,
                      const MV_FILTER_TYPE mv_filter,
                      const OPTFLOW_METHOD method, MV *mvs);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_OPTICAL_FLOW_H_

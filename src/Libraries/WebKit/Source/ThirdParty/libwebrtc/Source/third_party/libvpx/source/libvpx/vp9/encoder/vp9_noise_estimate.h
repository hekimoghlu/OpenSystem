/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 27, 2025.
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
#ifndef VPX_VP9_ENCODER_VP9_NOISE_ESTIMATE_H_
#define VPX_VP9_ENCODER_VP9_NOISE_ESTIMATE_H_

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_skin_detection.h"
#include "vpx_scale/yv12config.h"

#if CONFIG_VP9_TEMPORAL_DENOISING
#include "vp9/encoder/vp9_denoiser.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VAR_HIST_BINS 20

typedef enum noise_level { kLowLow, kLow, kMedium, kHigh } NOISE_LEVEL;

typedef struct noise_estimate {
  int enabled;
  NOISE_LEVEL level;
  int value;
  int thresh;
  int adapt_thresh;
  int count;
  int last_w;
  int last_h;
  int num_frames_estimate;
} NOISE_ESTIMATE;

struct VP9_COMP;

void vp9_noise_estimate_init(NOISE_ESTIMATE *const ne, int width, int height);

NOISE_LEVEL vp9_noise_estimate_extract_level(NOISE_ESTIMATE *const ne);

void vp9_update_noise_estimate(struct VP9_COMP *const cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_NOISE_ESTIMATE_H_

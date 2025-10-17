/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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
#ifndef AOM_AV1_ENCODER_SALIENCY_MAP_H_
#define AOM_AV1_ENCODER_SALIENCY_MAP_H_
#include "av1/encoder/encoder.h"

typedef struct saliency_feature_map {
  double *buf;  // stores values of the map in 1D array
  int height;
  int width;
} saliency_feature_map;

int av1_set_saliency_map(AV1_COMP *cpi);
#if !CONFIG_REALTIME_ONLY
double av1_setup_motion_ratio(AV1_COMP *cpi);
#endif
int av1_setup_sm_rdmult_scaling_factor(AV1_COMP *cpi, double motion_ratio);

#endif  // AOM_AV1_ENCODER_SALIENCY_MAP_H_

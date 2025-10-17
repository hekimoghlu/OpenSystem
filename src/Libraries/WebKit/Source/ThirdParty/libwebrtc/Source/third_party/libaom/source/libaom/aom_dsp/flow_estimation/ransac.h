/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
#ifndef AOM_AOM_DSP_FLOW_ESTIMATION_RANSAC_H_
#define AOM_AOM_DSP_FLOW_ESTIMATION_RANSAC_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <stdbool.h>

#include "aom_dsp/flow_estimation/flow_estimation.h"

#ifdef __cplusplus
extern "C" {
#endif

bool ransac(const Correspondence *matched_points, int npoints,
            TransformationType type, MotionModel *motion_models,
            int num_desired_motions, bool *mem_alloc_failed);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_RANSAC_H_

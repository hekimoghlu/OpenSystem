/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#ifndef AOM_AOM_DSP_FLOW_ESTIMATION_CORNER_DETECT_H_
#define AOM_AOM_DSP_FLOW_ESTIMATION_CORNER_DETECT_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <memory.h>

#include "aom_dsp/pyramid.h"
#include "aom_util/aom_pthread.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_CORNERS 4096

typedef struct corner_list {
#if CONFIG_MULTITHREAD
  // Mutex which is used to prevent the corner list from being computed twice
  // at the same time
  //
  // Semantics:
  // * This mutex must be held whenever reading or writing the `valid` flag
  //
  // * This mutex must also be held while computing the image pyramid,
  //   to ensure that only one thread may do so at a time.
  //
  // * However, once you have read the valid flag and seen a true value,
  //   it is safe to drop the mutex and read from the remaining fields.
  //   This is because, once the image pyramid is computed, its contents
  //   will not be changed until the parent frame buffer is recycled,
  //   which will not happen until there are no more outstanding references
  //   to the frame buffer.
  pthread_mutex_t mutex;
#endif  // CONFIG_MULTITHREAD
  // Flag indicating whether the corner list contains valid data
  bool valid;
  // Number of corners found
  int num_corners;
  // (x, y) coordinates of each corner
  int corners[2 * MAX_CORNERS];
} CornerList;

size_t av1_get_corner_list_size(void);

CornerList *av1_alloc_corner_list(void);

bool av1_compute_corner_list(const YV12_BUFFER_CONFIG *frame, int bit_depth,
                             int downsample_level, CornerList *corners);

#ifndef NDEBUG
// Check if a corner list has already been computed.
// This is mostly a debug helper - as it is necessary to hold corners->mutex
// while reading the valid flag, we cannot just write:
//   assert(corners->valid);
// This function allows the check to be correctly written as:
//   assert(aom_is_corner_list_valid(corners));
bool aom_is_corner_list_valid(CornerList *corners);
#endif

void av1_invalidate_corner_list(CornerList *corners);

void av1_free_corner_list(CornerList *corners);

#ifdef __cplusplus
}
#endif

#endif  // AOM_AOM_DSP_FLOW_ESTIMATION_CORNER_DETECT_H_

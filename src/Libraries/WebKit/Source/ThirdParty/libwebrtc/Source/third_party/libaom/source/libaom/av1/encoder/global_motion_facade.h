/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#ifndef AOM_AV1_ENCODER_GLOBAL_MOTION_FACADE_H_
#define AOM_AV1_ENCODER_GLOBAL_MOTION_FACADE_H_

#ifdef __cplusplus
extern "C" {
#endif
struct yv12_buffer_config;
struct AV1_COMP;

// Allocates memory for members of GlobalMotionData.
static inline void gm_alloc_data(AV1_COMP *cpi, GlobalMotionData *gm_data) {
  AV1_COMMON *cm = &cpi->common;
  GlobalMotionInfo *gm_info = &cpi->gm_info;

  CHECK_MEM_ERROR(cm, gm_data->segment_map,
                  aom_malloc(sizeof(*gm_data->segment_map) *
                             gm_info->segment_map_w * gm_info->segment_map_h));

  av1_zero_array(gm_data->motion_models, RANSAC_NUM_MOTIONS);
  for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
    CHECK_MEM_ERROR(cm, gm_data->motion_models[m].inliers,
                    aom_malloc(sizeof(*gm_data->motion_models[m].inliers) * 2 *
                               MAX_CORNERS));
  }
}

// Deallocates the memory allocated for members of GlobalMotionData.
static inline void gm_dealloc_data(GlobalMotionData *gm_data) {
  aom_free(gm_data->segment_map);
  gm_data->segment_map = NULL;
  for (int m = 0; m < RANSAC_NUM_MOTIONS; m++) {
    aom_free(gm_data->motion_models[m].inliers);
    gm_data->motion_models[m].inliers = NULL;
  }
}

void av1_compute_gm_for_valid_ref_frames(
    AV1_COMP *cpi, struct aom_internal_error_info *error_info,
    YV12_BUFFER_CONFIG *ref_buf[REF_FRAMES], int frame,
    MotionModel *motion_models, uint8_t *segment_map, int segment_map_w,
    int segment_map_h);
void av1_compute_global_motion_facade(struct AV1_COMP *cpi);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_ENCODER_GLOBAL_MOTION_FACADE_H_

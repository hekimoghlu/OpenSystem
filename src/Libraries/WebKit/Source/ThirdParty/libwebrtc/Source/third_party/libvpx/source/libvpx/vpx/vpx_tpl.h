/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
/*!\file
 * \brief Describes the TPL stats descriptor and associated operations
 *
 */
#ifndef VPX_VPX_VPX_TPL_H_
#define VPX_VPX_VPX_TPL_H_

#include "./vpx_integer.h"
#include "./vpx_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

/*!\brief Current ABI version number
 *
 * \internal
 * If this file is altered in any way that changes the ABI, this value
 * must be bumped.  Examples include, but are not limited to, changing
 * types, removing or reassigning enums, adding/removing/rearranging
 * fields to structures
 */
#define VPX_TPL_ABI_VERSION 4 /**<\hideinitializer*/

/*!\brief Temporal dependency model stats for each block before propagation */
typedef struct VpxTplBlockStats {
  int16_t row;            /**< Pixel row of the top left corner */
  int16_t col;            /**< Pixel col of the top left corner */
  int64_t intra_cost;     /**< Intra cost */
  int64_t inter_cost;     /**< Inter cost */
  int16_t mv_r;           /**< Motion vector row in pixel */
  int16_t mv_c;           /**< Motion vector col in pixel */
  int64_t srcrf_rate;     /**< Rate from source ref frame */
  int64_t srcrf_dist;     /**< Distortion from source ref frame */
  int64_t pred_error;     /**< Prediction error */
  int64_t inter_pred_err; /**< Inter prediction error */
  int64_t intra_pred_err; /**< Intra prediction error */
  int ref_frame_index;    /**< Ref frame index in the ref frame buffer */
} VpxTplBlockStats;

/*!\brief Temporal dependency model stats for each frame before propagation */
typedef struct VpxTplFrameStats {
  int frame_width;  /**< Frame width */
  int frame_height; /**< Frame height */
  int num_blocks;   /**< Number of blocks. Size of block_stats_list */
  VpxTplBlockStats *block_stats_list; /**< List of tpl stats for each block */
} VpxTplFrameStats;

/*!\brief Temporal dependency model stats for each GOP before propagation */
typedef struct VpxTplGopStats {
  int size; /**< GOP size, also the size of frame_stats_list. */
  VpxTplFrameStats *frame_stats_list; /**< List of tpl stats for each frame */
} VpxTplGopStats;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPX_VPX_TPL_H_

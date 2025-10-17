/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#ifndef VPX_VP9_ENCODER_VP9_CONTEXT_TREE_H_
#define VPX_VP9_ENCODER_VP9_CONTEXT_TREE_H_

#include "vp9/common/vp9_blockd.h"
#include "vp9/encoder/vp9_block.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP9_COMP;
struct VP9Common;
struct ThreadData;

// Structure to hold snapshot of coding context during the mode picking process
typedef struct {
  MODE_INFO mic;
  MB_MODE_INFO_EXT mbmi_ext;
  uint8_t *zcoeff_blk;
  tran_low_t *coeff[MAX_MB_PLANE][3];
  tran_low_t *qcoeff[MAX_MB_PLANE][3];
  tran_low_t *dqcoeff[MAX_MB_PLANE][3];
  uint16_t *eobs[MAX_MB_PLANE][3];

  // dual buffer pointers, 0: in use, 1: best in store
  tran_low_t *coeff_pbuf[MAX_MB_PLANE][3];
  tran_low_t *qcoeff_pbuf[MAX_MB_PLANE][3];
  tran_low_t *dqcoeff_pbuf[MAX_MB_PLANE][3];
  uint16_t *eobs_pbuf[MAX_MB_PLANE][3];

  int is_coded;
  int num_4x4_blk;
  int skip;
  int pred_pixel_ready;
  // For current partition, only if all Y, U, and V transform blocks'
  // coefficients are quantized to 0, skippable is set to 0.
  int skippable;
  uint8_t skip_txfm[MAX_MB_PLANE << 2];
  int best_mode_index;
  int hybrid_pred_diff;
  int comp_pred_diff;
  int single_pred_diff;
  int64_t best_filter_diff[SWITCHABLE_FILTER_CONTEXTS];

  // TODO(jingning) Use RD_COST struct here instead. This involves a boarder
  // scope of refactoring.
  int rate;
  int64_t dist;
  int64_t rdcost;

#if CONFIG_VP9_TEMPORAL_DENOISING
  unsigned int newmv_sse;
  unsigned int zeromv_sse;
  unsigned int zeromv_lastref_sse;
  PREDICTION_MODE best_sse_inter_mode;
  int_mv best_sse_mv;
  MV_REFERENCE_FRAME best_reference_frame;
  MV_REFERENCE_FRAME best_zeromv_reference_frame;
  int sb_skip_denoising;
#endif

  // motion vector cache for adaptive motion search control in partition
  // search loop
  MV pred_mv[MAX_REF_FRAMES];
  INTERP_FILTER pred_interp_filter;

  // Used for the machine learning-based early termination
  int32_t sum_y_eobs;
  // Skip certain ref frames during RD search of rectangular partitions.
  uint8_t skip_ref_frame_mask;
} PICK_MODE_CONTEXT;

typedef struct PC_TREE {
  int index;
  PARTITION_TYPE partitioning;
  BLOCK_SIZE block_size;
  PICK_MODE_CONTEXT none;
  PICK_MODE_CONTEXT horizontal[2];
  PICK_MODE_CONTEXT vertical[2];
  union {
    struct PC_TREE *split[4];
    PICK_MODE_CONTEXT *leaf_split[4];
  } u;
  // Obtained from a simple motion search. Used by the ML based partition search
  // speed feature.
  MV mv;
} PC_TREE;

void vp9_setup_pc_tree(struct VP9Common *cm, struct ThreadData *td);
void vp9_free_pc_tree(struct ThreadData *td);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_CONTEXT_TREE_H_

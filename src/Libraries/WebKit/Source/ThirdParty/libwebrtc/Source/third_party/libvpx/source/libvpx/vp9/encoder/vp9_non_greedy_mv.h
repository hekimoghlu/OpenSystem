/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#ifndef VPX_VP9_ENCODER_VP9_NON_GREEDY_MV_H_
#define VPX_VP9_ENCODER_VP9_NON_GREEDY_MV_H_

#include "vp9/common/vp9_enums.h"
#include "vp9/common/vp9_blockd.h"
#include "vpx_scale/yv12config.h"
#include "vpx_dsp/variance.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NB_MVS_NUM 4
#define LOG2_PRECISION 20
#define MF_LOCAL_STRUCTURE_SIZE 4
#define SQUARE_BLOCK_SIZES 4

typedef enum Status { STATUS_OK = 0, STATUS_FAILED = 1 } Status;

typedef struct MotionField {
  int ready;
  BLOCK_SIZE bsize;
  int block_rows;
  int block_cols;
  int block_num;  // block_num == block_rows * block_cols
  int (*local_structure)[MF_LOCAL_STRUCTURE_SIZE];
  int_mv *mf;
  int *set_mv;
  int mv_log_scale;
} MotionField;

typedef struct MotionFieldInfo {
  int frame_num;
  int allocated;
  MotionField (*motion_field_array)[MAX_INTER_REF_FRAMES][SQUARE_BLOCK_SIZES];
} MotionFieldInfo;

typedef struct {
  float row, col;
} FloatMV;

static INLINE int get_square_block_idx(BLOCK_SIZE bsize) {
  if (bsize == BLOCK_4X4) {
    return 0;
  }
  if (bsize == BLOCK_8X8) {
    return 1;
  }
  if (bsize == BLOCK_16X16) {
    return 2;
  }
  if (bsize == BLOCK_32X32) {
    return 3;
  }
  assert(0 && "ERROR: non-square block size");
  return -1;
}

static INLINE BLOCK_SIZE square_block_idx_to_bsize(int square_block_idx) {
  if (square_block_idx == 0) {
    return BLOCK_4X4;
  }
  if (square_block_idx == 1) {
    return BLOCK_8X8;
  }
  if (square_block_idx == 2) {
    return BLOCK_16X16;
  }
  if (square_block_idx == 3) {
    return BLOCK_32X32;
  }
  assert(0 && "ERROR: invalid square_block_idx");
  return BLOCK_INVALID;
}

Status vp9_alloc_motion_field_info(MotionFieldInfo *motion_field_info,
                                   int frame_num, int mi_rows, int mi_cols);

Status vp9_alloc_motion_field(MotionField *motion_field, BLOCK_SIZE bsize,
                              int block_rows, int block_cols);

void vp9_free_motion_field(MotionField *motion_field);

void vp9_free_motion_field_info(MotionFieldInfo *motion_field_info);

int64_t vp9_nb_mvs_inconsistency(const MV *mv, const int_mv *nb_full_mvs,
                                 int mv_num);

void vp9_get_smooth_motion_field(const MV *search_mf,
                                 const int (*M)[MF_LOCAL_STRUCTURE_SIZE],
                                 int rows, int cols, BLOCK_SIZE bize,
                                 float alpha, int num_iters, MV *smooth_mf);

void vp9_get_local_structure(const YV12_BUFFER_CONFIG *cur_frame,
                             const YV12_BUFFER_CONFIG *ref_frame,
                             const MV *search_mf,
                             const vp9_variance_fn_ptr_t *fn_ptr, int rows,
                             int cols, BLOCK_SIZE bsize,
                             int (*M)[MF_LOCAL_STRUCTURE_SIZE]);

MotionField *vp9_motion_field_info_get_motion_field(
    MotionFieldInfo *motion_field_info, int frame_idx, int rf_idx,
    BLOCK_SIZE bsize);

void vp9_motion_field_mi_set_mv(MotionField *motion_field, int mi_row,
                                int mi_col, int_mv mv);

void vp9_motion_field_reset_mvs(MotionField *motion_field);

int_mv vp9_motion_field_get_mv(const MotionField *motion_field, int brow,
                               int bcol);
int_mv vp9_motion_field_mi_get_mv(const MotionField *motion_field, int mi_row,
                                  int mi_col);
int vp9_motion_field_is_mv_set(const MotionField *motion_field, int brow,
                               int bcol);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // VPX_VP9_ENCODER_VP9_NON_GREEDY_MV_H_

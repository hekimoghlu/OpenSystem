/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef AOM_AV1_COMMON_OBMC_H_
#define AOM_AV1_COMMON_OBMC_H_

typedef void (*overlappable_nb_visitor_t)(MACROBLOCKD *xd, int rel_mi_row,
                                          int rel_mi_col, uint8_t op_mi_size,
                                          int dir, MB_MODE_INFO *nb_mi,
                                          void *fun_ctxt, const int num_planes);

static inline void foreach_overlappable_nb_above(const AV1_COMMON *cm,
                                                 MACROBLOCKD *xd, int nb_max,
                                                 overlappable_nb_visitor_t fun,
                                                 void *fun_ctxt) {
  if (!xd->up_available) return;

  const int num_planes = av1_num_planes(cm);
  int nb_count = 0;
  const int mi_col = xd->mi_col;
  // prev_row_mi points into the mi array, starting at the beginning of the
  // previous row.
  MB_MODE_INFO **prev_row_mi = xd->mi - mi_col - 1 * xd->mi_stride;
  const int end_col = AOMMIN(mi_col + xd->width, cm->mi_params.mi_cols);
  uint8_t mi_step;
  for (int above_mi_col = mi_col; above_mi_col < end_col && nb_count < nb_max;
       above_mi_col += mi_step) {
    MB_MODE_INFO **above_mi = prev_row_mi + above_mi_col;
    mi_step =
        AOMMIN(mi_size_wide[above_mi[0]->bsize], mi_size_wide[BLOCK_64X64]);
    // If we're considering a block with width 4, it should be treated as
    // half of a pair of blocks with chroma information in the second. Move
    // above_mi_col back to the start of the pair if needed, set above_mbmi
    // to point at the block with chroma information, and set mi_step to 2 to
    // step over the entire pair at the end of the iteration.
    if (mi_step == 1) {
      above_mi_col &= ~1;
      above_mi = prev_row_mi + above_mi_col + 1;
      mi_step = 2;
    }
    if (is_neighbor_overlappable(*above_mi)) {
      ++nb_count;
      fun(xd, 0, above_mi_col - mi_col, AOMMIN(xd->width, mi_step), 0,
          *above_mi, fun_ctxt, num_planes);
    }
  }
}

static inline void foreach_overlappable_nb_left(const AV1_COMMON *cm,
                                                MACROBLOCKD *xd, int nb_max,
                                                overlappable_nb_visitor_t fun,
                                                void *fun_ctxt) {
  if (!xd->left_available) return;

  const int num_planes = av1_num_planes(cm);
  int nb_count = 0;
  // prev_col_mi points into the mi array, starting at the top of the
  // previous column
  const int mi_row = xd->mi_row;
  MB_MODE_INFO **prev_col_mi = xd->mi - 1 - mi_row * xd->mi_stride;
  const int end_row = AOMMIN(mi_row + xd->height, cm->mi_params.mi_rows);
  uint8_t mi_step;
  for (int left_mi_row = mi_row; left_mi_row < end_row && nb_count < nb_max;
       left_mi_row += mi_step) {
    MB_MODE_INFO **left_mi = prev_col_mi + left_mi_row * xd->mi_stride;
    mi_step =
        AOMMIN(mi_size_high[left_mi[0]->bsize], mi_size_high[BLOCK_64X64]);
    if (mi_step == 1) {
      left_mi_row &= ~1;
      left_mi = prev_col_mi + (left_mi_row + 1) * xd->mi_stride;
      mi_step = 2;
    }
    if (is_neighbor_overlappable(*left_mi)) {
      ++nb_count;
      fun(xd, left_mi_row - mi_row, 0, AOMMIN(xd->height, mi_step), 1, *left_mi,
          fun_ctxt, num_planes);
    }
  }
}

#endif  // AOM_AV1_COMMON_OBMC_H_

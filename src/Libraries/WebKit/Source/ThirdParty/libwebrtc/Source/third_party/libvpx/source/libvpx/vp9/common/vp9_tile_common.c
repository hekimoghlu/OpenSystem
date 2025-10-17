/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
#include "vp9/common/vp9_tile_common.h"
#include "vp9/common/vp9_onyxc_int.h"
#include "vpx_dsp/vpx_dsp_common.h"

#define MIN_TILE_WIDTH_B64 4
#define MAX_TILE_WIDTH_B64 64

static int get_tile_offset(int idx, int mis, int log2) {
  const int sb_cols = mi_cols_aligned_to_sb(mis) >> MI_BLOCK_SIZE_LOG2;
  const int offset = ((idx * sb_cols) >> log2) << MI_BLOCK_SIZE_LOG2;
  return VPXMIN(offset, mis);
}

void vp9_tile_set_row(TileInfo *tile, const VP9_COMMON *cm, int row) {
  tile->mi_row_start = get_tile_offset(row, cm->mi_rows, cm->log2_tile_rows);
  tile->mi_row_end = get_tile_offset(row + 1, cm->mi_rows, cm->log2_tile_rows);
}

void vp9_tile_set_col(TileInfo *tile, const VP9_COMMON *cm, int col) {
  tile->mi_col_start = get_tile_offset(col, cm->mi_cols, cm->log2_tile_cols);
  tile->mi_col_end = get_tile_offset(col + 1, cm->mi_cols, cm->log2_tile_cols);
}

void vp9_tile_init(TileInfo *tile, const VP9_COMMON *cm, int row, int col) {
  vp9_tile_set_row(tile, cm, row);
  vp9_tile_set_col(tile, cm, col);
}

static int get_min_log2_tile_cols(const int sb64_cols) {
  int min_log2 = 0;
  while ((MAX_TILE_WIDTH_B64 << min_log2) < sb64_cols) ++min_log2;
  return min_log2;
}

static int get_max_log2_tile_cols(const int sb64_cols) {
  int max_log2 = 1;
  while ((sb64_cols >> max_log2) >= MIN_TILE_WIDTH_B64) ++max_log2;
  return max_log2 - 1;
}

void vp9_get_tile_n_bits(int mi_cols, int *min_log2_tile_cols,
                         int *max_log2_tile_cols) {
  const int sb64_cols = mi_cols_aligned_to_sb(mi_cols) >> MI_BLOCK_SIZE_LOG2;
  *min_log2_tile_cols = get_min_log2_tile_cols(sb64_cols);
  *max_log2_tile_cols = get_max_log2_tile_cols(sb64_cols);
  assert(*min_log2_tile_cols <= *max_log2_tile_cols);
}

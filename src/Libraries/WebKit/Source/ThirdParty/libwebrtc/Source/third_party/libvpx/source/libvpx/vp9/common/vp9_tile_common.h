/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#ifndef VPX_VP9_COMMON_VP9_TILE_COMMON_H_
#define VPX_VP9_COMMON_VP9_TILE_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

struct VP9Common;

typedef struct TileInfo {
  int mi_row_start, mi_row_end;
  int mi_col_start, mi_col_end;
} TileInfo;

// initializes 'tile->mi_(row|col)_(start|end)' for (row, col) based on
// 'cm->log2_tile_(rows|cols)' & 'cm->mi_(rows|cols)'
void vp9_tile_init(TileInfo *tile, const struct VP9Common *cm, int row,
                   int col);

void vp9_tile_set_row(TileInfo *tile, const struct VP9Common *cm, int row);
void vp9_tile_set_col(TileInfo *tile, const struct VP9Common *cm, int col);

void vp9_get_tile_n_bits(int mi_cols, int *min_log2_tile_cols,
                         int *max_log2_tile_cols);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_TILE_COMMON_H_

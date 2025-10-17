/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 27, 2024.
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
#ifndef VPX_VP9_ENCODER_VP9_RDOPT_H_
#define VPX_VP9_ENCODER_VP9_RDOPT_H_

#include "vp9/common/vp9_blockd.h"

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_context_tree.h"

#ifdef __cplusplus
extern "C" {
#endif

struct TileInfo;
struct VP9_COMP;
struct macroblock;
struct RD_COST;

void vp9_rd_pick_intra_mode_sb(struct VP9_COMP *cpi, struct macroblock *x,
                               struct RD_COST *rd_cost, BLOCK_SIZE bsize,
                               PICK_MODE_CONTEXT *ctx, int64_t best_rd);

#if !CONFIG_REALTIME_ONLY
void vp9_rd_pick_inter_mode_sb(struct VP9_COMP *cpi,
                               struct TileDataEnc *tile_data,
                               struct macroblock *x, int mi_row, int mi_col,
                               struct RD_COST *rd_cost, BLOCK_SIZE bsize,
                               PICK_MODE_CONTEXT *ctx, int64_t best_rd_so_far);

void vp9_rd_pick_inter_mode_sb_seg_skip(
    struct VP9_COMP *cpi, struct TileDataEnc *tile_data, struct macroblock *x,
    struct RD_COST *rd_cost, BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx,
    int64_t best_rd_so_far);
#endif

int vp9_internal_image_edge(struct VP9_COMP *cpi);
int vp9_active_h_edge(struct VP9_COMP *cpi, int mi_row, int mi_step);
int vp9_active_v_edge(struct VP9_COMP *cpi, int mi_col, int mi_step);
int vp9_active_edge_sb(struct VP9_COMP *cpi, int mi_row, int mi_col);

#if !CONFIG_REALTIME_ONLY
void vp9_rd_pick_inter_mode_sub8x8(struct VP9_COMP *cpi,
                                   struct TileDataEnc *tile_data,
                                   struct macroblock *x, int mi_row, int mi_col,
                                   struct RD_COST *rd_cost, BLOCK_SIZE bsize,
                                   PICK_MODE_CONTEXT *ctx,
                                   int64_t best_rd_so_far);
#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_RDOPT_H_

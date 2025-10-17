/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
#ifndef VPX_VP9_ENCODER_VP9_PICKMODE_H_
#define VPX_VP9_ENCODER_VP9_PICKMODE_H_

#include "vp9/encoder/vp9_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_pick_intra_mode(VP9_COMP *cpi, MACROBLOCK *x, RD_COST *rd_cost,
                         BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx);

void vp9_pick_inter_mode(VP9_COMP *cpi, MACROBLOCK *x, TileDataEnc *tile_data,
                         int mi_row, int mi_col, RD_COST *rd_cost,
                         BLOCK_SIZE bsize, PICK_MODE_CONTEXT *ctx);

void vp9_pick_inter_mode_sub8x8(VP9_COMP *cpi, MACROBLOCK *x, int mi_row,
                                int mi_col, RD_COST *rd_cost, BLOCK_SIZE bsize,
                                PICK_MODE_CONTEXT *ctx);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_PICKMODE_H_

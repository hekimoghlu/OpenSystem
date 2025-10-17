/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 12, 2025.
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
#ifndef VPX_VP9_ENCODER_VP9_ENCODEMB_H_
#define VPX_VP9_ENCODER_VP9_ENCODEMB_H_

#include "./vpx_config.h"
#include "vp9/encoder/vp9_block.h"

#ifdef __cplusplus
extern "C" {
#endif

struct encode_b_args {
  MACROBLOCK *x;
  int enable_trellis_opt;
  double trellis_opt_thresh;
  int *sse_calc_done;
  int64_t *sse;
  ENTROPY_CONTEXT *ta;
  ENTROPY_CONTEXT *tl;
  int8_t *skip;
#if CONFIG_MISMATCH_DEBUG
  int mi_row;
  int mi_col;
  int output_enabled;
#endif
};
int vp9_optimize_b(MACROBLOCK *mb, int plane, int block, TX_SIZE tx_size,
                   int ctx);
void vp9_encode_sb(MACROBLOCK *x, BLOCK_SIZE bsize, int mi_row, int mi_col,
                   int output_enabled);
void vp9_encode_sby_pass1(MACROBLOCK *x, BLOCK_SIZE bsize);
void vp9_xform_quant_fp(MACROBLOCK *x, int plane, int block, int row, int col,
                        BLOCK_SIZE plane_bsize, TX_SIZE tx_size);
void vp9_xform_quant_dc(MACROBLOCK *x, int plane, int block, int row, int col,
                        BLOCK_SIZE plane_bsize, TX_SIZE tx_size);
void vp9_xform_quant(MACROBLOCK *x, int plane, int block, int row, int col,
                     BLOCK_SIZE plane_bsize, TX_SIZE tx_size);

void vp9_subtract_plane(MACROBLOCK *x, BLOCK_SIZE bsize, int plane);

void vp9_encode_block_intra(int plane, int block, int row, int col,
                            BLOCK_SIZE plane_bsize, TX_SIZE tx_size, void *arg);

void vp9_encode_intra_block_plane(MACROBLOCK *x, BLOCK_SIZE bsize, int plane,
                                  int enable_trellis_opt);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ENCODEMB_H_

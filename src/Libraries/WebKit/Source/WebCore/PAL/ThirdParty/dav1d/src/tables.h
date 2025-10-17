/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 1, 2024.
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
#ifndef DAV1D_SRC_TABLES_H
#define DAV1D_SRC_TABLES_H

#include <stdint.h>

#include "common/intops.h"

#include "src/levels.h"

extern const uint8_t dav1d_al_part_ctx[2][N_BL_LEVELS][N_PARTITIONS];
extern const uint8_t /* enum BlockSize */
                     dav1d_block_sizes[N_BL_LEVELS][N_PARTITIONS][2];
// width, height (in 4px blocks), log2 versions of these two
extern const uint8_t dav1d_block_dimensions[N_BS_SIZES][4];
typedef struct TxfmInfo {
    // width, height (in 4px blocks), log2 of them, min/max of log2, sub, pad
    uint8_t w, h, lw, lh, min, max, sub, ctx;
} TxfmInfo;
extern const TxfmInfo dav1d_txfm_dimensions[N_RECT_TX_SIZES];
extern const uint8_t /* enum (Rect)TxfmSize */
                     dav1d_max_txfm_size_for_bs[N_BS_SIZES][4 /* y, 420, 422, 444 */];
extern const uint8_t /* enum TxfmType */
                     dav1d_txtp_from_uvmode[N_UV_INTRA_PRED_MODES];

extern const uint8_t /* enum InterPredMode */
                     dav1d_comp_inter_pred_modes[N_COMP_INTER_PRED_MODES][2];

extern const uint8_t dav1d_partition_type_count[N_BL_LEVELS];
extern const uint8_t /* enum TxfmType */ dav1d_tx_types_per_set[40];

extern const uint8_t dav1d_filter_mode_to_y_mode[5];
extern const uint8_t dav1d_ymode_size_context[N_BS_SIZES];
extern const uint8_t dav1d_lo_ctx_offsets[3][5][5];
extern const uint8_t dav1d_skip_ctx[5][5];
extern const uint8_t /* enum TxClass */
                     dav1d_tx_type_class[N_TX_TYPES_PLUS_LL];
extern const uint8_t /* enum Filter2d */
                     dav1d_filter_2d[DAV1D_N_FILTERS /* h */][DAV1D_N_FILTERS /* v */];
extern const uint8_t /* enum Dav1dFilterMode */ dav1d_filter_dir[N_2D_FILTERS][2];
extern const uint8_t dav1d_intra_mode_context[N_INTRA_PRED_MODES];
extern const uint8_t dav1d_wedge_ctx_lut[N_BS_SIZES];

static const unsigned cfl_allowed_mask =
    (1 << BS_32x32) |
    (1 << BS_32x16) |
    (1 << BS_32x8) |
    (1 << BS_16x32) |
    (1 << BS_16x16) |
    (1 << BS_16x8) |
    (1 << BS_16x4) |
    (1 << BS_8x32) |
    (1 << BS_8x16) |
    (1 << BS_8x8) |
    (1 << BS_8x4) |
    (1 << BS_4x16) |
    (1 << BS_4x8) |
    (1 << BS_4x4);

static const unsigned wedge_allowed_mask =
    (1 << BS_32x32) |
    (1 << BS_32x16) |
    (1 << BS_32x8) |
    (1 << BS_16x32) |
    (1 << BS_16x16) |
    (1 << BS_16x8) |
    (1 << BS_8x32) |
    (1 << BS_8x16) |
    (1 << BS_8x8);

static const unsigned interintra_allowed_mask =
    (1 << BS_32x32) |
    (1 << BS_32x16) |
    (1 << BS_16x32) |
    (1 << BS_16x16) |
    (1 << BS_16x8) |
    (1 << BS_8x16) |
    (1 << BS_8x8);

extern const Dav1dWarpedMotionParams dav1d_default_wm_params;

extern const int8_t dav1d_cdef_directions[12][2];

extern const uint16_t dav1d_sgr_params[16][2];
extern const uint8_t dav1d_sgr_x_by_x[256];

extern const int8_t dav1d_mc_subpel_filters[6][15][8];
extern const int8_t dav1d_mc_warp_filter[193][8];
extern const int8_t dav1d_resize_filter[64][8];

extern const uint8_t dav1d_sm_weights[128];
extern const uint16_t dav1d_dr_intra_derivative[44];
extern const int8_t dav1d_filter_intra_taps[5][64];

extern const uint8_t dav1d_obmc_masks[64];

extern const int16_t dav1d_gaussian_sequence[2048]; // for fgs

#endif /* DAV1D_SRC_TABLES_H */

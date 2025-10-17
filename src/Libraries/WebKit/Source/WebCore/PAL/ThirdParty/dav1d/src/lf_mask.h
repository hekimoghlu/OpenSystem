/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#ifndef DAV1D_SRC_LF_MASK_H
#define DAV1D_SRC_LF_MASK_H

#include <stddef.h>
#include <stdint.h>

#include "src/levels.h"

typedef struct Av1FilterLUT {
    uint8_t e[64];
    uint8_t i[64];
    uint64_t sharp[2];
} Av1FilterLUT;

typedef struct Av1RestorationUnit {
    uint8_t /* enum Dav1dRestorationType */ type;
    int8_t filter_h[3];
    int8_t filter_v[3];
    uint8_t sgr_idx;
    int8_t sgr_weights[2];
} Av1RestorationUnit;

// each struct describes one 128x128 area (1 or 4 SBs), pre-superres-scaling
typedef struct Av1Filter {
    // each bit is 1 col
    uint16_t filter_y[2 /* 0=col, 1=row */][32][3][2];
    uint16_t filter_uv[2 /* 0=col, 1=row */][32][2][2];
    int8_t cdef_idx[4]; // -1 means "unset"
    uint16_t noskip_mask[16][2]; // for 8x8 blocks, but stored on a 4x8 basis
} Av1Filter;

// each struct describes one 128x128 area (1 or 4 SBs), post-superres-scaling
typedef struct Av1Restoration {
    Av1RestorationUnit lr[3][4];
} Av1Restoration;

void dav1d_create_lf_mask_intra(Av1Filter *lflvl, uint8_t (*level_cache)[4],
                                const ptrdiff_t b4_stride,
                                const uint8_t (*level)[8][2], int bx, int by,
                                int iw, int ih, enum BlockSize bs,
                                enum RectTxfmSize ytx, enum RectTxfmSize uvtx,
                                enum Dav1dPixelLayout layout, uint8_t *ay,
                                uint8_t *ly, uint8_t *auv, uint8_t *luv);
void dav1d_create_lf_mask_inter(Av1Filter *lflvl, uint8_t (*level_cache)[4],
                                const ptrdiff_t b4_stride,
                                const uint8_t (*level)[8][2], int bx, int by,
                                int iw, int ih, int skip_inter,
                                enum BlockSize bs, enum RectTxfmSize max_ytx,
                                const uint16_t *tx_mask, enum RectTxfmSize uvtx,
                                enum Dav1dPixelLayout layout, uint8_t *ay,
                                uint8_t *ly, uint8_t *auv, uint8_t *luv);
void dav1d_calc_eih(Av1FilterLUT *lim_lut, int filter_sharpness);
void dav1d_calc_lf_values(uint8_t (*values)[4][8][2], const Dav1dFrameHeader *hdr,
                          const int8_t lf_delta[4]);

#endif /* DAV1D_SRC_LF_MASK_H */

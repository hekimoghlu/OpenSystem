/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#ifndef DAV1D_SRC_FILM_GRAIN_H
#define DAV1D_SRC_FILM_GRAIN_H

#include "common/bitdepth.h"

#include "src/levels.h"

#define GRAIN_WIDTH 82
#define GRAIN_HEIGHT 73
#define BLOCK_SIZE 32
#if !defined(BITDEPTH) || BITDEPTH == 8
#define SCALING_SIZE 256
typedef int8_t entry;
#else
#define SCALING_SIZE 4096
typedef int16_t entry;
#endif

#define decl_generate_grain_y_fn(name) \
void (name)(entry buf[][GRAIN_WIDTH], \
            const Dav1dFilmGrainData *const data HIGHBD_DECL_SUFFIX)
typedef decl_generate_grain_y_fn(*generate_grain_y_fn);

#define decl_generate_grain_uv_fn(name) \
void (name)(entry buf[][GRAIN_WIDTH], \
            const entry buf_y[][GRAIN_WIDTH], \
            const Dav1dFilmGrainData *const data, const intptr_t uv HIGHBD_DECL_SUFFIX)
typedef decl_generate_grain_uv_fn(*generate_grain_uv_fn);

#define decl_fgy_32x32xn_fn(name) \
void (name)(pixel *dst_row, const pixel *src_row, ptrdiff_t stride, \
            const Dav1dFilmGrainData *data, \
            size_t pw, const uint8_t scaling[SCALING_SIZE], \
            const entry grain_lut[][GRAIN_WIDTH], \
            int bh, int row_num HIGHBD_DECL_SUFFIX)
typedef decl_fgy_32x32xn_fn(*fgy_32x32xn_fn);

#define decl_fguv_32x32xn_fn(name) \
void (name)(pixel *dst_row, const pixel *src_row, ptrdiff_t stride, \
            const Dav1dFilmGrainData *data, int pw, \
            const uint8_t scaling[SCALING_SIZE], \
            const entry grain_lut[][GRAIN_WIDTH], int bh, int row_num, \
            const pixel *luma_row, ptrdiff_t luma_stride, \
            int uv_pl, int is_id HIGHBD_DECL_SUFFIX)
typedef decl_fguv_32x32xn_fn(*fguv_32x32xn_fn);

typedef struct Dav1dFilmGrainDSPContext {
    generate_grain_y_fn generate_grain_y;
    generate_grain_uv_fn generate_grain_uv[3];

    fgy_32x32xn_fn fgy_32x32xn;
    fguv_32x32xn_fn fguv_32x32xn[3];
} Dav1dFilmGrainDSPContext;

bitfn_decls(void dav1d_film_grain_dsp_init, Dav1dFilmGrainDSPContext *c);
bitfn_decls(void dav1d_film_grain_dsp_init_arm, Dav1dFilmGrainDSPContext *c);
bitfn_decls(void dav1d_film_grain_dsp_init_x86, Dav1dFilmGrainDSPContext *c);

#endif /* DAV1D_SRC_FILM_GRAIN_H */

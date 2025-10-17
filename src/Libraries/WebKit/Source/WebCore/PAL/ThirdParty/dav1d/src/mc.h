/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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
#ifndef DAV1D_SRC_MC_H
#define DAV1D_SRC_MC_H

#include <stdint.h>
#include <stddef.h>

#include "common/bitdepth.h"

#include "src/levels.h"

#define decl_mc_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const pixel *src, ptrdiff_t src_stride, \
            int w, int h, int mx, int my HIGHBD_DECL_SUFFIX)
typedef decl_mc_fn(*mc_fn);

#define decl_mc_scaled_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const pixel *src, ptrdiff_t src_stride, \
            int w, int h, int mx, int my, int dx, int dy HIGHBD_DECL_SUFFIX)
typedef decl_mc_scaled_fn(*mc_scaled_fn);

#define decl_warp8x8_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const pixel *src, ptrdiff_t src_stride, \
            const int16_t *abcd, int mx, int my HIGHBD_DECL_SUFFIX)
typedef decl_warp8x8_fn(*warp8x8_fn);

#define decl_mct_fn(name) \
void (name)(int16_t *tmp, const pixel *src, ptrdiff_t src_stride, \
            int w, int h, int mx, int my HIGHBD_DECL_SUFFIX)
typedef decl_mct_fn(*mct_fn);

#define decl_mct_scaled_fn(name) \
void (name)(int16_t *tmp, const pixel *src, ptrdiff_t src_stride, \
            int w, int h, int mx, int my, int dx, int dy HIGHBD_DECL_SUFFIX)
typedef decl_mct_scaled_fn(*mct_scaled_fn);

#define decl_warp8x8t_fn(name) \
void (name)(int16_t *tmp, const ptrdiff_t tmp_stride, \
            const pixel *src, ptrdiff_t src_stride, \
            const int16_t *abcd, int mx, int my HIGHBD_DECL_SUFFIX)
typedef decl_warp8x8t_fn(*warp8x8t_fn);

#define decl_avg_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const int16_t *tmp1, const int16_t *tmp2, int w, int h \
            HIGHBD_DECL_SUFFIX)
typedef decl_avg_fn(*avg_fn);

#define decl_w_avg_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const int16_t *tmp1, const int16_t *tmp2, int w, int h, int weight \
            HIGHBD_DECL_SUFFIX)
typedef decl_w_avg_fn(*w_avg_fn);

#define decl_mask_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const int16_t *tmp1, const int16_t *tmp2, int w, int h, \
            const uint8_t *mask HIGHBD_DECL_SUFFIX)
typedef decl_mask_fn(*mask_fn);

#define decl_w_mask_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const int16_t *tmp1, const int16_t *tmp2, int w, int h, \
            uint8_t *mask, int sign HIGHBD_DECL_SUFFIX)
typedef decl_w_mask_fn(*w_mask_fn);

#define decl_blend_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, const pixel *tmp, \
            int w, int h, const uint8_t *mask)
typedef decl_blend_fn(*blend_fn);

#define decl_blend_dir_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, const pixel *tmp, int w, int h)
typedef decl_blend_dir_fn(*blend_dir_fn);

#define decl_emu_edge_fn(name) \
void (name)(intptr_t bw, intptr_t bh, intptr_t iw, intptr_t ih, intptr_t x, intptr_t y, \
            pixel *dst, ptrdiff_t dst_stride, const pixel *src, ptrdiff_t src_stride)
typedef decl_emu_edge_fn(*emu_edge_fn);

#define decl_resize_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const pixel *src, ptrdiff_t src_stride, \
            int dst_w, int h, int src_w, int dx, int mx HIGHBD_DECL_SUFFIX)
typedef decl_resize_fn(*resize_fn);

typedef struct Dav1dMCDSPContext {
    mc_fn mc[N_2D_FILTERS];
    mc_scaled_fn mc_scaled[N_2D_FILTERS];
    mct_fn mct[N_2D_FILTERS];
    mct_scaled_fn mct_scaled[N_2D_FILTERS];
    avg_fn avg;
    w_avg_fn w_avg;
    mask_fn mask;
    w_mask_fn w_mask[3 /* 444, 422, 420 */];
    blend_fn blend;
    blend_dir_fn blend_v;
    blend_dir_fn blend_h;
    warp8x8_fn warp8x8;
    warp8x8t_fn warp8x8t;
    emu_edge_fn emu_edge;
    resize_fn resize;
} Dav1dMCDSPContext;

bitfn_decls(void dav1d_mc_dsp_init, Dav1dMCDSPContext *c);
bitfn_decls(void dav1d_mc_dsp_init_arm, Dav1dMCDSPContext *c);
bitfn_decls(void dav1d_mc_dsp_init_x86, Dav1dMCDSPContext *c);

#endif /* DAV1D_SRC_MC_H */

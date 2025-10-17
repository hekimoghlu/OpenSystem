/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 27, 2023.
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
#include "config.h"

#include "src/mc.h"
#include "src/cpu.h"

decl_mc_fn(BF(dav1d_put_8tap_regular, neon));
decl_mc_fn(BF(dav1d_put_8tap_regular_smooth, neon));
decl_mc_fn(BF(dav1d_put_8tap_regular_sharp, neon));
decl_mc_fn(BF(dav1d_put_8tap_smooth, neon));
decl_mc_fn(BF(dav1d_put_8tap_smooth_regular, neon));
decl_mc_fn(BF(dav1d_put_8tap_smooth_sharp, neon));
decl_mc_fn(BF(dav1d_put_8tap_sharp, neon));
decl_mc_fn(BF(dav1d_put_8tap_sharp_regular, neon));
decl_mc_fn(BF(dav1d_put_8tap_sharp_smooth, neon));
decl_mc_fn(BF(dav1d_put_bilin, neon));

decl_mct_fn(BF(dav1d_prep_8tap_regular, neon));
decl_mct_fn(BF(dav1d_prep_8tap_regular_smooth, neon));
decl_mct_fn(BF(dav1d_prep_8tap_regular_sharp, neon));
decl_mct_fn(BF(dav1d_prep_8tap_smooth, neon));
decl_mct_fn(BF(dav1d_prep_8tap_smooth_regular, neon));
decl_mct_fn(BF(dav1d_prep_8tap_smooth_sharp, neon));
decl_mct_fn(BF(dav1d_prep_8tap_sharp, neon));
decl_mct_fn(BF(dav1d_prep_8tap_sharp_regular, neon));
decl_mct_fn(BF(dav1d_prep_8tap_sharp_smooth, neon));
decl_mct_fn(BF(dav1d_prep_bilin, neon));

decl_avg_fn(BF(dav1d_avg, neon));
decl_w_avg_fn(BF(dav1d_w_avg, neon));
decl_mask_fn(BF(dav1d_mask, neon));
decl_blend_fn(BF(dav1d_blend, neon));
decl_blend_dir_fn(BF(dav1d_blend_h, neon));
decl_blend_dir_fn(BF(dav1d_blend_v, neon));

decl_w_mask_fn(BF(dav1d_w_mask_444, neon));
decl_w_mask_fn(BF(dav1d_w_mask_422, neon));
decl_w_mask_fn(BF(dav1d_w_mask_420, neon));

decl_warp8x8_fn(BF(dav1d_warp_affine_8x8, neon));
decl_warp8x8t_fn(BF(dav1d_warp_affine_8x8t, neon));

decl_emu_edge_fn(BF(dav1d_emu_edge, neon));

void bitfn(dav1d_mc_dsp_init_arm)(Dav1dMCDSPContext *const c) {
#define init_mc_fn(type, name, suffix) \
    c->mc[type] = BF(dav1d_put_##name, suffix)
#define init_mct_fn(type, name, suffix) \
    c->mct[type] = BF(dav1d_prep_##name, suffix)
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_ARM_CPU_FLAG_NEON)) return;

    init_mc_fn (FILTER_2D_8TAP_REGULAR,        8tap_regular,        neon);
    init_mc_fn (FILTER_2D_8TAP_REGULAR_SMOOTH, 8tap_regular_smooth, neon);
    init_mc_fn (FILTER_2D_8TAP_REGULAR_SHARP,  8tap_regular_sharp,  neon);
    init_mc_fn (FILTER_2D_8TAP_SMOOTH_REGULAR, 8tap_smooth_regular, neon);
    init_mc_fn (FILTER_2D_8TAP_SMOOTH,         8tap_smooth,         neon);
    init_mc_fn (FILTER_2D_8TAP_SMOOTH_SHARP,   8tap_smooth_sharp,   neon);
    init_mc_fn (FILTER_2D_8TAP_SHARP_REGULAR,  8tap_sharp_regular,  neon);
    init_mc_fn (FILTER_2D_8TAP_SHARP_SMOOTH,   8tap_sharp_smooth,   neon);
    init_mc_fn (FILTER_2D_8TAP_SHARP,          8tap_sharp,          neon);
    init_mc_fn (FILTER_2D_BILINEAR,            bilin,               neon);

    init_mct_fn(FILTER_2D_8TAP_REGULAR,        8tap_regular,        neon);
    init_mct_fn(FILTER_2D_8TAP_REGULAR_SMOOTH, 8tap_regular_smooth, neon);
    init_mct_fn(FILTER_2D_8TAP_REGULAR_SHARP,  8tap_regular_sharp,  neon);
    init_mct_fn(FILTER_2D_8TAP_SMOOTH_REGULAR, 8tap_smooth_regular, neon);
    init_mct_fn(FILTER_2D_8TAP_SMOOTH,         8tap_smooth,         neon);
    init_mct_fn(FILTER_2D_8TAP_SMOOTH_SHARP,   8tap_smooth_sharp,   neon);
    init_mct_fn(FILTER_2D_8TAP_SHARP_REGULAR,  8tap_sharp_regular,  neon);
    init_mct_fn(FILTER_2D_8TAP_SHARP_SMOOTH,   8tap_sharp_smooth,   neon);
    init_mct_fn(FILTER_2D_8TAP_SHARP,          8tap_sharp,          neon);
    init_mct_fn(FILTER_2D_BILINEAR,            bilin,               neon);

    c->avg = BF(dav1d_avg, neon);
    c->w_avg = BF(dav1d_w_avg, neon);
    c->mask = BF(dav1d_mask, neon);
    c->blend = BF(dav1d_blend, neon);
    c->blend_h = BF(dav1d_blend_h, neon);
    c->blend_v = BF(dav1d_blend_v, neon);
    c->w_mask[0] = BF(dav1d_w_mask_444, neon);
    c->w_mask[1] = BF(dav1d_w_mask_422, neon);
    c->w_mask[2] = BF(dav1d_w_mask_420, neon);
    c->warp8x8 = BF(dav1d_warp_affine_8x8, neon);
    c->warp8x8t = BF(dav1d_warp_affine_8x8t, neon);
    c->emu_edge = BF(dav1d_emu_edge, neon);
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 8, 2023.
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
#include "src/cpu.h"
#include "src/ipred.h"

decl_angular_ipred_fn(BF(dav1d_ipred_dc, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_dc_128, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_dc_top, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_dc_left, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_h, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_v, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_paeth, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_smooth, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_smooth_v, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_smooth_h, neon));
decl_angular_ipred_fn(BF(dav1d_ipred_filter, neon));

decl_cfl_pred_fn(BF(dav1d_ipred_cfl, neon));
decl_cfl_pred_fn(BF(dav1d_ipred_cfl_128, neon));
decl_cfl_pred_fn(BF(dav1d_ipred_cfl_top, neon));
decl_cfl_pred_fn(BF(dav1d_ipred_cfl_left, neon));

decl_cfl_ac_fn(BF(dav1d_ipred_cfl_ac_420, neon));
decl_cfl_ac_fn(BF(dav1d_ipred_cfl_ac_422, neon));
decl_cfl_ac_fn(BF(dav1d_ipred_cfl_ac_444, neon));

decl_pal_pred_fn(BF(dav1d_pal_pred, neon));

COLD void bitfn(dav1d_intra_pred_dsp_init_arm)(Dav1dIntraPredDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_ARM_CPU_FLAG_NEON)) return;

    c->intra_pred[DC_PRED]       = BF(dav1d_ipred_dc, neon);
    c->intra_pred[DC_128_PRED]   = BF(dav1d_ipred_dc_128, neon);
    c->intra_pred[TOP_DC_PRED]   = BF(dav1d_ipred_dc_top, neon);
    c->intra_pred[LEFT_DC_PRED]  = BF(dav1d_ipred_dc_left, neon);
    c->intra_pred[HOR_PRED]      = BF(dav1d_ipred_h, neon);
    c->intra_pred[VERT_PRED]     = BF(dav1d_ipred_v, neon);
    c->intra_pred[PAETH_PRED]    = BF(dav1d_ipred_paeth, neon);
    c->intra_pred[SMOOTH_PRED]   = BF(dav1d_ipred_smooth, neon);
    c->intra_pred[SMOOTH_V_PRED] = BF(dav1d_ipred_smooth_v, neon);
    c->intra_pred[SMOOTH_H_PRED] = BF(dav1d_ipred_smooth_h, neon);
    c->intra_pred[FILTER_PRED]   = BF(dav1d_ipred_filter, neon);

    c->cfl_pred[DC_PRED]         = BF(dav1d_ipred_cfl, neon);
    c->cfl_pred[DC_128_PRED]     = BF(dav1d_ipred_cfl_128, neon);
    c->cfl_pred[TOP_DC_PRED]     = BF(dav1d_ipred_cfl_top, neon);
    c->cfl_pred[LEFT_DC_PRED]    = BF(dav1d_ipred_cfl_left, neon);

    c->cfl_ac[DAV1D_PIXEL_LAYOUT_I420 - 1] = BF(dav1d_ipred_cfl_ac_420, neon);
    c->cfl_ac[DAV1D_PIXEL_LAYOUT_I422 - 1] = BF(dav1d_ipred_cfl_ac_422, neon);
    c->cfl_ac[DAV1D_PIXEL_LAYOUT_I444 - 1] = BF(dav1d_ipred_cfl_ac_444, neon);

    c->pal_pred                  = BF(dav1d_pal_pred, neon);
}

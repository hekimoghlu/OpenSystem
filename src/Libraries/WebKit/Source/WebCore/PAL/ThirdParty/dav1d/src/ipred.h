/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef DAV1D_SRC_IPRED_H
#define DAV1D_SRC_IPRED_H

#include <stddef.h>

#include "common/bitdepth.h"

#include "src/levels.h"

/*
 * Intra prediction.
 * - a is the angle (in degrees) for directional intra predictors. For other
 *   modes, it is ignored;
 * - topleft is the same as the argument given to dav1d_prepare_intra_edges(),
 *   see ipred_prepare.h for more detailed documentation.
 */
#define decl_angular_ipred_fn(name) \
void (name)(pixel *dst, ptrdiff_t stride, const pixel *topleft, \
            int width, int height, int angle, int max_width, int max_height \
            HIGHBD_DECL_SUFFIX)
typedef decl_angular_ipred_fn(*angular_ipred_fn);

/*
 * Create a subsampled Y plane with the DC subtracted.
 * - w/h_pad is the edge of the width/height that extends outside the visible
 *   portion of the frame in 4px units;
 * - ac has a stride of 16.
 */
#define decl_cfl_ac_fn(name) \
void (name)(int16_t *ac, const pixel *y, ptrdiff_t stride, \
            int w_pad, int h_pad, int cw, int ch)
typedef decl_cfl_ac_fn(*cfl_ac_fn);

/*
 * dst[x,y] += alpha * ac[x,y]
 * - alpha contains a q3 scalar in [-16,16] range;
 */
#define decl_cfl_pred_fn(name) \
void (name)(pixel *dst, ptrdiff_t stride, const pixel *topleft, \
            int width, int height, const int16_t *ac, int alpha \
            HIGHBD_DECL_SUFFIX)
typedef decl_cfl_pred_fn(*cfl_pred_fn);

/*
 * dst[x,y] = pal[idx[x,y]]
 * - palette indices are [0-7]
 * - only 16-byte alignment is guaranteed for idx.
 */
#define decl_pal_pred_fn(name) \
void (name)(pixel *dst, ptrdiff_t stride, const uint16_t *pal, \
            const uint8_t *idx, int w, int h)
typedef decl_pal_pred_fn(*pal_pred_fn);

typedef struct Dav1dIntraPredDSPContext {
    angular_ipred_fn intra_pred[N_IMPL_INTRA_PRED_MODES];

    // chroma-from-luma
    cfl_ac_fn cfl_ac[3 /* 420, 422, 444 */];
    cfl_pred_fn cfl_pred[DC_128_PRED + 1];

    // palette
    pal_pred_fn pal_pred;
} Dav1dIntraPredDSPContext;

bitfn_decls(void dav1d_intra_pred_dsp_init, Dav1dIntraPredDSPContext *c);
bitfn_decls(void dav1d_intra_pred_dsp_init_arm, Dav1dIntraPredDSPContext *c);
bitfn_decls(void dav1d_intra_pred_dsp_init_x86, Dav1dIntraPredDSPContext *c);

#endif /* DAV1D_SRC_IPRED_H */

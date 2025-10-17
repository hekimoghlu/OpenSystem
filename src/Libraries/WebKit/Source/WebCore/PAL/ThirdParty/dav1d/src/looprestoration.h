/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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
#ifndef DAV1D_SRC_LOOPRESTORATION_H
#define DAV1D_SRC_LOOPRESTORATION_H

#include <stdint.h>
#include <stddef.h>

#include "common/bitdepth.h"

enum LrEdgeFlags {
    LR_HAVE_LEFT = 1 << 0,
    LR_HAVE_RIGHT = 1 << 1,
    LR_HAVE_TOP = 1 << 2,
    LR_HAVE_BOTTOM = 1 << 3,
};

#ifdef BITDEPTH
typedef const pixel (*const_left_pixel_row)[4];
#else
typedef const void *const_left_pixel_row;
#endif

typedef union LooprestorationParams {
    ALIGN(int16_t filter[2][8], 16);
    struct {
        uint32_t s0, s1;
        int16_t w0, w1;
    } sgr;
} LooprestorationParams;

// Although the spec applies restoration filters over 4x4 blocks,
// they can be applied to a bigger surface.
//    * w is constrained by the restoration unit size (w <= 256)
//    * h is constrained by the stripe height (h <= 64)
// The filter functions are allowed to do aligned writes past the right
// edge of the buffer, aligned up to the minimum loop restoration unit size
// (which is 32 pixels for subsampled chroma and 64 pixels for luma).
#define decl_lr_filter_fn(name) \
void (name)(pixel *dst, ptrdiff_t dst_stride, \
            const_left_pixel_row left, \
            const pixel *lpf, int w, int h, \
            const LooprestorationParams *params, \
            enum LrEdgeFlags edges HIGHBD_DECL_SUFFIX)
typedef decl_lr_filter_fn(*looprestorationfilter_fn);

typedef struct Dav1dLoopRestorationDSPContext {
    looprestorationfilter_fn wiener[2]; /* 7-tap, 5-tap */
    looprestorationfilter_fn sgr[3]; /* 5x5, 3x3, mix */
} Dav1dLoopRestorationDSPContext;

bitfn_decls(void dav1d_loop_restoration_dsp_init, Dav1dLoopRestorationDSPContext *c, int bpc);
bitfn_decls(void dav1d_loop_restoration_dsp_init_arm, Dav1dLoopRestorationDSPContext *c, int bpc);
bitfn_decls(void dav1d_loop_restoration_dsp_init_x86, Dav1dLoopRestorationDSPContext *c, int bpc);
bitfn_decls(void dav1d_loop_restoration_dsp_init_ppc, Dav1dLoopRestorationDSPContext *c, int bpc);

#endif /* DAV1D_SRC_LOOPRESTORATION_H */

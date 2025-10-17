/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 20, 2022.
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
#ifndef DAV1D_SRC_LOOPFILTER_H
#define DAV1D_SRC_LOOPFILTER_H

#include <stdint.h>
#include <stddef.h>

#include "common/bitdepth.h"

#include "src/levels.h"
#include "src/lf_mask.h"

#define decl_loopfilter_sb_fn(name) \
void (name)(pixel *dst, ptrdiff_t stride, const uint32_t *mask, \
            const uint8_t (*lvl)[4], ptrdiff_t lvl_stride, \
            const Av1FilterLUT *lut, int w HIGHBD_DECL_SUFFIX)
typedef decl_loopfilter_sb_fn(*loopfilter_sb_fn);

typedef struct Dav1dLoopFilterDSPContext {
    /*
     * dimension 1: plane (0=luma, 1=chroma)
     * dimension 2: 0=col-edge filter (h), 1=row-edge filter (v)
     *
     * dst/stride are aligned by 32
     */
    loopfilter_sb_fn loop_filter_sb[2][2];
} Dav1dLoopFilterDSPContext;

bitfn_decls(void dav1d_loop_filter_dsp_init, Dav1dLoopFilterDSPContext *c);
bitfn_decls(void dav1d_loop_filter_dsp_init_arm, Dav1dLoopFilterDSPContext *c);
bitfn_decls(void dav1d_loop_filter_dsp_init_x86, Dav1dLoopFilterDSPContext *c);

#endif /* DAV1D_SRC_LOOPFILTER_H */

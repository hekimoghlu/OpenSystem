/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 11, 2022.
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
#ifndef DAV1D_SRC_CDEF_H
#define DAV1D_SRC_CDEF_H

#include <stddef.h>
#include <stdint.h>

#include "common/bitdepth.h"

enum CdefEdgeFlags {
    CDEF_HAVE_LEFT = 1 << 0,
    CDEF_HAVE_RIGHT = 1 << 1,
    CDEF_HAVE_TOP = 1 << 2,
    CDEF_HAVE_BOTTOM = 1 << 3,
};

#ifdef BITDEPTH
typedef const pixel (*const_left_pixel_row_2px)[2];
#else
typedef const void *const_left_pixel_row_2px;
#endif

// CDEF operates entirely on pre-filter data; if bottom/right edges are
// present (according to $edges), then the pre-filter data is located in
// $dst. However, the edge pixels above $dst may be post-filter, so in
// order to get access to pre-filter top pixels, use $top.
#define decl_cdef_fn(name) \
void (name)(pixel *dst, ptrdiff_t stride, const_left_pixel_row_2px left, \
            const pixel *top, const pixel *bottom, \
            int pri_strength, int sec_strength, \
            int dir, int damping, enum CdefEdgeFlags edges HIGHBD_DECL_SUFFIX)
typedef decl_cdef_fn(*cdef_fn);

#define decl_cdef_dir_fn(name) \
int (name)(const pixel *dst, ptrdiff_t dst_stride, unsigned *var HIGHBD_DECL_SUFFIX)
typedef decl_cdef_dir_fn(*cdef_dir_fn);

typedef struct Dav1dCdefDSPContext {
    cdef_dir_fn dir;
    cdef_fn fb[3 /* 444/luma, 422, 420 */];
} Dav1dCdefDSPContext;

bitfn_decls(void dav1d_cdef_dsp_init, Dav1dCdefDSPContext *c);
bitfn_decls(void dav1d_cdef_dsp_init_arm, Dav1dCdefDSPContext *c);
bitfn_decls(void dav1d_cdef_dsp_init_ppc, Dav1dCdefDSPContext *c);
bitfn_decls(void dav1d_cdef_dsp_init_x86, Dav1dCdefDSPContext *c);

#endif /* DAV1D_SRC_CDEF_H */

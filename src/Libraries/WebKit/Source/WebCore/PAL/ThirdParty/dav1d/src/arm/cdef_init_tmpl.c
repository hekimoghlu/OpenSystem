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
#include "src/cpu.h"
#include "src/cdef.h"

decl_cdef_dir_fn(BF(dav1d_cdef_find_dir, neon));

void BF(dav1d_cdef_padding4, neon)(uint16_t *tmp, const pixel *src,
                                   ptrdiff_t src_stride, const pixel (*left)[2],
                                   const pixel *const top,
                                   const pixel *const bottom, int h,
                                   enum CdefEdgeFlags edges);
void BF(dav1d_cdef_padding8, neon)(uint16_t *tmp, const pixel *src,
                                   ptrdiff_t src_stride, const pixel (*left)[2],
                                   const pixel *const top,
                                   const pixel *const bottom, int h,
                                   enum CdefEdgeFlags edges);

// Passing edges to this function, to allow it to switch to a more
// optimized version for fully edged cases. Using size_t for edges,
// to avoid ABI differences for passing more than one argument on the stack.
void BF(dav1d_cdef_filter4, neon)(pixel *dst, ptrdiff_t dst_stride,
                                  const uint16_t *tmp, int pri_strength,
                                  int sec_strength, int dir, int damping, int h,
                                  size_t edges HIGHBD_DECL_SUFFIX);
void BF(dav1d_cdef_filter8, neon)(pixel *dst, ptrdiff_t dst_stride,
                                  const uint16_t *tmp, int pri_strength,
                                  int sec_strength, int dir, int damping, int h,
                                  size_t edges HIGHBD_DECL_SUFFIX);

#define DEFINE_FILTER(w, h, tmp_stride)                                      \
static void                                                                  \
cdef_filter_##w##x##h##_neon(pixel *dst, const ptrdiff_t stride,             \
                             const pixel (*left)[2],                         \
                             const pixel *const top,                         \
                             const pixel *const bottom,                      \
                             const int pri_strength, const int sec_strength, \
                             const int dir, const int damping,               \
                             const enum CdefEdgeFlags edges                  \
                             HIGHBD_DECL_SUFFIX)                             \
{                                                                            \
    ALIGN_STK_16(uint16_t, tmp_buf, 12 * tmp_stride + 8,);                   \
    uint16_t *tmp = tmp_buf + 2 * tmp_stride + 8;                            \
    BF(dav1d_cdef_padding##w, neon)(tmp, dst, stride,                        \
                                    left, top, bottom, h, edges);            \
    BF(dav1d_cdef_filter##w, neon)(dst, stride, tmp, pri_strength,           \
                                   sec_strength, dir, damping, h, edges      \
                                   HIGHBD_TAIL_SUFFIX);                      \
}

DEFINE_FILTER(8, 8, 16)
DEFINE_FILTER(4, 8, 8)
DEFINE_FILTER(4, 4, 8)


COLD void bitfn(dav1d_cdef_dsp_init_arm)(Dav1dCdefDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_ARM_CPU_FLAG_NEON)) return;

    c->dir = BF(dav1d_cdef_find_dir, neon);
    c->fb[0] = cdef_filter_8x8_neon;
    c->fb[1] = cdef_filter_4x8_neon;
    c->fb[2] = cdef_filter_4x4_neon;
}

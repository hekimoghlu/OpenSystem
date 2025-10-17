/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#include "src/filmgrain.h"

#define decl_fg_fns(ext)                                         \
decl_generate_grain_y_fn(BF(dav1d_generate_grain_y, ext));       \
decl_generate_grain_uv_fn(BF(dav1d_generate_grain_uv_420, ext)); \
decl_generate_grain_uv_fn(BF(dav1d_generate_grain_uv_422, ext)); \
decl_generate_grain_uv_fn(BF(dav1d_generate_grain_uv_444, ext)); \
decl_fgy_32x32xn_fn(BF(dav1d_fgy_32x32xn, ext));                 \
decl_fguv_32x32xn_fn(BF(dav1d_fguv_32x32xn_i420, ext));          \
decl_fguv_32x32xn_fn(BF(dav1d_fguv_32x32xn_i422, ext));          \
decl_fguv_32x32xn_fn(BF(dav1d_fguv_32x32xn_i444, ext))

decl_fg_fns(ssse3);
decl_fg_fns(avx2);
decl_fg_fns(avx512icl);

COLD void bitfn(dav1d_film_grain_dsp_init_x86)(Dav1dFilmGrainDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_X86_CPU_FLAG_SSSE3)) return;

    c->generate_grain_y = BF(dav1d_generate_grain_y, ssse3);
    c->generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420 - 1] = BF(dav1d_generate_grain_uv_420, ssse3);
    c->fgy_32x32xn = BF(dav1d_fgy_32x32xn, ssse3);
    c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420 - 1] = BF(dav1d_fguv_32x32xn_i420, ssse3);
    c->generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422 - 1] = BF(dav1d_generate_grain_uv_422, ssse3);
    c->generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444 - 1] = BF(dav1d_generate_grain_uv_444, ssse3);
    c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422 - 1] = BF(dav1d_fguv_32x32xn_i422, ssse3);
    c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444 - 1] = BF(dav1d_fguv_32x32xn_i444, ssse3);

#if ARCH_X86_64
    if (!(flags & DAV1D_X86_CPU_FLAG_AVX2)) return;

    c->generate_grain_y = BF(dav1d_generate_grain_y, avx2);
    c->generate_grain_uv[DAV1D_PIXEL_LAYOUT_I420 - 1] = BF(dav1d_generate_grain_uv_420, avx2);
    c->generate_grain_uv[DAV1D_PIXEL_LAYOUT_I422 - 1] = BF(dav1d_generate_grain_uv_422, avx2);
    c->generate_grain_uv[DAV1D_PIXEL_LAYOUT_I444 - 1] = BF(dav1d_generate_grain_uv_444, avx2);

    if (!(flags & DAV1D_X86_CPU_FLAG_SLOW_GATHER)) {
        c->fgy_32x32xn = BF(dav1d_fgy_32x32xn, avx2);
        c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420 - 1] = BF(dav1d_fguv_32x32xn_i420, avx2);
        c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422 - 1] = BF(dav1d_fguv_32x32xn_i422, avx2);
        c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444 - 1] = BF(dav1d_fguv_32x32xn_i444, avx2);
    }

    if (!(flags & DAV1D_X86_CPU_FLAG_AVX512ICL)) return;

    c->fgy_32x32xn = BF(dav1d_fgy_32x32xn, avx512icl);
    c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I420 - 1] = BF(dav1d_fguv_32x32xn_i420, avx512icl);
    c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I422 - 1] = BF(dav1d_fguv_32x32xn_i422, avx512icl);
    c->fguv_32x32xn[DAV1D_PIXEL_LAYOUT_I444 - 1] = BF(dav1d_fguv_32x32xn_i444, avx512icl);
#endif
}

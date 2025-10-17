/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
#include "src/loopfilter.h"

#define decl_loopfilter_sb_fns(ext) \
decl_loopfilter_sb_fn(BF(dav1d_lpf_h_sb_y, ext)); \
decl_loopfilter_sb_fn(BF(dav1d_lpf_v_sb_y, ext)); \
decl_loopfilter_sb_fn(BF(dav1d_lpf_h_sb_uv, ext)); \
decl_loopfilter_sb_fn(BF(dav1d_lpf_v_sb_uv, ext))

decl_loopfilter_sb_fns(ssse3);
decl_loopfilter_sb_fns(avx2);
decl_loopfilter_sb_fns(avx512icl);

COLD void bitfn(dav1d_loop_filter_dsp_init_x86)(Dav1dLoopFilterDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_X86_CPU_FLAG_SSSE3)) return;

    c->loop_filter_sb[0][0] = BF(dav1d_lpf_h_sb_y, ssse3);
    c->loop_filter_sb[0][1] = BF(dav1d_lpf_v_sb_y, ssse3);
    c->loop_filter_sb[1][0] = BF(dav1d_lpf_h_sb_uv, ssse3);
    c->loop_filter_sb[1][1] = BF(dav1d_lpf_v_sb_uv, ssse3);

#if ARCH_X86_64
    if (!(flags & DAV1D_X86_CPU_FLAG_AVX2)) return;

    c->loop_filter_sb[0][0] = BF(dav1d_lpf_h_sb_y, avx2);
    c->loop_filter_sb[0][1] = BF(dav1d_lpf_v_sb_y, avx2);
    c->loop_filter_sb[1][0] = BF(dav1d_lpf_h_sb_uv, avx2);
    c->loop_filter_sb[1][1] = BF(dav1d_lpf_v_sb_uv, avx2);

    if (!(flags & DAV1D_X86_CPU_FLAG_AVX512ICL)) return;

#if BITDEPTH == 8
    c->loop_filter_sb[0][0] = BF(dav1d_lpf_h_sb_y, avx512icl);
    c->loop_filter_sb[0][1] = BF(dav1d_lpf_v_sb_y, avx512icl);
    c->loop_filter_sb[1][0] = BF(dav1d_lpf_h_sb_uv, avx512icl);
    c->loop_filter_sb[1][1] = BF(dav1d_lpf_v_sb_uv, avx512icl);
#endif
#endif
}

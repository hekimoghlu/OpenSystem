/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 22, 2025.
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
#include "src/looprestoration.h"

#include "common/intops.h"

#define decl_wiener_filter_fns(ext) \
decl_lr_filter_fn(BF(dav1d_wiener_filter7, ext)); \
decl_lr_filter_fn(BF(dav1d_wiener_filter5, ext))

#define decl_sgr_filter_fns(ext) \
decl_lr_filter_fn(BF(dav1d_sgr_filter_5x5, ext)); \
decl_lr_filter_fn(BF(dav1d_sgr_filter_3x3, ext)); \
decl_lr_filter_fn(BF(dav1d_sgr_filter_mix, ext))

decl_wiener_filter_fns(sse2);
decl_wiener_filter_fns(ssse3);
decl_wiener_filter_fns(avx2);
decl_wiener_filter_fns(avx512icl);
decl_sgr_filter_fns(ssse3);
decl_sgr_filter_fns(avx2);
decl_sgr_filter_fns(avx512icl);

COLD void bitfn(dav1d_loop_restoration_dsp_init_x86)(Dav1dLoopRestorationDSPContext *const c,
                                                     const int bpc)
{
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_X86_CPU_FLAG_SSE2)) return;
#if BITDEPTH == 8
    c->wiener[0] = BF(dav1d_wiener_filter7, sse2);
    c->wiener[1] = BF(dav1d_wiener_filter5, sse2);
#endif

    if (!(flags & DAV1D_X86_CPU_FLAG_SSSE3)) return;
    c->wiener[0] = BF(dav1d_wiener_filter7, ssse3);
    c->wiener[1] = BF(dav1d_wiener_filter5, ssse3);
    if (bpc <= 10) {
        c->sgr[0] = BF(dav1d_sgr_filter_5x5, ssse3);
        c->sgr[1] = BF(dav1d_sgr_filter_3x3, ssse3);
        c->sgr[2] = BF(dav1d_sgr_filter_mix, ssse3);
    }

#if ARCH_X86_64
    if (!(flags & DAV1D_X86_CPU_FLAG_AVX2)) return;

    c->wiener[0] = BF(dav1d_wiener_filter7, avx2);
    c->wiener[1] = BF(dav1d_wiener_filter5, avx2);
    if (bpc <= 10) {
        c->sgr[0] = BF(dav1d_sgr_filter_5x5, avx2);
        c->sgr[1] = BF(dav1d_sgr_filter_3x3, avx2);
        c->sgr[2] = BF(dav1d_sgr_filter_mix, avx2);
    }

    if (!(flags & DAV1D_X86_CPU_FLAG_AVX512ICL)) return;

    c->wiener[0] = BF(dav1d_wiener_filter7, avx512icl);
#if BITDEPTH == 8
    /* With VNNI we don't need a 5-tap version. */
    c->wiener[1] = c->wiener[0];
#else
    c->wiener[1] = BF(dav1d_wiener_filter5, avx512icl);
#endif
    if (bpc <= 10) {
        c->sgr[0] = BF(dav1d_sgr_filter_5x5, avx512icl);
        c->sgr[1] = BF(dav1d_sgr_filter_3x3, avx512icl);
        c->sgr[2] = BF(dav1d_sgr_filter_mix, avx512icl);
    }
#endif
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#define decl_cdef_fns(ext) \
    decl_cdef_fn(BF(dav1d_cdef_filter_4x4, ext)); \
    decl_cdef_fn(BF(dav1d_cdef_filter_4x8, ext)); \
    decl_cdef_fn(BF(dav1d_cdef_filter_8x8, ext))

decl_cdef_fns(avx512icl);
decl_cdef_fns(avx2);
decl_cdef_fns(sse4);
decl_cdef_fns(ssse3);
decl_cdef_fns(sse2);

decl_cdef_dir_fn(BF(dav1d_cdef_dir, avx2));
decl_cdef_dir_fn(BF(dav1d_cdef_dir, sse4));
decl_cdef_dir_fn(BF(dav1d_cdef_dir, ssse3));

COLD void bitfn(dav1d_cdef_dsp_init_x86)(Dav1dCdefDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

#if BITDEPTH == 8
    if (!(flags & DAV1D_X86_CPU_FLAG_SSE2)) return;

    c->fb[0] = BF(dav1d_cdef_filter_8x8, sse2);
    c->fb[1] = BF(dav1d_cdef_filter_4x8, sse2);
    c->fb[2] = BF(dav1d_cdef_filter_4x4, sse2);
#endif

    if (!(flags & DAV1D_X86_CPU_FLAG_SSSE3)) return;

    c->dir = BF(dav1d_cdef_dir, ssse3);
    c->fb[0] = BF(dav1d_cdef_filter_8x8, ssse3);
    c->fb[1] = BF(dav1d_cdef_filter_4x8, ssse3);
    c->fb[2] = BF(dav1d_cdef_filter_4x4, ssse3);

    if (!(flags & DAV1D_X86_CPU_FLAG_SSE41)) return;

    c->dir = BF(dav1d_cdef_dir, sse4);
#if BITDEPTH == 8
    c->fb[0] = BF(dav1d_cdef_filter_8x8, sse4);
    c->fb[1] = BF(dav1d_cdef_filter_4x8, sse4);
    c->fb[2] = BF(dav1d_cdef_filter_4x4, sse4);
#endif

#if ARCH_X86_64
    if (!(flags & DAV1D_X86_CPU_FLAG_AVX2)) return;

    c->dir = BF(dav1d_cdef_dir, avx2);
    c->fb[0] = BF(dav1d_cdef_filter_8x8, avx2);
    c->fb[1] = BF(dav1d_cdef_filter_4x8, avx2);
    c->fb[2] = BF(dav1d_cdef_filter_4x4, avx2);

    if (!(flags & DAV1D_X86_CPU_FLAG_AVX512ICL)) return;

#if BITDEPTH == 8
    c->fb[0] = BF(dav1d_cdef_filter_8x8, avx512icl);
    c->fb[1] = BF(dav1d_cdef_filter_4x8, avx512icl);
    c->fb[2] = BF(dav1d_cdef_filter_4x4, avx512icl);
#endif
#endif
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 30, 2022.
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
#include "src/refmvs.h"

decl_splat_mv_fn(dav1d_splat_mv_sse2);
decl_splat_mv_fn(dav1d_splat_mv_avx2);
decl_splat_mv_fn(dav1d_splat_mv_avx512icl);

COLD void dav1d_refmvs_dsp_init_x86(Dav1dRefmvsDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_X86_CPU_FLAG_SSE2)) return;

    c->splat_mv = dav1d_splat_mv_sse2;

#if ARCH_X86_64
    if (!(flags & DAV1D_X86_CPU_FLAG_AVX2)) return;

    c->splat_mv = dav1d_splat_mv_avx2;

    if (!(flags & DAV1D_X86_CPU_FLAG_AVX512ICL)) return;

    c->splat_mv = dav1d_splat_mv_avx512icl;
#endif
}

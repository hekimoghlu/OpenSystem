/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "tests/checkasm/checkasm.h"
#include "src/refmvs.h"

static void check_splat_mv(const Dav1dRefmvsDSPContext *const c) {
    ALIGN_STK_64(refmvs_block, c_buf, 32 * 32,);
    ALIGN_STK_64(refmvs_block, a_buf, 32 * 32,);
    refmvs_block *c_dst[32];
    refmvs_block *a_dst[32];
    const size_t stride = 32 * sizeof(refmvs_block);

    for (int i = 0; i < 32; i++) {
        c_dst[i] = c_buf + 32 * i;
        a_dst[i] = a_buf + 32 * i;
    }

    declare_func(void, refmvs_block **rr, const refmvs_block *rmv,
                 int bx4, int bw4, int bh4);

    for (int w = 1; w <= 32; w *= 2) {
        if (check_func(c->splat_mv, "splat_mv_w%d", w)) {
            const int h_min = imax(w / 4, 1);
            const int h_max = imin(w * 4, 32);
            const int w_uint32 = w * sizeof(refmvs_block) / sizeof(uint32_t);
            for (int h = h_min; h <= h_max; h *= 2) {
                const int offset = (w * rnd()) & 31;
                union {
                    refmvs_block rmv;
                    uint32_t u32[3];
                } ALIGN(tmp, 16);
                tmp.u32[0] = rnd();
                tmp.u32[1] = rnd();
                tmp.u32[2] = rnd();

                call_ref(c_dst, &tmp.rmv, offset, w, h);
                call_new(a_dst, &tmp.rmv, offset, w, h);
                checkasm_check(uint32_t, (uint32_t*)(c_buf + offset), stride,
                                         (uint32_t*)(a_buf + offset), stride,
                                         w_uint32, h, "dst");

                bench_new(a_dst, &tmp.rmv, 0, w, h);
            }
        }
    }
    report("splat_mv");
}

void checkasm_check_refmvs(void) {
    Dav1dRefmvsDSPContext c;
    dav1d_refmvs_dsp_init(&c);

    check_splat_mv(&c);
}

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
#include "tests/checkasm/checkasm.h"

#include <string.h>
#include <stdio.h>

#include "common/dump.h"

#include "src/levels.h"
#include "src/cdef.h"

static int to_binary(int x) { /* 0-15 -> 0000-1111 */
    return (x & 1) + 5 * (x & 2) + 25 * (x & 4) + 125 * (x & 8);
}

static void init_tmp(pixel *buf, int n, const int bitdepth_max) {
    const int fill_type = rnd() & 7;
    if (fill_type == 0)
        while (n--) /* check for cdef_filter underflows */
            *buf++ = rnd() & 1;
    else if (fill_type == 1)
        while (n--) /* check for cdef_filter overflows */
            *buf++ = bitdepth_max - (rnd() & 1);
    else
        while (n--)
            *buf++ = rnd() & bitdepth_max;
}

static void check_cdef_filter(const cdef_fn fn, const int w, const int h) {
    ALIGN_STK_64(pixel, c_src,   16 * 10 + 16, ), *const c_dst = c_src + 8;
    ALIGN_STK_64(pixel, a_src,   16 * 10 + 16, ), *const a_dst = a_src + 8;
    ALIGN_STK_64(pixel, top_buf, 16 *  2 + 16, ), *const top = top_buf + 8;
    ALIGN_STK_64(pixel, bot_buf, 16 *  2 + 16, ), *const bot = bot_buf + 8;
    ALIGN_STK_16(pixel, left, 8,[2]);
    const ptrdiff_t stride = 16 * sizeof(pixel);

    declare_func(void, pixel *dst, ptrdiff_t dst_stride, const pixel (*left)[2],
                 const pixel *top, const pixel *bot, int pri_strength,
                 int sec_strength, int dir, int damping,
                 enum CdefEdgeFlags edges HIGHBD_DECL_SUFFIX);

    if (check_func(fn, "cdef_filter_%dx%d_%dbpc", w, h, BITDEPTH)) {
        for (int dir = 0; dir < 8; dir++) {
            for (enum CdefEdgeFlags edges = 0x0; edges <= 0xf; edges++) {
#if BITDEPTH == 16
                const int bitdepth_max = rnd() & 1 ? 0x3ff : 0xfff;
#else
                const int bitdepth_max = 0xff;
#endif
                const int bitdepth_min_8 = bitdepth_from_max(bitdepth_max) - 8;

                init_tmp(c_src, 16 * 10 + 16, bitdepth_max);
                init_tmp(top_buf, 16 * 2 + 16, bitdepth_max);
                init_tmp(bot_buf, 16 * 2 + 16, bitdepth_max);
                init_tmp((pixel *) left, 8 * 2, bitdepth_max);
                memcpy(a_src, c_src, (16 * 10 + 16) * sizeof(pixel));

                const int lvl = 1 + (rnd() % 62);
                const int damping = 3 + (rnd() & 3) + bitdepth_min_8 - (w == 4 || (rnd() & 1));
                int pri_strength = (lvl >> 2) << bitdepth_min_8;
                int sec_strength = lvl & 3;
                sec_strength += sec_strength == 3;
                sec_strength <<= bitdepth_min_8;
                call_ref(c_dst, stride, left, top, bot, pri_strength, sec_strength,
                         dir, damping, edges HIGHBD_TAIL_SUFFIX);
                call_new(a_dst, stride, left, top, bot, pri_strength, sec_strength,
                         dir, damping, edges HIGHBD_TAIL_SUFFIX);
                if (checkasm_check_pixel(c_dst, stride, a_dst, stride, w, h, "dst")) {
                    fprintf(stderr, "strength = %d:%d, dir = %d, damping = %d, edges = %04d\n",
                            pri_strength, sec_strength, dir, damping, to_binary(edges));
                    return;
                }
                if (dir == 7 && (edges == 0x5 || edges == 0xa || edges == 0xf)) {
                    /* Benchmark a fixed set of cases to get consistent results:
                     *  1) top/left edges and pri_strength only
                     *  2) bottom/right edges and sec_strength only
                     *  3) all edges and both pri_strength and sec_strength
                     */
                    pri_strength = (edges & 1) << bitdepth_min_8;
                    sec_strength = (edges & 2) << bitdepth_min_8;
                    bench_new(a_dst, stride, left, top, bot, pri_strength,
                              sec_strength, dir, damping, edges HIGHBD_TAIL_SUFFIX);
                }
            }
        }
    }
}

static void check_cdef_direction(const cdef_dir_fn fn) {
    ALIGN_STK_64(pixel, src, 8 * 8,);

    declare_func(int, pixel *src, ptrdiff_t dst_stride, unsigned *var
                 HIGHBD_DECL_SUFFIX);

    if (check_func(fn, "cdef_dir_%dbpc", BITDEPTH)) {
        unsigned c_var, a_var;
#if BITDEPTH == 16
        const int bitdepth_max = rnd() & 1 ? 0x3ff : 0xfff;
#else
        const int bitdepth_max = 0xff;
#endif
        init_tmp(src, 64, bitdepth_max);

        const int c_dir = call_ref(src, 8 * sizeof(pixel), &c_var HIGHBD_TAIL_SUFFIX);
        const int a_dir = call_new(src, 8 * sizeof(pixel), &a_var HIGHBD_TAIL_SUFFIX);
        if (c_var != a_var || c_dir != a_dir) {
            if (fail()) {
                hex_fdump(stderr, src, 8 * sizeof(pixel), 8, 8, "src");
                fprintf(stderr, "c_dir %d a_dir %d\n", c_dir, a_dir);
            }
        }
        bench_new(src, 8 * sizeof(pixel), &a_var HIGHBD_TAIL_SUFFIX);
    }
    report("cdef_dir");
}

void bitfn(checkasm_check_cdef)(void) {
    Dav1dCdefDSPContext c;
    bitfn(dav1d_cdef_dsp_init)(&c);

    check_cdef_direction(c.dir);

    check_cdef_filter(c.fb[0], 8, 8);
    check_cdef_filter(c.fb[1], 4, 8);
    check_cdef_filter(c.fb[2], 4, 4);
    report("cdef_filter");
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 14, 2023.
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

decl_loopfilter_sb_fn(BF(dav1d_lpf_h_sb_y, neon));
decl_loopfilter_sb_fn(BF(dav1d_lpf_v_sb_y, neon));
decl_loopfilter_sb_fn(BF(dav1d_lpf_h_sb_uv, neon));
decl_loopfilter_sb_fn(BF(dav1d_lpf_v_sb_uv, neon));

COLD void bitfn(dav1d_loop_filter_dsp_init_arm)(Dav1dLoopFilterDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_ARM_CPU_FLAG_NEON)) return;

    c->loop_filter_sb[0][0] = BF(dav1d_lpf_h_sb_y, neon);
    c->loop_filter_sb[0][1] = BF(dav1d_lpf_v_sb_y, neon);
    c->loop_filter_sb[1][0] = BF(dav1d_lpf_h_sb_uv, neon);
    c->loop_filter_sb[1][1] = BF(dav1d_lpf_v_sb_uv, neon);
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 6, 2025.
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
#include "src/ipred.h"

#define decl_fn(type, name) \
    decl_##type##_fn(BF(dav1d_##name, ssse3)); \
    decl_##type##_fn(BF(dav1d_##name, avx2)); \
    decl_##type##_fn(BF(dav1d_##name, avx512icl))
#define init_fn(type0, type1, name, suffix) \
    c->type0[type1] = BF(dav1d_##name, suffix)

#define init_angular_ipred_fn(type, name, suffix) \
    init_fn(intra_pred, type, name, suffix)
#define init_cfl_pred_fn(type, name, suffix) \
    init_fn(cfl_pred, type, name, suffix)
#define init_cfl_ac_fn(type, name, suffix) \
    init_fn(cfl_ac, type, name, suffix)

decl_fn(angular_ipred, ipred_dc);
decl_fn(angular_ipred, ipred_dc_128);
decl_fn(angular_ipred, ipred_dc_top);
decl_fn(angular_ipred, ipred_dc_left);
decl_fn(angular_ipred, ipred_h);
decl_fn(angular_ipred, ipred_v);
decl_fn(angular_ipred, ipred_paeth);
decl_fn(angular_ipred, ipred_smooth);
decl_fn(angular_ipred, ipred_smooth_h);
decl_fn(angular_ipred, ipred_smooth_v);
decl_fn(angular_ipred, ipred_z1);
decl_fn(angular_ipred, ipred_z2);
decl_fn(angular_ipred, ipred_z3);
decl_fn(angular_ipred, ipred_filter);

decl_fn(cfl_pred, ipred_cfl);
decl_fn(cfl_pred, ipred_cfl_128);
decl_fn(cfl_pred, ipred_cfl_top);
decl_fn(cfl_pred, ipred_cfl_left);

decl_fn(cfl_ac, ipred_cfl_ac_420);
decl_fn(cfl_ac, ipred_cfl_ac_422);
decl_fn(cfl_ac, ipred_cfl_ac_444);

decl_fn(pal_pred, pal_pred);

COLD void bitfn(dav1d_intra_pred_dsp_init_x86)(Dav1dIntraPredDSPContext *const c) {
    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_X86_CPU_FLAG_SSSE3)) return;

    init_angular_ipred_fn(DC_PRED,       ipred_dc,       ssse3);
    init_angular_ipred_fn(DC_128_PRED,   ipred_dc_128,   ssse3);
    init_angular_ipred_fn(TOP_DC_PRED,   ipred_dc_top,   ssse3);
    init_angular_ipred_fn(LEFT_DC_PRED,  ipred_dc_left,  ssse3);
    init_angular_ipred_fn(HOR_PRED,      ipred_h,        ssse3);
    init_angular_ipred_fn(VERT_PRED,     ipred_v,        ssse3);
    init_angular_ipred_fn(PAETH_PRED,    ipred_paeth,    ssse3);
    init_angular_ipred_fn(SMOOTH_PRED,   ipred_smooth,   ssse3);
    init_angular_ipred_fn(SMOOTH_H_PRED, ipred_smooth_h, ssse3);
    init_angular_ipred_fn(SMOOTH_V_PRED, ipred_smooth_v, ssse3);
    init_angular_ipred_fn(FILTER_PRED,   ipred_filter,   ssse3);

    init_cfl_pred_fn(DC_PRED,      ipred_cfl,      ssse3);
    init_cfl_pred_fn(DC_128_PRED,  ipred_cfl_128,  ssse3);
    init_cfl_pred_fn(TOP_DC_PRED,  ipred_cfl_top,  ssse3);
    init_cfl_pred_fn(LEFT_DC_PRED, ipred_cfl_left, ssse3);

    init_cfl_ac_fn(DAV1D_PIXEL_LAYOUT_I420 - 1, ipred_cfl_ac_420, ssse3);
    init_cfl_ac_fn(DAV1D_PIXEL_LAYOUT_I422 - 1, ipred_cfl_ac_422, ssse3);
    init_cfl_ac_fn(DAV1D_PIXEL_LAYOUT_I444 - 1, ipred_cfl_ac_444, ssse3);

    c->pal_pred = BF(dav1d_pal_pred, ssse3);

#if ARCH_X86_64
    if (!(flags & DAV1D_X86_CPU_FLAG_AVX2)) return;

    init_angular_ipred_fn(DC_PRED,       ipred_dc,       avx2);
    init_angular_ipred_fn(DC_128_PRED,   ipred_dc_128,   avx2);
    init_angular_ipred_fn(TOP_DC_PRED,   ipred_dc_top,   avx2);
    init_angular_ipred_fn(LEFT_DC_PRED,  ipred_dc_left,  avx2);
    init_angular_ipred_fn(HOR_PRED,      ipred_h,        avx2);
    init_angular_ipred_fn(VERT_PRED,     ipred_v,        avx2);
    init_angular_ipred_fn(PAETH_PRED,    ipred_paeth,    avx2);
    init_angular_ipred_fn(SMOOTH_PRED,   ipred_smooth,   avx2);
    init_angular_ipred_fn(SMOOTH_H_PRED, ipred_smooth_h, avx2);
    init_angular_ipred_fn(SMOOTH_V_PRED, ipred_smooth_v, avx2);
    init_angular_ipred_fn(Z1_PRED,       ipred_z1,       avx2);
    init_angular_ipred_fn(Z2_PRED,       ipred_z2,       avx2);
    init_angular_ipred_fn(Z3_PRED,       ipred_z3,       avx2);
    init_angular_ipred_fn(FILTER_PRED,   ipred_filter,   avx2);

    init_cfl_pred_fn(DC_PRED,      ipred_cfl,      avx2);
    init_cfl_pred_fn(DC_128_PRED,  ipred_cfl_128,  avx2);
    init_cfl_pred_fn(TOP_DC_PRED,  ipred_cfl_top,  avx2);
    init_cfl_pred_fn(LEFT_DC_PRED, ipred_cfl_left, avx2);

    init_cfl_ac_fn(DAV1D_PIXEL_LAYOUT_I420 - 1, ipred_cfl_ac_420, avx2);
    init_cfl_ac_fn(DAV1D_PIXEL_LAYOUT_I422 - 1, ipred_cfl_ac_422, avx2);
    init_cfl_ac_fn(DAV1D_PIXEL_LAYOUT_I444 - 1, ipred_cfl_ac_444, avx2);

    c->pal_pred = BF(dav1d_pal_pred, avx2);

    if (!(flags & DAV1D_X86_CPU_FLAG_AVX512ICL)) return;

#if BITDEPTH == 8
    init_angular_ipred_fn(DC_PRED,       ipred_dc,       avx512icl);
    init_angular_ipred_fn(DC_128_PRED,   ipred_dc_128,   avx512icl);
    init_angular_ipred_fn(TOP_DC_PRED,   ipred_dc_top,   avx512icl);
    init_angular_ipred_fn(LEFT_DC_PRED,  ipred_dc_left,  avx512icl);
    init_angular_ipred_fn(HOR_PRED,      ipred_h,        avx512icl);
    init_angular_ipred_fn(VERT_PRED,     ipred_v,        avx512icl);
#endif
    init_angular_ipred_fn(PAETH_PRED,    ipred_paeth,    avx512icl);
    init_angular_ipred_fn(SMOOTH_PRED,   ipred_smooth,   avx512icl);
    init_angular_ipred_fn(SMOOTH_H_PRED, ipred_smooth_h, avx512icl);
    init_angular_ipred_fn(SMOOTH_V_PRED, ipred_smooth_v, avx512icl);
    init_angular_ipred_fn(FILTER_PRED,   ipred_filter,   avx512icl);

    c->pal_pred = BF(dav1d_pal_pred, avx512icl);
#endif
}

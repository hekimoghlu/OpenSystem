/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "src/itx.h"

#define decl_itx2_fns(w, h, opt) \
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_dct_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_identity_identity_##w##x##h, opt))

#define decl_itx12_fns(w, h, opt) \
decl_itx2_fns(w, h, opt); \
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_adst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_flipadst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_identity_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_adst_dct_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_adst_adst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_adst_flipadst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_flipadst_dct_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_flipadst_adst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_flipadst_flipadst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_identity_dct_##w##x##h, opt))

#define decl_itx16_fns(w, h, opt) \
decl_itx12_fns(w, h, opt); \
decl_itx_fn(BF(dav1d_inv_txfm_add_adst_identity_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_flipadst_identity_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_identity_adst_##w##x##h, opt)); \
decl_itx_fn(BF(dav1d_inv_txfm_add_identity_flipadst_##w##x##h, opt))

#define decl_itx17_fns(w, h, opt) \
decl_itx16_fns(w, h, opt); \
decl_itx_fn(BF(dav1d_inv_txfm_add_wht_wht_##w##x##h, opt))

decl_itx17_fns( 4,  4, neon);
decl_itx16_fns( 4,  8, neon);
decl_itx16_fns( 4, 16, neon);
decl_itx16_fns( 8,  4, neon);
decl_itx16_fns( 8,  8, neon);
decl_itx16_fns( 8, 16, neon);
decl_itx2_fns ( 8, 32, neon);
decl_itx16_fns(16,  4, neon);
decl_itx16_fns(16,  8, neon);
decl_itx12_fns(16, 16, neon);
decl_itx2_fns (16, 32, neon);
decl_itx2_fns (32,  8, neon);
decl_itx2_fns (32, 16, neon);
decl_itx2_fns (32, 32, neon);

decl_itx_fn(BF(dav1d_inv_txfm_add_dct_dct_16x64, neon));
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_dct_32x64, neon));
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_dct_64x16, neon));
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_dct_64x32, neon));
decl_itx_fn(BF(dav1d_inv_txfm_add_dct_dct_64x64, neon));

COLD void bitfn(dav1d_itx_dsp_init_arm)(Dav1dInvTxfmDSPContext *const c, int bpc) {
#define assign_itx_fn(pfx, w, h, type, type_enum, ext) \
    c->itxfm_add[pfx##TX_##w##X##h][type_enum] = \
        BF(dav1d_inv_txfm_add_##type##_##w##x##h, ext)

#define assign_itx1_fn(pfx, w, h, ext) \
    assign_itx_fn(pfx, w, h, dct_dct,           DCT_DCT,           ext)

#define assign_itx2_fn(pfx, w, h, ext) \
    assign_itx1_fn(pfx, w, h, ext); \
    assign_itx_fn(pfx, w, h, identity_identity, IDTX,              ext)

#define assign_itx12_fn(pfx, w, h, ext) \
    assign_itx2_fn(pfx, w, h, ext); \
    assign_itx_fn(pfx, w, h, dct_adst,          ADST_DCT,          ext); \
    assign_itx_fn(pfx, w, h, dct_flipadst,      FLIPADST_DCT,      ext); \
    assign_itx_fn(pfx, w, h, dct_identity,      H_DCT,             ext); \
    assign_itx_fn(pfx, w, h, adst_dct,          DCT_ADST,          ext); \
    assign_itx_fn(pfx, w, h, adst_adst,         ADST_ADST,         ext); \
    assign_itx_fn(pfx, w, h, adst_flipadst,     FLIPADST_ADST,     ext); \
    assign_itx_fn(pfx, w, h, flipadst_dct,      DCT_FLIPADST,      ext); \
    assign_itx_fn(pfx, w, h, flipadst_adst,     ADST_FLIPADST,     ext); \
    assign_itx_fn(pfx, w, h, flipadst_flipadst, FLIPADST_FLIPADST, ext); \
    assign_itx_fn(pfx, w, h, identity_dct,      V_DCT,             ext)

#define assign_itx16_fn(pfx, w, h, ext) \
    assign_itx12_fn(pfx, w, h, ext); \
    assign_itx_fn(pfx, w, h, adst_identity,     H_ADST,            ext); \
    assign_itx_fn(pfx, w, h, flipadst_identity, H_FLIPADST,        ext); \
    assign_itx_fn(pfx, w, h, identity_adst,     V_ADST,            ext); \
    assign_itx_fn(pfx, w, h, identity_flipadst, V_FLIPADST,        ext)

#define assign_itx17_fn(pfx, w, h, ext) \
    assign_itx16_fn(pfx, w, h, ext); \
    assign_itx_fn(pfx, w, h, wht_wht,           WHT_WHT,           ext)

    const unsigned flags = dav1d_get_cpu_flags();

    if (!(flags & DAV1D_ARM_CPU_FLAG_NEON)) return;

    if (bpc > 10) return;

    assign_itx17_fn( ,  4,  4, neon);
    assign_itx16_fn(R,  4,  8, neon);
    assign_itx16_fn(R,  4, 16, neon);
    assign_itx16_fn(R,  8,  4, neon);
    assign_itx16_fn( ,  8,  8, neon);
    assign_itx16_fn(R,  8, 16, neon);
    assign_itx2_fn (R,  8, 32, neon);
    assign_itx16_fn(R, 16,  4, neon);
    assign_itx16_fn(R, 16,  8, neon);
    assign_itx12_fn( , 16, 16, neon);
    assign_itx2_fn (R, 16, 32, neon);
    assign_itx1_fn (R, 16, 64, neon);
    assign_itx2_fn (R, 32,  8, neon);
    assign_itx2_fn (R, 32, 16, neon);
    assign_itx2_fn ( , 32, 32, neon);
    assign_itx1_fn (R, 32, 64, neon);
    assign_itx1_fn (R, 64, 16, neon);
    assign_itx1_fn (R, 64, 32, neon);
    assign_itx1_fn ( , 64, 64, neon);
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 12, 2024.
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
#ifndef VPX_VPX_DSP_LOONGARCH_TXFM_MACROS_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_TXFM_MACROS_LSX_H_

#include "vpx_util/loongson_intrinsics.h"

#define DOTP_CONST_PAIR(reg0, reg1, cnst0, cnst1, out0, out1)         \
  do {                                                                \
    __m128i s0_m, s1_m, s2_m, s3_m, s4_m, s5_m;                       \
    __m128i k0_m, k1_m, k2_m, k3_m;                                   \
                                                                      \
    k0_m = __lsx_vreplgr2vr_h(cnst0);                                 \
    k1_m = __lsx_vreplgr2vr_h(cnst1);                                 \
    k2_m = __lsx_vpackev_h(k1_m, k0_m);                               \
                                                                      \
    DUP2_ARG2(__lsx_vilvl_h, reg1, reg0, reg0, reg1, s5_m, s3_m);     \
    DUP2_ARG2(__lsx_vilvh_h, reg1, reg0, reg0, reg1, s4_m, s2_m);     \
                                                                      \
    DUP2_ARG2(__lsx_vmulwev_w_h, s5_m, k0_m, s4_m, k0_m, s1_m, s0_m); \
    k3_m = __lsx_vmulwod_w_h(s5_m, k1_m);                             \
    s1_m = __lsx_vsub_w(s1_m, k3_m);                                  \
    k3_m = __lsx_vmulwod_w_h(s4_m, k1_m);                             \
    s0_m = __lsx_vsub_w(s0_m, k3_m);                                  \
                                                                      \
    out0 = __lsx_vssrarni_h_w(s0_m, s1_m, DCT_CONST_BITS);            \
                                                                      \
    DUP2_ARG2(__lsx_vdp2_w_h, s3_m, k2_m, s2_m, k2_m, s1_m, s0_m);    \
    out1 = __lsx_vssrarni_h_w(s0_m, s1_m, DCT_CONST_BITS);            \
  } while (0)

#define DOT_SHIFT_RIGHT_PCK_H(in0, in1, in2, in3)                \
  do {                                                           \
    __m128i tp0_m, tp1_m;                                        \
                                                                 \
    DUP2_ARG2(__lsx_vdp2_w_h, in0, in2, in1, in2, tp1_m, tp0_m); \
    in3 = __lsx_vssrarni_h_w(tp1_m, tp0_m, DCT_CONST_BITS);      \
  } while (0)

#endif  // VPX_VPX_DSP_LOONGARCH_TXFM_MACROS_LSX_H_

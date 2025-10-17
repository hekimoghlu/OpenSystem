/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 18, 2024.
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
#ifndef VPX_VPX_DSP_LOONGARCH_VARIANCE_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_VARIANCE_LSX_H_

#include "vpx_util/loongson_intrinsics.h"

#define HADD_SW_S32(in0, in1)                  \
  do {                                         \
    __m128i res0_m;                            \
                                               \
    res0_m = __lsx_vhaddw_d_w(in0, in0);       \
    res0_m = __lsx_vhaddw_q_d(res0_m, res0_m); \
    in1 = __lsx_vpickve2gr_w(res0_m, 0);       \
  } while (0)

#define HORIZ_2TAP_FILT_UH(in0, in1, mask, coeff, shift, in2) \
  do {                                                        \
    __m128i tmp0_m, tmp1_m;                                   \
                                                              \
    tmp0_m = __lsx_vshuf_b(in1, in0, mask);                   \
    tmp1_m = __lsx_vdp2_h_bu(tmp0_m, coeff);                  \
    in2 = __lsx_vsrari_h(tmp1_m, shift);                      \
  } while (0)

#define CALC_MSE_B(src, ref, var)                                         \
  do {                                                                    \
    __m128i src_l0_m, src_l1_m;                                           \
    __m128i res_l0_m, res_l1_m;                                           \
                                                                          \
    src_l0_m = __lsx_vilvl_b(src, ref);                                   \
    src_l1_m = __lsx_vilvh_b(src, ref);                                   \
    DUP2_ARG2(__lsx_vhsubw_hu_bu, src_l0_m, src_l0_m, src_l1_m, src_l1_m, \
              res_l0_m, res_l1_m);                                        \
    var = __lsx_vdp2add_w_h(var, res_l0_m, res_l0_m);                     \
    var = __lsx_vdp2add_w_h(var, res_l1_m, res_l1_m);                     \
  } while (0)

#define CALC_MSE_AVG_B(src, ref, var, sub)                                \
  do {                                                                    \
    __m128i src_l0_m, src_l1_m;                                           \
    __m128i res_l0_m, res_l1_m;                                           \
                                                                          \
    src_l0_m = __lsx_vilvl_b(src, ref);                                   \
    src_l1_m = __lsx_vilvh_b(src, ref);                                   \
    DUP2_ARG2(__lsx_vhsubw_hu_bu, src_l0_m, src_l0_m, src_l1_m, src_l1_m, \
              res_l0_m, res_l1_m);                                        \
    var = __lsx_vdp2add_w_h(var, res_l0_m, res_l0_m);                     \
    var = __lsx_vdp2add_w_h(var, res_l1_m, res_l1_m);                     \
    sub = __lsx_vadd_h(sub, res_l0_m);                                    \
    sub = __lsx_vadd_h(sub, res_l1_m);                                    \
  } while (0)

#endif  // VPX_VPX_DSP_LOONGARCH_VARIANCE_LSX_H_

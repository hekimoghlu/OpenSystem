/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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
#ifndef VPX_VP9_ENCODER_MIPS_MSA_VP9_FDCT_MSA_H_
#define VPX_VP9_ENCODER_MIPS_MSA_VP9_FDCT_MSA_H_

#include "vpx_dsp/mips/fwd_txfm_msa.h"
#include "vpx_dsp/mips/txfm_macros_msa.h"
#include "vpx_ports/mem.h"

#define VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, out0, out1, out2,  \
                  out3, out4, out5, out6, out7)                              \
  {                                                                          \
    v8i16 cnst0_m, cnst1_m, cnst2_m, cnst3_m, cnst4_m;                       \
    v8i16 vec0_m, vec1_m, vec2_m, vec3_m, s0_m, s1_m;                        \
    v8i16 coeff0_m = { cospi_2_64,  cospi_6_64,  cospi_10_64, cospi_14_64,   \
                       cospi_18_64, cospi_22_64, cospi_26_64, cospi_30_64 }; \
    v8i16 coeff1_m = { cospi_8_64,  -cospi_8_64,  cospi_16_64, -cospi_16_64, \
                       cospi_24_64, -cospi_24_64, 0,           0 };          \
                                                                             \
    SPLATI_H2_SH(coeff0_m, 0, 7, cnst0_m, cnst1_m);                          \
    cnst2_m = -cnst0_m;                                                      \
    ILVEV_H2_SH(cnst0_m, cnst1_m, cnst1_m, cnst2_m, cnst0_m, cnst1_m);       \
    SPLATI_H2_SH(coeff0_m, 4, 3, cnst2_m, cnst3_m);                          \
    cnst4_m = -cnst2_m;                                                      \
    ILVEV_H2_SH(cnst2_m, cnst3_m, cnst3_m, cnst4_m, cnst2_m, cnst3_m);       \
                                                                             \
    ILVRL_H2_SH(in0, in7, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(in4, in3, vec3_m, vec2_m);                                   \
    DOT_ADD_SUB_SRARI_PCK(vec0_m, vec1_m, vec2_m, vec3_m, cnst0_m, cnst1_m,  \
                          cnst2_m, cnst3_m, in7, in0, in4, in3);             \
                                                                             \
    SPLATI_H2_SH(coeff0_m, 2, 5, cnst0_m, cnst1_m);                          \
    cnst2_m = -cnst0_m;                                                      \
    ILVEV_H2_SH(cnst0_m, cnst1_m, cnst1_m, cnst2_m, cnst0_m, cnst1_m);       \
    SPLATI_H2_SH(coeff0_m, 6, 1, cnst2_m, cnst3_m);                          \
    cnst4_m = -cnst2_m;                                                      \
    ILVEV_H2_SH(cnst2_m, cnst3_m, cnst3_m, cnst4_m, cnst2_m, cnst3_m);       \
                                                                             \
    ILVRL_H2_SH(in2, in5, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(in6, in1, vec3_m, vec2_m);                                   \
                                                                             \
    DOT_ADD_SUB_SRARI_PCK(vec0_m, vec1_m, vec2_m, vec3_m, cnst0_m, cnst1_m,  \
                          cnst2_m, cnst3_m, in5, in2, in6, in1);             \
    BUTTERFLY_4(in7, in0, in2, in5, s1_m, s0_m, in2, in5);                   \
    out7 = -s0_m;                                                            \
    out0 = s1_m;                                                             \
                                                                             \
    SPLATI_H4_SH(coeff1_m, 0, 4, 1, 5, cnst0_m, cnst1_m, cnst2_m, cnst3_m);  \
                                                                             \
    ILVEV_H2_SH(cnst3_m, cnst0_m, cnst1_m, cnst2_m, cnst3_m, cnst2_m);       \
    cnst0_m = __msa_ilvev_h(cnst1_m, cnst0_m);                               \
    cnst1_m = cnst0_m;                                                       \
                                                                             \
    ILVRL_H2_SH(in4, in3, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(in6, in1, vec3_m, vec2_m);                                   \
    DOT_ADD_SUB_SRARI_PCK(vec0_m, vec1_m, vec2_m, vec3_m, cnst0_m, cnst2_m,  \
                          cnst3_m, cnst1_m, out1, out6, s0_m, s1_m);         \
                                                                             \
    SPLATI_H2_SH(coeff1_m, 2, 3, cnst0_m, cnst1_m);                          \
    cnst1_m = __msa_ilvev_h(cnst1_m, cnst0_m);                               \
                                                                             \
    ILVRL_H2_SH(in2, in5, vec1_m, vec0_m);                                   \
    ILVRL_H2_SH(s0_m, s1_m, vec3_m, vec2_m);                                 \
    out3 = DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst0_m);                   \
    out4 = DOT_SHIFT_RIGHT_PCK_H(vec0_m, vec1_m, cnst1_m);                   \
    out2 = DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst0_m);                   \
    out5 = DOT_SHIFT_RIGHT_PCK_H(vec2_m, vec3_m, cnst1_m);                   \
                                                                             \
    out1 = -out1;                                                            \
    out3 = -out3;                                                            \
    out5 = -out5;                                                            \
  }

#define VP9_FADST4(in0, in1, in2, in3, out0, out1, out2, out3)              \
  {                                                                         \
    v4i32 s0_m, s1_m, s2_m, s3_m, constant_m;                               \
    v4i32 in0_r_m, in1_r_m, in2_r_m, in3_r_m;                               \
                                                                            \
    UNPCK_R_SH_SW(in0, in0_r_m);                                            \
    UNPCK_R_SH_SW(in1, in1_r_m);                                            \
    UNPCK_R_SH_SW(in2, in2_r_m);                                            \
    UNPCK_R_SH_SW(in3, in3_r_m);                                            \
                                                                            \
    constant_m = __msa_fill_w(sinpi_4_9);                                   \
    MUL2(in0_r_m, constant_m, in3_r_m, constant_m, s1_m, s0_m);             \
                                                                            \
    constant_m = __msa_fill_w(sinpi_1_9);                                   \
    s0_m += in0_r_m * constant_m;                                           \
    s1_m -= in1_r_m * constant_m;                                           \
                                                                            \
    constant_m = __msa_fill_w(sinpi_2_9);                                   \
    s0_m += in1_r_m * constant_m;                                           \
    s1_m += in3_r_m * constant_m;                                           \
                                                                            \
    s2_m = in0_r_m + in1_r_m - in3_r_m;                                     \
                                                                            \
    constant_m = __msa_fill_w(sinpi_3_9);                                   \
    MUL2(in2_r_m, constant_m, s2_m, constant_m, s3_m, in1_r_m);             \
                                                                            \
    in0_r_m = s0_m + s3_m;                                                  \
    s2_m = s1_m - s3_m;                                                     \
    s3_m = s1_m - s0_m + s3_m;                                              \
                                                                            \
    SRARI_W4_SW(in0_r_m, in1_r_m, s2_m, s3_m, DCT_CONST_BITS);              \
    PCKEV_H4_SH(in0_r_m, in0_r_m, in1_r_m, in1_r_m, s2_m, s2_m, s3_m, s3_m, \
                out0, out1, out2, out3);                                    \
  }
#endif  // VPX_VP9_ENCODER_MIPS_MSA_VP9_FDCT_MSA_H_

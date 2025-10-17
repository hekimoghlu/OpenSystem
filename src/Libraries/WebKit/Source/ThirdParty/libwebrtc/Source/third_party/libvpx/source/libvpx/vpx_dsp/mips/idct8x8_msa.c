/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/mips/inv_txfm_msa.h"

void vpx_idct8x8_64_add_msa(const int16_t *input, uint8_t *dst,
                            int32_t dst_stride) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;

  /* load vector elements of 8x8 block */
  LD_SH8(input, 8, in0, in1, in2, in3, in4, in5, in6, in7);

  /* rows transform */
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  /* 1D idct8x8 */
  VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                 in4, in5, in6, in7);
  /* columns transform */
  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  /* 1D idct8x8 */
  VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                 in4, in5, in6, in7);
  /* final rounding (add 2^4, divide by 2^5) and shift */
  SRARI_H4_SH(in0, in1, in2, in3, 5);
  SRARI_H4_SH(in4, in5, in6, in7, 5);
  /* add block and store 8x8 */
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in0, in1, in2, in3);
  dst += (4 * dst_stride);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in4, in5, in6, in7);
}

void vpx_idct8x8_12_add_msa(const int16_t *input, uint8_t *dst,
                            int32_t dst_stride) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;
  v8i16 s0, s1, s2, s3, s4, s5, s6, s7, k0, k1, k2, k3, m0, m1, m2, m3;
  v4i32 tmp0, tmp1, tmp2, tmp3;
  v8i16 zero = { 0 };

  /* load vector elements of 8x8 block */
  LD_SH8(input, 8, in0, in1, in2, in3, in4, in5, in6, in7);
  TRANSPOSE8X4_SH_SH(in0, in1, in2, in3, in0, in1, in2, in3);

  /* stage1 */
  ILVL_H2_SH(in3, in0, in2, in1, s0, s1);
  k0 = VP9_SET_COSPI_PAIR(cospi_28_64, -cospi_4_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_4_64, cospi_28_64);
  k2 = VP9_SET_COSPI_PAIR(-cospi_20_64, cospi_12_64);
  k3 = VP9_SET_COSPI_PAIR(cospi_12_64, cospi_20_64);
  DOTP_SH4_SW(s0, s0, s1, s1, k0, k1, k2, k3, tmp0, tmp1, tmp2, tmp3);
  SRARI_W4_SW(tmp0, tmp1, tmp2, tmp3, DCT_CONST_BITS);
  PCKEV_H2_SH(zero, tmp0, zero, tmp1, s0, s1);
  PCKEV_H2_SH(zero, tmp2, zero, tmp3, s2, s3);
  BUTTERFLY_4(s0, s1, s3, s2, s4, s7, s6, s5);

  /* stage2 */
  ILVR_H2_SH(in3, in1, in2, in0, s1, s0);
  k0 = VP9_SET_COSPI_PAIR(cospi_16_64, cospi_16_64);
  k1 = VP9_SET_COSPI_PAIR(cospi_16_64, -cospi_16_64);
  k2 = VP9_SET_COSPI_PAIR(cospi_24_64, -cospi_8_64);
  k3 = VP9_SET_COSPI_PAIR(cospi_8_64, cospi_24_64);
  DOTP_SH4_SW(s0, s0, s1, s1, k0, k1, k2, k3, tmp0, tmp1, tmp2, tmp3);
  SRARI_W4_SW(tmp0, tmp1, tmp2, tmp3, DCT_CONST_BITS);
  PCKEV_H2_SH(zero, tmp0, zero, tmp1, s0, s1);
  PCKEV_H2_SH(zero, tmp2, zero, tmp3, s2, s3);
  BUTTERFLY_4(s0, s1, s2, s3, m0, m1, m2, m3);

  /* stage3 */
  s0 = __msa_ilvr_h(s6, s5);

  k1 = VP9_SET_COSPI_PAIR(-cospi_16_64, cospi_16_64);
  DOTP_SH2_SW(s0, s0, k1, k0, tmp0, tmp1);
  SRARI_W2_SW(tmp0, tmp1, DCT_CONST_BITS);
  PCKEV_H2_SH(zero, tmp0, zero, tmp1, s2, s3);

  /* stage4 */
  BUTTERFLY_8(m0, m1, m2, m3, s4, s2, s3, s7, in0, in1, in2, in3, in4, in5, in6,
              in7);
  TRANSPOSE4X8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  VP9_IDCT8x8_1D(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                 in4, in5, in6, in7);

  /* final rounding (add 2^4, divide by 2^5) and shift */
  SRARI_H4_SH(in0, in1, in2, in3, 5);
  SRARI_H4_SH(in4, in5, in6, in7, 5);

  /* add block and store 8x8 */
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in0, in1, in2, in3);
  dst += (4 * dst_stride);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, in4, in5, in6, in7);
}

void vpx_idct8x8_1_add_msa(const int16_t *input, uint8_t *dst,
                           int32_t dst_stride) {
  int16_t out;
  int32_t val;
  v8i16 vec;

  out = ROUND_POWER_OF_TWO((input[0] * cospi_16_64), DCT_CONST_BITS);
  out = ROUND_POWER_OF_TWO((out * cospi_16_64), DCT_CONST_BITS);
  val = ROUND_POWER_OF_TWO(out, 5);
  vec = __msa_fill_h(val);

  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, vec, vec, vec, vec);
  dst += (4 * dst_stride);
  VP9_ADDBLK_ST8x4_UB(dst, dst_stride, vec, vec, vec, vec);
}

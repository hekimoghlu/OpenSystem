/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 18, 2023.
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
#include "./vp8_rtcd.h"
#include "vpx_util/loongson_intrinsics.h"
#include "vp8/encoder/block.h"

int32_t vp8_block_error_lsx(int16_t *coeff_ptr, int16_t *dq_coeff_ptr) {
  int32_t err = 0;
  __m128i dq_coeff0, dq_coeff1, coeff0, coeff1;
  __m128i reg0, reg1, reg2, reg3, error;

  DUP4_ARG2(__lsx_vld, coeff_ptr, 0, coeff_ptr, 16, dq_coeff_ptr, 0,
            dq_coeff_ptr, 16, coeff0, coeff1, dq_coeff0, dq_coeff1);
  DUP2_ARG2(__lsx_vsubwev_w_h, coeff0, dq_coeff0, coeff1, dq_coeff1, reg0,
            reg2);
  DUP2_ARG2(__lsx_vsubwod_w_h, coeff0, dq_coeff0, coeff1, dq_coeff1, reg1,
            reg3);
  error = __lsx_vmul_w(reg0, reg0);
  DUP2_ARG3(__lsx_vmadd_w, error, reg1, reg1, error, reg2, reg2, error, error);
  error = __lsx_vmadd_w(error, reg3, reg3);
  error = __lsx_vhaddw_d_w(error, error);
  err = __lsx_vpickve2gr_w(error, 0);
  err += __lsx_vpickve2gr_w(error, 2);
  return err;
}

int32_t vp8_mbblock_error_lsx(MACROBLOCK *mb, int32_t dc) {
  BLOCK *be;
  BLOCKD *bd;
  int16_t *coeff, *dq_coeff;
  int32_t err = 0;
  uint32_t loop_cnt;
  __m128i src0, src1, src2, src3;
  __m128i tmp0, tmp1, tmp2, tmp3;
  __m128i reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7, error;
  __m128i mask0 = __lsx_vldi(0xFF);
  __m128i zero = __lsx_vldi(0);

  if (dc == 1) {
    mask0 = __lsx_vinsgr2vr_w(mask0, 0, 0);
  }

  for (loop_cnt = 0; loop_cnt < 8; loop_cnt++) {
    int32_t loop_tmp = loop_cnt << 1;
    be = &mb->block[loop_tmp];
    bd = &mb->e_mbd.block[loop_tmp];
    coeff = be->coeff;
    dq_coeff = bd->dqcoeff;
    DUP4_ARG2(__lsx_vld, coeff, 0, coeff, 16, dq_coeff, 0, dq_coeff, 16, src0,
              src1, tmp0, tmp1);
    be = &mb->block[loop_tmp + 1];
    bd = &mb->e_mbd.block[loop_tmp + 1];
    coeff = be->coeff;
    dq_coeff = bd->dqcoeff;
    DUP4_ARG2(__lsx_vld, coeff, 0, coeff, 16, dq_coeff, 0, dq_coeff, 16, src2,
              src3, tmp2, tmp3);
    DUP4_ARG2(__lsx_vsubwev_w_h, src0, tmp0, src1, tmp1, src2, tmp2, src3, tmp3,
              reg0, reg2, reg4, reg6);
    DUP4_ARG2(__lsx_vsubwod_w_h, src0, tmp0, src1, tmp1, src2, tmp2, src3, tmp3,
              reg1, reg3, reg5, reg7);
    DUP2_ARG3(__lsx_vbitsel_v, zero, reg0, mask0, zero, reg4, mask0, reg0,
              reg4);
    error = __lsx_vmul_w(reg0, reg0);
    DUP4_ARG3(__lsx_vmadd_w, error, reg1, reg1, error, reg2, reg2, error, reg3,
              reg3, error, reg4, reg4, error, error, error, error);
    DUP2_ARG3(__lsx_vmadd_w, error, reg5, reg5, error, reg6, reg6, error,
              error);
    error = __lsx_vmadd_w(error, reg7, reg7);
    error = __lsx_vhaddw_d_w(error, error);
    error = __lsx_vhaddw_q_d(error, error);
    err += __lsx_vpickve2gr_w(error, 0);
  }
  return err;
}

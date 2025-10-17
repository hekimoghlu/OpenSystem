/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 3, 2024.
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
#include <assert.h>

#include "./vp9_rtcd.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/ppc/inv_txfm_vsx.h"
#include "vpx_dsp/ppc/bitdepth_conversion_vsx.h"

#include "vp9/common/vp9_enums.h"

void vp9_iht4x4_16_add_vsx(const tran_low_t *input, uint8_t *dest, int stride,
                           int tx_type) {
  int16x8_t in[2], out[2];

  in[0] = load_tran_low(0, input);
  in[1] = load_tran_low(8 * sizeof(*input), input);

  switch (tx_type) {
    case DCT_DCT:
      vpx_idct4_vsx(in, out);
      vpx_idct4_vsx(out, in);
      break;
    case ADST_DCT:
      vpx_idct4_vsx(in, out);
      vp9_iadst4_vsx(out, in);
      break;
    case DCT_ADST:
      vp9_iadst4_vsx(in, out);
      vpx_idct4_vsx(out, in);
      break;
    default:
      assert(tx_type == ADST_ADST);
      vp9_iadst4_vsx(in, out);
      vp9_iadst4_vsx(out, in);
      break;
  }

  vpx_round_store4x4_vsx(in, out, dest, stride);
}

void vp9_iht8x8_64_add_vsx(const tran_low_t *input, uint8_t *dest, int stride,
                           int tx_type) {
  int16x8_t in[8], out[8];

  // load input data
  in[0] = load_tran_low(0, input);
  in[1] = load_tran_low(8 * sizeof(*input), input);
  in[2] = load_tran_low(2 * 8 * sizeof(*input), input);
  in[3] = load_tran_low(3 * 8 * sizeof(*input), input);
  in[4] = load_tran_low(4 * 8 * sizeof(*input), input);
  in[5] = load_tran_low(5 * 8 * sizeof(*input), input);
  in[6] = load_tran_low(6 * 8 * sizeof(*input), input);
  in[7] = load_tran_low(7 * 8 * sizeof(*input), input);

  switch (tx_type) {
    case DCT_DCT:
      vpx_idct8_vsx(in, out);
      vpx_idct8_vsx(out, in);
      break;
    case ADST_DCT:
      vpx_idct8_vsx(in, out);
      vp9_iadst8_vsx(out, in);
      break;
    case DCT_ADST:
      vp9_iadst8_vsx(in, out);
      vpx_idct8_vsx(out, in);
      break;
    default:
      assert(tx_type == ADST_ADST);
      vp9_iadst8_vsx(in, out);
      vp9_iadst8_vsx(out, in);
      break;
  }

  vpx_round_store8x8_vsx(in, dest, stride);
}

void vp9_iht16x16_256_add_vsx(const tran_low_t *input, uint8_t *dest,
                              int stride, int tx_type) {
  int16x8_t in0[16], in1[16];

  LOAD_INPUT16(load_tran_low, input, 0, 8 * sizeof(*input), in0);
  LOAD_INPUT16(load_tran_low, input, 8 * 8 * 2 * sizeof(*input),
               8 * sizeof(*input), in1);

  switch (tx_type) {
    case DCT_DCT:
      vpx_idct16_vsx(in0, in1);
      vpx_idct16_vsx(in0, in1);
      break;
    case ADST_DCT:
      vpx_idct16_vsx(in0, in1);
      vpx_iadst16_vsx(in0, in1);
      break;
    case DCT_ADST:
      vpx_iadst16_vsx(in0, in1);
      vpx_idct16_vsx(in0, in1);
      break;
    default:
      assert(tx_type == ADST_ADST);
      vpx_iadst16_vsx(in0, in1);
      vpx_iadst16_vsx(in0, in1);
      break;
  }

  vpx_round_store16x16_vsx(in0, in1, dest, stride);
}

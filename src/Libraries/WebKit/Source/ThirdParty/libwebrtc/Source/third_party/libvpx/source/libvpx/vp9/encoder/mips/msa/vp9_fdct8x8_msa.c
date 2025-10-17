/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 23, 2023.
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
#include "vp9/common/vp9_enums.h"
#include "vp9/encoder/mips/msa/vp9_fdct_msa.h"

void vp9_fht8x8_msa(const int16_t *input, int16_t *output, int32_t stride,
                    int32_t tx_type) {
  v8i16 in0, in1, in2, in3, in4, in5, in6, in7;

  LD_SH8(input, stride, in0, in1, in2, in3, in4, in5, in6, in7);
  SLLI_4V(in0, in1, in2, in3, 2);
  SLLI_4V(in4, in5, in6, in7, 2);

  switch (tx_type) {
    case DCT_DCT:
      VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      break;
    case ADST_DCT:
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      break;
    case DCT_ADST:
      VP9_FDCT8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      break;
    case ADST_ADST:
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2,
                         in3, in4, in5, in6, in7);
      VP9_ADST8(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3, in4,
                in5, in6, in7);
      break;
    default: assert(0); break;
  }

  TRANSPOSE8x8_SH_SH(in0, in1, in2, in3, in4, in5, in6, in7, in0, in1, in2, in3,
                     in4, in5, in6, in7);
  SRLI_AVE_S_4V_H(in0, in1, in2, in3, in4, in5, in6, in7);
  ST_SH8(in0, in1, in2, in3, in4, in5, in6, in7, output, 8);
}

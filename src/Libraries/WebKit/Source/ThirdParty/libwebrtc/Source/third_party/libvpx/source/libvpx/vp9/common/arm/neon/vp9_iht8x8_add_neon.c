/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 13, 2023.
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
#include <arm_neon.h>
#include <assert.h>

#include "./vp9_rtcd.h"
#include "./vpx_config.h"
#include "vp9/common/vp9_common.h"
#include "vp9/common/arm/neon/vp9_iht_neon.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"

void vp9_iht8x8_64_add_neon(const tran_low_t *input, uint8_t *dest, int stride,
                            int tx_type) {
  const int16x8_t cospis = vld1q_s16(kCospi);
  const int16x4_t cospis0 = vget_low_s16(cospis);   // cospi 0, 8, 16, 24
  const int16x4_t cospis1 = vget_high_s16(cospis);  // cospi 4, 12, 20, 28
  int16x8_t a[8];

  a[0] = load_tran_low_to_s16q(input + 0 * 8);
  a[1] = load_tran_low_to_s16q(input + 1 * 8);
  a[2] = load_tran_low_to_s16q(input + 2 * 8);
  a[3] = load_tran_low_to_s16q(input + 3 * 8);
  a[4] = load_tran_low_to_s16q(input + 4 * 8);
  a[5] = load_tran_low_to_s16q(input + 5 * 8);
  a[6] = load_tran_low_to_s16q(input + 6 * 8);
  a[7] = load_tran_low_to_s16q(input + 7 * 8);

  transpose_s16_8x8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);

  switch (tx_type) {
    case DCT_DCT:
      idct8x8_64_1d_bd8_kernel(cospis0, cospis1, a);
      transpose_s16_8x8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
      idct8x8_64_1d_bd8_kernel(cospis0, cospis1, a);
      break;

    case ADST_DCT:
      idct8x8_64_1d_bd8_kernel(cospis0, cospis1, a);
      transpose_s16_8x8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
      iadst8(a);
      break;

    case DCT_ADST:
      iadst8(a);
      transpose_s16_8x8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
      idct8x8_64_1d_bd8_kernel(cospis0, cospis1, a);
      break;

    default:
      assert(tx_type == ADST_ADST);
      iadst8(a);
      transpose_s16_8x8(&a[0], &a[1], &a[2], &a[3], &a[4], &a[5], &a[6], &a[7]);
      iadst8(a);
      break;
  }

  idct8x8_add8x8_neon(a, dest, stride);
}

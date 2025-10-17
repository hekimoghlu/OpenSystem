/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 23, 2023.
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

#include "./vpx_config.h"
#include "./vpx_dsp_rtcd.h"
#include "vpx_dsp/arm/idct_neon.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/transpose_neon.h"
#include "vpx_dsp/txfm_common.h"

void vpx_idct8x8_64_add_neon(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const int16x8_t cospis = vld1q_s16(kCospi);
  const int16x4_t cospis0 = vget_low_s16(cospis);   // cospi 0, 8, 16, 24
  const int16x4_t cospis1 = vget_high_s16(cospis);  // cospi 4, 12, 20, 28
  int16x8_t a[8];

  a[0] = load_tran_low_to_s16q(input);
  a[1] = load_tran_low_to_s16q(input + 8);
  a[2] = load_tran_low_to_s16q(input + 16);
  a[3] = load_tran_low_to_s16q(input + 24);
  a[4] = load_tran_low_to_s16q(input + 32);
  a[5] = load_tran_low_to_s16q(input + 40);
  a[6] = load_tran_low_to_s16q(input + 48);
  a[7] = load_tran_low_to_s16q(input + 56);

  idct8x8_64_1d_bd8(cospis0, cospis1, a);
  idct8x8_64_1d_bd8(cospis0, cospis1, a);
  idct8x8_add8x8_neon(a, dest, stride);
}

void vpx_idct8x8_12_add_neon(const tran_low_t *input, uint8_t *dest,
                             int stride) {
  const int16x8_t cospis = vld1q_s16(kCospi);
  const int16x8_t cospisd = vaddq_s16(cospis, cospis);
  const int16x4_t cospis0 = vget_low_s16(cospis);     // cospi 0, 8, 16, 24
  const int16x4_t cospisd0 = vget_low_s16(cospisd);   // doubled 0, 8, 16, 24
  const int16x4_t cospisd1 = vget_high_s16(cospisd);  // doubled 4, 12, 20, 28
  int16x4_t a[8];
  int16x8_t b[8];

  a[0] = load_tran_low_to_s16d(input);
  a[1] = load_tran_low_to_s16d(input + 8);
  a[2] = load_tran_low_to_s16d(input + 16);
  a[3] = load_tran_low_to_s16d(input + 24);

  idct8x8_12_pass1_bd8(cospis0, cospisd0, cospisd1, a);
  idct8x8_12_pass2_bd8(cospis0, cospisd0, cospisd1, a, b);
  idct8x8_add8x8_neon(b, dest, stride);
}

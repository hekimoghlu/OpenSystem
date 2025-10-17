/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 9, 2023.
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

#include "config/av1_rtcd.h"

#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"

void av1_round_shift_array_neon(int32_t *arr, int size, int bit) {
  assert(!(size % 4));
  if (!bit) return;
  const int32x4_t dup_bits_n_32x4 = vdupq_n_s32((int32_t)(-bit));
  for (int i = 0; i < size; i += 4) {
    int32x4_t tmp_q_s32 = vld1q_s32(arr);
    tmp_q_s32 = vrshlq_s32(tmp_q_s32, dup_bits_n_32x4);
    vst1q_s32(arr, tmp_q_s32);
    arr += 4;
  }
}

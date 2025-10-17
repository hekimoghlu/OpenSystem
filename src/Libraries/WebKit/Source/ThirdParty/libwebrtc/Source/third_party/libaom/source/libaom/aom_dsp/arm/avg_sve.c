/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 16, 2022.
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

#include "config/aom_config.h"
#include "config/aom_dsp_rtcd.h"
#include "aom/aom_integer.h"
#include "aom_dsp/arm/aom_neon_sve_bridge.h"
#include "aom_dsp/arm/mem_neon.h"
#include "aom_ports/mem.h"

int aom_vector_var_sve(const int16_t *ref, const int16_t *src, int bwl) {
  assert(bwl >= 2 && bwl <= 5);
  int width = 4 << bwl;

  int64x2_t sse_s64[2] = { vdupq_n_s64(0), vdupq_n_s64(0) };
  int16x8_t v_mean[2] = { vdupq_n_s16(0), vdupq_n_s16(0) };

  do {
    int16x8_t r0 = vld1q_s16(ref);
    int16x8_t s0 = vld1q_s16(src);

    // diff: dynamic range [-510, 510] 10 (signed) bits.
    int16x8_t diff0 = vsubq_s16(r0, s0);
    // v_mean: dynamic range 16 * diff -> [-8160, 8160], 14 (signed) bits.
    v_mean[0] = vaddq_s16(v_mean[0], diff0);

    // v_sse: dynamic range 2 * 16 * diff^2 -> [0, 8,323,200], 24 (signed) bits.
    sse_s64[0] = aom_sdotq_s16(sse_s64[0], diff0, diff0);

    int16x8_t r1 = vld1q_s16(ref + 8);
    int16x8_t s1 = vld1q_s16(src + 8);

    // diff: dynamic range [-510, 510] 10 (signed) bits.
    int16x8_t diff1 = vsubq_s16(r1, s1);
    // v_mean: dynamic range 16 * diff -> [-8160, 8160], 14 (signed) bits.
    v_mean[1] = vaddq_s16(v_mean[1], diff1);

    // v_sse: dynamic range 2 * 16 * diff^2 -> [0, 8,323,200], 24 (signed) bits.
    sse_s64[1] = aom_sdotq_s16(sse_s64[1], diff1, diff1);

    ref += 16;
    src += 16;
    width -= 16;
  } while (width != 0);

  // Dynamic range [0, 65280], 16 (unsigned) bits.
  const uint32_t mean_abs = abs(vaddlvq_s16(vaddq_s16(v_mean[0], v_mean[1])));
  const int64_t sse = vaddvq_s64(vaddq_s64(sse_s64[0], sse_s64[1]));

  // (mean_abs * mean_abs): dynamic range 32 (unsigned) bits.
  return (int)(sse - ((mean_abs * mean_abs) >> (bwl + 2)));
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 26, 2025.
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
#include "./vp8_rtcd.h"

static INLINE void vp8_loop_filter_simple_horizontal_edge_neon(
    unsigned char *s, int p, const unsigned char *blimit) {
  uint8_t *sp;
  uint8x16_t qblimit, q0u8;
  uint8x16_t q5u8, q6u8, q7u8, q8u8, q9u8, q10u8, q14u8, q15u8;
  int16x8_t q2s16, q3s16, q13s16;
  int8x8_t d8s8, d9s8;
  int8x16_t q2s8, q3s8, q4s8, q10s8, q11s8, q14s8;

  qblimit = vdupq_n_u8(*blimit);

  sp = s - (p << 1);
  q5u8 = vld1q_u8(sp);
  sp += p;
  q6u8 = vld1q_u8(sp);
  sp += p;
  q7u8 = vld1q_u8(sp);
  sp += p;
  q8u8 = vld1q_u8(sp);

  q15u8 = vabdq_u8(q6u8, q7u8);
  q14u8 = vabdq_u8(q5u8, q8u8);

  q15u8 = vqaddq_u8(q15u8, q15u8);
  q14u8 = vshrq_n_u8(q14u8, 1);
  q0u8 = vdupq_n_u8(0x80);
  q13s16 = vdupq_n_s16(3);
  q15u8 = vqaddq_u8(q15u8, q14u8);

  q5u8 = veorq_u8(q5u8, q0u8);
  q6u8 = veorq_u8(q6u8, q0u8);
  q7u8 = veorq_u8(q7u8, q0u8);
  q8u8 = veorq_u8(q8u8, q0u8);

  q15u8 = vcgeq_u8(qblimit, q15u8);

  q2s16 = vsubl_s8(vget_low_s8(vreinterpretq_s8_u8(q7u8)),
                   vget_low_s8(vreinterpretq_s8_u8(q6u8)));
  q3s16 = vsubl_s8(vget_high_s8(vreinterpretq_s8_u8(q7u8)),
                   vget_high_s8(vreinterpretq_s8_u8(q6u8)));

  q4s8 = vqsubq_s8(vreinterpretq_s8_u8(q5u8), vreinterpretq_s8_u8(q8u8));

  q2s16 = vmulq_s16(q2s16, q13s16);
  q3s16 = vmulq_s16(q3s16, q13s16);

  q10u8 = vdupq_n_u8(3);
  q9u8 = vdupq_n_u8(4);

  q2s16 = vaddw_s8(q2s16, vget_low_s8(q4s8));
  q3s16 = vaddw_s8(q3s16, vget_high_s8(q4s8));

  d8s8 = vqmovn_s16(q2s16);
  d9s8 = vqmovn_s16(q3s16);
  q4s8 = vcombine_s8(d8s8, d9s8);

  q14s8 = vandq_s8(q4s8, vreinterpretq_s8_u8(q15u8));

  q2s8 = vqaddq_s8(q14s8, vreinterpretq_s8_u8(q10u8));
  q3s8 = vqaddq_s8(q14s8, vreinterpretq_s8_u8(q9u8));
  q2s8 = vshrq_n_s8(q2s8, 3);
  q3s8 = vshrq_n_s8(q3s8, 3);

  q11s8 = vqaddq_s8(vreinterpretq_s8_u8(q6u8), q2s8);
  q10s8 = vqsubq_s8(vreinterpretq_s8_u8(q7u8), q3s8);

  q6u8 = veorq_u8(vreinterpretq_u8_s8(q11s8), q0u8);
  q7u8 = veorq_u8(vreinterpretq_u8_s8(q10s8), q0u8);

  vst1q_u8(s, q7u8);
  s -= p;
  vst1q_u8(s, q6u8);
  return;
}

void vp8_loop_filter_bhs_neon(unsigned char *y_ptr, int y_stride,
                              const unsigned char *blimit) {
  y_ptr += y_stride * 4;
  vp8_loop_filter_simple_horizontal_edge_neon(y_ptr, y_stride, blimit);
  y_ptr += y_stride * 4;
  vp8_loop_filter_simple_horizontal_edge_neon(y_ptr, y_stride, blimit);
  y_ptr += y_stride * 4;
  vp8_loop_filter_simple_horizontal_edge_neon(y_ptr, y_stride, blimit);
  return;
}

void vp8_loop_filter_mbhs_neon(unsigned char *y_ptr, int y_stride,
                               const unsigned char *blimit) {
  vp8_loop_filter_simple_horizontal_edge_neon(y_ptr, y_stride, blimit);
  return;
}

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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

#include "./vpx_dsp_rtcd.h"
#include "./vpx_config.h"

#include "vpx/vpx_integer.h"
#include "vpx_dsp/arm/mem_neon.h"
#include "vpx_dsp/arm/sum_neon.h"
#include "vpx_ports/mem.h"

static INLINE uint32_t highbd_mse8_8xh_neon_dotprod(const uint16_t *src_ptr,
                                                    int src_stride,
                                                    const uint16_t *ref_ptr,
                                                    int ref_stride, int h) {
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h / 2;
  do {
    uint16x8_t s0, s1, r0, r1;
    uint8x16_t s, r, diff;

    s0 = vld1q_u16(src_ptr);
    src_ptr += src_stride;
    s1 = vld1q_u16(src_ptr);
    src_ptr += src_stride;
    r0 = vld1q_u16(ref_ptr);
    ref_ptr += ref_stride;
    r1 = vld1q_u16(ref_ptr);
    ref_ptr += ref_stride;

    s = vcombine_u8(vmovn_u16(s0), vmovn_u16(s1));
    r = vcombine_u8(vmovn_u16(r0), vmovn_u16(r1));

    diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, diff, diff);
  } while (--i != 0);

  return horizontal_add_uint32x4(sse_u32);
}

static INLINE uint32_t highbd_mse8_16xh_neon_dotprod(const uint16_t *src_ptr,
                                                     int src_stride,
                                                     const uint16_t *ref_ptr,
                                                     int ref_stride, int h) {
  uint32x4_t sse_u32 = vdupq_n_u32(0);

  int i = h;
  do {
    uint16x8_t s0, s1, r0, r1;
    uint8x16_t s, r, diff;

    s0 = vld1q_u16(src_ptr);
    s1 = vld1q_u16(src_ptr + 8);
    r0 = vld1q_u16(ref_ptr);
    r1 = vld1q_u16(ref_ptr + 8);

    s = vcombine_u8(vmovn_u16(s0), vmovn_u16(s1));
    r = vcombine_u8(vmovn_u16(r0), vmovn_u16(r1));

    diff = vabdq_u8(s, r);
    sse_u32 = vdotq_u32(sse_u32, diff, diff);

    src_ptr += src_stride;
    ref_ptr += ref_stride;
  } while (--i != 0);

  return horizontal_add_uint32x4(sse_u32);
}

#define HIGHBD_MSE_WXH_NEON_DOTPROD(w, h)                                      \
  uint32_t vpx_highbd_8_mse##w##x##h##_neon_dotprod(                           \
      const uint8_t *src_ptr, int src_stride, const uint8_t *ref_ptr,          \
      int ref_stride, uint32_t *sse) {                                         \
    uint16_t *src = CONVERT_TO_SHORTPTR(src_ptr);                              \
    uint16_t *ref = CONVERT_TO_SHORTPTR(ref_ptr);                              \
    *sse =                                                                     \
        highbd_mse8_##w##xh_neon_dotprod(src, src_stride, ref, ref_stride, h); \
    return *sse;                                                               \
  }

HIGHBD_MSE_WXH_NEON_DOTPROD(16, 16)
HIGHBD_MSE_WXH_NEON_DOTPROD(16, 8)
HIGHBD_MSE_WXH_NEON_DOTPROD(8, 16)
HIGHBD_MSE_WXH_NEON_DOTPROD(8, 8)

#undef HIGHBD_MSE_WXH_NEON_DOTPROD

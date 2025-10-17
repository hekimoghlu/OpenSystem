/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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
#ifndef VPX_VPX_DSP_PPC_BITDEPTH_CONVERSION_VSX_H_
#define VPX_VPX_DSP_PPC_BITDEPTH_CONVERSION_VSX_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_dsp/ppc/types_vsx.h"

// Load 8 16 bit values. If the source is 32 bits then pack down with
// saturation.
static INLINE int16x8_t load_tran_low(int32_t c, const tran_low_t *s) {
#if CONFIG_VP9_HIGHBITDEPTH
  int32x4_t u = vec_vsx_ld(c, s);
  int32x4_t v = vec_vsx_ld(c, s + 4);
  return vec_packs(u, v);
#else
  return vec_vsx_ld(c, s);
#endif
}

// Store 8 16 bit values. If the destination is 32 bits then sign extend the
// values by multiplying by 1.
static INLINE void store_tran_low(int16x8_t v, int32_t c, tran_low_t *s) {
#if CONFIG_VP9_HIGHBITDEPTH
  const int16x8_t one = vec_splat_s16(1);
  const int32x4_t even = vec_mule(v, one);
  const int32x4_t odd = vec_mulo(v, one);
  const int32x4_t high = vec_mergeh(even, odd);
  const int32x4_t low = vec_mergel(even, odd);
  vec_vsx_st(high, c, s);
  vec_vsx_st(low, c, s + 4);
#else
  vec_vsx_st(v, c, s);
#endif
}

#endif  // VPX_VPX_DSP_PPC_BITDEPTH_CONVERSION_VSX_H_

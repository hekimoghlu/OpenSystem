/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 20, 2024.
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
#ifndef VPX_VPX_DSP_LOONGARCH_BITDEPTH_CONVERSION_LSX_H_
#define VPX_VPX_DSP_LOONGARCH_BITDEPTH_CONVERSION_LSX_H_

#include "./vpx_config.h"
#include "vpx/vpx_integer.h"
#include "vpx_dsp/vpx_dsp_common.h"
#include "vpx_util/loongson_intrinsics.h"

static INLINE __m128i load_tran_low(const tran_low_t *s) {
#if CONFIG_VP9_HIGHBITDEPTH
  __m128i v0_m = __lsx_vld(s, 0);
  __m128i v1_m = __lsx_vld(s + 4, 0);
  return __lsx_vsrlni_h_w(v0_m, v1_m, 0);
#else
  return __lsx_vld(s, 0);
#endif
}

static INLINE void store_tran_low(__m128i v, tran_low_t *s, int32_t c) {
#if CONFIG_VP9_HIGHBITDEPTH
  __m128i v0_m, v1_m;
  v1_m = __lsx_vexth_w_h(v);
  v0_m = __lsx_vsllwil_w_h(v, 0);
  __lsx_vst(v0_m, s + c, 0);
  __lsx_vst(v1_m, s + c + 4, 0);
#else
  __lsx_vst(v, s + c, 0);
#endif
}

#endif  // VPX_VPX_DSP_LOONGARCH_BITDEPTH_CONVERSION_LSX_H_

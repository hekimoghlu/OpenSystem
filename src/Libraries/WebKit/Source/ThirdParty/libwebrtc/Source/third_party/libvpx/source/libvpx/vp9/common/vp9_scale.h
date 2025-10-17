/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#ifndef VPX_VP9_COMMON_VP9_SCALE_H_
#define VPX_VP9_COMMON_VP9_SCALE_H_

#include "vp9/common/vp9_mv.h"
#include "vpx_dsp/vpx_convolve.h"

#ifdef __cplusplus
extern "C" {
#endif

#define REF_SCALE_SHIFT 14
#define REF_NO_SCALE (1 << REF_SCALE_SHIFT)
#define REF_INVALID_SCALE (-1)

struct scale_factors {
  int x_scale_fp;  // horizontal fixed point scale factor
  int y_scale_fp;  // vertical fixed point scale factor
  int x_step_q4;
  int y_step_q4;

  int (*scale_value_x)(int val, const struct scale_factors *sf);
  int (*scale_value_y)(int val, const struct scale_factors *sf);

  convolve_fn_t predict[2][2][2];  // horiz, vert, avg
#if CONFIG_VP9_HIGHBITDEPTH
  highbd_convolve_fn_t highbd_predict[2][2][2];  // horiz, vert, avg
#endif
};

MV32 vp9_scale_mv(const MV *mv, int x, int y, const struct scale_factors *sf);

#if CONFIG_VP9_HIGHBITDEPTH
void vp9_setup_scale_factors_for_frame(struct scale_factors *sf, int other_w,
                                       int other_h, int this_w, int this_h,
                                       int use_highbd);
#else
void vp9_setup_scale_factors_for_frame(struct scale_factors *sf, int other_w,
                                       int other_h, int this_w, int this_h);
#endif

static INLINE int vp9_is_valid_scale(const struct scale_factors *sf) {
  return sf->x_scale_fp != REF_INVALID_SCALE &&
         sf->y_scale_fp != REF_INVALID_SCALE;
}

static INLINE int vp9_is_scaled(const struct scale_factors *sf) {
  return vp9_is_valid_scale(sf) &&
         (sf->x_scale_fp != REF_NO_SCALE || sf->y_scale_fp != REF_NO_SCALE);
}

static INLINE int valid_ref_frame_size(int ref_width, int ref_height,
                                       int this_width, int this_height) {
  return 2 * this_width >= ref_width && 2 * this_height >= ref_height &&
         this_width <= 16 * ref_width && this_height <= 16 * ref_height;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_COMMON_VP9_SCALE_H_

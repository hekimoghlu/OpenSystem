/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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
#ifndef AOM_AV1_COMMON_SCALE_H_
#define AOM_AV1_COMMON_SCALE_H_

#include "av1/common/convolve.h"
#include "av1/common/mv.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SCALE_NUMERATOR 8

#define REF_SCALE_SHIFT 14
#define REF_NO_SCALE (1 << REF_SCALE_SHIFT)
#define REF_INVALID_SCALE -1

struct scale_factors {
  int x_scale_fp;  // horizontal fixed point scale factor
  int y_scale_fp;  // vertical fixed point scale factor
  int x_step_q4;
  int y_step_q4;
};

// Note: Expect val to be in q4 precision
static inline int av1_scaled_x(int val, const struct scale_factors *sf) {
  const int off =
      (sf->x_scale_fp - (1 << REF_SCALE_SHIFT)) * (1 << (SUBPEL_BITS - 1));
  const int64_t tval = (int64_t)val * sf->x_scale_fp + off;
  return (int)ROUND_POWER_OF_TWO_SIGNED_64(tval,
                                           REF_SCALE_SHIFT - SCALE_EXTRA_BITS);
}

// Note: Expect val to be in q4 precision
static inline int av1_scaled_y(int val, const struct scale_factors *sf) {
  const int off =
      (sf->y_scale_fp - (1 << REF_SCALE_SHIFT)) * (1 << (SUBPEL_BITS - 1));
  const int64_t tval = (int64_t)val * sf->y_scale_fp + off;
  return (int)ROUND_POWER_OF_TWO_SIGNED_64(tval,
                                           REF_SCALE_SHIFT - SCALE_EXTRA_BITS);
}

// Note: Expect val to be in q4 precision
static inline int av1_unscaled_value(int val, const struct scale_factors *sf) {
  (void)sf;
  return val * (1 << SCALE_EXTRA_BITS);
}

MV32 av1_scale_mv(const MV *mv, int x, int y, const struct scale_factors *sf);

void av1_setup_scale_factors_for_frame(struct scale_factors *sf, int other_w,
                                       int other_h, int this_w, int this_h);

static inline int av1_is_valid_scale(const struct scale_factors *sf) {
  assert(sf != NULL);
  return sf->x_scale_fp != REF_INVALID_SCALE &&
         sf->y_scale_fp != REF_INVALID_SCALE;
}

static inline int av1_is_scaled(const struct scale_factors *sf) {
  assert(sf != NULL);
  return av1_is_valid_scale(sf) &&
         (sf->x_scale_fp != REF_NO_SCALE || sf->y_scale_fp != REF_NO_SCALE);
}

// See AV1 spec, Section 6.8.6. Frame size with refs semantics.
static inline int valid_ref_frame_size(int ref_width, int ref_height,
                                       int this_width, int this_height) {
  return 2 * this_width >= ref_width && 2 * this_height >= ref_height &&
         this_width <= 16 * ref_width && this_height <= 16 * ref_height;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_AV1_COMMON_SCALE_H_

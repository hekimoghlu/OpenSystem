/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#ifndef VPX_VP8_ENCODER_RATECTRL_H_
#define VPX_VP8_ENCODER_RATECTRL_H_

#include "onyx_int.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void vp8_save_coding_context(VP8_COMP *cpi);
extern void vp8_restore_coding_context(VP8_COMP *cpi);

extern void vp8_setup_key_frame(VP8_COMP *cpi);
extern void vp8_update_rate_correction_factors(VP8_COMP *cpi, int damp_var);
extern int vp8_regulate_q(VP8_COMP *cpi, int target_bits_per_frame);
extern void vp8_adjust_key_frame_context(VP8_COMP *cpi);
extern void vp8_compute_frame_size_bounds(VP8_COMP *cpi,
                                          int *frame_under_shoot_limit,
                                          int *frame_over_shoot_limit);

/* return of 0 means drop frame */
extern int vp8_pick_frame_size(VP8_COMP *cpi);

extern int vp8_drop_encodedframe_overshoot(VP8_COMP *cpi, int Q);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_RATECTRL_H_

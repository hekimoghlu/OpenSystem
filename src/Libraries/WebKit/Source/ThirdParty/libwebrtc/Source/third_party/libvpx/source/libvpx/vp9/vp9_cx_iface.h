/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 30, 2023.
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
#ifndef VPX_VP9_VP9_CX_IFACE_H_
#define VPX_VP9_VP9_CX_IFACE_H_
#include "vp9/encoder/vp9_encoder.h"
#include "vp9/common/vp9_onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

VP9EncoderConfig vp9_get_encoder_config(int frame_width, int frame_height,
                                        vpx_rational_t frame_rate,
                                        int target_bitrate, int encode_speed,
                                        int target_level,
                                        vpx_enc_pass enc_pass);

void vp9_dump_encoder_config(const VP9EncoderConfig *oxcf, FILE *fp);

FRAME_INFO vp9_get_frame_info(const VP9EncoderConfig *oxcf);

static INLINE int64_t
timebase_units_to_ticks(const vpx_rational64_t *timestamp_ratio, int64_t n) {
  return n * timestamp_ratio->num / timestamp_ratio->den;
}

static INLINE int64_t
ticks_to_timebase_units(const vpx_rational64_t *timestamp_ratio, int64_t n) {
  int64_t round = timestamp_ratio->num / 2;
  if (round > 0) --round;
  return (n * timestamp_ratio->den + round) / timestamp_ratio->num;
}

void vp9_set_first_pass_stats(VP9EncoderConfig *oxcf,
                              const vpx_fixed_buf_t *stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_VP9_CX_IFACE_H_

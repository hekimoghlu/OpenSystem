/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 25, 2022.
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
#ifndef VPX_VP9_DECODER_VP9_DECODEFRAME_H_
#define VPX_VP9_DECODER_VP9_DECODEFRAME_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/common/vp9_enums.h"

struct VP9Decoder;
struct vpx_read_bit_buffer;

int vp9_read_sync_code(struct vpx_read_bit_buffer *const rb);
void vp9_read_frame_size(struct vpx_read_bit_buffer *rb, int *width,
                         int *height);
BITSTREAM_PROFILE vp9_read_profile(struct vpx_read_bit_buffer *rb);

void vp9_decode_frame(struct VP9Decoder *pbi, const uint8_t *data,
                      const uint8_t *data_end, const uint8_t **p_data_end);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_DECODER_VP9_DECODEFRAME_H_

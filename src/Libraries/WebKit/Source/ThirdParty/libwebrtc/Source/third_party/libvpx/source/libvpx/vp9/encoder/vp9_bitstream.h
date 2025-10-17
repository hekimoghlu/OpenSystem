/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#ifndef VPX_VP9_ENCODER_VP9_BITSTREAM_H_
#define VPX_VP9_ENCODER_VP9_BITSTREAM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/encoder/vp9_encoder.h"

typedef struct VP9BitstreamWorkerData {
  uint8_t *dest;
  size_t dest_size;
  vpx_writer bit_writer;
  int tile_idx;
  unsigned int max_mv_magnitude;
  // The size of interp_filter_selected in VP9_COMP is actually
  // MAX_REFERENCE_FRAMES x SWITCHABLE. But when encoding tiles, all we ever do
  // is increment the very first index (index 0) for the first dimension. Hence
  // this is sufficient.
  int interp_filter_selected[1][SWITCHABLE];
  DECLARE_ALIGNED(16, MACROBLOCKD, xd);
} VP9BitstreamWorkerData;

int vp9_get_refresh_mask(VP9_COMP *cpi);

void vp9_bitstream_encode_tiles_buffer_dealloc(VP9_COMP *const cpi);

void vp9_pack_bitstream(VP9_COMP *cpi, uint8_t *dest, size_t dest_size,
                        size_t *size);

static INLINE int vp9_preserve_existing_gf(VP9_COMP *cpi) {
  return cpi->refresh_golden_frame && cpi->rc.is_src_frame_alt_ref &&
         !cpi->use_svc;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_BITSTREAM_H_

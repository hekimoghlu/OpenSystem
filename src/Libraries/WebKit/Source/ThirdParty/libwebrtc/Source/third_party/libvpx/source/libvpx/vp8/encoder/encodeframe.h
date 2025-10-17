/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 18, 2025.
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
#ifndef VPX_VP8_ENCODER_ENCODEFRAME_H_
#define VPX_VP8_ENCODER_ENCODEFRAME_H_

#include "vp8/encoder/tokenize.h"

#ifdef __cplusplus
extern "C" {
#endif

struct VP8_COMP;
struct macroblock;

void vp8_activity_masking(struct VP8_COMP *cpi, MACROBLOCK *x);

void vp8_build_block_offsets(struct macroblock *x);

void vp8_setup_block_ptrs(struct macroblock *x);

void vp8_encode_frame(struct VP8_COMP *cpi);

int vp8cx_encode_inter_macroblock(struct VP8_COMP *cpi, struct macroblock *x,
                                  TOKENEXTRA **t, int recon_yoffset,
                                  int recon_uvoffset, int mb_row, int mb_col);

int vp8cx_encode_intra_macroblock(struct VP8_COMP *cpi, struct macroblock *x,
                                  TOKENEXTRA **t);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_ENCODEFRAME_H_

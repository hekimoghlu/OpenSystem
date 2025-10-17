/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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
#ifndef VPX_VP9_ENCODER_VP9_AQ_VARIANCE_H_
#define VPX_VP9_ENCODER_VP9_AQ_VARIANCE_H_

#include "vp9/encoder/vp9_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

unsigned int vp9_vaq_segment_id(int energy);
void vp9_vaq_frame_setup(VP9_COMP *cpi);

void vp9_get_sub_block_energy(VP9_COMP *cpi, MACROBLOCK *mb, int mi_row,
                              int mi_col, BLOCK_SIZE bsize, int *min_e,
                              int *max_e);
int vp9_block_energy(VP9_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bs);

double vp9_log_block_var(VP9_COMP *cpi, MACROBLOCK *x, BLOCK_SIZE bs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_AQ_VARIANCE_H_

/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 19, 2022.
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
#ifndef VPX_VP8_ENCODER_BITSTREAM_H_
#define VPX_VP8_ENCODER_BITSTREAM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp8/encoder/treewriter.h"
#include "vp8/encoder/tokenize.h"

void vp8_pack_tokens(vp8_writer *w, const TOKENEXTRA *p, int xcount);
void vp8_convert_rfct_to_prob(struct VP8_COMP *const cpi);
void vp8_calc_ref_frame_costs(int *ref_frame_cost, int prob_intra,
                              int prob_last, int prob_garf);
int vp8_estimate_entropy_savings(struct VP8_COMP *cpi);
void vp8_update_coef_probs(struct VP8_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_BITSTREAM_H_

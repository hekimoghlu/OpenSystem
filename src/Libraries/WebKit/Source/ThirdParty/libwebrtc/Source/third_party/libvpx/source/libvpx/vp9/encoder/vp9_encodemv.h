/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 13, 2022.
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
#ifndef VPX_VP9_ENCODER_VP9_ENCODEMV_H_
#define VPX_VP9_ENCODER_VP9_ENCODEMV_H_

#include "vp9/encoder/vp9_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

void vp9_entropy_mv_init(void);

void vp9_write_nmv_probs(VP9_COMMON *cm, int usehp, vpx_writer *w,
                         nmv_context_counts *const counts);

void vp9_encode_mv(VP9_COMP *cpi, vpx_writer *w, const MV *mv, const MV *ref,
                   const nmv_context *mvctx, int usehp,
                   unsigned int *const max_mv_magnitude);

void vp9_build_nmv_cost_table(int *mvjoint, int *mvcost[2],
                              const nmv_context *ctx, int usehp);

void vp9_update_mv_count(ThreadData *td);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP9_ENCODER_VP9_ENCODEMV_H_

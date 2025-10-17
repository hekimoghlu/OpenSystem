/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#ifndef VPX_VP8_ENCODER_PICKINTER_H_
#define VPX_VP8_ENCODER_PICKINTER_H_
#include "vpx_config.h"
#include "vp8/common/onyxc_int.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void vp8_pick_inter_mode(VP8_COMP *cpi, MACROBLOCK *x, int recon_yoffset,
                                int recon_uvoffset, int *returnrate,
                                int *returndistortion, int *returnintra,
                                int mb_row, int mb_col);
extern void vp8_pick_intra_mode(MACROBLOCK *x, int *rate);

extern int vp8_get_inter_mbpred_error(MACROBLOCK *mb,
                                      const vp8_variance_fn_ptr_t *vfp,
                                      unsigned int *sse, int_mv this_mv);
#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_ENCODER_PICKINTER_H_

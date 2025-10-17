/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
#ifndef VPX_VP8_DECODER_ERROR_CONCEALMENT_H_
#define VPX_VP8_DECODER_ERROR_CONCEALMENT_H_

#include "onyxd_int.h"
#include "ec_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocate memory for the overlap lists */
int vp8_alloc_overlap_lists(VP8D_COMP *pbi);

/* Deallocate the overlap lists */
void vp8_de_alloc_overlap_lists(VP8D_COMP *pbi);

/* Estimate all missing motion vectors. */
void vp8_estimate_missing_mvs(VP8D_COMP *pbi);

/* Functions for spatial MV interpolation */

/* Interpolates all motion vectors for a macroblock mb at position
 * (mb_row, mb_col). */
void vp8_interpolate_motion(MACROBLOCKD *mb, int mb_row, int mb_col,
                            int mb_rows, int mb_cols);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VP8_DECODER_ERROR_CONCEALMENT_H_
